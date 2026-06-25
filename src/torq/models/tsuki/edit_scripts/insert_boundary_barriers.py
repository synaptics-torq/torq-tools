#!/usr/bin/env python3
"""Insert Pad+Slice fusion barriers on specified tensor names.

Used to prevent TileAndFuse from creating fusion groups that span
the encoder-decoder boundary in an unsplit model.
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _get_shape_and_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            dtype = item.type.tensor_type.elem_type
            if item.type.tensor_type.HasField("shape"):
                shape = [d.dim_value for d in item.type.tensor_type.shape.dim]
                return shape, dtype
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims), init.data_type
    return None, None


def insert_barriers(model, tensor_names):
    graph = model.graph
    nodes = list(graph.node)
    inserted = 0

    for tname in tensor_names:
        shape, dtype = _get_shape_and_dtype(model, tname)
        if shape is None:
            print(f"  Skipping {tname}: shape not found")
            continue
        if all(d <= 1 for d in shape):
            continue

        consumers = [n for n in nodes if tname in list(n.input)]
        if not consumers:
            continue

        ndim = len(shape)
        if ndim == 0:
            continue

        padded_name = f"{tname}__boundary_padded"
        barrier_name = f"{tname}__boundary_barrier"

        pad_values = [1] + [0] * (ndim - 1) + [0] * ndim
        pads_const = f"{tname}__boundary_pads"
        graph.initializer.append(numpy_helper.from_array(
            np.array(pad_values, dtype=np.int64), name=pads_const))

        pad_node = helper.make_node(
            "Pad", [tname, pads_const], [padded_name],
            name=f"{tname}__boundary_pad", mode="constant")

        padded_shape = [shape[0] + 1] + shape[1:]
        graph.value_info.append(
            helper.make_tensor_value_info(padded_name, dtype, padded_shape))

        starts_name = f"{tname}__boundary_starts"
        ends_name = f"{tname}__boundary_ends"
        axes_name = f"{tname}__boundary_axes"
        graph.initializer.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=starts_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([shape[0] + 1], dtype=np.int64), name=ends_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=axes_name))

        slice_node = helper.make_node(
            "Slice", [padded_name, starts_name, ends_name, axes_name],
            [barrier_name], name=f"{tname}__boundary_slice")

        graph.value_info.append(
            helper.make_tensor_value_info(barrier_name, dtype, shape))

        for n in nodes:
            new_inputs = []
            for inp in n.input:
                new_inputs.append(barrier_name if inp == tname else inp)
            del n.input[:]
            n.input.extend(new_inputs)

        producer_idx = -1
        for i, n in enumerate(nodes):
            if tname in list(n.output):
                producer_idx = i
                break

        if producer_idx >= 0:
            nodes.insert(producer_idx + 1, pad_node)
            nodes.insert(producer_idx + 2, slice_node)
        else:
            nodes.insert(0, pad_node)
            nodes.insert(1, slice_node)

        inserted += 1

    del graph.node[:]
    graph.node.extend(nodes)
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--tensor-names", nargs="+", help="Tensor names to barrier")
    parser.add_argument("--tensor-names-file", help="File with one tensor name per line")
    parser.add_argument("--from-split-model", help="Get boundary tensor names from split model inputs")
    args = parser.parse_args()

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    tensor_names = []
    if args.tensor_names:
        tensor_names = args.tensor_names
    if args.tensor_names_file:
        with open(args.tensor_names_file) as f:
            tensor_names.extend(line.strip() for line in f if line.strip())
    if args.from_split_model:
        split_model = onnx.load(args.from_split_model)
        split_inits = set(init.name for init in split_model.graph.initializer)
        graph_inputs = set(inp.name for inp in model.graph.input)
        model_inits = set(init.name for init in model.graph.initializer)
        node_outputs = set()
        for node in model.graph.node:
            for o in node.output:
                node_outputs.add(o)
        for inp in split_model.graph.input:
            if inp.name not in split_inits and inp.name in node_outputs:
                tensor_names.append(inp.name)

    print(f"Inserting barriers on {len(tensor_names)} boundary tensors")
    count = insert_barriers(model, tensor_names)
    print(f"Inserted {count} barriers")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
