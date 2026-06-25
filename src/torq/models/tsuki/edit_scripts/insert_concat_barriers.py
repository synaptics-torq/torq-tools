#!/usr/bin/env python3
"""Insert Pad+Slice fusion barriers after Concat ops with mixed input sizes.

When TileAndFuse fuses a Concat (with dissimilar input dim sizes) with downstream
Mul/Add ops, it can trigger a rank mismatch assertion in AddOp::getKernelEncoding().
This script inserts a Pad(1)+Slice barrier on the Concat output, which prevents
the fusion without changing semantics.

Targets: Concat ops where the concatenated dimension has inputs of different sizes
(e.g., 256, 32, 1, 1 → 290). Equal-size Concat ops are not affected.

Usage:
    python3 scripts/insert_concat_barriers.py -i model.onnx -o model_barriers.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    return None


def _get_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    return TensorProto.BFLOAT16


def find_mixed_concats(model):
    """Find Concat ops where inputs have different sizes on the concat axis."""
    nodes = list(model.graph.node)
    targets = []

    for i, n in enumerate(nodes):
        if n.op_type != "Concat":
            continue

        axis = 0
        for attr in n.attribute:
            if attr.name == "axis":
                axis = attr.i

        input_shapes = [_get_shape(model, inp) for inp in n.input]
        if any(s is None for s in input_shapes):
            continue

        concat_dim_sizes = [s[axis] for s in input_shapes]
        if len(set(concat_dim_sizes)) <= 1:
            continue

        out_shape = _get_shape(model, n.output[0])
        if out_shape is None:
            continue

        targets.append({
            "node_idx": i,
            "node_name": n.name,
            "tensor_name": n.output[0],
            "shape": out_shape,
            "concat_dim_sizes": concat_dim_sizes,
            "axis": axis,
        })

    return targets


def insert_pad_slice_barrier(model, target):
    """Insert Pad(1,dim=0)+Slice(1:2,dim=0) barrier on a tensor."""
    graph = model.graph
    nodes = list(graph.node)

    tensor_name = target["tensor_name"]
    shape = target["shape"]
    dtype = _get_dtype(model, tensor_name)

    padded_name = f"{tensor_name}__concat_padded"
    barrier_name = f"{tensor_name}__concat_barrier"

    ndim = len(shape)
    pad_values = [1] + [0] * (ndim - 1) + [0] * ndim
    pads_init = numpy_helper.from_array(
        np.array(pad_values, dtype=np.int64), name=f"{tensor_name}__concat_barrier_pads")
    graph.initializer.append(pads_init)

    pad_node = helper.make_node(
        "Pad", [tensor_name, f"{tensor_name}__concat_barrier_pads"], [padded_name],
        name=f"{tensor_name}__concat_barrier_pad", mode="constant")

    padded_shape = [shape[0] + 1] + shape[1:]
    graph.value_info.append(helper.make_tensor_value_info(padded_name, dtype, padded_shape))

    starts_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=f"{tensor_name}__concat_barrier_starts")
    ends_init = numpy_helper.from_array(
        np.array([shape[0] + 1], dtype=np.int64), name=f"{tensor_name}__concat_barrier_ends")
    axes_init = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name=f"{tensor_name}__concat_barrier_axes")
    graph.initializer.append(starts_init)
    graph.initializer.append(ends_init)
    graph.initializer.append(axes_init)

    slice_node = helper.make_node(
        "Slice",
        [padded_name, f"{tensor_name}__concat_barrier_starts",
         f"{tensor_name}__concat_barrier_ends", f"{tensor_name}__concat_barrier_axes"],
        [barrier_name],
        name=f"{tensor_name}__concat_barrier_slice")
    graph.value_info.append(helper.make_tensor_value_info(barrier_name, dtype, shape))

    for n in nodes:
        for j, inp in enumerate(n.input):
            if inp == tensor_name and tensor_name not in n.output:
                n.input[j] = barrier_name

    for i, n in enumerate(nodes):
        if tensor_name in n.output:
            nodes.insert(i + 1, pad_node)
            nodes.insert(i + 2, slice_node)
            break

    del graph.node[:]
    graph.node.extend(nodes)


def main():
    parser = argparse.ArgumentParser(description="Insert barriers after mixed-size Concat ops")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--name-prefix", default=None,
                        help="Only target Concat nodes whose name starts with this prefix")
    parser.add_argument("--dry-run", action="store_true", help="Just print targets")
    args = parser.parse_args()

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    targets = find_mixed_concats(model)
    if args.name_prefix:
        targets = [t for t in targets if t["node_name"].startswith(args.name_prefix)]
    print(f"Found {len(targets)} mixed-size Concat op(s):")
    for t in targets:
        print(f"  [{t['node_idx']}] {t['node_name']}: "
              f"axis={t['axis']} sizes={t['concat_dim_sizes']} → {t['shape']}")

    if args.dry_run:
        return

    for t in reversed(targets):
        print(f"  Inserting barrier on {t['tensor_name']}")
        insert_pad_slice_barrier(model, t)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed: {e}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
