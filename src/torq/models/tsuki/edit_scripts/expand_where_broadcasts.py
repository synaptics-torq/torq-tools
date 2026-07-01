#!/usr/bin/env python3
"""Expand Where op broadcast inputs to eliminate implicit broadcasting.

The torq compiler has a bug where Where ops with mixed broadcast shapes
(e.g., cond=[1,9], then=[1,1], else=[1,9]) incorrectly broadcast all
inputs as scalars. This script explicitly adds Expand ops before Where
so all three inputs have the same shape.
"""
import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import argparse
import sys


def get_shapes(model):
    shapes = {}
    for vi in model.graph.value_info:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for vi in model.graph.input:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for vi in model.graph.output:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return shapes


def get_dtypes(model):
    dtypes = {}
    for vi in model.graph.value_info:
        dtypes[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.input:
        dtypes[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.output:
        dtypes[vi.name] = vi.type.tensor_type.elem_type
    return dtypes


def broadcast_shape(shapes_list):
    """Compute the broadcast output shape from a list of shapes."""
    max_ndim = max(len(s) for s in shapes_list if s)
    result = [1] * max_ndim
    for shape in shapes_list:
        if not shape:
            continue
        padded = [1] * (max_ndim - len(shape)) + list(shape)
        for i in range(max_ndim):
            if padded[i] != 1:
                result[i] = padded[i]
    return result


def expand_where_broadcasts(model):
    shapes = get_shapes(model)
    dtypes = get_dtypes(model)
    graph = model.graph

    new_nodes = []
    new_value_info = []
    new_initializers = []
    expanded_count = 0

    for node in graph.node:
        if node.op_type != 'Where':
            new_nodes.append(node)
            continue

        cond_name, then_name, else_name = node.input[0], node.input[1], node.input[2]
        cond_shape = shapes.get(cond_name, [])
        then_shape = shapes.get(then_name, [])
        else_shape = shapes.get(else_name, [])

        if not cond_shape and not then_shape and not else_shape:
            new_nodes.append(node)
            continue

        target_shape = broadcast_shape([s for s in [cond_shape, then_shape, else_shape] if s])

        needs_expand = False
        new_inputs = list(node.input)

        for idx, (inp_name, inp_shape) in enumerate([
            (cond_name, cond_shape),
            (then_name, then_shape),
            (else_name, else_shape),
        ]):
            if not inp_shape or inp_shape == target_shape:
                continue

            needs_expand = True
            expanded_name = f"{inp_name}__where_expanded"
            shape_const_name = f"{inp_name}__where_target_shape"

            new_initializers.append(numpy_helper.from_array(
                np.array(target_shape, dtype=np.int64), shape_const_name))

            expand_node = helper.make_node(
                "Expand", [inp_name, shape_const_name], [expanded_name],
                name=f"expand_where_{node.name}_{idx}")
            new_nodes.append(expand_node)

            dt = dtypes.get(inp_name, TensorProto.FLOAT)
            new_value_info.append(helper.make_tensor_value_info(
                expanded_name, dt, target_shape))

            new_inputs[idx] = expanded_name
            expanded_count += 1

        if needs_expand:
            new_where = helper.make_node(
                "Where", new_inputs, list(node.output), name=node.name)
            for attr in node.attribute:
                new_where.attribute.append(attr)
            new_nodes.append(new_where)
            print(f"  Expanded {node.name}: cond={cond_shape} then={then_shape} "
                  f"else={else_shape} -> {target_shape}")
        else:
            new_nodes.append(node)

    if expanded_count == 0:
        print("No Where broadcasts found.")
        return model, 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.value_info.extend(new_value_info)
    graph.initializer.extend(new_initializers)

    return model, expanded_count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    args = p.parse_args()

    print(f"Loading {args.input}...")
    model = onnx.load(args.input)

    model, count = expand_where_broadcasts(model)
    print(f"Expanded {count} Where broadcast inputs")

    print(f"Saving {args.output}...")
    onnx.save(model, args.output)
    print("Done.")


if __name__ == '__main__':
    main()
