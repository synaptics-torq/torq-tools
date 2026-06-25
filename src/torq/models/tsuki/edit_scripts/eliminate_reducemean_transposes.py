#!/usr/bin/env python3
"""Eliminate ReduceMean axis transposes.

Replaces the pattern:
  Transpose(0,2,1) -> ReduceMean(axis=-1, keepdims=0) -> Unsqueeze -> Transpose(0,2,1)
with:
  ReduceMean(axis=1, keepdims=1)

This eliminates 2 Transpose + 1 Unsqueeze per ReduceMean, saving host dispatch overhead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def eliminate_reducemean_transposes(model):
    graph = model.graph

    output_to_node = {}
    for n in graph.node:
        for o in n.output:
            if o:
                output_to_node[o] = n

    input_to_nodes = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                input_to_nodes.setdefault(inp, []).append(n)

    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        t = vi.type.tensor_type
        if t.HasField('shape'):
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in t.shape.dim]
            shapes[vi.name] = dims

    inits = {init.name: init for init in graph.initializer}

    axis_to_last = [n for n in graph.node
                    if n.op_type == 'Transpose' and 'axis_to_last' in n.name]

    nodes_to_remove = set()
    replacement_map = {}  # id(first_removed_node) -> list of replacement nodes
    new_initializers = []
    new_value_info = []
    count = 0

    for atl in axis_to_last:
        perm = tuple(atl.attribute[0].ints) if atl.attribute else None
        if perm != (0, 2, 1):
            continue

        consumers = input_to_nodes.get(atl.output[0], [])
        if len(consumers) != 1 or consumers[0].op_type != 'ReduceMean':
            continue
        rm = consumers[0]

        rm_consumers = input_to_nodes.get(rm.output[0], [])
        if len(rm_consumers) != 1 or rm_consumers[0].op_type != 'Unsqueeze':
            continue
        unsq = rm_consumers[0]

        unsq_consumers = input_to_nodes.get(unsq.output[0], [])
        if len(unsq_consumers) != 1 or unsq_consumers[0].op_type != 'Transpose':
            continue
        restore = unsq_consumers[0]

        restore_perm = tuple(restore.attribute[0].ints) if restore.attribute else None
        if restore_perm != (0, 2, 1):
            continue
        if 'restore_axis' not in restore.name:
            continue

        ncl_input = atl.input[0]
        final_output = restore.output[0]
        inp_shape = shapes.get(ncl_input, None)
        if inp_shape is None or len(inp_shape) != 3:
            print(f"  SKIP {atl.name}: no shape for {ncl_input}")
            continue

        B, C, L = inp_shape
        out_shape = [B, 1, L]

        prefix = atl.name.replace('_axis_to_last', '')
        axes_name = f"{prefix}_reduce_axis_1"
        new_initializers.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=axes_name
        ))

        new_rm = helper.make_node(
            "ReduceMean",
            [ncl_input, axes_name],
            [final_output],
            name=f"{prefix}_reducemean_ncl",
            keepdims=1,
        )
        replacement_map[id(atl)] = [new_rm]

        new_value_info.append(helper.make_tensor_value_info(
            final_output, TensorProto.BFLOAT16, out_shape
        ))

        nodes_to_remove.add(id(atl))
        nodes_to_remove.add(id(rm))
        nodes_to_remove.add(id(unsq))
        nodes_to_remove.add(id(restore))
        count += 1

    if count == 0:
        return model, 0

    all_nodes = []
    for n in graph.node:
        if id(n) in nodes_to_remove:
            if id(n) in replacement_map:
                all_nodes.extend(replacement_map[id(n)])
            continue
        all_nodes.append(n)

    graph.ClearField("node")
    graph.node.extend(all_nodes)

    for init in new_initializers:
        graph.initializer.append(init)

    existing_vi = {vi.name for vi in graph.value_info}
    for vi in new_value_info:
        if vi.name not in existing_vi:
            graph.value_info.append(vi)

    # Clean up value_info for removed intermediate tensors
    removed_outputs = set()
    for n in [atl, rm, unsq]:
        for o in n.output:
            if o and o != final_output:
                removed_outputs.add(o)

    return model, count


def main():
    parser = argparse.ArgumentParser(description="Eliminate ReduceMean axis transposes")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    model = onnx.load(args.input)
    print(f"Nodes before: {len(model.graph.node)}")

    model, count = eliminate_reducemean_transposes(model)

    print(f"Eliminated {count} ReduceMean transpose chains ({count * 4} nodes removed, {count} added)")
    print(f"Nodes after: {len(model.graph.node)}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
