#!/usr/bin/env python3
"""Split Part B into B1 and B2 at the boundary tensors.

B1: everything that produces `full_static_tail_gate_1172_cat_206` and `slice_1245`
B2: everything downstream, plus pass-through inputs (x20_context, x20_mask_axis2, x20_length)

Usage:
    python3 scripts/split_part_b.py -i part_b.onnx -o /tmp/part_b_split/
"""
import argparse
import os
from collections import defaultdict

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

BOUNDARY_TENSORS = {"full_static_tail_gate_1172_cat_206", "slice_1245"}


def get_value_info(name, graph, inferred_shapes):
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name == name:
            return vi
    if name in inferred_shapes:
        return inferred_shapes[name]
    for init in graph.initializer:
        if init.name == name:
            return helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)


def split_part_b(model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model = onnx.load(model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    graph = model.graph

    inferred_shapes = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        inferred_shapes[vi.name] = vi

    all_nodes = list(graph.node)
    init_names = {i.name for i in graph.initializer}
    init_map = {i.name: i for i in graph.initializer}

    output_to_node = {}
    for i, node in enumerate(all_nodes):
        for out in node.output:
            output_to_node[out] = i

    # BFS backward from boundary tensors to find all B1 nodes
    b1_node_indices = set()
    queue = list(BOUNDARY_TENSORS)
    visited = set()
    while queue:
        tensor_name = queue.pop()
        if tensor_name in visited:
            continue
        visited.add(tensor_name)
        if tensor_name in output_to_node:
            idx = output_to_node[tensor_name]
            if idx not in b1_node_indices:
                b1_node_indices.add(idx)
                for inp in all_nodes[idx].input:
                    if inp and inp not in init_names:
                        queue.append(inp)

    b2_node_indices = set(range(len(all_nodes))) - b1_node_indices

    # Collect tensors produced by B1 that are consumed by B2 (boundary tensors)
    b1_outputs_set = set()
    for idx in b1_node_indices:
        for out in all_nodes[idx].output:
            b1_outputs_set.add(out)

    b2_inputs_from_b1 = set()
    for idx in b2_node_indices:
        for inp in all_nodes[idx].input:
            if inp and inp in b1_outputs_set:
                b2_inputs_from_b1.add(inp)

    # Build B1
    b1_nodes = [all_nodes[i] for i in sorted(b1_node_indices)]
    b1_all_inputs = set()
    b1_all_outputs = set()
    for n in b1_nodes:
        for inp in n.input:
            if inp:
                b1_all_inputs.add(inp)
        for out in n.output:
            b1_all_outputs.add(out)

    b1_graph_inputs = []
    b1_inits = []
    for name in sorted(b1_all_inputs - b1_all_outputs):
        if name in init_names:
            b1_inits.append(init_map[name])
        else:
            vi = get_value_info(name, graph, inferred_shapes)
            b1_graph_inputs.append(vi)

    b1_graph_outputs = []
    for name in sorted(b2_inputs_from_b1):
        vi = get_value_info(name, graph, inferred_shapes)
        b1_graph_outputs.append(vi)

    b1_vi = []
    for vi in graph.value_info:
        if vi.name in b1_all_outputs or vi.name in b1_all_inputs:
            if vi.name not in {v.name for v in b1_graph_inputs} and \
               vi.name not in {v.name for v in b1_graph_outputs} and \
               vi.name not in init_names:
                b1_vi.append(vi)

    b1_graph = helper.make_graph(b1_nodes, "part_b1", b1_graph_inputs, b1_graph_outputs,
                                  initializer=b1_inits, value_info=b1_vi)
    b1_model = helper.make_model(b1_graph, opset_imports=model.opset_import)
    b1_model.ir_version = model.ir_version

    b1_path = os.path.join(out_dir, "part_b1.onnx")
    onnx.save(b1_model, b1_path)
    print(f"B1: {len(b1_nodes)} nodes, {len(b1_graph_inputs)} inputs, "
          f"{len(b1_graph_outputs)} outputs -> {b1_path}")

    # Build B2
    b2_nodes = [all_nodes[i] for i in sorted(b2_node_indices)]
    b2_all_inputs = set()
    b2_all_outputs = set()
    for n in b2_nodes:
        for inp in n.input:
            if inp:
                b2_all_inputs.add(inp)
        for out in n.output:
            b2_all_outputs.add(out)

    b2_graph_inputs = []
    b2_inits = []
    for name in sorted(b2_all_inputs - b2_all_outputs):
        if name in init_names:
            b2_inits.append(init_map[name])
        else:
            vi = get_value_info(name, graph, inferred_shapes)
            b2_graph_inputs.append(vi)

    b2_graph_outputs = []
    orig_output_names = {o.name for o in graph.output}
    for name in sorted(b2_all_outputs & orig_output_names):
        vi = get_value_info(name, graph, inferred_shapes)
        b2_graph_outputs.append(vi)
    if not b2_graph_outputs:
        for out in graph.output:
            b2_graph_outputs.append(out)

    b2_vi = []
    for vi in graph.value_info:
        if vi.name in b2_all_outputs or vi.name in b2_all_inputs:
            if vi.name not in {v.name for v in b2_graph_inputs} and \
               vi.name not in {v.name for v in b2_graph_outputs} and \
               vi.name not in init_names:
                b2_vi.append(vi)

    b2_graph = helper.make_graph(b2_nodes, "part_b2", b2_graph_inputs, b2_graph_outputs,
                                  initializer=b2_inits, value_info=b2_vi)
    b2_model = helper.make_model(b2_graph, opset_imports=model.opset_import)
    b2_model.ir_version = model.ir_version

    b2_path = os.path.join(out_dir, "part_b2.onnx")
    onnx.save(b2_model, b2_path)
    print(f"B2: {len(b2_nodes)} nodes, {len(b2_graph_inputs)} inputs, "
          f"{len(b2_graph_outputs)} outputs -> {b2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()
    split_part_b(args.input, args.output_dir)
