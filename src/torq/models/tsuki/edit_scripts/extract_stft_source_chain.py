#!/usr/bin/env python3
"""Extract the stft_source computation chain from Part A and save as separate fp32 ONNX.

Removes a specific set of nodes (identified by a reference submodel or by tracing
backwards from stft_source to given cut-point tensors), adds the cut-point tensors
as new graph outputs, and builds a standalone fp32 submodel for ORT execution.

Usage:
    python3 scripts/extract_stft_source_chain.py \
        -i part_a.onnx -o part_a_no_stft.onnx \
        --submodel stft_source_fp32.onnx \
        --reference /tmp/acc_sub6/stft_source_submodel.onnx
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def trace_back_to_cuts(graph, target_tensor, cut_tensors):
    """Trace backwards from target_tensor, stopping at cut_tensors. Returns node names."""
    tensor_to_producer = {}
    node_inputs_map = {}
    for n in graph.node:
        for out in n.output:
            tensor_to_producer[out] = n.name
        node_inputs_map[n.name] = list(n.input)

    init_names = {i.name for i in graph.initializer}
    visited_tensors = set()
    chain = set()

    def visit(tname):
        if tname in visited_tensors or tname in cut_tensors or tname in init_names:
            return
        visited_tensors.add(tname)
        if tname in tensor_to_producer:
            nname = tensor_to_producer[tname]
            chain.add(nname)
            for inp in node_inputs_map.get(nname, []):
                visit(inp)

    visit(target_tensor)
    return chain


def get_chain_from_reference(full_graph, ref_model_path):
    """Get chain node names by matching reference submodel node names to full model."""
    ref = onnx.load(ref_model_path, load_external_data=False)
    full_names = {n.name for n in full_graph.node}
    ref_names = {n.name for n in ref.graph.node}
    matched = ref_names & full_names
    unmatched = ref_names - full_names

    ref_inits = {i.name for i in ref.graph.initializer}
    cut_tensors = set()
    for inp in ref.graph.input:
        if inp.name not in ref_inits:
            cut_tensors.add(inp.name)

    return matched, unmatched, cut_tensors


def find_cut_tensors(graph, chain_nodes):
    """Find tensors that cross the chain boundary (produced outside, consumed inside)."""
    init_names = {i.name for i in graph.initializer}
    graph_input_names = {i.name for i in graph.input}
    node_inputs = {n.name: list(n.input) for n in graph.node}
    tensor_to_producer = {}
    for n in graph.node:
        for out in n.output:
            tensor_to_producer[out] = n.name

    internal_cuts = set()
    graph_input_cuts = set()
    for nname in chain_nodes:
        for inp in node_inputs.get(nname, []):
            if inp in init_names:
                continue
            if inp in graph_input_names:
                graph_input_cuts.add(inp)
                continue
            producer = tensor_to_producer.get(inp)
            if producer and producer not in chain_nodes:
                internal_cuts.add(inp)
            elif producer is None:
                internal_cuts.add(inp)

    return internal_cuts, graph_input_cuts


def get_tensor_type_info(graph, tensor_name):
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == tensor_name and vi.type.HasField("tensor_type"):
            tt = vi.type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim] if tt.HasField("shape") else []
            return tt.elem_type, shape
    return None, None


def convert_bf16_to_fp32_initializer(init):
    if init.data_type != TensorProto.BFLOAT16:
        return init
    raw = init.raw_data
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    shape = list(init.dims) if init.dims else [len(f32)]
    return numpy_helper.from_array(f32.reshape(shape), name=init.name)


def convert_type_to_fp32(elem_type):
    if elem_type == TensorProto.BFLOAT16:
        return TensorProto.FLOAT
    return elem_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input Part A ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output Part A ONNX (chain removed)")
    parser.add_argument("--submodel", required=True, help="Output stft_source submodel (fp32)")
    parser.add_argument("--reference", default=None,
                        help="Reference submodel to identify chain nodes by name matching")
    args = parser.parse_args()

    print("Loading model...")
    model = onnx.load(args.input)
    graph = model.graph

    output_names = {o.name for o in graph.output}
    if "stft_source" not in output_names:
        print("ERROR: stft_source not found in graph outputs", file=sys.stderr)
        sys.exit(1)

    # Identify chain nodes
    if args.reference:
        print(f"Using reference submodel: {args.reference}")
        chain_nodes, unmatched, ref_cuts = get_chain_from_reference(graph, args.reference)
        print(f"  Matched nodes: {len(chain_nodes)}")
        if unmatched:
            print(f"  Unmatched (extraction artifacts): {len(unmatched)}: {sorted(unmatched)}")
    else:
        print("ERROR: --reference is required (provides the node names to extract)", file=sys.stderr)
        sys.exit(1)

    # Find cut-point tensors
    internal_cuts, graph_input_cuts = find_cut_tensors(graph, chain_nodes)
    print(f"  Cut-point internal tensors: {len(internal_cuts)}")
    for t in sorted(internal_cuts):
        print(f"    {t}")
    print(f"  Graph inputs consumed by chain: {sorted(graph_input_cuts)}")

    # Verify no chain outputs are consumed outside the chain
    chain_tensors = set()
    chain_node_list = []
    for n in graph.node:
        if n.name in chain_nodes:
            chain_tensors.update(n.output)
            chain_node_list.append(n)

    shared_out = set()
    for n in graph.node:
        if n.name not in chain_nodes:
            for inp in n.input:
                if inp in chain_tensors:
                    shared_out.add(inp)
    if shared_out:
        print(f"  WARNING: {len(shared_out)} chain outputs consumed by non-chain nodes:")
        for t in sorted(shared_out):
            print(f"    {t}")
        print("  These tensors will also be added as Part A outputs.")

    # Find chain-exclusive initializers
    all_init_names = {i.name for i in graph.initializer}
    init_used_by_chain = set()
    init_used_by_others = set()
    for n in graph.node:
        for inp in n.input:
            if inp in all_init_names:
                if n.name in chain_nodes:
                    init_used_by_chain.add(inp)
                else:
                    init_used_by_others.add(inp)
    exclusive_inits = init_used_by_chain - init_used_by_others
    print(f"  Chain-exclusive initializers: {len(exclusive_inits)}")

    # =========================================================================
    # Build stft_source submodel (fp32)
    # =========================================================================
    print("\nBuilding stft_source submodel...")

    sub_inputs = []
    for tname in sorted(internal_cuts | shared_out):
        elem_type, shape = get_tensor_type_info(graph, tname)
        if elem_type is None:
            print(f"  WARNING: no type info for '{tname}', defaulting to FLOAT []")
            elem_type = TensorProto.FLOAT
            shape = []
        sub_inputs.append(
            helper.make_tensor_value_info(tname, convert_type_to_fp32(elem_type), shape)
        )

    for tname in sorted(graph_input_cuts):
        elem_type, shape = get_tensor_type_info(graph, tname)
        if elem_type is None:
            elem_type = TensorProto.FLOAT
            shape = []
        sub_inputs.append(
            helper.make_tensor_value_info(tname, convert_type_to_fp32(elem_type), shape)
        )

    stft_elem_type, stft_shape = get_tensor_type_info(graph, "stft_source")
    if stft_elem_type is None:
        for o in graph.output:
            if o.name == "stft_source" and o.type.HasField("tensor_type"):
                tt = o.type.tensor_type
                stft_elem_type = tt.elem_type
                stft_shape = [d.dim_value for d in tt.shape.dim]
                break
    sub_outputs = [helper.make_tensor_value_info(
        "stft_source", convert_type_to_fp32(stft_elem_type or TensorProto.FLOAT),
        stft_shape or []
    )]

    # Copy and convert nodes
    sub_nodes = []
    for n in chain_node_list:
        new_node = onnx.NodeProto()
        new_node.CopyFrom(n)
        for attr in new_node.attribute:
            if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                attr.i = TensorProto.FLOAT
        sub_nodes.append(new_node)

    # Copy and convert initializers
    sub_initializers = []
    for init in graph.initializer:
        if init.name in init_used_by_chain:
            sub_initializers.append(convert_bf16_to_fp32_initializer(init))

    # Copy value_info for chain-internal tensors
    sub_input_names = {inp.name for inp in sub_inputs}
    sub_value_info = []
    for vi in graph.value_info:
        if vi.name in chain_tensors and vi.name not in sub_input_names and vi.name != "stft_source":
            new_vi = onnx.ValueInfoProto()
            new_vi.CopyFrom(vi)
            if new_vi.type.HasField("tensor_type"):
                if new_vi.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                    new_vi.type.tensor_type.elem_type = TensorProto.FLOAT
            sub_value_info.append(new_vi)

    sub_graph = helper.make_graph(
        sub_nodes, "stft_source_chain", sub_inputs, sub_outputs,
        initializer=sub_initializers, value_info=sub_value_info,
    )
    sub_model = helper.make_model(sub_graph, opset_imports=model.opset_import)
    sub_model.ir_version = model.ir_version

    onnx.save(sub_model, args.submodel)
    print(f"  Saved submodel: {args.submodel}")
    print(f"    Nodes: {len(sub_nodes)}, Initializers: {len(sub_initializers)}")
    print(f"    Inputs: {len(sub_inputs)}, Output: stft_source")

    # =========================================================================
    # Build modified Part A (chain removed)
    # =========================================================================
    print("\nBuilding modified Part A...")

    remaining_nodes = [n for n in graph.node if n.name not in chain_nodes]
    remaining_inits = [i for i in graph.initializer if i.name not in exclusive_inits]
    remaining_vi = [vi for vi in graph.value_info if vi.name not in chain_tensors]

    new_outputs = []
    for o in graph.output:
        if o.name != "stft_source":
            new_outputs.append(o)

    # Add cut-point tensors as new outputs (these become vmfb outputs)
    for tname in sorted(internal_cuts | shared_out):
        elem_type, shape = get_tensor_type_info(graph, tname)
        if elem_type is None:
            elem_type = TensorProto.FLOAT
            shape = []
        new_outputs.append(helper.make_tensor_value_info(tname, elem_type, shape))

    new_graph = helper.make_graph(
        remaining_nodes, graph.name, list(graph.input), new_outputs,
        initializer=remaining_inits, value_info=remaining_vi,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    onnx.save(new_model, args.output)
    print(f"  Saved modified Part A: {args.output}")
    print(f"    Nodes: {len(remaining_nodes)} (was {len(graph.node)})")
    print(f"    Outputs: {len(new_outputs)} (was {len(graph.output)})")

    # Print manifest-compatible output specs
    print("\n=== New Part A output specs (for manifest) ===")
    dtype_map = {
        TensorProto.FLOAT: "float32", TensorProto.BFLOAT16: "float32",
        TensorProto.INT64: "int64", TensorProto.INT32: "int32",
        TensorProto.INT8: "int8", TensorProto.BOOL: "int8",
    }
    for o in new_outputs:
        if o.type.HasField("tensor_type"):
            tt = o.type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim] if tt.HasField("shape") else []
            dtype = dtype_map.get(tt.elem_type, "float32")
            print(f'  {{"name": "{o.name}", "shape": {shape}, "dtype": "{dtype}"}}')

    print("\n=== stft_source submodel input specs ===")
    for inp in sub_inputs:
        if inp.type.HasField("tensor_type"):
            tt = inp.type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim] if tt.HasField("shape") else []
            dtype = dtype_map.get(tt.elem_type, "float32")
            print(f'  {inp.name}: {dtype} {shape}')

    print("\nDone.")


if __name__ == "__main__":
    main()
