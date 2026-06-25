#!/usr/bin/env python3
"""Chunk attention MatMul+Softmax blocks along the query dimension.

Finds patterns: MatMul(Q,K^T) → [optional Add(bias)] → Softmax → MatMul(attn,V)
and splits Q along the sequence dimension into N chunks, producing N independent
smaller attention computations that are Concat'd back together.

This is the same "qchunk" strategy used for model B attention blocks.

Usage:
    python3 scripts/chunk_attention.py -i model.onnx -o model_chunked.onnx --num-chunks 8
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    return None


def _get_elem_type(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    return TensorProto.BFLOAT16


def _node_by_output(nodes, name):
    for n in nodes:
        if name in n.output:
            return n
    return None


def _consumers(nodes, name):
    return [n for n in nodes if name in n.input]


def find_attention_blocks(model):
    """Find MatMul→[Add]→Softmax→MatMul attention patterns."""
    nodes = list(model.graph.node)
    blocks = []

    for softmax in nodes:
        if softmax.op_type != "Softmax":
            continue

        softmax_input = softmax.input[0]
        softmax_output = softmax.output[0]

        add_node = _node_by_output(nodes, softmax_input)
        if add_node and add_node.op_type == "Add":
            qk_matmul_output = None
            bias_input = None
            for inp in add_node.input:
                producer = _node_by_output(nodes, inp)
                if producer and producer.op_type == "MatMul":
                    qk_matmul_output = inp
                else:
                    bias_input = inp
            if qk_matmul_output is None:
                continue
            qk_matmul = _node_by_output(nodes, qk_matmul_output)
        elif add_node and add_node.op_type == "MatMul":
            qk_matmul = add_node
            add_node = None
            bias_input = None
        else:
            continue

        attn_v_consumers = _consumers(nodes, softmax_output)
        attn_v_matmul = None
        for c in attn_v_consumers:
            if c.op_type == "MatMul":
                attn_v_matmul = c
                break
        if attn_v_matmul is None:
            continue

        q_name = qk_matmul.input[0]
        kt_name = qk_matmul.input[1]
        q_shape = _get_shape(model, q_name)
        kt_shape = _get_shape(model, kt_name)
        qk_shape = _get_shape(model, qk_matmul.output[0])

        if q_shape is None or kt_shape is None or qk_shape is None:
            continue
        if len(q_shape) != 4 or len(qk_shape) != 4:
            continue
        if qk_shape[-2] != qk_shape[-1]:
            continue

        v_name = attn_v_matmul.input[1] if attn_v_matmul.input[0] == softmax_output else attn_v_matmul.input[0]

        blocks.append({
            "qk_matmul": qk_matmul,
            "add_node": add_node,
            "bias_input": bias_input,
            "softmax": softmax,
            "attn_v_matmul": attn_v_matmul,
            "q_name": q_name,
            "kt_name": kt_name,
            "v_name": v_name,
            "q_shape": q_shape,
            "seq_len": q_shape[-2],
        })

    return blocks


def chunk_attention_block(model, block, num_chunks):
    """Replace one attention block with per-head chunked version.

    Splits heads first (axis=1), then chunks Q per head (axis=-2).
    Per-head Q-chunks Concat to [B,1,S,D_v], then heads Concat to [B,H,S,D_v].
    This matches the golden model structure and avoids SDIM tile size crashes.
    """
    graph = model.graph
    nodes = list(graph.node)
    seq_len = block["seq_len"]
    chunk_size = seq_len // num_chunks
    assert seq_len % num_chunks == 0, f"seq_len {seq_len} not divisible by {num_chunks}"

    q_name = block["q_name"]
    kt_name = block["kt_name"]
    v_name = block["v_name"]
    q_shape = block["q_shape"]
    qk_matmul = block["qk_matmul"]
    add_node = block["add_node"]
    softmax = block["softmax"]
    attn_v_matmul = block["attn_v_matmul"]
    bias_input = block["bias_input"]

    base_name = qk_matmul.name or qk_matmul.output[0]
    final_output = attn_v_matmul.output[0]

    B, H, S, D_q = q_shape
    kt_shape = _get_shape(model, kt_name)
    D_k = kt_shape[-1]
    v_shape = _get_shape(model, v_name)
    D_v = v_shape[-1]

    dtype = _get_elem_type(model, q_name)

    new_nodes = []

    # --- Split Q, K^T, V along head dimension (axis=1) ---
    head_split_sizes_name = f"{base_name}_head_split_sizes"
    graph.initializer.append(numpy_helper.from_array(
        np.array([1] * H, dtype=np.int64), name=head_split_sizes_name))

    q_head_names = [f"{base_name}_q_head_{h}" for h in range(H)]
    new_nodes.append(helper.make_node(
        "Split", [q_name, head_split_sizes_name], q_head_names,
        name=f"{base_name}_split_q_heads", axis=1))
    for n in q_head_names:
        graph.value_info.append(helper.make_tensor_value_info(
            n, dtype, [B, 1, S, D_q]))

    kt_head_names = [f"{base_name}_kt_head_{h}" for h in range(H)]
    new_nodes.append(helper.make_node(
        "Split", [kt_name, head_split_sizes_name], kt_head_names,
        name=f"{base_name}_split_kt_heads", axis=1))
    kt_per_head_shape = list(kt_shape)
    kt_per_head_shape[1] = 1
    for n in kt_head_names:
        graph.value_info.append(helper.make_tensor_value_info(
            n, dtype, kt_per_head_shape))

    v_head_names = [f"{base_name}_v_head_{h}" for h in range(H)]
    new_nodes.append(helper.make_node(
        "Split", [v_name, head_split_sizes_name], v_head_names,
        name=f"{base_name}_split_v_heads", axis=1))
    for n in v_head_names:
        graph.value_info.append(helper.make_tensor_value_info(
            n, dtype, [B, 1, S, D_v]))

    if add_node and bias_input:
        bias_shape = _get_shape(model, bias_input)
        if bias_shape and len(bias_shape) == 4 and bias_shape[1] == H:
            bias_head_names = [f"{base_name}_bias_head_{h}" for h in range(H)]
            new_nodes.append(helper.make_node(
                "Split", [bias_input, head_split_sizes_name], bias_head_names,
                name=f"{base_name}_split_bias_heads", axis=1))
            for n in bias_head_names:
                graph.value_info.append(helper.make_tensor_value_info(
                    n, dtype,
                    [bias_shape[0], 1, bias_shape[2], bias_shape[3]]))
        else:
            bias_head_names = [bias_input] * H
    else:
        bias_head_names = [None] * H

    # --- Per-head: split Q into chunks, do attention, concat chunks ---
    q_chunk_split_sizes_name = f"{base_name}_q_chunk_split_sizes"
    graph.initializer.append(numpy_helper.from_array(
        np.array([chunk_size] * num_chunks, dtype=np.int64),
        name=q_chunk_split_sizes_name))

    head_outputs = []
    attn_name = attn_v_matmul.name or final_output

    for h in range(H):
        q_h = q_head_names[h]
        kt_h = kt_head_names[h]
        v_h = v_head_names[h]
        bias_h = bias_head_names[h]

        q_chunk_names = [f"{base_name}_head_{h}_q_chunk_{c}" for c in range(num_chunks)]
        new_nodes.append(helper.make_node(
            "Split", [q_h, q_chunk_split_sizes_name], q_chunk_names,
            name=f"{base_name}_head_{h}_split_query_chunks", axis=-2))
        for n in q_chunk_names:
            graph.value_info.append(helper.make_tensor_value_info(
                n, dtype, [B, 1, chunk_size, D_q]))

        chunk_outputs = []
        for c in range(num_chunks):
            sfx = f"_head_{h}_query_chunk_{c}"

            qk_out = f"{qk_matmul.output[0]}{sfx}"
            new_nodes.append(helper.make_node(
                "MatMul", [q_chunk_names[c], kt_h], [qk_out],
                name=f"{qk_matmul.name}{sfx}"))
            graph.value_info.append(helper.make_tensor_value_info(
                qk_out, dtype, [B, 1, chunk_size, D_k]))

            if add_node and bias_h:
                add_out = f"{add_node.output[0]}{sfx}"
                new_nodes.append(helper.make_node(
                    "Add", [qk_out, bias_h], [add_out],
                    name=f"{add_node.name}{sfx}"))
                graph.value_info.append(helper.make_tensor_value_info(
                    add_out, dtype, [B, 1, chunk_size, D_k]))
                sm_in = add_out
            else:
                sm_in = qk_out

            sm_out = f"{softmax.output[0]}{sfx}"
            attrs = {}
            for attr in softmax.attribute:
                attrs[attr.name] = attr.i if attr.type == 2 else attr.f
            new_nodes.append(helper.make_node(
                "Softmax", [sm_in], [sm_out],
                name=f"{softmax.name}{sfx}",
                axis=attrs.get("axis", -1)))
            graph.value_info.append(helper.make_tensor_value_info(
                sm_out, dtype, [B, 1, chunk_size, D_k]))

            av_out = f"{attn_name}{sfx}"
            new_nodes.append(helper.make_node(
                "MatMul", [sm_out, v_h], [av_out],
                name=f"{attn_name}{sfx}"))
            graph.value_info.append(helper.make_tensor_value_info(
                av_out, dtype, [B, 1, chunk_size, D_v]))
            chunk_outputs.append(av_out)

        head_concat_out = f"{attn_name}_head_{h}_concat_query_chunks"
        new_nodes.append(helper.make_node(
            "Concat", chunk_outputs, [head_concat_out],
            name=f"{attn_name}_head_{h}_concat_query_chunks", axis=-2))
        graph.value_info.append(helper.make_tensor_value_info(
            head_concat_out, dtype, [B, 1, S, D_v]))
        head_outputs.append(head_concat_out)

    # --- Concat all heads back ---
    new_nodes.append(helper.make_node(
        "Concat", head_outputs, [final_output],
        name=f"{attn_name}_concat_heads", axis=1))

    old_names = {qk_matmul.name, softmax.name, attn_v_matmul.name}
    if add_node:
        old_names.add(add_node.name)

    insert_idx = next(i for i, n in enumerate(nodes) if n.name in old_names)
    remaining = [n for n in nodes if n.name not in old_names]
    remaining[insert_idx:insert_idx] = new_nodes

    del graph.node[:]
    graph.node.extend(remaining)


def main():
    parser = argparse.ArgumentParser(description="Chunk attention blocks along query dimension")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--num-chunks", type=int, default=8, help="Number of query chunks")
    parser.add_argument("--dry-run", action="store_true", help="Just print found blocks")
    parser.add_argument("--min-seq-len", type=int, default=0,
                        help="Only chunk blocks with seq_len >= this value")
    args = parser.parse_args()

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    blocks = find_attention_blocks(model)
    print(f"Found {len(blocks)} attention block(s):")
    for i, b in enumerate(blocks):
        print(f"  [{i}] {b['qk_matmul'].name}: Q{b['q_shape']} → "
              f"[{b['seq_len']}×{b['seq_len']}] → {b['attn_v_matmul'].name}")

    if args.dry_run:
        return

    eligible = [b for b in blocks if b["seq_len"] >= args.min_seq_len
                and b["seq_len"] % args.num_chunks == 0]
    if len(eligible) < len(blocks):
        skipped = len(blocks) - len(eligible)
        print(f"Skipping {skipped} block(s) with seq_len < {args.min_seq_len} "
              f"or not divisible by {args.num_chunks}")

    for b in reversed(eligible):
        print(f"Chunking {b['qk_matmul'].name} into {args.num_chunks} chunks "
              f"(chunk_size={b['seq_len'] // args.num_chunks})")
        chunk_attention_block(model, b, args.num_chunks)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed after chunking: {e}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
