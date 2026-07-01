#!/usr/bin/env python3
"""Replace all BOOL-producing ops with arithmetic equivalents that output INT8.

The torq compiler's TileAndFuse pass fuses comparison ops (producing i1) with
downstream consumers (i8/bf16/i32), causing "Input strides must match" errors
on torq_hl.elementwisebinary due to mixed element sizes.

This script eliminates ALL i1 intermediates by replacing:
  - Less(a, b)           → Clip(Clip(Sub(b, a), -1, 1), 0, 1) → Cast(INT8)
  - Greater(a, b)        → Clip(Clip(Sub(a, b), -1, 1), 0, 1) → Cast(INT8)
  - GreaterOrEqual(a, b) → Sub(1, Clip(Clip(Sub(b, a), -1, 1), 0, 1)) → Cast(INT8)
  - LessOrEqual(a, b)    → Sub(1, Clip(Clip(Sub(a, b), -1, 1), 0, 1)) → Cast(INT8)
  - And(a_i8, b_i8)      → Mul(a, b)
  - Or(a_i8, b_i8)       → Max(a, b)
  - Not(a_i8)            → Sub(1, a)

The Cast(BOOL→INT8) chains added by cast_bool_to_int8.py (step 18) are absorbed.

Usage:
    python3 scripts/eliminate_bool_ops.py -i model.onnx -o model_nobool.onnx
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

INT8 = TensorProto.INT8
INT32 = TensorProto.INT32
INT64 = TensorProto.INT64
BOOL = TensorProto.BOOL
BFLOAT16 = TensorProto.BFLOAT16
FLOAT = TensorProto.FLOAT


def _get_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return None


def _get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _node_by_output(nodes, name):
    for n in nodes:
        if name in n.output:
            return n
    return None


def _consumers(nodes, name):
    return [n for n in nodes if name in n.input]


def _ensure_const(graph, name, value, dtype):
    for init in graph.initializer:
        if init.name == name:
            return
    np_dtype = {INT8: np.int8, INT32: np.int32, INT64: np.int64,
                BFLOAT16: np.float32, FLOAT: np.float32}[dtype]
    graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np_dtype), name=name))


def _add_vi(graph, name, dtype, shape):
    for vi in graph.value_info:
        if vi.name == name:
            return
    graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))


def _broadcast_shape(s1, s2):
    r = max(len(s1), len(s2))
    s1 = [1] * (r - len(s1)) + s1
    s2 = [1] * (r - len(s2)) + s2
    return [max(a, b) for a, b in zip(s1, s2)]


def find_bool_chains(model):
    """Find all bool-producing ops and their Cast(INT8) chains."""
    nodes = list(model.graph.node)
    chains = []

    for i, n in enumerate(nodes):
        if n.op_type not in ("Less", "Greater", "GreaterOrEqual", "LessOrEqual",
                             "Equal", "And", "Or", "Not", "Xor"):
            continue

        bool_output = n.output[0]
        cast_chain = []
        current = bool_output

        for j in range(i + 1, min(i + 4, len(nodes))):
            nj = nodes[j]
            if nj.op_type == "Cast" and nj.input[0] == current:
                cast_to = None
                for attr in nj.attribute:
                    if attr.name == "to":
                        cast_to = attr.i
                if cast_to in (INT8, INT32):
                    cast_chain.append(j)
                    current = nj.output[0]
                else:
                    break
            else:
                break

        chains.append({
            "op_idx": i,
            "op_node": n,
            "cast_indices": cast_chain,
            "final_output": current,
        })

    return chains


MAX_ELEMENTS = 16384


def _total_elements(shape):
    if not shape:
        return 1
    r = 1
    for d in shape:
        r *= d
    return r


def _replace_comparison_single(model, name, op, a_name, b_name, final_output,
                                work_dtype, out_shape):
    """Core Sub→Sign→Clip→Cast replacement for one chunk (or full tensor)."""
    graph = model.graph
    new_nodes = []

    if op in ("Less", "GreaterOrEqual"):
        sub_out = f"{name}__diff"
        sub_node = helper.make_node("Sub", [b_name, a_name], [sub_out], name=f"{name}__sub")
        new_nodes.append(sub_node)
        _add_vi(graph, sub_out, work_dtype, out_shape)
    else:
        sub_out = f"{name}__diff"
        sub_node = helper.make_node("Sub", [a_name, b_name], [sub_out], name=f"{name}__sub")
        new_nodes.append(sub_node)
        _add_vi(graph, sub_out, work_dtype, out_shape)

    # Sign(x) is broken on the compiler for positive integers (returns 0 instead of 1).
    # Use Clip(x, -1, 1) instead — equivalent for integer tensors.
    clip_neg1_name = f"__clip_neg1_{work_dtype}"
    clip_pos1_name = f"__clip_pos1_{work_dtype}"
    _ensure_const(graph, clip_neg1_name, -1, work_dtype)
    _ensure_const(graph, clip_pos1_name, 1, work_dtype)

    sign_out = f"{name}__sign"
    sign_node = helper.make_node("Clip", [sub_out, clip_neg1_name, clip_pos1_name],
                                 [sign_out], name=f"{name}__sign")
    new_nodes.append(sign_node)
    _add_vi(graph, sign_out, work_dtype, out_shape)

    clip_min_name = f"__clip_zero_{work_dtype}"
    clip_max_name = f"__clip_one_{work_dtype}"
    _ensure_const(graph, clip_min_name, 0, work_dtype)
    _ensure_const(graph, clip_max_name, 1, work_dtype)

    clipped_out = f"{name}__clipped"
    clip_node = helper.make_node("Clip", [sign_out, clip_min_name, clip_max_name],
                                 [clipped_out], name=f"{name}__clip")
    new_nodes.append(clip_node)
    _add_vi(graph, clipped_out, work_dtype, out_shape)

    if op in ("GreaterOrEqual", "LessOrEqual"):
        one_name = f"__one_{work_dtype}"
        _ensure_const(graph, one_name, 1, work_dtype)
        inv_out = f"{name}__inv"
        inv_node = helper.make_node("Sub", [one_name, clipped_out], [inv_out], name=f"{name}__inv")
        new_nodes.append(inv_node)
        _add_vi(graph, inv_out, work_dtype, out_shape)
        result = inv_out
    else:
        result = clipped_out

    if work_dtype != INT8:
        cast_node = helper.make_node("Cast", [result], [final_output],
                                     name=f"{name}__to_i8", to=INT8)
        new_nodes.append(cast_node)
    else:
        rename_node = helper.make_node("Identity", [result], [final_output],
                                       name=f"{name}__identity")
        new_nodes.append(rename_node)

    return new_nodes


def _find_split_axis(shape):
    """Find the axis with the largest dimension for chunking."""
    if not shape:
        return -1, 0
    best_axis = 0
    best_dim = shape[0]
    for i, d in enumerate(shape):
        if d > best_dim:
            best_dim = d
            best_axis = i
    return best_axis, best_dim


def replace_comparison(model, chain, nodes):
    """Replace Less/Greater/GreaterOrEqual/LessOrEqual with arithmetic."""
    graph = model.graph
    n = chain["op_node"]
    op = n.op_type
    final_output = chain["final_output"]
    name = n.name or f"{op}_{chain['op_idx']}"

    a_name, b_name = n.input[0], n.input[1]
    a_dtype = _get_dtype(model, a_name)
    b_dtype = _get_dtype(model, b_name)
    a_shape = _get_shape(model, a_name) or []
    b_shape = _get_shape(model, b_name) or []
    orig_out_shape = _get_shape(model, n.output[0])
    out_shape = orig_out_shape if orig_out_shape else _broadcast_shape(a_shape, b_shape)

    work_dtype = a_dtype or b_dtype or INT64

    total_elems = _total_elements(out_shape)
    if total_elems <= MAX_ELEMENTS:
        return _replace_comparison_single(model, name, op, a_name, b_name,
                                          final_output, work_dtype, out_shape)

    split_axis, split_dim = _find_split_axis(out_shape)
    num_chunks = (total_elems + MAX_ELEMENTS - 1) // MAX_ELEMENTS
    num_chunks = min(num_chunks, split_dim)
    chunk_size = split_dim // num_chunks
    remainder = split_dim % num_chunks

    print(f"    Chunking {name}: {out_shape} → {num_chunks} chunks along axis {split_axis} "
          f"(chunk_size={chunk_size}, remainder={remainder})")

    new_nodes = []
    chunk_outputs = []

    for ci in range(num_chunks):
        start = ci * chunk_size + min(ci, remainder)
        end = start + chunk_size + (1 if ci < remainder else 0)
        this_size = end - start
        chunk_shape = list(out_shape)
        chunk_shape[split_axis] = this_size

        starts_name = f"{name}__chunk{ci}_starts"
        ends_name = f"{name}__chunk{ci}_ends"
        axes_name = f"{name}__chunk{ci}_axes"
        graph.initializer.append(numpy_helper.from_array(
            np.array([start], dtype=np.int64), name=starts_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([end], dtype=np.int64), name=ends_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([split_axis], dtype=np.int64), name=axes_name))

        a_chunk_name = a_name
        b_chunk_name = b_name

        a_needs_slice = len(a_shape) > split_axis and a_shape[split_axis] > 1
        b_needs_slice = len(b_shape) > split_axis and b_shape[split_axis] > 1

        if a_needs_slice:
            a_chunk_name = f"{name}__chunk{ci}_a"
            slice_a = helper.make_node("Slice", [a_name, starts_name, ends_name, axes_name],
                                       [a_chunk_name], name=f"{name}__chunk{ci}_slice_a")
            new_nodes.append(slice_a)
            a_chunk_shape = list(a_shape)
            a_chunk_shape[split_axis] = this_size
            _add_vi(graph, a_chunk_name, a_dtype, a_chunk_shape)

        if b_needs_slice:
            b_chunk_name = f"{name}__chunk{ci}_b"
            slice_b = helper.make_node("Slice", [b_name, starts_name, ends_name, axes_name],
                                       [b_chunk_name], name=f"{name}__chunk{ci}_slice_b")
            new_nodes.append(slice_b)
            b_chunk_shape = list(b_shape)
            b_chunk_shape[split_axis] = this_size
            _add_vi(graph, b_chunk_name, b_dtype, b_chunk_shape)

        chunk_out = f"{name}__chunk{ci}_out"
        chunk_nodes = _replace_comparison_single(
            model, f"{name}__chunk{ci}", op, a_chunk_name, b_chunk_name,
            chunk_out, work_dtype, chunk_shape)
        new_nodes.extend(chunk_nodes)
        _add_vi(graph, chunk_out, INT8, chunk_shape)
        chunk_outputs.append(chunk_out)

    barrier_raw = f"{name}__concat_raw"
    concat_node = helper.make_node("Concat", chunk_outputs, [barrier_raw],
                                   name=f"{name}__concat", axis=split_axis)
    new_nodes.append(concat_node)
    _add_vi(graph, barrier_raw, INT8, out_shape)

    ndim = len(out_shape)
    pads_val = [0] * ndim + [1] + [0] * (ndim - 1)
    pads_name = f"{name}__concat_barrier_pads"
    graph.initializer.append(numpy_helper.from_array(
        np.array(pads_val, dtype=np.int64), pads_name))
    padded_name = f"{name}__concat_barrier_padded"
    padded_shape = list(out_shape)
    padded_shape[0] += 1
    pad_node = helper.make_node(
        "Pad", [barrier_raw, pads_name], [padded_name],
        name=f"{name}__concat_barrier_pad")
    new_nodes.append(pad_node)
    _add_vi(graph, padded_name, INT8, padded_shape)

    starts_b = f"{name}__concat_barrier_starts"
    ends_b = f"{name}__concat_barrier_ends"
    axes_b = f"{name}__concat_barrier_axes"
    graph.initializer.append(numpy_helper.from_array(
        np.array([1], dtype=np.int64), starts_b))
    graph.initializer.append(numpy_helper.from_array(
        np.array([1 + out_shape[0]], dtype=np.int64), ends_b))
    graph.initializer.append(numpy_helper.from_array(
        np.array([0], dtype=np.int64), axes_b))
    slice_node = helper.make_node(
        "Slice", [padded_name, starts_b, ends_b, axes_b],
        [final_output], name=f"{name}__concat_barrier_slice")
    new_nodes.append(slice_node)

    return new_nodes


def replace_logic(model, chain, nodes):
    """Replace And/Or/Not with INT8 arithmetic."""
    graph = model.graph
    n = chain["op_node"]
    op = n.op_type
    final_output = chain["final_output"]
    name = n.name or f"{op}_{chain['op_idx']}"

    new_nodes = []

    if op == "And":
        a, b = n.input[0], n.input[1]
        mul_node = helper.make_node("Mul", [a, b], [final_output], name=f"{name}__mul")
        new_nodes.append(mul_node)

    elif op == "Or":
        a, b = n.input[0], n.input[1]
        max_node = helper.make_node("Max", [a, b], [final_output], name=f"{name}__max")
        new_nodes.append(max_node)

    elif op == "Not":
        a = n.input[0]
        one_name = f"__one_i8"
        _ensure_const(graph, one_name, 1, INT8)
        sub_node = helper.make_node("Sub", [one_name, a], [final_output], name=f"{name}__sub")
        new_nodes.append(sub_node)

    elif op == "Xor":
        a, b = n.input[0], n.input[1]
        sub_out = f"{name}__xor_sub"
        sub_node = helper.make_node("Sub", [a, b], [sub_out], name=f"{name}__sub")
        new_nodes.append(sub_node)
        a_shape = _get_shape(model, a) or []
        b_shape = _get_shape(model, b) or []
        _add_vi(graph, sub_out, INT8, _broadcast_shape(a_shape, b_shape))
        abs_node = helper.make_node("Abs", [sub_out], [final_output], name=f"{name}__abs")
        new_nodes.append(abs_node)

    return new_nodes


def apply_fixes(model):
    graph = model.graph
    nodes = list(graph.node)
    chains = find_bool_chains(model)

    if not chains:
        print("No bool-producing ops found.")
        return model, 0

    indices_to_remove = set()
    insertions = []

    for chain in reversed(chains):
        op = chain["op_node"].op_type
        idx = chain["op_idx"]

        if op in ("Less", "Greater", "GreaterOrEqual", "LessOrEqual", "Equal"):
            new_nodes = replace_comparison(model, chain, nodes)
        elif op in ("And", "Or", "Not", "Xor"):
            new_nodes = replace_logic(model, chain, nodes)
        else:
            continue

        indices_to_remove.add(idx)
        for ci in chain["cast_indices"]:
            indices_to_remove.add(ci)

        insertions.append((idx, new_nodes))

        name = chain["op_node"].name or f"{op}_{idx}"
        print(f"  [{idx:3d}] {op:20s} {name} → {len(new_nodes)} arithmetic ops "
              f"(removed {1 + len(chain['cast_indices'])} nodes)")

    final_nodes = []
    insert_map = {pos: newnodes for pos, newnodes in insertions}

    for i, n in enumerate(nodes):
        if i in insert_map:
            final_nodes.extend(insert_map[i])
        if i not in indices_to_remove:
            final_nodes.append(n)

    del graph.node[:]
    graph.node.extend(final_nodes)

    out_shape = _get_shape(model, chains[0]["final_output"]) or []
    _add_vi(graph, chains[0]["final_output"], INT8, out_shape)

    return model, len(chains)


def main():
    parser = argparse.ArgumentParser(description="Eliminate BOOL ops with arithmetic equivalents")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--dry-run", action="store_true", help="Just list ops")
    parser.add_argument("--max-elements", type=int, default=16384,
                        help="Chunk comparisons with more than this many output elements")
    args = parser.parse_args()

    global MAX_ELEMENTS
    MAX_ELEMENTS = args.max_elements

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    chains = find_bool_chains(model)
    print(f"Found {len(chains)} bool-producing op(s):")
    for c in chains:
        n = c["op_node"]
        print(f"  [{c['op_idx']:3d}] {n.op_type:20s} {n.name} → final: {c['final_output']} "
              f"(+{len(c['cast_indices'])} casts)")

    if args.dry_run:
        return

    model, count = apply_fixes(model)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed: {e}")

    onnx.save(model, args.output)
    print(f"\nReplaced {count} ops. Saved to {args.output}")


if __name__ == "__main__":
    main()
