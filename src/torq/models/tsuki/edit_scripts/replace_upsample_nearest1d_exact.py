#!/usr/bin/env python3
"""Replace upsample_nearest1d Gather nodes with mathematically exact alternatives.

Two strategies based on index source:

1. Static initializer indices matching a repeat pattern:
   Reshape+Expand+Reshape (+ Slice+Concat for tail=1).
   Mathematically identical by construction.

2. Dynamic (computed) indices with a Min-clamped pattern:
   Reshape+Expand (same as strategy 1) + arithmetic clamping correction.
   The Expand approximates the upsampling with integer scale. The clamping
   correction uses the original Min clamp value (e.g., real_x4_last) to
   overwrite positions beyond the clamp point with the correct clamped value.
   No Gather ops are produced — avoids the DEQW compiler bug entirely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def const_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def const_f32(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)


def get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def static_shapes(graph):
    out = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                ok = False
                break
        if ok:
            out[vi.name] = dims
    for init in graph.initializer:
        out.setdefault(init.name, list(init.dims))
    return out


def elem_types(graph):
    out = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        out.setdefault(init.name, init.data_type)
    return out


def find_clamp_tensor(graph, indices_name, shapes):
    """Find the Min-clamp value tensor feeding into the indices.

    Pattern: Min(unclipped_indices, clamp_value) → indices
    Returns the clamp_value tensor name (the scalar input to Min), or None.
    """
    for node in graph.node:
        if node.op_type == "Min" and indices_name in node.output:
            for inp in node.input:
                s = shapes.get(inp)
                if s is not None and len(s) == 0:
                    return inp
            for inp in node.input:
                s = shapes.get(inp)
                if s is not None and len(s) == 1 and s[0] == 1:
                    return inp
    return None


def find_targets(graph, shapes):
    init_names = {i.name for i in graph.initializer}
    init_data = {i.name: i for i in graph.initializer}

    targets = []
    for node in graph.node:
        if node.op_type != "Gather":
            continue
        if "upsample_nearest1d" not in (node.name or "").lower():
            continue
        if get_attr_int(node, "axis", 0) != 2:
            continue
        d = shapes.get(node.input[0])
        o = shapes.get(node.output[0])
        if not d or not o or len(d) != 3 or len(o) != 3:
            continue
        if d[0] != o[0] or d[1] != o[1] or d[0] != 1:
            continue
        L_in, L_out = d[2], o[2]
        if L_in <= 0 or L_out <= L_in:
            continue
        scale = L_out // L_in
        tail = L_out - scale * L_in
        if scale < 2 or tail not in (0, 1):
            continue

        idx_name = node.input[1]
        is_static_repeat = False
        clamp_tensor = None

        if idx_name in init_names:
            idx_array = numpy_helper.to_array(init_data[idx_name]).flatten()
            expected = np.repeat(np.arange(L_in, dtype=idx_array.dtype), scale)
            if tail:
                expected = np.concatenate([expected, [L_in - 1]])
            if np.array_equal(idx_array, expected):
                is_static_repeat = True
        else:
            clamp_tensor = find_clamp_tensor(graph, idx_name, shapes)

        targets.append({
            "node": node,
            "name": node.name,
            "data": node.input[0],
            "indices": node.input[1],
            "out": node.output[0],
            "N": d[0], "C": d[1], "L_in": L_in, "L_out": L_out,
            "scale": scale, "tail": tail,
            "is_static_repeat": is_static_repeat,
            "clamp_tensor": clamp_tensor,
        })
    return targets


def _make_uniq(used_names):
    def uniq(base):
        name = base
        n = 0
        while name in used_names:
            n += 1
            name = f"{base}__{n}"
        used_names.add(name)
        return name
    return uniq


def make_expand_nodes(target, elem_type, used_names, output_name=None):
    """Reshape+Expand for repeat-pattern upsampling.

    If output_name is given, uses that as the final output tensor name
    instead of target["out"]. Returns (nodes, inits, vis).
    """
    data = target["data"]
    out = output_name or target["out"]
    N, C, L_in = target["N"], target["C"], target["L_in"]
    scale, tail = target["scale"], target["tail"]
    L_main = L_in * scale
    prefix = target["out"]
    uniq = _make_uniq(used_names)

    inits, nodes, vis = [], [], []

    shape_4d = uniq(f"{prefix}__shape_4d")
    inits.append(const_i64(shape_4d, [N, C, L_in, 1]))
    data_4d = uniq(f"{prefix}__data_4d")
    nodes.append(helper.make_node("Reshape", [data, shape_4d], [data_4d],
                                  name=uniq(f"{prefix}__reshape_in")))
    vis.append(helper.make_tensor_value_info(data_4d, elem_type, [N, C, L_in, 1]))

    expand_to = uniq(f"{prefix}__expand_to")
    inits.append(const_i64(expand_to, [N, C, L_in, scale]))
    expanded = uniq(f"{prefix}__expanded")
    nodes.append(helper.make_node("Expand", [data_4d, expand_to], [expanded],
                                  name=uniq(f"{prefix}__expand")))
    vis.append(helper.make_tensor_value_info(expanded, elem_type, [N, C, L_in, scale]))

    shape_flat = uniq(f"{prefix}__shape_flat")
    inits.append(const_i64(shape_flat, [N, C, L_main]))

    if tail == 0:
        nodes.append(helper.make_node("Reshape", [expanded, shape_flat], [out],
                                      name=uniq(f"{prefix}__reshape_out")))
    else:
        flat = uniq(f"{prefix}__flat")
        nodes.append(helper.make_node("Reshape", [expanded, shape_flat], [flat],
                                      name=uniq(f"{prefix}__reshape_flat")))
        vis.append(helper.make_tensor_value_info(flat, elem_type, [N, C, L_main]))

        starts = uniq(f"{prefix}__tail_starts")
        ends = uniq(f"{prefix}__tail_ends")
        axes = uniq(f"{prefix}__tail_axes")
        inits.extend([
            const_i64(starts, [L_in - 1]),
            const_i64(ends, [L_in]),
            const_i64(axes, [2]),
        ])
        tail_slice = uniq(f"{prefix}__tail_slice")
        nodes.append(helper.make_node("Slice", [data, starts, ends, axes],
                                      [tail_slice],
                                      name=uniq(f"{prefix}__tail_slice_op")))
        vis.append(helper.make_tensor_value_info(tail_slice, elem_type, [N, C, 1]))
        nodes.append(helper.make_node("Concat", [flat, tail_slice], [out],
                                      name=uniq(f"{prefix}__concat_tail"),
                                      axis=2))

    return nodes, inits, vis


def make_clamped_expand_replacement(target, elem_type, used_names):
    """Reshape+Expand + arithmetic clamping for dynamic-index upsampling.

    Produces the same result as Gather(data, Min(indices, clamp_val), axis=2)
    without using Gather:

    1. Reshape+Expand gives unclamped result (same as integer-scale nearest-neighbor)
    2. Arithmetic mask identifies positions beyond the clamp boundary
    3. Dynamic Slice extracts the clamped value from data
    4. Blend: result = unclamped * (1-mask) + clamped_value * mask

    All ops are elementwise/Slice/Expand — no Gather, no DEQW.
    """
    data = target["data"]
    final_out = target["out"]
    N, C, L_in = target["N"], target["C"], target["L_in"]
    L_out = target["L_out"]
    scale = target["scale"]
    clamp_tensor = target["clamp_tensor"]
    prefix = final_out
    uniq = _make_uniq(used_names)

    all_nodes, all_inits, all_vis = [], [], []

    # Step 1: Reshape+Expand → unclamped intermediate
    unclamped = uniq(f"{prefix}__unclamped")
    e_nodes, e_inits, e_vis = make_expand_nodes(target, elem_type, used_names,
                                                 output_name=unclamped)
    all_nodes.extend(e_nodes)
    all_inits.extend(e_inits)
    all_vis.extend(e_vis)
    all_vis.append(helper.make_tensor_value_info(unclamped, elem_type, [N, C, L_out]))

    # Step 2: Compute clamp boundary in output space
    # clamp_start = (clamp_val + 1) * scale
    # Output positions >= clamp_start should use the clamped value
    one_i64 = uniq(f"{prefix}__one_i64")
    all_inits.append(const_i64(one_i64, 1))

    clamp_plus_one = uniq(f"{prefix}__clamp_plus_one")
    all_nodes.append(helper.make_node("Add", [clamp_tensor, one_i64], [clamp_plus_one],
                                      name=uniq(f"{prefix}__add_one")))
    all_vis.append(helper.make_tensor_value_info(clamp_plus_one, TensorProto.INT64, []))

    scale_i64 = uniq(f"{prefix}__scale_i64")
    all_inits.append(const_i64(scale_i64, scale))

    clamp_start = uniq(f"{prefix}__clamp_start")
    all_nodes.append(helper.make_node("Mul", [clamp_plus_one, scale_i64], [clamp_start],
                                      name=uniq(f"{prefix}__mul_scale")))
    all_vis.append(helper.make_tensor_value_info(clamp_start, TensorProto.INT64, []))

    # Step 3: Compute mask as bf16 {0.0, 1.0} using arithmetic (no BOOL ops)
    # diff = range - clamp_start  (INT64, exact)
    # mask = Clip(Sign(Cast(diff, bf16)) + 1, 0, 1)  (bf16)
    # mask=0 where range < clamp_start, mask=1 where range >= clamp_start
    range_i64 = uniq(f"{prefix}__range_i64")
    all_inits.append(numpy_helper.from_array(
        np.arange(L_out, dtype=np.int64), range_i64))

    diff_i64 = uniq(f"{prefix}__diff_i64")
    all_nodes.append(helper.make_node("Sub", [range_i64, clamp_start], [diff_i64],
                                      name=uniq(f"{prefix}__sub_range_clamp")))
    all_vis.append(helper.make_tensor_value_info(diff_i64, TensorProto.INT64, [L_out]))

    diff_f32 = uniq(f"{prefix}__diff_f32")
    all_nodes.append(helper.make_node("Cast", [diff_i64], [diff_f32],
                                      name=uniq(f"{prefix}__cast_diff_f32"),
                                      to=TensorProto.FLOAT))
    all_vis.append(helper.make_tensor_value_info(diff_f32, TensorProto.FLOAT, [L_out]))

    sign_f32 = uniq(f"{prefix}__sign_f32")
    all_nodes.append(helper.make_node("Sign", [diff_f32], [sign_f32],
                                      name=uniq(f"{prefix}__sign")))
    all_vis.append(helper.make_tensor_value_info(sign_f32, TensorProto.FLOAT, [L_out]))

    one_f32 = uniq(f"{prefix}__one_f32")
    all_inits.append(const_f32(one_f32, 1.0))

    sign_plus_one = uniq(f"{prefix}__sign_plus_one")
    all_nodes.append(helper.make_node("Add", [sign_f32, one_f32], [sign_plus_one],
                                      name=uniq(f"{prefix}__add_sign_one")))
    all_vis.append(helper.make_tensor_value_info(sign_plus_one, TensorProto.FLOAT, [L_out]))

    zero_f32 = uniq(f"{prefix}__zero_f32")
    all_inits.append(const_f32(zero_f32, 0.0))

    mask_f32 = uniq(f"{prefix}__mask_f32")
    all_nodes.append(helper.make_node("Clip", [sign_plus_one, zero_f32, one_f32],
                                      [mask_f32],
                                      name=uniq(f"{prefix}__clip_mask")))
    all_vis.append(helper.make_tensor_value_info(mask_f32, TensorProto.FLOAT, [L_out]))

    mask = uniq(f"{prefix}__mask")
    all_nodes.append(helper.make_node("Cast", [mask_f32], [mask],
                                      name=uniq(f"{prefix}__cast_mask"),
                                      to=elem_type))
    all_vis.append(helper.make_tensor_value_info(mask, elem_type, [L_out]))

    # Step 4: Extract clamped value using one-hot ReduceSum
    # Avoids dynamic Slice (compiler bug with --torq-convert-dtypes) and
    # avoids integer Clip(0,1) (compiled to i1, triggers strides mismatch).
    # Computes abs(diff) in INT64 (exact), converts to bf16, then uses
    # Relu(1 - abs) to get the one-hot indicator in bf16.
    range_lin = uniq(f"{prefix}__range_lin")
    all_inits.append(numpy_helper.from_array(
        np.arange(L_in, dtype=np.int64), range_lin))

    diff_lin = uniq(f"{prefix}__diff_lin")
    all_nodes.append(helper.make_node("Sub", [range_lin, clamp_tensor], [diff_lin],
                                      name=uniq(f"{prefix}__sub_onehot")))
    all_vis.append(helper.make_tensor_value_info(diff_lin, TensorProto.INT64, [L_in]))

    abs_lin = uniq(f"{prefix}__abs_lin")
    all_nodes.append(helper.make_node("Abs", [diff_lin], [abs_lin],
                                      name=uniq(f"{prefix}__abs_onehot")))
    all_vis.append(helper.make_tensor_value_info(abs_lin, TensorProto.INT64, [L_in]))

    abs_bf16 = uniq(f"{prefix}__abs_bf16")
    all_nodes.append(helper.make_node("Cast", [abs_lin], [abs_bf16],
                                      name=uniq(f"{prefix}__cast_abs"),
                                      to=elem_type))
    all_vis.append(helper.make_tensor_value_info(abs_bf16, elem_type, [L_in]))

    one_for_ind = uniq(f"{prefix}__one_for_ind")
    all_inits.append(const_f32(one_for_ind, 1.0))
    one_ind_cast = uniq(f"{prefix}__one_ind_cast")
    all_nodes.append(helper.make_node("Cast", [one_for_ind], [one_ind_cast],
                                      name=uniq(f"{prefix}__cast_one_ind"),
                                      to=elem_type))
    all_vis.append(helper.make_tensor_value_info(one_ind_cast, elem_type, []))

    raw_indicator = uniq(f"{prefix}__raw_indicator")
    all_nodes.append(helper.make_node("Sub", [one_ind_cast, abs_bf16], [raw_indicator],
                                      name=uniq(f"{prefix}__sub_one_abs")))
    all_vis.append(helper.make_tensor_value_info(raw_indicator, elem_type, [L_in]))

    indicator = uniq(f"{prefix}__indicator")
    all_nodes.append(helper.make_node("Relu", [raw_indicator], [indicator],
                                      name=uniq(f"{prefix}__relu_indicator")))
    all_vis.append(helper.make_tensor_value_info(indicator, elem_type, [L_in]))

    selected = uniq(f"{prefix}__selected")
    all_nodes.append(helper.make_node("Mul", [data, indicator], [selected],
                                      name=uniq(f"{prefix}__mul_onehot")))
    all_vis.append(helper.make_tensor_value_info(selected, elem_type, [N, C, L_in]))

    reduce_axes = uniq(f"{prefix}__reduce_axes")
    all_inits.append(const_i64(reduce_axes, [2]))

    clamp_val = uniq(f"{prefix}__clamp_val")
    all_nodes.append(helper.make_node("ReduceSum", [selected, reduce_axes], [clamp_val],
                                      name=uniq(f"{prefix}__reduce_clamp_val"),
                                      keepdims=1))
    all_vis.append(helper.make_tensor_value_info(clamp_val, elem_type, [N, C, 1]))

    # Step 5: Blend unclamped and clamped values
    # result = unclamped * (1 - mask) + clamp_val * mask
    # Broadcasting: mask [L_out] broadcasts with [N,C,L_out]
    #               clamp_val [N,C,1] broadcasts with [L_out] → [N,C,L_out]
    inv_mask = uniq(f"{prefix}__inv_mask")
    one_for_inv = uniq(f"{prefix}__one_for_inv")
    all_inits.append(const_f32(one_for_inv, 1.0))
    one_cast = uniq(f"{prefix}__one_cast")
    all_nodes.append(helper.make_node("Cast", [one_for_inv], [one_cast],
                                      name=uniq(f"{prefix}__cast_one_inv"),
                                      to=elem_type))
    all_vis.append(helper.make_tensor_value_info(one_cast, elem_type, []))

    all_nodes.append(helper.make_node("Sub", [one_cast, mask], [inv_mask],
                                      name=uniq(f"{prefix}__sub_inv_mask")))
    all_vis.append(helper.make_tensor_value_info(inv_mask, elem_type, [L_out]))

    unclamped_masked = uniq(f"{prefix}__unclamped_masked")
    all_nodes.append(helper.make_node("Mul", [unclamped, inv_mask], [unclamped_masked],
                                      name=uniq(f"{prefix}__mul_unclamped")))
    all_vis.append(helper.make_tensor_value_info(unclamped_masked, elem_type, [N, C, L_out]))

    clamped_masked = uniq(f"{prefix}__clamped_masked")
    all_nodes.append(helper.make_node("Mul", [clamp_val, mask], [clamped_masked],
                                      name=uniq(f"{prefix}__mul_clamped")))
    all_vis.append(helper.make_tensor_value_info(clamped_masked, elem_type, [N, C, L_out]))

    all_nodes.append(helper.make_node("Add", [unclamped_masked, clamped_masked],
                                      [final_out],
                                      name=uniq(f"{prefix}__add_blend")))

    return all_nodes, all_inits, all_vis


def rewrite_model(model):
    graph = model.graph
    shapes = static_shapes(graph)
    types = elem_types(graph)

    targets = find_targets(graph, shapes)
    if not targets:
        print("No upsample_nearest1d Gather nodes found")
        return []

    by_id = {id(t["node"]): t for t in targets}
    used = set(shapes.keys()) | {n.name for n in graph.node if n.name}
    used.update(init.name for init in graph.initializer)

    new_nodes = []
    new_inits = []
    new_vis = []

    for node in graph.node:
        t = by_id.get(id(node))
        if t is None:
            new_nodes.append(node)
            continue

        et = types.get(t["out"], TensorProto.BFLOAT16)

        if t["is_static_repeat"]:
            print(f"  {t['name']}: static repeat -> Reshape+Expand "
                  f"(L_in={t['L_in']} scale={t['scale']} tail={t['tail']})")
            n, i, v = make_expand_nodes(t, et, used)
        elif t["clamp_tensor"]:
            print(f"  {t['name']}: dynamic clamped -> Expand+ClampCorrection "
                  f"(L_in={t['L_in']} scale={t['scale']} tail={t['tail']} "
                  f"clamp={t['clamp_tensor']})")
            n, i, v = make_clamped_expand_replacement(t, et, used)
        else:
            print(f"  {t['name']}: dynamic unclamped -> Reshape+Expand only "
                  f"(L_in={t['L_in']} scale={t['scale']} tail={t['tail']}) "
                  f"WARNING: no clamping found, accuracy may differ")
            n, i, v = make_expand_nodes(t, et, used)

        new_nodes.extend(n)
        new_inits.extend(i)
        new_vis.extend(v)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    targets = rewrite_model(model)

    print(f"Rewrote {len(targets)} upsample Gather nodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))

    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")

    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
