#!/usr/bin/env python3
"""Replace `ScatterElements` with a manual per-position `Slice + Where + Concat`
decomposition when the scatter is small.

Motivating crash: torq-compile's ScatterElements lowering emits index-arithmetic
(`arith.remsi`) for runtime indices into big tensors and trips an SSA dominance
violation. Decomposing into a small fan-out of structural ops sidesteps it.

Targets the narrow safe pattern:
  - data shape [N1, ..., Nk], axis = A
  - indices/updates shape [Ni1, ..., Nik] where the K-th dim is small
    (the "scatter count" — we generate one Where per element along that dim)
  - reduction = "none"

For each scatter position k along the dimension we fan out on, we emit:
  - Slice(input, axis=other_axis, start=k, end=k+1)  → input slice for channel k
  - Slice(indices, axis=other_axis, start=k, end=k+1) → pos_k (runtime int64, [1,1,1])
  - Slice(updates, axis=other_axis, start=k, end=k+1) → upd_k (runtime bf16, [1,1,1])
  - Equal(arange, pos_k_broadcast) → mask of shape [1, N_A, 1] (bool)
  - Where(mask, upd_k_broadcast, input_slice) → updated slice
Finally Concat all the updated slices along other_axis.

`arange` is a single shared INT64 initializer over the scatter axis (no Range op).

CLI:
  python3 scripts/replace_small_scatter_elements.py --input <onnx> --output <onnx>
                                                     [--target-output <name>]
                                                     [--max-positions 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# Maximum positions we're willing to fan out for. Above this, fan-out gets large.
DEFAULT_MAX_POSITIONS = 16


def const_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def get_attr_str(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            v = attr.s
            return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
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


def find_targets(graph, shapes, target_output, max_positions):
    """Find ScatterElements ops we'll rewrite."""
    targets = []
    for node in graph.node:
        if node.op_type != "ScatterElements":
            continue
        if target_output and node.output[0] != target_output:
            continue
        if get_attr_str(node, "reduction", "none") != "none":
            continue
        data_shape = shapes.get(node.input[0])
        idx_shape = shapes.get(node.input[1])
        upd_shape = shapes.get(node.input[2])
        out_shape = shapes.get(node.output[0])
        if not data_shape or not idx_shape or not upd_shape or not out_shape:
            continue
        if idx_shape != upd_shape:
            continue
        axis = get_attr_int(node, "axis", 0)
        if axis < 0:
            axis += len(data_shape)
        # Find the "fan-out" axis: the non-`axis` dim that's > 1
        fan_out_axis = None
        for a in range(len(data_shape)):
            if a == axis:
                continue
            if idx_shape[a] > 1:
                if fan_out_axis is not None:
                    # multiple > 1 dims; out of scope
                    fan_out_axis = None
                    break
                fan_out_axis = a
        if fan_out_axis is None:
            # All non-axis dims are 1 → only one scatter position. Still works.
            # Default to the last non-axis dim for fan-out.
            for a in reversed(range(len(data_shape))):
                if a != axis:
                    fan_out_axis = a; break
        if fan_out_axis is None:
            continue
        positions = idx_shape[fan_out_axis]
        if positions > max_positions:
            continue
        targets.append({
            "node": node,
            "name": node.name,
            "data": node.input[0],
            "indices": node.input[1],
            "updates": node.input[2],
            "out": node.output[0],
            "axis": axis,
            "fan_axis": fan_out_axis,
            "data_shape": data_shape,
            "idx_shape": idx_shape,
            "positions": positions,
        })
    return targets


def make_replacement(target, types, used_names):
    """Emit (nodes, initializers, value_infos) for one ScatterElements rewrite."""
    data = target["data"]
    idx_full = target["indices"]
    upd_full = target["updates"]
    out = target["out"]
    axis = target["axis"]
    fan_axis = target["fan_axis"]
    data_shape = target["data_shape"]
    n_pos = target["positions"]
    prefix = out
    data_dtype = types.get(data, TensorProto.BFLOAT16)

    def uniq(base):
        name = base
        i = 0
        while name in used_names:
            i += 1
            name = f"{base}__{i}"
        used_names.add(name)
        return name

    inits = []
    nodes = []
    vis = []

    # Shared arange [N_axis] INT64
    n_axis_dim = data_shape[axis]
    arange_name = uniq(f"{prefix}__arange_axis_{axis}")
    inits.append(numpy_helper.from_array(
        np.arange(n_axis_dim, dtype=np.int64), name=arange_name))

    # Shape of arange after reshape into the data layout
    # We want mask shape to be broadcastable to [1, ..., N_axis, ..., 1]
    arange_reshape_target = [1] * len(data_shape)
    arange_reshape_target[axis] = n_axis_dim
    arange_reshape_name = uniq(f"{prefix}__arange_reshape_shape")
    inits.append(const_i64(arange_reshape_name, arange_reshape_target))
    arange_reshaped = uniq(f"{prefix}__arange_reshaped")
    nodes.append(helper.make_node(
        "Reshape", [arange_name, arange_reshape_name], [arange_reshaped],
        name=uniq(f"{prefix}__arange_reshape")))
    vis.append(helper.make_tensor_value_info(arange_reshaped, TensorProto.INT64, arange_reshape_target))

    per_channel_outputs = []
    # For each scatter position k along fan_axis
    for k in range(n_pos):
        # Slice the data on fan_axis from k to k+1 → shape with that dim = 1
        starts_name = uniq(f"{prefix}__slice_starts_{k}")
        ends_name = uniq(f"{prefix}__slice_ends_{k}")
        axes_name = uniq(f"{prefix}__slice_axes_{k}")
        steps_name = uniq(f"{prefix}__slice_steps_{k}")
        inits.extend([
            const_i64(starts_name, [k]),
            const_i64(ends_name, [k + 1]),
            const_i64(axes_name, [fan_axis]),
            const_i64(steps_name, [1]),
        ])
        data_slice = uniq(f"{prefix}__data_slice_{k}")
        nodes.append(helper.make_node(
            "Slice", [data, starts_name, ends_name, axes_name, steps_name],
            [data_slice], name=uniq(f"{prefix}__data_slice_op_{k}")))
        sliced_shape = list(data_shape); sliced_shape[fan_axis] = 1
        vis.append(helper.make_tensor_value_info(data_slice, data_dtype, sliced_shape))

        # Slice indices and updates the same way
        idx_slice = uniq(f"{prefix}__idx_slice_{k}")
        nodes.append(helper.make_node(
            "Slice", [idx_full, starts_name, ends_name, axes_name, steps_name],
            [idx_slice], name=uniq(f"{prefix}__idx_slice_op_{k}")))
        idx_sliced_shape = list(target["idx_shape"]); idx_sliced_shape[fan_axis] = 1
        vis.append(helper.make_tensor_value_info(idx_slice, TensorProto.INT64, idx_sliced_shape))

        upd_slice = uniq(f"{prefix}__upd_slice_{k}")
        nodes.append(helper.make_node(
            "Slice", [upd_full, starts_name, ends_name, axes_name, steps_name],
            [upd_slice], name=uniq(f"{prefix}__upd_slice_op_{k}")))
        vis.append(helper.make_tensor_value_info(upd_slice, data_dtype, idx_sliced_shape))

        # Build mask: Equal(arange_reshaped, idx_slice_broadcast)
        # arange_reshaped has 1 on every non-axis dim and N_axis on `axis`.
        # idx_slice has 1 on every dim. Broadcast → same shape as data_slice.
        mask_name = uniq(f"{prefix}__mask_{k}")
        nodes.append(helper.make_node(
            "Equal", [arange_reshaped, idx_slice], [mask_name],
            name=uniq(f"{prefix}__equal_{k}")))
        vis.append(helper.make_tensor_value_info(mask_name, TensorProto.BOOL, sliced_shape))

        # Where(mask, upd_slice_broadcast, data_slice) — Where broadcasts naturally.
        merged = uniq(f"{prefix}__merged_{k}")
        nodes.append(helper.make_node(
            "Where", [mask_name, upd_slice, data_slice], [merged],
            name=uniq(f"{prefix}__where_{k}")))
        vis.append(helper.make_tensor_value_info(merged, data_dtype, sliced_shape))

        per_channel_outputs.append(merged)

    # Final Concat along fan_axis
    nodes.append(helper.make_node(
        "Concat", per_channel_outputs, [out],
        name=uniq(f"{prefix}__concat_fan"), axis=fan_axis))

    return nodes, inits, vis


def prune_unused(graph):
    producer = {o: n for n in graph.node for o in n.output if o}
    keep = {o.name for o in graph.output} | {i.name for i in graph.input}
    keep_nodes = set()
    work = list(keep)
    while work:
        t = work.pop()
        p = producer.get(t)
        if p is None or id(p) in keep_nodes: continue
        keep_nodes.add(id(p))
        for i in p.input:
            if i and i not in keep:
                keep.add(i); work.append(i)
        for o in p.output:
            if o: keep.add(o)

    kn = [n for n in graph.node if id(n) in keep_nodes]
    dn = len(graph.node) - len(kn); del graph.node[:]; graph.node.extend(kn)
    ki = [init for init in graph.initializer if init.name in keep]
    di = len(graph.initializer) - len(ki); del graph.initializer[:]; graph.initializer.extend(ki)
    kv = [vi for vi in graph.value_info if vi.name in keep]
    dv = len(graph.value_info) - len(kv); del graph.value_info[:]; graph.value_info.extend(kv)
    return dn, di, dv


def rewrite_model(model, target_output, max_positions):
    graph = model.graph
    shapes = static_shapes(graph)
    types = elem_types(graph)

    targets = find_targets(graph, shapes, target_output, max_positions)
    if not targets:
        return [], 0, 0, 0

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
        nodes, inits, vis = make_replacement(t, types, used)
        new_nodes.extend(nodes)
        new_inits.extend(inits)
        new_vis.extend(vis)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)

    dropped = prune_unused(graph)
    return targets, *dropped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-output", default=None,
                   help="Only rewrite ScatterElements with this output name.")
    p.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shape-infer", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    targets, dn, di, dv = rewrite_model(model, args.target_output, args.max_positions)

    print(f"ScatterElements rewrites: {len(targets)}")
    for t in targets:
        print(f"  {t['name']:40s}  data_shape={t['data_shape']} axis={t['axis']} "
              f"fan_axis={t['fan_axis']} positions={t['positions']}")
    print(f"Pruned by reachability: nodes={dn} initializers={di} value_info={dv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        onnx.checker.check_model(model); print("ONNX checker: OK")
    if args.shape_infer:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        print("Strict shape inference: OK")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
