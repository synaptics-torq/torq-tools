#!/usr/bin/env python3
"""Replace `Slice(data, [dyn_idx], [dyn_idx+1], [axis], [1])` with an
element-wise one-hot extraction: Equal + Mul + ReduceSum.

The motivating crash: torq-compile hits "Unsupported dynamic offsets" when
lowering a Slice whose start/end indices are computed at runtime. This happens
for the edge-padding pattern that extracts the last valid audio sample:
`audio[:, sample_len-1 : sample_len]`.

The one-hot workaround avoids dynamic tensor indexing entirely:
  1. Expand the scalar index to the axis dimension size: [N]
  2. Equal(range(0..N-1), expanded_idx) → one-hot bool mask [N]
  3. Unsqueeze to match data rank
  4. Cast bool → data dtype
  5. Mul(data, mask) → zeros except at target position
  6. ReduceSum(axis) → extracted element [1]

Targets the narrow safe pattern:
  - Slice with 5 inputs (data, starts, ends, axes, steps)
  - Output has exactly 1 on the slice axis (single-element extraction)
  - starts is NOT a constant initializer (dynamic index)
  - starts has shape [1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def const_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def build_static_shapes(graph):
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


def build_types(graph):
    out = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        out.setdefault(init.name, init.data_type)
    return out


def find_targets(graph, shapes):
    inits = {init.name for init in graph.initializer}
    targets = []
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) != 5:
            continue
        data_name, starts_name, ends_name, axes_name, steps_name = node.input
        data_shape = shapes.get(data_name)
        out_shape = shapes.get(node.output[0])
        starts_shape = shapes.get(starts_name)
        if not data_shape or not out_shape or not starts_shape:
            continue
        if starts_shape != [1]:
            continue
        # Skip constant-index slices — those compile fine
        if starts_name in inits:
            continue
        # Only target single-element extraction (output axis dim == 1)
        if len(data_shape) != len(out_shape):
            continue
        axis_dim = None
        for i in range(len(data_shape)):
            if data_shape[i] != out_shape[i]:
                if out_shape[i] == 1 and axis_dim is None:
                    axis_dim = i
                else:
                    axis_dim = None
                    break
        if axis_dim is None:
            continue

        targets.append({
            "node": node,
            "name": node.name,
            "data": data_name,
            "starts": starts_name,
            "out": node.output[0],
            "axis": axis_dim,
            "axis_size": data_shape[axis_dim],
            "data_shape": data_shape,
            "out_shape": out_shape,
            "data_dtype": None,  # filled below
        })
    return targets


def make_replacement(target, used_names, data_dtype):
    prefix = target["name"] or target["out"]
    axis = target["axis"]
    axis_size = target["axis_size"]
    data = target["data"]
    starts = target["starts"]
    out = target["out"]
    data_shape = target["data_shape"]

    inits = []
    nodes = []
    vis = []

    def uniq(base):
        name = base
        n = 0
        while name in used_names:
            n += 1
            name = f"{base}__{n}"
        used_names.add(name)
        return name

    # The starts tensor has shape [1] — we need the scalar index value.
    # Squeeze it to scalar, then Expand to the axis dimension size.
    squeeze_axes_name = uniq(f"{prefix}__squeeze_axes")
    inits.append(const_i64(squeeze_axes_name, [0]))
    idx_scalar = uniq(f"{prefix}__idx_scalar")
    nodes.append(helper.make_node(
        "Squeeze", [starts, squeeze_axes_name], [idx_scalar],
        name=uniq(f"{prefix}__squeeze_idx"),
    ))

    expand_shape_name = uniq(f"{prefix}__expand_shape")
    inits.append(const_i64(expand_shape_name, [axis_size]))
    idx_expanded = uniq(f"{prefix}__idx_expanded")
    nodes.append(helper.make_node(
        "Expand", [idx_scalar, expand_shape_name], [idx_expanded],
        name=uniq(f"{prefix}__expand_idx"),
    ))

    # Range constant [0, 1, ..., axis_size-1]
    range_name = uniq(f"{prefix}__range_{axis_size}")
    inits.append(numpy_helper.from_array(
        np.arange(axis_size, dtype=np.int64), range_name))

    # Equal(range, idx_expanded) → one-hot bool mask
    mask_1d = uniq(f"{prefix}__onehot_mask")
    nodes.append(helper.make_node(
        "Equal", [range_name, idx_expanded], [mask_1d],
        name=uniq(f"{prefix}__equal"),
    ))

    # Unsqueeze to match data rank: insert dims for all axes except the target
    unsqueeze_axes = [i for i in range(len(data_shape)) if i != axis]
    if unsqueeze_axes:
        unsqueeze_axes_name = uniq(f"{prefix}__unsqueeze_axes")
        inits.append(const_i64(unsqueeze_axes_name, unsqueeze_axes))
        mask_nd = uniq(f"{prefix}__mask_nd")
        nodes.append(helper.make_node(
            "Unsqueeze", [mask_1d, unsqueeze_axes_name], [mask_nd],
            name=uniq(f"{prefix}__unsqueeze_mask"),
        ))
    else:
        mask_nd = mask_1d

    # Cast bool → data dtype
    mask_typed = uniq(f"{prefix}__mask_typed")
    nodes.append(helper.make_node(
        "Cast", [mask_nd], [mask_typed],
        name=uniq(f"{prefix}__cast_mask"),
        to=data_dtype,
    ))

    # Mul(data, mask) — zeros except at target position
    masked = uniq(f"{prefix}__masked")
    nodes.append(helper.make_node(
        "Mul", [data, mask_typed], [masked],
        name=uniq(f"{prefix}__mul"),
    ))

    # ReduceSum along the target axis → collapse to 1
    reduce_axes_name = uniq(f"{prefix}__reduce_axes")
    inits.append(const_i64(reduce_axes_name, [axis]))
    nodes.append(helper.make_node(
        "ReduceSum", [masked, reduce_axes_name], [out],
        name=uniq(f"{prefix}__reduce_sum"),
        keepdims=1,
    ))

    return nodes, inits, vis


def prune_unused(graph):
    producer = {o: n for n in graph.node for o in n.output if o}
    keep_tensors = {o.name for o in graph.output} | {i.name for i in graph.input}
    keep_nodes = set()
    work = list(keep_tensors)
    while work:
        t = work.pop()
        p = producer.get(t)
        if p is None:
            continue
        nid = id(p)
        if nid in keep_nodes:
            continue
        keep_nodes.add(nid)
        for inp in p.input:
            if inp and inp not in keep_tensors:
                keep_tensors.add(inp)
                work.append(inp)
        for outp in p.output:
            if outp:
                keep_tensors.add(outp)

    kn = [n for n in graph.node if id(n) in keep_nodes]
    dropped_n = len(graph.node) - len(kn)
    del graph.node[:]
    graph.node.extend(kn)

    ki = [init for init in graph.initializer if init.name in keep_tensors]
    dropped_i = len(graph.initializer) - len(ki)
    del graph.initializer[:]
    graph.initializer.extend(ki)

    kv = [vi for vi in graph.value_info if vi.name in keep_tensors]
    dropped_v = len(graph.value_info) - len(kv)
    del graph.value_info[:]
    graph.value_info.extend(kv)

    return dropped_n, dropped_i, dropped_v


def rewrite_model(model):
    graph = model.graph
    shapes = build_static_shapes(graph)
    types = build_types(graph)
    targets = find_targets(graph, shapes)
    if not targets:
        return [], 0, 0, 0

    for t in targets:
        t["data_dtype"] = types.get(t["data"], TensorProto.BFLOAT16)

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
        nodes, inits, vis = make_replacement(t, used, t["data_dtype"])
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
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shape-infer", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    targets, dn, di, dv = rewrite_model(model)

    print(f"Dynamic Slice → one-hot extraction rewrites: {len(targets)}")
    for t in targets:
        print(f"  {t['name']:50s}  data_shape={t['data_shape']}  "
              f"axis={t['axis']}  axis_size={t['axis_size']}  "
              f"starts={t['starts']}")
    print(f"Pruned by reachability: nodes={dn} initializers={di} value_info={dv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    if args.shape_infer:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        print("Strict shape inference: OK")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
