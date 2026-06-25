#!/usr/bin/env python3
"""Replace `Gather(data, [idx], axis=A)` → size-1-on-axis-A output with an
equivalent `Slice(data, start=[idx], end=[idx+1], axes=[A], steps=[1])`.

The motivating crash: torq-compile's `Gather → tensor.extract` lowering hits
an SSA dominance violation when the indices are produced at runtime against a
large data tensor (`tensor<1x96000xbf16>` for our audio edge-padding case).
Slicing produces the same `[N1, ..., 1, ..., Nk]` shape via pure shape ops
that the compiler already handles cleanly.

Targets only the narrow safe pattern:
  - 2 inputs (data, indices)
  - indices shape is exactly `[1]`
  - output's `axis` dim is 1, and all other dims match the data tensor
This is the single-element gather that's interchangeable with a Slice.

For dynamic indices, we emit `Slice` with the indices tensor as `starts`, then
build `ends = indices + 1` via an `Add`. For constant indices, the bounds are
folded into constant initializers (cheaper for the runtime).
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


def shape_of(graph, shapes, name):
    if name in shapes:
        return shapes[name]
    return None


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
    inits = {init.name: init for init in graph.initializer}
    targets = []
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) != 2:
            continue
        data_shape = shapes.get(node.input[0])
        idx_shape = shapes.get(node.input[1])
        out_shape = shapes.get(node.output[0])
        if data_shape is None or idx_shape is None or out_shape is None:
            continue
        is_scalar = idx_shape == []
        if idx_shape != [1] and not is_scalar:
            continue
        if is_scalar:
            idx_name = node.input[1]
            if idx_name not in inits:
                continue
        axis = get_attr_int(node, "axis", 0)
        if axis < 0:
            axis += len(data_shape)
        if axis >= len(data_shape):
            continue
        if is_scalar:
            expected_out = list(data_shape)
            del expected_out[axis]
        else:
            expected_out = list(data_shape)
            expected_out[axis] = 1
        if list(out_shape) != expected_out:
            continue
        idx_name = node.input[1]
        is_const = idx_name in inits
        targets.append({
            "node": node,
            "name": node.name,
            "data": node.input[0],
            "indices": idx_name,
            "out": node.output[0],
            "axis": axis,
            "data_shape": data_shape,
            "is_const_index": is_const,
            "is_scalar_index": is_scalar,
            "const_value": (int(numpy_helper.to_array(inits[idx_name]).reshape(-1)[0])
                            if is_const else None),
        })
    return targets


def make_replacement(target, used_names, out_dtype):
    """Emit (nodes, initializers, value_infos) replacing one Gather with a Slice."""
    prefix = target["out"]
    axis = target["axis"]
    data = target["data"]
    out = target["out"]

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

    axes_name = uniq(f"{prefix}__slice_axes")
    steps_name = uniq(f"{prefix}__slice_steps")
    inits.append(const_i64(axes_name, [axis]))
    inits.append(const_i64(steps_name, [1]))

    needs_squeeze = target.get("is_scalar_index", False)
    slice_out = uniq(f"{prefix}__sliced") if needs_squeeze else out

    if target["is_const_index"]:
        k = target["const_value"]
        starts_name = uniq(f"{prefix}__slice_starts")
        ends_name = uniq(f"{prefix}__slice_ends")
        inits.append(const_i64(starts_name, [k]))
        inits.append(const_i64(ends_name, [k + 1]))
        nodes.append(helper.make_node(
            "Slice",
            [data, starts_name, ends_name, axes_name, steps_name],
            [slice_out],
            name=uniq(f"{prefix}__as_slice"),
        ))
    else:
        one_name = uniq(f"{prefix}__one_i64")
        inits.append(const_i64(one_name, [1]))
        ends_name = uniq(f"{prefix}__slice_ends")
        nodes.append(helper.make_node(
            "Add",
            [target["indices"], one_name],
            [ends_name],
            name=uniq(f"{prefix}__compute_ends"),
        ))
        vis.append(helper.make_tensor_value_info(ends_name, TensorProto.INT64, [1]))

        nodes.append(helper.make_node(
            "Slice",
            [data, target["indices"], ends_name, axes_name, steps_name],
            [slice_out],
            name=uniq(f"{prefix}__as_slice"),
        ))

    if needs_squeeze:
        sq_axes_name = uniq(f"{prefix}__squeeze_axes")
        inits.append(const_i64(sq_axes_name, [axis]))
        nodes.append(helper.make_node(
            "Squeeze",
            [slice_out, sq_axes_name],
            [out],
            name=uniq(f"{prefix}__squeeze"),
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
                keep_tensors.add(inp); work.append(inp)
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
        print("ASDF", node.name)
        nodes, inits, vis = make_replacement(t, used, types.get(t["out"], TensorProto.BFLOAT16))
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

    print(f"Single-index Gather → Slice rewrites: {len(targets)}")
    for t in targets:
        idx = f"const[{t['const_value']}]" if t["is_const_index"] else f"dyn({t['indices']})"
        print(f"  {t['name']:50s}  data_shape={t['data_shape']}  axis={t['axis']}  idx={idx}")
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
