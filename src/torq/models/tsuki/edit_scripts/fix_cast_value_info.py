#!/usr/bin/env python3
"""Propagate Cast output types through stale value_info.

When a pass inserts a ``Cast`` between two ops but forgets to update the
downstream ``value_info``, the model claims tensors are still the *pre-cast*
type. A compiler that trusts ``value_info`` over the ``Cast.to`` attribute then
emits the wrong dtype to downstream ops (e.g. ``Unsqueeze`` on ``BOOL`` when
the intended type is ``BFLOAT16``).

For every ``Cast`` whose output's stored type disagrees with its ``to`` attr,
this script:
  1. fixes the cast output's ``value_info``,
  2. forward-propagates the new type through elementwise / shape-only ops as
     long as the descendant still carries the OLD (pre-cast) type.
Propagation stops at another ``Cast`` (honours its own ``to``), at an op that
is not in the transparent list, or when the descendant's stored type already
differs from the OLD type.
"""

import argparse
from pathlib import Path

import onnx


# Ops whose output element type equals their first input's element type.
# Conservative list; expand as needed.
TRANSPARENT_OPS = frozenset({
    "Unsqueeze", "Squeeze", "Reshape", "Transpose", "Expand", "Tile",
    "Slice", "Gather", "GatherElements", "GatherND", "Concat", "Identity",
    "Pad", "Flatten", "Split", "DepthToSpace", "SpaceToDepth",
    "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Sum", "Mean",
    "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Exp", "Log", "Abs", "Neg",
    "Sqrt", "Reciprocal", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
    "Floor", "Ceil", "Round", "Erf", "Sign", "Softmax", "LogSoftmax",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceSumSquare", "ReduceLogSum",
    "ReduceLogSumExp", "Clip", "PRelu", "Selu", "Elu", "HardSigmoid",
    "ThresholdedRelu",
})


def dtype_name(t):
    try:
        return onnx.TensorProto.DataType.Name(int(t))
    except ValueError:
        return str(t)


def fix_model(model):
    graph = model.graph

    vi_map = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            vi_map[vi.name] = vi
    init_types = {init.name: int(init.data_type) for init in graph.initializer}

    consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)

    fixes = []

    def propagate(tensor_name, old_type, new_type, visited):
        if tensor_name in visited:
            return
        visited.add(tensor_name)
        for cnode in consumers.get(tensor_name, []):
            if cnode.op_type == "Cast":
                continue  # has its own `to`
            if cnode.op_type not in TRANSPARENT_OPS:
                continue
            for out in cnode.output:
                if not out:
                    continue
                vi = vi_map.get(out)
                if vi is None or not vi.type.HasField("tensor_type"):
                    continue
                cur = int(vi.type.tensor_type.elem_type)
                if cur == old_type:
                    vi.type.tensor_type.elem_type = new_type
                    fixes.append({
                        "tensor": out,
                        "via": cnode.op_type,
                        "via_node": cnode.name or "<unnamed>",
                        "from": dtype_name(old_type),
                        "to": dtype_name(new_type),
                        "kind": "propagated",
                    })
                    propagate(out, old_type, new_type, visited)
                # else: already a different (likely correct) type — leave alone

    for node in graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = None
        for attr in node.attribute:
            if attr.name == "to":
                to_attr = int(attr.i)
                break
        if to_attr is None:
            continue
        out_name = node.output[0]
        out_vi = vi_map.get(out_name)
        if out_vi is None or not out_vi.type.HasField("tensor_type"):
            continue
        cur = int(out_vi.type.tensor_type.elem_type)
        if cur == to_attr:
            continue

        inp_name = node.input[0]
        if inp_name in vi_map and vi_map[inp_name].type.HasField("tensor_type"):
            old_type = int(vi_map[inp_name].type.tensor_type.elem_type)
        elif inp_name in init_types:
            old_type = init_types[inp_name]
        else:
            old_type = cur  # best guess: stored output type matched input pre-cast

        out_vi.type.tensor_type.elem_type = to_attr
        fixes.append({
            "tensor": out_name,
            "via": "Cast",
            "via_node": node.name or "<unnamed>",
            "from": dtype_name(cur),
            "to": dtype_name(to_attr),
            "kind": "cast_output",
        })
        propagate(out_name, old_type, to_attr, set())

    return fixes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Output path (defaults to overwriting --input).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print fixes but do not write.")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    fixes = fix_model(model)

    cast_out = sum(1 for f in fixes if f["kind"] == "cast_output")
    prop = sum(1 for f in fixes if f["kind"] == "propagated")
    print(f"Cast outputs corrected: {cast_out}")
    print(f"Descendants re-typed:   {prop}")
    for f in fixes:
        print(f"  [{f['kind']:>10}] {f['tensor']}: {f['from']} -> {f['to']}"
              f"  (via {f['via']} '{f['via_node']}')")

    if args.dry_run:
        return
    out = args.output or args.input
    onnx.save(model, str(out))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
