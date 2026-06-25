#!/usr/bin/env python3
"""Wrap every INT64 x INT64 -> INT64 Mul with Cast(BF16) -> Mul(BF16) -> Cast(INT64).

Target hardware does not support integer Mul. Each affected node is replaced
with a 4-node group:

    a_int64 ──┐
              ├─ Cast(BF16) ─┐
                              ├─ Mul(BF16) ─ Cast(INT64) ─ <original output>
    b_int64 ─ Cast(BF16) ────┘

SAFETY CAVEAT: BF16 represents integers exactly only in [0, 256]. The script
does NOT prove that operand or product values stay in that band — it assumes
they do (this is the case for the tsuki model's index-arithmetic Muls). For
any Mul whose values can grow past 256 this rewrite will silently corrupt
results. Use FP32 instead in that case.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_type_map(graph):
    types = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        types.setdefault(init.name, init.data_type)
    return types


def build_shape_map(graph):
    shapes = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append(None)
        shapes[vi.name] = dims
    for init in graph.initializer:
        shapes.setdefault(init.name, list(init.dims))
    return shapes


def make_value_info(name, elem_type, dims):
    if dims is None:
        vi = onnx.ValueInfoProto()
        vi.name = name
        vi.type.tensor_type.elem_type = elem_type
        return vi
    return helper.make_tensor_value_info(name, elem_type, dims)


def unique(base, used):
    name = base
    i = 0
    while name in used:
        i += 1
        name = f"{base}__{i}"
    used.add(name)
    return name


def patch(model):
    graph = model.graph
    types = build_type_map(graph)
    shapes = build_shape_map(graph)
    used = set(types.keys()) | {n.name for n in graph.node if n.name}

    INT64 = TensorProto.INT64
    BF16 = TensorProto.BFLOAT16

    patched = []
    new_nodes = []
    new_vis = []

    for node in graph.node:
        is_target = (
            node.op_type == "Mul"
            and len(node.input) == 2
            and types.get(node.input[0]) == INT64
            and types.get(node.input[1]) == INT64
            and len(node.output) == 1
            and types.get(node.output[0]) == INT64
        )
        if not is_target:
            new_nodes.append(node)
            continue

        a, b = node.input[0], node.input[1]
        out = node.output[0]
        op_name = node.name or out

        a_bf = unique(f"{a}_bf16_for_{op_name}", used)
        b_bf = unique(f"{b}_bf16_for_{op_name}", used)
        mul_bf = unique(f"{out}_bf16", used)

        new_nodes.append(helper.make_node(
            "Cast", [a], [a_bf], name=f"{op_name}__cast_a_bf16", to=BF16))
        new_nodes.append(helper.make_node(
            "Cast", [b], [b_bf], name=f"{op_name}__cast_b_bf16", to=BF16))

        mul_bf16 = onnx.NodeProto()
        mul_bf16.CopyFrom(node)
        del mul_bf16.input[:]
        mul_bf16.input.extend([a_bf, b_bf])
        del mul_bf16.output[:]
        mul_bf16.output.append(mul_bf)
        new_nodes.append(mul_bf16)

        new_nodes.append(helper.make_node(
            "Cast", [mul_bf], [out], name=f"{op_name}__cast_back_int64", to=INT64))

        new_vis.append(make_value_info(a_bf, BF16, shapes.get(a)))
        new_vis.append(make_value_info(b_bf, BF16, shapes.get(b)))
        new_vis.append(make_value_info(mul_bf, BF16, shapes.get(out)))

        patched.append({
            "node": op_name,
            "a": a,
            "b": b,
            "out": out,
            "shape": shapes.get(out),
        })

    if patched:
        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.value_info.extend(new_vis)

    return patched


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    patched = patch(model)

    print(f"INT64 Muls wrapped in BF16: {len(patched)}")
    for p in patched:
        print(f"  {p['node']:50s} shape={p['shape']}  in=[{p['a']}, {p['b']}] -> {p['out']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker passed")
    print(f"Wrote model: {args.output}")


if __name__ == "__main__":
    main()
