#!/usr/bin/env python3
"""Wrap Where and ArgMax nodes in FP32 dtype islands.

The target hardware does not run these ops in BF16. Each affected node is
rewritten to do its compute in FP32, surrounded by Casts at the island boundary:

  Where:   X, Y (BF16) ──Cast(FP32)──► Where(FP32) ──Cast(BF16)──► out (BF16)
                               (condition is passed through unchanged)
  ArgMax:  data (BF16) ──Cast(FP32)──► ArgMax(FP32) ──► indices (INT64; unchanged)

Idempotent: Where/ArgMax whose data inputs are already non-BF16 are left alone.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


BF16 = TensorProto.BFLOAT16
FP32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def build_types(graph):
    types = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        types.setdefault(init.name, init.data_type)
    return types


def build_shapes(graph):
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


def unique(base, used):
    name = base
    i = 0
    while name in used:
        i += 1
        name = f"{base}__{i}"
    used.add(name)
    return name


def make_vi(name, dtype, dims):
    if dims is None:
        vi = onnx.ValueInfoProto()
        vi.name = name
        vi.type.tensor_type.elem_type = dtype
        return vi
    return helper.make_tensor_value_info(name, dtype, dims)


def wrap_where(node, types, shapes, used, new_nodes, new_vis):
    cond, x, y = node.input[0], node.input[1], node.input[2]
    out = node.output[0]
    cond_bf16 = types.get(cond) == BF16
    x_bf16 = types.get(x) == BF16
    y_bf16 = types.get(y) == BF16
    if not (cond_bf16 or x_bf16 or y_bf16):
        return None
    op_name = node.name or out

    # Condition must be BOOL per ONNX spec; cast it back if the bool->bf16
    # pipeline retyped it to BF16.
    cond_new = cond
    if cond_bf16:
        cond_new = unique(f"{cond}_bool_for_{op_name}", used)
        new_nodes.append(helper.make_node(
            "Cast", [cond], [cond_new], name=f"{op_name}__cast_cond_bool", to=BOOL))
        new_vis.append(make_vi(cond_new, BOOL, shapes.get(cond)))

    # X/Y go to FP32 for the compute island.
    x_fp = unique(f"{x}_fp32_for_{op_name}", used) if x_bf16 else x
    y_fp = unique(f"{y}_fp32_for_{op_name}", used) if y_bf16 else y
    if x_bf16:
        new_nodes.append(helper.make_node(
            "Cast", [x], [x_fp], name=f"{op_name}__cast_x_fp32", to=FP32))
        new_vis.append(make_vi(x_fp, FP32, shapes.get(x)))
    if y_bf16:
        new_nodes.append(helper.make_node(
            "Cast", [y], [y_fp], name=f"{op_name}__cast_y_fp32", to=FP32))
        new_vis.append(make_vi(y_fp, FP32, shapes.get(y)))

    # Where output dtype follows X/Y. Run the island in FP32 only if X/Y were
    # BF16; otherwise keep the original output dtype to avoid a needless Cast.
    island_in_fp32 = x_bf16 or y_bf16
    out_fp = unique(f"{out}_fp32", used) if island_in_fp32 else out

    where_island = onnx.NodeProto()
    where_island.CopyFrom(node)
    del where_island.input[:]
    where_island.input.extend([cond_new, x_fp, y_fp])
    del where_island.output[:]
    where_island.output.append(out_fp)
    new_nodes.append(where_island)
    if island_in_fp32:
        new_vis.append(make_vi(out_fp, FP32, shapes.get(out)))
        new_nodes.append(helper.make_node(
            "Cast", [out_fp], [out], name=f"{op_name}__cast_back_bf16", to=BF16))

    return {"op": "Where", "node": op_name, "shape": shapes.get(out),
            "cond_was": "BF16->BOOL" if cond_bf16 else "BOOL",
            "xy_was": "BF16->FP32" if (x_bf16 or y_bf16) else "kept"}


def wrap_argmax(node, types, shapes, used, new_nodes, new_vis):
    data = node.input[0]
    if types.get(data) != BF16:
        return None
    out = node.output[0]
    op_name = node.name or out

    data_fp = unique(f"{data}_fp32_for_{op_name}", used)
    new_nodes.append(helper.make_node(
        "Cast", [data], [data_fp], name=f"{op_name}__cast_in_fp32", to=FP32))
    new_vis.append(make_vi(data_fp, FP32, shapes.get(data)))

    argmax_fp32 = onnx.NodeProto()
    argmax_fp32.CopyFrom(node)
    del argmax_fp32.input[:]
    argmax_fp32.input.append(data_fp)
    # ArgMax output is INT64 regardless of input dtype — no output Cast.
    new_nodes.append(argmax_fp32)
    return {"op": "ArgMax", "node": op_name, "shape": shapes.get(out)}


def patch(model, target_ops):
    graph = model.graph
    types = build_types(graph)
    shapes = build_shapes(graph)
    used = set(types.keys()) | {n.name for n in graph.node if n.name}

    patched = []
    new_nodes = []
    new_vis = []

    for node in graph.node:
        if node.op_type not in target_ops:
            new_nodes.append(node)
            continue
        if node.op_type == "Where":
            result = wrap_where(node, types, shapes, used, new_nodes, new_vis)
        elif node.op_type == "ArgMax":
            result = wrap_argmax(node, types, shapes, used, new_nodes, new_vis)
        else:
            new_nodes.append(node)
            continue
        if result is None:
            new_nodes.append(node)
        else:
            patched.append(result)

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
    parser.add_argument("--ops", default="Where,ArgMax",
                        help="Comma-separated op types to wrap (default: Where,ArgMax)")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    target_ops = {o.strip() for o in args.ops.split(",") if o.strip()}
    model = onnx.load(str(args.input))
    patched = patch(model, target_ops)

    print(f"Ops wrapped: {len(patched)}")
    for p in patched:
        extras = ""
        if p["op"] == "Where":
            extras = f"  cond={p['cond_was']}  X/Y={p['xy_was']}"
        print(f"  {p['op']:8s} {p['node']:50s} shape={p['shape']}{extras}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker passed")
    print(f"Wrote model: {args.output}")


if __name__ == "__main__":
    main()
