#!/usr/bin/env python3
"""Replace bool logical ops (Not, Or, And, Xor) with INT8 arithmetic equivalents.

The torq compiler's NSS/CSS/host backends can't handle i1 elementwise ops
with broadcasting ("Input strides must match" on torq_hl.elementwisebinary
with memref<...xi1>).

This script rewrites:
  Not(x_bool)        → Sub(1_i8, Cast(x, INT8))  → Cast(result, BOOL)
  Or(a_bool, b_bool) → Max(Cast(a, INT8), Cast(b, INT8)) → Cast(result, BOOL)
  And(a_bool, b_bool)→ Min(Cast(a, INT8), Cast(b, INT8)) → Cast(result, BOOL)
  Xor(a_bool, b_bool)→ Sub(Max(...), Min(...))             → Cast(result, BOOL)

If the ONLY consumer of the result is a Where (on the condition slot) or a
Cast to a numeric type, the final Cast(→BOOL) is skipped since Where accepts
i8-as-bool and the Cast can absorb the type change.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


BOOL = TensorProto.BOOL
INT8 = TensorProto.INT8


def _find_bool_logic_ops(graph):
    """Find Not/Or/And/Xor ops operating on BOOL tensors."""
    bool_tensors = set()
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == BOOL:
            bool_tensors.add(vi.name)
    for init in graph.initializer:
        if init.data_type == BOOL:
            bool_tensors.add(init.name)
    for node in graph.node:
        if node.op_type in ("Less", "LessOrEqual", "Greater", "GreaterOrEqual",
                            "Equal", "Not", "And", "Or", "Xor", "IsNaN", "IsInf"):
            for o in node.output:
                if o:
                    bool_tensors.add(o)
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == BOOL:
                    for o in node.output:
                        if o:
                            bool_tensors.add(o)

    targets = []
    for i, node in enumerate(graph.node):
        if node.op_type in ("Not", "Or", "And", "Xor"):
            has_bool_input = any(inp in bool_tensors for inp in node.input if inp)
            if has_bool_input:
                targets.append((i, node))
    return targets


def _consumer_map(graph):
    consumers = {}
    for i, node in enumerate(graph.node):
        for slot, inp in enumerate(node.input):
            if inp:
                consumers.setdefault(inp, []).append((i, slot, node))
    return consumers


def _ensure_one_const(graph, name, value, dtype=INT8):
    for init in graph.initializer:
        if init.name == name:
            return
    graph.initializer.append(
        numpy_helper.from_array(np.array(value, dtype=np.int8), name=name))


def apply_fix(model):
    graph = model.graph
    targets = _find_bool_logic_ops(graph)
    if not targets:
        print("No bool logic ops found to replace.")
        return model, 0

    consumers = _consumer_map(graph)
    nodes = list(graph.node)
    _ensure_one_const(graph, "__one_i8", 1, INT8)

    replaced = 0
    for idx, node in reversed(targets):
        op = node.op_type
        node_name = node.name or f"{op}_{idx}"
        out_name = node.output[0]

        replacement_nodes = []

        if op == "Not":
            inp = node.input[0]
            cast_name = f"{node_name}__cast_i8"
            cast_node = helper.make_node("Cast", [inp], [cast_name],
                                         name=f"{node_name}__to_i8", to=INT8)
            sub_out = f"{node_name}__sub_result"
            sub_node = helper.make_node("Sub", ["__one_i8", cast_name], [sub_out],
                                        name=f"{node_name}__sub")
            cast_back = helper.make_node("Cast", [sub_out], [out_name],
                                         name=f"{node_name}__to_bool", to=BOOL)
            replacement_nodes = [cast_node, sub_node, cast_back]

        elif op == "Or":
            # Or(a,b) for {0,1}: Clip(Add(a,b), 0, 1)
            # Using Add+Clip instead of Max because Max with broadcasting
            # lowers to linalg.generic which NSS can't handle.
            a, b = node.input[0], node.input[1]
            a_i8 = f"{node_name}__a_i8"
            b_i8 = f"{node_name}__b_i8"
            cast_a = helper.make_node("Cast", [a], [a_i8],
                                      name=f"{node_name}__cast_a", to=INT8)
            cast_b = helper.make_node("Cast", [b], [b_i8],
                                      name=f"{node_name}__cast_b", to=INT8)
            add_out = f"{node_name}__add_result"
            add_node = helper.make_node("Add", [a_i8, b_i8], [add_out],
                                        name=f"{node_name}__add")
            _ensure_one_const(graph, "__zero_i8", 0, INT8)
            clip_out = f"{node_name}__clip_result"
            clip_node = helper.make_node("Clip", [add_out, "__zero_i8", "__one_i8"],
                                         [clip_out], name=f"{node_name}__clip")
            cast_back = helper.make_node("Cast", [clip_out], [out_name],
                                         name=f"{node_name}__to_bool", to=BOOL)
            replacement_nodes = [cast_a, cast_b, add_node, clip_node, cast_back]

        elif op == "And":
            # And(a,b) for {0,1}: Mul(a,b)
            a, b = node.input[0], node.input[1]
            a_i8 = f"{node_name}__a_i8"
            b_i8 = f"{node_name}__b_i8"
            cast_a = helper.make_node("Cast", [a], [a_i8],
                                      name=f"{node_name}__cast_a", to=INT8)
            cast_b = helper.make_node("Cast", [b], [b_i8],
                                      name=f"{node_name}__cast_b", to=INT8)
            mul_out = f"{node_name}__mul_result"
            mul_node = helper.make_node("Mul", [a_i8, b_i8], [mul_out],
                                        name=f"{node_name}__mul")
            cast_back = helper.make_node("Cast", [mul_out], [out_name],
                                         name=f"{node_name}__to_bool", to=BOOL)
            replacement_nodes = [cast_a, cast_b, mul_node, cast_back]

        elif op == "Xor":
            a, b = node.input[0], node.input[1]
            a_i8 = f"{node_name}__a_i8"
            b_i8 = f"{node_name}__b_i8"
            cast_a = helper.make_node("Cast", [a], [a_i8],
                                      name=f"{node_name}__cast_a", to=INT8)
            cast_b = helper.make_node("Cast", [b], [b_i8],
                                      name=f"{node_name}__cast_b", to=INT8)
            # XOR(a,b) = (a+b) - 2*min(a,b)  for {0,1} values
            # simpler: abs(a - b)
            sub_out = f"{node_name}__sub"
            sub_node = helper.make_node("Sub", [a_i8, b_i8], [sub_out],
                                        name=f"{node_name}__sub")
            abs_out = f"{node_name}__abs"
            abs_node = helper.make_node("Abs", [sub_out], [abs_out],
                                        name=f"{node_name}__abs")
            cast_back = helper.make_node("Cast", [abs_out], [out_name],
                                         name=f"{node_name}__to_bool", to=BOOL)
            replacement_nodes = [cast_a, cast_b, sub_node, abs_node, cast_back]

        nodes[idx:idx+1] = replacement_nodes
        replaced += 1
        print(f"  Replaced: {node_name} ({op}) → INT8 arithmetic ({len(replacement_nodes)} ops)")

    del graph.node[:]
    graph.node.extend(nodes)

    # Propagate shape info for new intermediate tensors.
    # Run shape inference to get shapes for inputs/outputs of replaced ops,
    # then add value_info for all new intermediates with the correct shapes.
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        inferred_vi = {vi.name: vi for vi in inferred.graph.value_info}
    except Exception:
        inferred_vi = {}

    shape_lookup = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shape_lookup[vi.name] = vi.type.tensor_type.shape
    for name, vi in inferred_vi.items():
        if name not in shape_lookup and vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shape_lookup[name] = vi.type.tensor_type.shape

    existing_vi_names = {vi.name for vi in graph.value_info}
    for n in nodes:
        for out_name in n.output:
            if out_name and out_name not in existing_vi_names and out_name in inferred_vi:
                graph.value_info.append(inferred_vi[out_name])

    return model, replaced


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    model, count = apply_fix(model)
    print(f"Replaced {count} bool logic op(s)")
    onnx.save(model, str(args.output))
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker WARN: {e}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
