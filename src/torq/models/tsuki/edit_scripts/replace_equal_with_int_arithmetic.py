#!/usr/bin/env python3
"""Replace Equal ops with INT arithmetic to avoid i1 output tensors.

The torq compiler crashes on Equal with broadcasting ("Input strides must match"
on i1 elementwisebinary). This replaces Equal(a, b) → BOOL with an arithmetic
equivalent that produces INT8 directly:

  Equal(a, b) → Cast(1 - Min(Abs(Sub(a, b)), 1), INT8)

For INT64 inputs: Sub → Abs → Clip(0,1) → Sub(1, _) → Cast(INT8)
All operations stay in INT64 until the final Cast, avoiding i1 entirely.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INT8 = TensorProto.INT8
INT32 = TensorProto.INT32
BOOL = TensorProto.BOOL


def apply_fix(model):
    graph = model.graph

    # Ensure constants exist
    _ensure_init(graph, "__one_i32", np.array(1, dtype=np.int32))
    _ensure_init(graph, "__zero_i32", np.array(0, dtype=np.int32))

    nodes = list(graph.node)
    new_nodes = []
    replaced = 0

    for i, n in enumerate(nodes):
        if n.op_type != "Equal":
            new_nodes.append(n)
            continue

        a, b = n.input[0], n.input[1]
        out = n.output[0]
        name = n.name or f"Equal_{i}"

        # Cast inputs to INT32 (Equal inputs may be INT64)
        a_i32 = f"{name}__a_i32"
        b_i32 = f"{name}__b_i32"
        cast_a = helper.make_node("Cast", [a], [a_i32], name=f"{name}__cast_a", to=INT32)
        cast_b = helper.make_node("Cast", [b], [b_i32], name=f"{name}__cast_b", to=INT32)

        # Sub(a, b) → INT32
        sub_out = f"{name}__sub"
        sub_node = helper.make_node("Sub", [a_i32, b_i32], [sub_out], name=f"{name}__sub")

        # Abs(sub) → INT32
        abs_out = f"{name}__abs"
        abs_node = helper.make_node("Abs", [sub_out], [abs_out], name=f"{name}__abs")

        # Clip(abs, 0, 1) → INT32 {0 or 1}: 0 where equal, 1 where not
        clip_out = f"{name}__clip"
        clip_node = helper.make_node("Clip", [abs_out, "__zero_i32", "__one_i32"],
                                     [clip_out], name=f"{name}__clip")

        # Sub(1, clip) → INT32 {1 or 0}: 1 where equal, 0 where not
        inv_out = f"{name}__inv"
        inv_node = helper.make_node("Sub", ["__one_i32", clip_out], [inv_out],
                                    name=f"{name}__inv")

        # Cast to INT8
        cast_node = helper.make_node("Cast", [inv_out], [out],
                                     name=f"{name}__cast_i8", to=INT8)

        new_nodes.extend([cast_a, cast_b, sub_node, abs_node, clip_node, inv_node, cast_node])
        replaced += 1
        print(f"  Replaced: {name} (Equal) → Sub+Abs+Clip+Sub+Cast ({out})")

    if replaced:
        # Insert Cast(INT8→BOOL) before Where ops that consume Equal outputs
        equal_outputs = {n.output[0] for _, n in enumerate(nodes) if n.op_type == "Equal" and n.output}
        final_nodes = []
        for n in new_nodes:
            if n.op_type == "Where" and n.input[0] in equal_outputs:
                cond = n.input[0]
                bool_cond = f"{cond}__to_bool"
                cast_back = helper.make_node(
                    "Cast", [cond], [bool_cond],
                    name=f"{n.name}__cast_cond_to_bool", to=BOOL)
                final_nodes.append(cast_back)
                n.input[0] = bool_cond
            final_nodes.append(n)

        del graph.node[:]
        graph.node.extend(final_nodes)

        # Update value_info/outputs: Equal outputs change from BOOL to INT8
        for vi in list(graph.value_info) + list(graph.output):
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == BOOL:
                vi.type.tensor_type.elem_type = INT8

        # Propagate shape info for new intermediate tensors
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            inferred_vi = {vi.name: vi for vi in inferred.graph.value_info}
        except Exception:
            inferred_vi = {}

        existing_vi_names = {vi.name for vi in graph.value_info}
        for n in final_nodes:
            for out_name in n.output:
                if out_name and out_name not in existing_vi_names and out_name in inferred_vi:
                    graph.value_info.append(inferred_vi[out_name])

    return model, replaced


def _ensure_init(graph, name, value):
    for init in graph.initializer:
        if init.name == name:
            return
    graph.initializer.append(numpy_helper.from_array(value, name=name))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    model, count = apply_fix(model)
    print(f"Replaced {count} Equal op(s)")
    onnx.save(model, str(args.output))
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker WARN: {e}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
