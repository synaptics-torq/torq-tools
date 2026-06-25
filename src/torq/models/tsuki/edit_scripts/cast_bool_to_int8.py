#!/usr/bin/env python3
"""Insert Cast(BOOL→INT8) after every bool-producing op and Cast(INT8→BOOL)
before every op that requires BOOL input (Where condition slot).

Eliminates all internal i1 tensors, which crash the torq compiler during
full-model compilation with 'Invalid element type' in LData::LData().

This is more general than replace_bool_logic_with_int8.py — it handles
comparison ops (Less, Equal, Greater, etc.) not just logic ops.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BOOL = TensorProto.BOOL
INT8 = TensorProto.INT8

BOOL_PRODUCING_OPS = {
    "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal",
    "And", "Or", "Xor", "Not", "IsNaN", "IsInf",
}

# Comparison ops that produce BOOL but can be replaced with INT8 arithmetic.
# Equal(a,b) → 1 - Min(Abs(Sub(a,b)), 1)  (all INT ops, no i1)
# Less(a,b) stays as Less since it may compile fine on some executors
COMPARISON_TO_ARITHMETIC = {"Equal"}

BOOL_CONDITION_SLOTS = {
    "Where": 0,
    "If": 0,
    "Compress": 1,
}


def apply_fix(model):
    graph = model.graph

    # Find all bool-producing ops and Cast-to-BOOL ops
    bool_outputs = set()
    for i, n in enumerate(graph.node):
        if n.op_type in BOOL_PRODUCING_OPS:
            for o in n.output:
                if o:
                    bool_outputs.add(o)
        elif n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == BOOL:
                    for o in n.output:
                        if o:
                            bool_outputs.add(o)

    # Also check value_info for bool tensors produced by transparent ops
    # (Unsqueeze, Expand, etc. that pass through bool)
    bool_vi = set()
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == BOOL:
            bool_vi.add(vi.name)

    # Propagate: if an op's input is bool and its output is also bool in value_info,
    # it's a transparent bool op
    changed = True
    while changed:
        changed = False
        for n in graph.node:
            if any(inp in bool_outputs for inp in n.input if inp):
                for o in n.output:
                    if o and o in bool_vi and o not in bool_outputs:
                        bool_outputs.add(o)
                        changed = True

    if not bool_outputs:
        print("No bool-producing ops found.")
        return model, 0

    print(f"Found {len(bool_outputs)} bool tensor(s)")

    # Run shape inference BEFORE modifying the graph so we get shapes for
    # the original tensor names (which will be renamed below).
    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
        inferred_vi = {vi.name: vi for vi in inferred_model.graph.value_info}
    except Exception:
        inferred_vi = {}

    # Build a shape lookup from all available sources
    shape_lookup = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shape_lookup[vi.name] = vi.type.tensor_type.shape
    for name, vi in inferred_vi.items():
        if name not in shape_lookup and vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shape_lookup[name] = vi.type.tensor_type.shape

    # Build replacement: for each bool output, create a renamed bool version
    # and a Cast to INT8 that takes the original name
    nodes = list(graph.node)
    new_nodes = []
    rename_map = {}  # old_bool_name → new_bool_name (the pre-cast version)
    cast_count = 0

    for i, n in enumerate(nodes):
        new_nodes.append(n)

        # After each bool-producing op, insert Cast(BOOL→INT8)
        for j, out in enumerate(n.output):
            if out and out in bool_outputs:
                bool_name = f"{out}__bool_raw"
                rename_map[out] = bool_name
                n.output[j] = bool_name

                cast_node = helper.make_node(
                    "Cast", [bool_name], [out],
                    name=f"{n.name}__cast_bool_to_i8",
                    to=INT8,
                )
                new_nodes.append(cast_node)
                cast_count += 1

    # Now fix consumers that need actual BOOL (Where condition slot).
    # Cast ANY non-BOOL condition input to BOOL, not just ones we converted —
    # earlier scripts (replace_equal_with_int_arithmetic) may have already
    # changed some tensors to INT8.
    all_int8_tensors = set(bool_outputs)
    # Also include any tensor whose value_info says INT8 or whose name ends
    # with a known INT8 suffix from our fixes
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == INT8:
            all_int8_tensors.add(vi.name)

    final_nodes = []
    cast_back_count = 0
    seen_casts = set()
    for n in new_nodes:
        if n.op_type in BOOL_CONDITION_SLOTS:
            slot = BOOL_CONDITION_SLOTS[n.op_type]
            if slot < len(n.input) and n.input[slot]:
                cond_name = n.input[slot]
                if cond_name in all_int8_tensors and cond_name not in seen_casts:
                    bool_cond = f"{cond_name}__back_to_bool"
                    cast_back = helper.make_node(
                        "Cast", [cond_name], [bool_cond],
                        name=f"{n.name}__cast_cond_to_bool",
                        to=BOOL,
                    )
                    final_nodes.append(cast_back)
                    n.input[slot] = bool_cond
                    seen_casts.add(cond_name)
                    cast_back_count += 1
        final_nodes.append(n)

    del graph.node[:]
    graph.node.extend(final_nodes)

    # Update value_info: change bool entries to INT8 and propagate shapes
    for vi in list(graph.value_info) + list(graph.output):
        if vi.name in bool_outputs:
            if vi.type.HasField("tensor_type"):
                vi.type.tensor_type.elem_type = INT8
                # If shape is missing, try to fill it from inference
                if not vi.type.tensor_type.HasField("shape") and vi.name in shape_lookup:
                    vi.type.tensor_type.shape.CopyFrom(shape_lookup[vi.name])

    # Add value_info for new intermediate tensors (__bool_raw, __back_to_bool)
    existing_vi_names = {vi.name for vi in graph.value_info}
    for orig_name in rename_map:
        bool_raw_name = rename_map[orig_name]  # e.g. "foo__bool_raw"
        # __bool_raw has BOOL type with the same shape as the original
        if bool_raw_name not in existing_vi_names and orig_name in shape_lookup:
            vi = helper.make_tensor_value_info(bool_raw_name, BOOL, None)
            vi.type.tensor_type.shape.CopyFrom(shape_lookup[orig_name])
            graph.value_info.append(vi)

    for cast_name in seen_casts:
        back_name = f"{cast_name}__back_to_bool"
        if back_name not in existing_vi_names and cast_name in shape_lookup:
            vi = helper.make_tensor_value_info(back_name, BOOL, None)
            vi.type.tensor_type.shape.CopyFrom(shape_lookup[cast_name])
            graph.value_info.append(vi)

    total = cast_count + cast_back_count
    print(f"Inserted {cast_count} Cast(BOOL→INT8) + {cast_back_count} Cast(INT8→BOOL)")
    return model, total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    model, count = apply_fix(model)
    print(f"Total Cast ops inserted: {count}")
    onnx.save(model, str(args.output))
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker WARN: {e}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
