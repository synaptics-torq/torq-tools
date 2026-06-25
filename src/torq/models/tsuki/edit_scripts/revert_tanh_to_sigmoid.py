#!/usr/bin/env python3
"""Revert tanh-based sigmoid decompositions back to Sigmoid ops.

Pattern matched:
  half_x    = Mul(x, 0.5)          # node named *_half_input
  tanh_out  = Tanh(half_x)         # node named *_tanh
  plus_one  = Add(tanh_out, 1.0)   # node named *_add_one
  sigmoid   = Mul(plus_one, 0.5)   # node named *_half_output

Replaced with:
  sigmoid   = Sigmoid(x)

The tanh decomposition triggers an APInt assertion in the torq compiler's
ConvertBf16ToUInt16Buffers pass when assigned to the host executor.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def find_tanh_sigmoid_patterns(graph):
    """Find Tanh ops that are part of sigmoid decompositions."""
    node_by_output = {}
    for i, n in enumerate(graph.node):
        for o in n.output:
            if o:
                node_by_output[o] = (i, n)

    consumers = {}
    for i, n in enumerate(graph.node):
        for inp in n.input:
            if inp:
                consumers.setdefault(inp, []).append((i, n))

    patterns = []
    for i, n in enumerate(graph.node):
        if n.op_type != "Tanh":
            continue

        tanh_input = n.input[0]
        tanh_output = n.output[0]

        # Check: input comes from Mul(x, 0.5)
        if tanh_input not in node_by_output:
            continue
        mul_half_idx, mul_half = node_by_output[tanh_input]
        if mul_half.op_type != "Mul":
            continue

        # Check: output feeds Add(tanh, 1.0)
        tanh_consumers = consumers.get(tanh_output, [])
        add_one = None
        add_one_idx = None
        for ci, cn in tanh_consumers:
            if cn.op_type == "Add":
                add_one_idx = ci
                add_one = cn
                break
        if add_one is None:
            continue

        # Check: Add feeds Mul(_, 0.5)
        add_output = add_one.output[0]
        add_consumers = consumers.get(add_output, [])
        mul_half_out = None
        mul_half_out_idx = None
        for ci, cn in add_consumers:
            if cn.op_type == "Mul":
                mul_half_out_idx = ci
                mul_half_out = cn
                break
        if mul_half_out is None:
            continue

        # Found the pattern. The original input to sigmoid is the
        # non-0.5 input of the first Mul.
        original_x = None
        for inp in mul_half.input:
            if inp != tanh_input:
                # Check if this is the 0.5 constant or the actual input
                is_half = False
                for init in graph.initializer:
                    if init.name == inp:
                        val = numpy_helper.to_array(init)
                        if val.size == 1 and abs(float(val.flat[0]) - 0.5) < 1e-6:
                            is_half = True
                        break
                if not is_half:
                    original_x = inp

        if original_x is None:
            # Both inputs could be the same — try the first
            original_x = mul_half.input[0]

        sigmoid_output = mul_half_out.output[0]

        patterns.append({
            "mul_half_idx": mul_half_idx,
            "tanh_idx": i,
            "add_one_idx": add_one_idx,
            "mul_half_out_idx": mul_half_out_idx,
            "original_x": original_x,
            "sigmoid_output": sigmoid_output,
            "tanh_name": n.name,
            "indices_to_remove": {mul_half_idx, i, add_one_idx, mul_half_out_idx},
        })

    return patterns


def apply_fix(model):
    graph = model.graph
    patterns = find_tanh_sigmoid_patterns(graph)
    if not patterns:
        print("No tanh-based sigmoid patterns found.")
        return model, 0

    # Collect all indices to remove
    all_remove = set()
    for p in patterns:
        all_remove |= p["indices_to_remove"]

    # Build new node list
    nodes = list(graph.node)
    new_nodes = []
    pattern_by_tanh_idx = {p["tanh_idx"]: p for p in patterns}

    for i, n in enumerate(nodes):
        if i in all_remove:
            # If this is the tanh node, insert the Sigmoid replacement
            if i in pattern_by_tanh_idx:
                p = pattern_by_tanh_idx[i]
                sigmoid_name = p["tanh_name"].replace("_tanh", "")
                if not sigmoid_name:
                    sigmoid_name = f"sigmoid_{i}"
                sigmoid_node = helper.make_node(
                    "Sigmoid",
                    inputs=[p["original_x"]],
                    outputs=[p["sigmoid_output"]],
                    name=sigmoid_name,
                )
                new_nodes.append(sigmoid_node)
                print(f"  Reverted: {p['tanh_name']} → Sigmoid {sigmoid_name!r}")
        else:
            new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model, len(patterns)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    model, count = apply_fix(model)
    print(f"Reverted {count} tanh→sigmoid decomposition(s)")
    onnx.save(model, str(args.output))
    try:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    except Exception as e:
        print(f"ONNX checker WARN: {e}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
