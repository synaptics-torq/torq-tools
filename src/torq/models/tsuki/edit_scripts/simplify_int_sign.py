#!/usr/bin/env python3
"""Remove redundant Clip(-1,1) step from integer Sign chains.

The eliminate_bool_ops.py script replaces comparisons with:
  diff = Sub(a, b)       # INT8/INT32
  sign = Clip(diff, -1, 1)  # acting as Sign
  binary = Clip(sign, 0, 1) # step function

For integer inputs, Clip(x, 0, 1) gives the same result as
Clip(Clip(x, -1, 1), 0, 1) because there are no values between
-1 and 0 or between 0 and 1. So the first Clip is redundant.

The integer Clip(-1, 1) has known compiler issues. This script
removes it by redirecting the second Clip to operate directly
on the Sub output.
"""
import argparse
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def simplify_int_sign(model):
    nodes_by_output = {}
    for n in model.graph.node:
        for o in n.output:
            nodes_by_output[o] = n

    # Get initializer values
    init_values = {}
    for init in model.graph.initializer:
        if len(init.dims) == 0:  # scalar
            if init.data_type == TensorProto.INT8:
                init_values[init.name] = int(np.frombuffer(init.raw_data, dtype=np.int8)[0])
            elif init.data_type == TensorProto.INT32:
                init_values[init.name] = int(np.frombuffer(init.raw_data, dtype=np.int32)[0])
            elif init.data_type == TensorProto.INT64:
                init_values[init.name] = int(np.frombuffer(init.raw_data, dtype=np.int64)[0])
            elif init.data_type == TensorProto.FLOAT:
                init_values[init.name] = float(np.frombuffer(init.raw_data, dtype=np.float32)[0])
            elif init.data_type == TensorProto.BFLOAT16:
                raw = np.frombuffer(init.raw_data, dtype=np.uint16)
                f = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
                init_values[init.name] = float(f[0])

    # Find pattern: Clip(Clip(x, -1, 1), 0, 1) where inputs are integers
    fixed = 0
    nodes_to_remove = set()
    for n in model.graph.node:
        if n.op_type != 'Clip':
            continue
        # Check if this is Clip(0, 1)
        if len(n.input) < 3:
            continue
        min_val = init_values.get(n.input[1])
        max_val = init_values.get(n.input[2])
        if min_val != 0 or max_val != 1:
            continue

        # Check if input is another Clip(-1, 1)
        inner_name = n.input[0]
        if inner_name not in nodes_by_output:
            continue
        inner = nodes_by_output[inner_name]
        if inner.op_type != 'Clip' or len(inner.input) < 3:
            continue
        inner_min = init_values.get(inner.input[1])
        inner_max = init_values.get(inner.input[2])
        if inner_min != -1 or inner_max != 1:
            continue

        # Found Clip(Clip(x, -1, 1), 0, 1) — check if inner Clip has only this consumer
        inner_output = inner.output[0]
        consumers = sum(1 for node in model.graph.node
                       if inner_output in node.input and node != n)
        if consumers > 0:
            print(f"  {n.name}: Clip(-1,1) has {consumers} other consumers, skipping")
            continue

        # Redirect this Clip to use the Sub output directly
        original_input = inner.input[0]
        n.input[0] = original_input
        nodes_to_remove.add(inner.name)
        print(f"  {n.name}: removed Clip(-1,1) step ({inner.name}), now Clip({original_input}, 0, 1)")
        fixed += 1

    # Remove the Clip(-1,1) nodes
    for node in list(model.graph.node):
        if node.name in nodes_to_remove:
            model.graph.node.remove(node)

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = simplify_int_sign(model)
    print(f"\nSimplified {count} integer Sign chain(s)")

    out_path = args.output or args.input
    onnx.save(model, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
