#!/usr/bin/env python3
"""Replace Round with Floor(x + 0.5) to avoid banker's rounding issues.

The torq compiler implements Round as round-half-to-even (banker's rounding),
which differs from ORT's behavior when bf16 truncation pushes values onto
exact .5 boundaries. Example: ORT computes 8.502 -> Round -> 9, but bf16
truncates 8.502 to 8.5, then banker's rounding gives 8 instead of 9.

Floor(x + 0.5) implements round-half-up, which always rounds .5 upward.
This matches ORT's effective behavior for these near-.5 values.
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def fix_round_half_up(model, target_names=None):
    nodes_to_replace = []

    for i, node in enumerate(model.graph.node):
        if node.op_type != 'Round':
            continue
        if target_names and node.name not in target_names:
            continue
        nodes_to_replace.append((i, node))

    if not nodes_to_replace:
        print("No Round nodes found to replace")
        return 0

    for idx, node in reversed(nodes_to_replace):
        inp = node.input[0]
        out = node.output[0]
        prefix = f"{out}__half_up"

        half_const_name = f"{prefix}_half"
        add_out_name = f"{prefix}_add"

        half_val = np.array(0.5, dtype=np.float32)
        bf16_raw = (np.float32(0.5).view(np.uint32) >> 16).astype(np.uint16)
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array(0.5).astype(np.float32).view(np.uint32).astype(np.uint32) >> 16,
                name="__discard"))

        for init in list(model.graph.initializer):
            if init.name == "__discard":
                model.graph.initializer.remove(init)
                break

        half_tensor = onnx.TensorProto()
        half_tensor.name = half_const_name
        half_tensor.data_type = TensorProto.BFLOAT16
        half_tensor.raw_data = np.uint16(0x3F00).tobytes()  # 0.5 in bf16
        model.graph.initializer.append(half_tensor)

        add_node = helper.make_node(
            "Add", inputs=[inp, half_const_name], outputs=[add_out_name],
            name=f"{prefix}_add_node")

        floor_node = helper.make_node(
            "Floor", inputs=[add_out_name], outputs=[out],
            name=node.name)

        inp_shape = None
        inp_dtype = None
        for vi in list(model.graph.value_info) + list(model.graph.input):
            if vi.name == inp:
                if vi.type.tensor_type.HasField('shape'):
                    inp_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                inp_dtype = vi.type.tensor_type.elem_type
                break

        if inp_shape:
            model.graph.value_info.append(
                helper.make_tensor_value_info(add_out_name, inp_dtype or TensorProto.BFLOAT16, inp_shape))

        model.graph.node.remove(node)
        model.graph.node.insert(idx, floor_node)
        model.graph.node.insert(idx, add_node)

        print(f"  Replaced {node.name}: Round({inp}) -> Floor({inp} + 0.5)")

    return len(nodes_to_replace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--names", nargs="+",
                        help="Only replace these Round node names (default: all)")
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = fix_round_half_up(model, target_names=set(args.names) if args.names else None)
    print(f"\nReplaced {count} Round node(s) with Floor(x + 0.5)")

    out_path = args.output or args.input
    onnx.save(model, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
