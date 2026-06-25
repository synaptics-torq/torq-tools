#!/usr/bin/env python3
"""Fix view_175 bf16 precision loss by pre-splitting into INT32 floor + fp32 frac.

view_175 contains 96K continuous position values (0 to ~319.498). In bf16, values
near 320 round to 320.0, causing off-by-one index errors after Cast(->INT64).

Fix: replace single view_175 constant with:
  - view_175__floor_int32: exact floor indices (INT32, values 0-319)
  - view_175__frac: fractional parts (fp32, all <1.0; hardware downcasts to bf16)

Rewires the Mul(view_175, mask) + Cast(->INT64) + Sub(frac) chain to use integer
operations for the index path and float for the fractional path.
"""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--fp32-ref", required=True,
                        help="Pre-bf16 model to get original fp32 view_175 values")
    args = parser.parse_args()

    m = onnx.load(args.input)
    ref = onnx.load(args.fp32_ref)

    orig = None
    for init in ref.graph.initializer:
        if init.name == "view_175":
            orig = numpy_helper.to_array(init).astype(np.float64)
            break
    if orig is None:
        print("view_175 not found in reference model, skipping")
        onnx.save(m, args.output)
        return

    floor_vals = np.floor(orig).astype(np.int32)
    frac_vals = (orig - floor_vals.astype(np.float64)).astype(np.float32)

    init_map = {init.name: init for init in m.graph.initializer}
    if "view_175" not in init_map:
        print("view_175 not found in input model, skipping")
        onnx.save(m, args.output)
        return

    # Replace view_175 initializer with floor INT32
    view_init = init_map["view_175"]
    view_init.data_type = TensorProto.INT32
    view_init.raw_data = floor_vals.tobytes()
    view_init.name = "view_175__floor_int32"

    # Add fractional part as fp32 initializer (hardware downcasts to bf16)
    frac_init = onnx.TensorProto()
    frac_init.name = "view_175__frac"
    frac_init.data_type = TensorProto.FLOAT
    frac_init.dims.extend(list(orig.shape))
    frac_init.raw_data = frac_vals.tobytes()
    m.graph.initializer.append(frac_init)

    # Find target nodes by tracing from view_175
    output_to_node = {}
    for i, n in enumerate(m.graph.node):
        for o in n.output:
            output_to_node[o] = (i, n)

    # 1) Mul that consumes view_175
    mul_idx = None
    mul_output = None
    mask_input = None
    for i, n in enumerate(m.graph.node):
        if n.op_type == "Mul" and "view_175" in list(n.input):
            mul_idx = i
            mul_output = n.output[0]
            mask_input = n.input[1] if n.input[0] == "view_175" else n.input[0]
            break

    # 2) Cast(->INT64) consuming the Mul output
    cast_idx = None
    for i, n in enumerate(m.graph.node):
        if (n.op_type == "Cast" and mul_output in list(n.input)
                and any(a.name == "to" and a.i == TensorProto.INT64
                        for a in n.attribute)):
            cast_idx = i
            break

    # 3) Sub computing fractional part from the Mul output
    sub_idx = None
    for i, n in enumerate(m.graph.node):
        if n.op_type == "Sub" and mul_output in list(n.input):
            sub_idx = i
            break

    if mul_idx is None or cast_idx is None or sub_idx is None:
        print(f"Could not find all target nodes "
              f"(mul={mul_idx}, cast={cast_idx}, sub={sub_idx}), skipping")
        onnx.save(m, args.output)
        return

    cast_output = m.graph.node[cast_idx].output[0]   # _to_copy_108
    sub_output = m.graph.node[sub_idx].output[0]      # sub_5413

    new_nodes = [
        helper.make_node("Cast", [mask_input], ["__view175_mask_int32"],
                         name="__cast_mask_for_view175", to=TensorProto.INT32),
        helper.make_node("Mul", ["view_175__floor_int32", "__view175_mask_int32"],
                         ["__view175_floor_masked_int32"],
                         name="__mul_floor_mask_int32"),
        helper.make_node("Cast", ["__view175_floor_masked_int32"], [cast_output],
                         name="__cast_floor_to_int64", to=TensorProto.INT64),
        helper.make_node("Mul", ["view_175__frac", mask_input],
                         [sub_output], name="__mul_frac_mask"),
    ]

    remove_set = {mul_idx, cast_idx, sub_idx}
    insert_pos = min(remove_set)

    nodes = list(m.graph.node)
    for idx in sorted(remove_set, reverse=True):
        nodes.pop(idx)
    for j, nn in enumerate(new_nodes):
        nodes.insert(insert_pos + j, nn)

    del m.graph.node[:]
    m.graph.node.extend(nodes)
    del m.graph.value_info[:]

    print(f"Fixed view_175: split into INT32 floor + fp32 frac")
    print(f"  Removed 3 nodes (Mul, Cast->INT64, Sub)")
    print(f"  Added 4 nodes (Cast mask->INT32, Mul floor, Cast->INT64, Mul frac)")
    print(f"  floor range: [{floor_vals.min()}, {floor_vals.max()}]")
    print(f"  frac range: [{frac_vals.min():.6f}, {frac_vals.max():.6f}]")

    onnx.save(m, args.output)


if __name__ == "__main__":
    main()
