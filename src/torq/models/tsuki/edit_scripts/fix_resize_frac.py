#!/usr/bin/env python3
"""Fix bf16 precision loss in Resize decomposition's frac(x) computation.

The Resize decomposition computes interpolation weights as:
  coord = offset * scale - 0.5
  floor_idx = int64(coord)
  frac = coord - bf16(floor_idx)

For scale=300 (96000→320 resize), coordinates reach 95849.5. bf16 can't
represent these values accurately (7-bit mantissa), causing both the floor
index and fractional weight to be wrong.

Fix: precompute exact floor indices (INT64) and fractional weights (BF16 0.5)
as constants, replacing the imprecise bf16 computation chain.
"""
import argparse
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def fix_resize_frac(model):
    nodes_by_output = {}
    for n in model.graph.node:
        for o in n.output:
            nodes_by_output[o] = n

    # Find the frac pattern: Sub(x, Cast(Cast(x, INT64), BF16))
    fixed = 0
    for n in list(model.graph.node):
        if n.op_type != 'Sub' or len(n.input) != 2:
            continue
        x_name = n.input[0]
        cast_bf16_name = n.input[1]
        if cast_bf16_name not in nodes_by_output:
            continue
        cast_bf16 = nodes_by_output[cast_bf16_name]
        if cast_bf16.op_type != 'Cast':
            continue
        if not any(a.name == 'to' and a.i == TensorProto.BFLOAT16 for a in cast_bf16.attribute):
            continue
        int64_name = cast_bf16.input[0]
        if int64_name not in nodes_by_output:
            continue
        cast_int64 = nodes_by_output[int64_name]
        if cast_int64.op_type != 'Cast':
            continue
        if not any(a.name == 'to' and a.i == TensorProto.INT64 for a in cast_int64.attribute):
            continue
        if cast_int64.input[0] != x_name:
            continue

        # Found pattern. Trace back to find scale and offset.
        print(f"Found frac pattern: {n.name}")
        print(f"  x = {x_name} (clamp output)")
        print(f"  floor = {int64_name} (INT64)")
        print(f"  frac = {n.output[0]}")

        # Trace back through clamp -> sub -> mul to find scale
        clamp_node = nodes_by_output.get(x_name)
        if not clamp_node or clamp_node.op_type != 'Clip':
            print(f"  WARNING: expected Clip, got {clamp_node.op_type if clamp_node else 'None'}")
            continue

        sub_node = nodes_by_output.get(clamp_node.input[0])
        if not sub_node or sub_node.op_type != 'Sub':
            print(f"  WARNING: expected Sub, got {sub_node.op_type if sub_node else 'None'}")
            continue

        coord_name = sub_node.input[0]
        add_node = nodes_by_output.get(coord_name)
        if not add_node or add_node.op_type != 'Add':
            print(f"  WARNING: expected Add (recombine), got {add_node.op_type if add_node else 'None'}")
            continue

        # Find the offset initializer
        offset_init = None
        for child_name in [add_node.input[0], add_node.input[1]]:
            mul_node = nodes_by_output.get(child_name)
            if mul_node and mul_node.op_type == 'Mul':
                for inp in mul_node.input:
                    for init in model.graph.initializer:
                        if init.name == inp and len(init.dims) == 1:
                            arr = np.frombuffer(init.raw_data, dtype=np.uint16)
                            f = np.frombuffer((arr.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
                            if f[0] == 0.5 and len(f) > 1 and f[1] == 1.5:
                                offset_init = init
                                break

        if offset_init is None:
            print("  WARNING: could not find offset initializer")
            continue

        offset_arr = np.frombuffer(offset_init.raw_data, dtype=np.uint16)
        offset_f32 = np.frombuffer((offset_arr.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
        n_out = len(offset_f32)
        print(f"  Output size: {n_out}")

        # Determine scale by tracing the truediv chain
        # The scale factor comes from truediv = (L * K) / L = K
        # Find the Div node
        scale_name = None
        for child_name in [add_node.input[0], add_node.input[1]]:
            mul_node = nodes_by_output.get(child_name)
            if mul_node and mul_node.op_type == 'Mul':
                for inp in mul_node.input:
                    if inp in nodes_by_output and nodes_by_output[inp].op_type == 'Div':
                        div_node = nodes_by_output[inp]
                        # Trace numerator: Cast(Mul(x, K))
                        num_cast = nodes_by_output.get(div_node.input[0])
                        if num_cast and num_cast.op_type == 'Cast':
                            mul_inner = nodes_by_output.get(num_cast.input[0])
                            if mul_inner and mul_inner.op_type == 'Mul':
                                for minp in mul_inner.input:
                                    for ini in model.graph.initializer:
                                        if ini.name == minp:
                                            if ini.data_type == TensorProto.INT64:
                                                k = np.frombuffer(ini.raw_data, dtype=np.int64)[0]
                                                scale_name = ini.name
                                                print(f"  Scale factor K = {k} (from {ini.name})")
                                            elif ini.data_type == TensorProto.FLOAT:
                                                k = np.frombuffer(ini.raw_data, dtype=np.float32)[0]
                                                scale_name = ini.name
                                                print(f"  Scale factor K = {k} (from {ini.name})")

        if scale_name is None:
            print("  WARNING: could not determine scale factor, trying heuristic")
            # Heuristic: compute from offset pattern
            # offset = [0.5, 1.5, ..., (n_out-1)+0.5]
            # coord = offset * scale, sub = coord - 0.5
            # clamp = max(0, sub)
            # For the pattern to make sense, scale must be integer
            # From the values we know: offset[0]=0.5, expected clamp[0] = 0.5*scale - 0.5
            # We can't determine scale without running inference
            print("  SKIPPING - cannot determine scale")
            continue

        scale = int(k)

        # Compute exact values in fp64
        offsets_exact = np.arange(n_out, dtype=np.float64) + 0.5
        coords_exact = offsets_exact * scale
        sub_exact = coords_exact - 0.5
        clamp_exact = np.maximum(0, sub_exact)
        floor_exact = np.floor(clamp_exact).astype(np.int64)
        frac_exact = clamp_exact - floor_exact

        print(f"  Floor indices: [{floor_exact[0]}, {floor_exact[1]}, ..., {floor_exact[-1]}]")
        print(f"  Frac values: [{frac_exact[0]}, {frac_exact[1]}, ..., {frac_exact[-1]}]")
        print(f"  Frac unique values: {np.unique(frac_exact)}")

        # Create floor index initializer (INT64)
        floor_init = numpy_helper.from_array(floor_exact, name=int64_name)
        model.graph.initializer.append(floor_init)

        # Create frac initializer (BF16)
        frac_f32 = frac_exact.astype(np.float32)
        frac_bf16_raw = np.frombuffer(frac_f32.tobytes(), dtype=np.uint32)
        frac_bf16_raw = ((frac_bf16_raw + 0x7FFF + ((frac_bf16_raw >> 16) & 1)) >> 16).astype(np.uint16)
        frac_init = TensorProto()
        frac_init.name = n.output[0]  # sub_5442
        frac_init.data_type = TensorProto.BFLOAT16
        frac_init.dims.extend([n_out])
        frac_init.raw_data = frac_bf16_raw.tobytes()
        model.graph.initializer.append(frac_init)

        # Remove the nodes we're replacing
        nodes_to_remove = {n.name, cast_bf16.name, cast_int64.name}
        print(f"  Removing nodes: {nodes_to_remove}")
        for node in list(model.graph.node):
            if node.name in nodes_to_remove:
                model.graph.node.remove(node)

        fixed += 1

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Fix bf16 precision loss in Resize frac computation")
    parser.add_argument("input", help="Input ONNX model")
    parser.add_argument("-o", "--output", help="Output ONNX model (default: overwrite input)")
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = fix_resize_frac(model)
    print(f"\nFixed {count} frac pattern(s)")

    out_path = args.output or args.input
    onnx.save(model, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
