#!/usr/bin/env python3
"""Fix add_18087 bf16 precision loss via bucket decomposition.

add_18087 = [0.5, 1.5, ..., 319.5] — half-integer positions. In bf16, 192 of 320
values are wrong by ±0.5 (all indices >= 128). This corrupts downstream interpolation.

Fix: decompose into offset + bucket, both exact in bf16:
  - offset: [0.5, 1.5, ..., 127.5, 0.5, ..., 127.5, 0.5, ..., 63.5] (all <= 127.5)
  - bucket: [0, 0, ..., 128, 128, ..., 256, 256, ...]  (powers of 2)

Replaces Mul(add_18087, truediv) with Mul(offset, truediv) + Mul(bucket, truediv).
"""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


BUCKET_BOUNDARIES = [0, 128, 256]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    m = onnx.load(args.input)

    init_map = {init.name: init for init in m.graph.initializer}
    if "add_18087" not in init_map:
        print("add_18087 not found, skipping")
        onnx.save(m, args.output)
        return

    orig = numpy_helper.to_array(init_map["add_18087"]).astype(np.float64)
    n_vals = len(orig)

    bucket = np.zeros(n_vals, dtype=np.float32)
    for b in sorted(BUCKET_BOUNDARIES, reverse=True):
        bucket[orig >= b + 0.5] = float(b)
    offset = (orig - bucket.astype(np.float64)).astype(np.float32)

    if offset.max() > 127.5:
        print(f"WARNING: offset max {offset.max()} > 127.5, bf16 may round")

    # Verify both are exact in bf16
    def bf16_roundtrip(arr):
        u32 = arr.view(np.uint32)
        r = u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
        bf16 = (r >> np.uint32(16)).astype(np.uint16)
        return (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32), bf16

    offset_back, offset_bf16 = bf16_roundtrip(offset)
    bucket_back, bucket_bf16 = bf16_roundtrip(bucket)
    assert np.allclose(offset_back, offset), "offset not exact in bf16"
    assert np.allclose(bucket_back, bucket), "bucket not exact in bf16"

    # Replace add_18087 with offset (bf16)
    add_init = init_map["add_18087"]
    add_init.data_type = TensorProto.BFLOAT16
    add_init.dims[:] = list(orig.shape)
    add_init.raw_data = offset_bf16.tobytes()
    add_init.name = "add_18087__offset_bf16"

    # Add bucket as bf16 initializer
    bucket_init = onnx.TensorProto()
    bucket_init.name = "add_18087__bucket_bf16"
    bucket_init.data_type = TensorProto.BFLOAT16
    bucket_init.dims.extend(list(orig.shape))
    bucket_init.raw_data = bucket_bf16.tobytes()
    m.graph.initializer.append(bucket_init)

    # Find Mul(add_18087, truediv) node
    mul_idx = None
    for i, n in enumerate(m.graph.node):
        if "add_18087" in list(n.input):
            mul_idx = i
            break

    if mul_idx is None:
        print("Could not find node consuming add_18087, skipping")
        onnx.save(m, args.output)
        return

    n = m.graph.node[mul_idx]
    truediv_input = n.input[1] if n.input[0] == "add_18087" else n.input[0]
    mul_output = n.output[0]

    new_nodes = [
        helper.make_node("Mul", ["add_18087__offset_bf16", truediv_input],
                         ["__add18087_offset_scaled"],
                         name="__mul_add18087_offset"),
        helper.make_node("Mul", ["add_18087__bucket_bf16", truediv_input],
                         ["__add18087_bucket_scaled"],
                         name="__mul_add18087_bucket"),
        helper.make_node("Add", ["__add18087_offset_scaled",
                                 "__add18087_bucket_scaled"],
                         [mul_output], name="__add_add18087_recombine"),
    ]

    nodes = list(m.graph.node)
    nodes.pop(mul_idx)
    for j, nn in enumerate(new_nodes):
        nodes.insert(mul_idx + j, nn)

    del m.graph.node[:]
    m.graph.node.extend(nodes)
    del m.graph.value_info[:]

    print(f"Fixed add_18087: bucket decomposition (offset + bucket)")
    print(f"  Removed 1 node (Mul)")
    print(f"  Added 3 nodes (Mul offset, Mul bucket, Add recombine)")
    print(f"  offset range: [{offset.min():.1f}, {offset.max():.1f}]")
    print(f"  bucket values: {sorted(set(bucket))}")

    onnx.save(m, args.output)


if __name__ == "__main__":
    main()
