#!/usr/bin/env python3
"""Squeeze batch dimensions from MatMul ops to make them rank-2.

The torq compiler's MarkPatternsForTileAndFuse pass marks rank-2 linalg.matmul
for NSS (via Conv2DMatmulPattern), but rank-3+ linalg.batch_matmul goes to CSS.

This script converts:
  4D MatMul [1,H,M,K] x [1,H,K,N] -> H x 2D MatMul [M,K] x [K,N]
  3D MatMul [1,M,K] x [1,K,N]     -> 2D MatMul [M,K] x [K,N]
  3D MatMul [1,M,K] x [K,N]       -> 2D MatMul [M,K] x [K,N]

Usage:
    python3 scripts/squeeze_batch_matmul.py -i model.onnx -o model_squeezed.onnx
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def get_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    return TensorProto.BFLOAT16


def node_by_output(nodes, name):
    for n in nodes:
        if name in n.output:
            return n
    return None


def consumers_of(nodes, name):
    return [n for n in nodes if name in n.input]


def make_reshape(input_name, output_name, target_shape, node_name, graph, dtype):
    shape_name = f"{node_name}__shape"
    graph.initializer.append(numpy_helper.from_array(
        np.array(target_shape, dtype=np.int64), name=shape_name))
    node = helper.make_node("Reshape", [input_name, shape_name], [output_name],
                            name=node_name)
    graph.value_info.append(helper.make_tensor_value_info(output_name, dtype, list(target_shape)))
    return node


def make_barrier(input_name, output_name, shape, node_name, graph, dtype):
    """Add Pad(1)+Slice(1:2) barrier to break fusion chains."""
    pad_out = f"{node_name}__pad"
    # Pad 1 element along dim 0: pads=[1,0,0,...,0,...] (begin0=1, rest=0)
    ndim = len(shape)
    pads = [0] * (2 * ndim)
    pads[0] = 1  # pad 1 at beginning of dim 0
    pads_name = f"{node_name}__pads"
    graph.initializer.append(numpy_helper.from_array(
        np.array(pads, dtype=np.int64), name=pads_name))

    pad_shape = list(shape)
    pad_shape[0] += 1
    pad_node = helper.make_node("Pad", [input_name, pads_name], [pad_out],
                                name=f"{node_name}__pad_node", mode="constant")
    graph.value_info.append(helper.make_tensor_value_info(pad_out, dtype, pad_shape))

    # Slice [1:2] along dim 0 to get back to original shape
    starts_name = f"{node_name}__starts"
    ends_name = f"{node_name}__ends"
    axes_name = f"{node_name}__slice_axes"
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=starts_name))
    graph.initializer.append(numpy_helper.from_array(np.array([1 + shape[0]], dtype=np.int64), name=ends_name))
    graph.initializer.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_name))

    slice_node = helper.make_node("Slice", [pad_out, starts_name, ends_name, axes_name],
                                  [output_name], name=f"{node_name}__slice_node")
    graph.value_info.append(helper.make_tensor_value_info(output_name, dtype, list(shape)))

    return [pad_node, slice_node]


def squeeze_3d_matmul(model, node):
    """Convert 3D MatMul with batch=1 to 2D MatMul.

    [1,M,K] x [1,K,N] -> [M,K] x [K,N] -> [M,N] -> [1,M,N]
    [1,M,K] x [K,N]   -> [M,K] x [K,N] -> [M,N] -> [1,M,N]
    """
    graph = model.graph
    a_name, b_name = node.input[0], node.input[1]
    out_name = node.output[0]
    a_shape = get_shape(model, a_name)
    b_shape = get_shape(model, b_name)
    out_shape = get_shape(model, out_name)
    dtype = get_dtype(model, a_name)

    if a_shape is None or b_shape is None or out_shape is None:
        return None

    base = node.name or out_name

    new_nodes = []

    # Reshape A from [1,M,K] to [M,K]
    a_sq_name = f"{base}__a_sq"
    new_nodes.append(make_reshape(a_name, a_sq_name, a_shape[1:],
                                  f"{base}__reshape_a", graph, dtype))

    # Reshape B if 3D
    if len(b_shape) == 3 and b_shape[0] == 1:
        b_sq_name = f"{base}__b_sq"
        b_dtype = get_dtype(model, b_name)
        new_nodes.append(make_reshape(b_name, b_sq_name, b_shape[1:],
                                      f"{base}__reshape_b", graph, b_dtype))
    elif len(b_shape) == 2:
        b_sq_name = b_name
    else:
        return None

    # 2D MatMul
    mm_out = f"{base}__mm2d"
    mm_shape = [a_shape[1], b_shape[-1]]
    new_nodes.append(helper.make_node("MatMul", [a_sq_name, b_sq_name], [mm_out],
                                      name=f"{base}__matmul_2d"))
    graph.value_info.append(helper.make_tensor_value_info(mm_out, dtype, mm_shape))

    # Reshape back to [1,M,N]
    new_nodes.append(make_reshape(mm_out, out_name, out_shape,
                                  f"{base}__reshape_out", graph, dtype))

    return new_nodes


def squeeze_4d_matmul_single_head(model, node):
    """Convert 4D MatMul with batch=1, heads=1 to 2D MatMul.

    [1,1,M,K] x [1,1,K,N] -> [M,K] x [K,N] -> [M,N] -> [1,1,M,N]
    """
    graph = model.graph
    a_name, b_name = node.input[0], node.input[1]
    out_name = node.output[0]
    a_shape = get_shape(model, a_name)
    b_shape = get_shape(model, b_name)
    out_shape = get_shape(model, out_name)
    dtype = get_dtype(model, a_name)

    if a_shape is None or b_shape is None or out_shape is None:
        return None

    base = node.name or out_name
    new_nodes = []

    # Reshape A from [1,1,M,K] to [M,K]
    a_sq_name = f"{base}__a_sq"
    new_nodes.append(make_reshape(a_name, a_sq_name, a_shape[2:],
                                  f"{base}__reshape_a", graph, dtype))

    # Reshape B from [1,1,K,N] to [K,N]
    b_sq_name = f"{base}__b_sq"
    b_dtype = get_dtype(model, b_name)
    new_nodes.append(make_reshape(b_name, b_sq_name, b_shape[2:],
                                  f"{base}__reshape_b", graph, b_dtype))

    # 2D MatMul
    mm_out = f"{base}__mm2d"
    mm_shape = [a_shape[2], b_shape[-1]]
    new_nodes.append(helper.make_node("MatMul", [a_sq_name, b_sq_name], [mm_out],
                                      name=f"{base}__matmul_2d"))
    graph.value_info.append(helper.make_tensor_value_info(mm_out, dtype, mm_shape))

    # Reshape back to [1,1,M,N]
    new_nodes.append(make_reshape(mm_out, out_name, out_shape,
                                  f"{base}__reshape_out", graph, dtype))

    return new_nodes


def split_heads_and_squeeze(model, node):
    """Convert 4D MatMul [1,H,M,K] x [1,H,K,N] to H x 2D MatMul.

    Split along head dim, squeeze to 2D, MatMul per head, unsqueeze+concat.
    """
    graph = model.graph
    a_name, b_name = node.input[0], node.input[1]
    out_name = node.output[0]
    a_shape = get_shape(model, a_name)
    b_shape = get_shape(model, b_name)
    out_shape = get_shape(model, out_name)
    dtype = get_dtype(model, a_name)
    b_dtype = get_dtype(model, b_name)

    if a_shape is None or b_shape is None or out_shape is None:
        return None
    if len(a_shape) != 4 or len(b_shape) != 4:
        return None

    H = a_shape[1]
    M, K_a = a_shape[2], a_shape[3]
    K_b, N = b_shape[2], b_shape[3]
    base = node.name or out_name

    new_nodes = []

    # Split sizes initializer (H x [1])
    split_sizes_name = f"{base}__head_split_sizes"
    graph.initializer.append(numpy_helper.from_array(
        np.array([1] * H, dtype=np.int64), name=split_sizes_name))

    # Split A along head dim
    a_head_names = [f"{base}__a_h{h}" for h in range(H)]
    new_nodes.append(helper.make_node("Split", [a_name, split_sizes_name], a_head_names,
                                      name=f"{base}__split_a", axis=1))
    for name in a_head_names:
        graph.value_info.append(helper.make_tensor_value_info(name, dtype, [1, 1, M, K_a]))

    # Split B along head dim
    b_head_names = [f"{base}__b_h{h}" for h in range(H)]
    new_nodes.append(helper.make_node("Split", [b_name, split_sizes_name], b_head_names,
                                      name=f"{base}__split_b", axis=1))
    for name in b_head_names:
        graph.value_info.append(helper.make_tensor_value_info(name, b_dtype, [1, 1, K_b, N]))

    # Per-head: reshape to 2D, matmul, reshape back to [1,1,M,N]
    head_4d_names = []
    for h in range(H):
        sfx = f"_h{h}"

        # Reshape A: [1,1,M,K] -> [M,K]
        a_sq = f"{base}__a_sq{sfx}"
        new_nodes.append(make_reshape(a_head_names[h], a_sq, [M, K_a],
                                      f"{base}__reshape_a{sfx}", graph, dtype))

        # Reshape B: [1,1,K,N] -> [K,N]
        b_sq = f"{base}__b_sq{sfx}"
        new_nodes.append(make_reshape(b_head_names[h], b_sq, [K_b, N],
                                      f"{base}__reshape_b{sfx}", graph, b_dtype))

        # 2D MatMul
        mm_out = f"{base}__mm2d{sfx}"
        mm_shape = [M, N]
        new_nodes.append(helper.make_node("MatMul", [a_sq, b_sq], [mm_out],
                                          name=f"{base}__matmul_2d{sfx}"))
        graph.value_info.append(helper.make_tensor_value_info(mm_out, dtype, mm_shape))

        # Fusion barrier after MatMul to prevent problematic fusion chains
        barrier_out = f"{base}__barrier{sfx}"
        new_nodes.extend(make_barrier(mm_out, barrier_out, mm_shape,
                                      f"{base}__barrier{sfx}", graph, dtype))

        # Reshape back to [1,1,M,N]
        us_out = f"{base}__us{sfx}"
        new_nodes.append(make_reshape(barrier_out, us_out, [1, 1, M, N],
                                      f"{base}__reshape_out{sfx}", graph, dtype))
        head_4d_names.append(us_out)

    # Concat heads: H x [1,1,M,N] -> [1,H,M,N]
    concat_raw = f"{base}__concat_heads_raw"
    new_nodes.append(helper.make_node("Concat", head_4d_names, [concat_raw],
                                      name=f"{base}__concat_heads", axis=1))
    graph.value_info.append(helper.make_tensor_value_info(concat_raw, dtype, [1, H, M, N]))

    new_nodes.extend(make_barrier(concat_raw, out_name, [1, H, M, N],
                                  f"{base}__concat_barrier", graph, dtype))

    return new_nodes


def process_model(model, only_3d=False, only_4d=False):
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    nodes = list(graph.node)

    converted_3d = 0
    converted_4d_single = 0
    converted_4d_multi = 0
    skipped = 0

    new_nodes = []
    for node in nodes:
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue

        a_shape = get_shape(model, node.input[0])
        b_shape = get_shape(model, node.input[1])

        if a_shape is None or b_shape is None:
            new_nodes.append(node)
            skipped += 1
            continue

        rank_a = len(a_shape)
        rank_b = len(b_shape)
        name = node.name or node.output[0]

        if not only_4d and rank_a == 3 and a_shape[0] == 1:
            result = squeeze_3d_matmul(model, node)
            if result:
                new_nodes.extend(result)
                converted_3d += 1
                print(f"  3D->2D: {name} {a_shape} x {b_shape}")
                continue

        if not only_3d and rank_a == 4 and a_shape[0] == 1 and a_shape[1] == 1:
            result = squeeze_4d_matmul_single_head(model, node)
            if result:
                new_nodes.extend(result)
                converted_4d_single += 1
                print(f"  4D(H=1)->2D: {name} {a_shape} x {b_shape}")
                continue

        if not only_3d and rank_a == 4 and rank_b == 4 and a_shape[0] == 1 and a_shape[1] > 1:
            result = split_heads_and_squeeze(model, node)
            if result:
                new_nodes.extend(result)
                converted_4d_multi += 1
                print(f"  4D(H={a_shape[1]})->per-head 2D: {name} {a_shape} x {b_shape}")
                continue

        new_nodes.append(node)
        if rank_a > 2 or rank_b > 2:
            skipped += 1
            print(f"  SKIP: {name} {a_shape} x {b_shape}")

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"\nConverted: {converted_3d} 3D, {converted_4d_single} 4D(H=1), "
          f"{converted_4d_multi} 4D(H>1), skipped: {skipped}")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--only-3d", action="store_true",
                        help="Only convert 3D MatMul (skip 4D head splitting)")
    parser.add_argument("--only-4d", action="store_true",
                        help="Only convert 4D MatMul (skip 3D)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    model = onnx.load(args.input)

    print("Converting batched MatMul to 2D MatMul...")
    model = process_model(model, only_3d=args.only_3d, only_4d=args.only_4d)

    print(f"Saving to {args.output}...")
    onnx.save(model, args.output)

    print("Checking model...")
    onnx.checker.check_model(model, full_check=False)
    print("Done.")


if __name__ == "__main__":
    main()
