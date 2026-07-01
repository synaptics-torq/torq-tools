#!/usr/bin/env python3
"""Extract Gemm bias into a separate Add node with a fusion barrier.

The torq compiler silently drops the bias from certain Gemm ops
(observed on matmul_320x32x128_bf16). This script converts:
    Y = Gemm(A, B, C)
into:
    tmp = Gemm(A, B)
    padded = Pad(tmp, [1,0,...])
    sliced = Slice(padded, starts=[1], ends=[M+1], axes=[0])
    Y = Add(sliced, C)

The Pad+Slice barrier prevents TileAndFuse from re-fusing the Add
back into the Gemm dispatch.

By default, processes ALL Gemm nodes with a bias input. Use --names
to target specific nodes.
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def get_output_shape(model, tensor_name):
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if vi.name == tensor_name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None


def get_gemm_k_dim(model, node):
    """Get the K (inner/reduction) dimension of a Gemm node."""
    weight_name = node.input[1]
    transB = 0
    for attr in node.attribute:
        if attr.name == 'transB':
            transB = attr.i
    for ini in model.graph.initializer:
        if ini.name == weight_name:
            # Weight is [K, N] or [N, K] depending on transB
            return ini.dims[0] if transB else ini.dims[0]
    return None


def get_gemm_weight_bytes(model, node):
    """Get the weight tensor size in bytes."""
    weight_name = node.input[1]
    for ini in model.graph.initializer:
        if ini.name == weight_name:
            numel = 1
            for d in ini.dims:
                numel *= d
            # bf16 = 2 bytes per element
            return numel * 2
    return None


def extract_gemm_bias(model, target_names=None, min_k=None, min_weight_bytes=None):
    new_nodes = []
    count = 0
    skipped = 0

    for i, n in enumerate(model.graph.node):
        if n.op_type != 'Gemm':
            continue
        if len(n.input) < 3 or n.input[2] == '':
            continue
        if target_names and n.name not in target_names:
            continue
        if min_k is not None:
            k = get_gemm_k_dim(model, n)
            if k is not None and k < min_k:
                skipped += 1
                continue
        if min_weight_bytes is not None:
            wb = get_gemm_weight_bytes(model, n)
            if wb is not None and wb < min_weight_bytes:
                skipped += 1
                continue

        bias_name = n.input[2]
        original_output = n.output[0]
        prefix = f"{original_output}__bias_barrier"

        shape = get_output_shape(model, original_output)
        if shape is None:
            input_name = n.input[0]
            weight_name = n.input[1]
            for ini in model.graph.initializer:
                if ini.name == weight_name:
                    n_dim = ini.dims[1]
                    break
            for vi in list(model.graph.value_info) + list(model.graph.input):
                if vi.name == input_name:
                    m_dim = vi.type.tensor_type.shape.dim[0].dim_value
                    break
            shape = [m_dim, n_dim]

        matmul_out = f"{prefix}_matmul"
        padded_out = f"{prefix}_padded"
        sliced_out = f"{prefix}_sliced"

        n.output[0] = matmul_out
        while len(n.input) > 2:
            n.input.pop()

        ndim = len(shape)
        pad_amounts = [1] + [0] * (ndim - 1) + [0] * ndim
        pads_name = f"{prefix}_pads"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(pad_amounts, dtype=np.int64), name=pads_name))

        padded_shape = list(shape)
        padded_shape[0] += 1

        starts_name = f"{prefix}_starts"
        ends_name = f"{prefix}_ends"
        axes_name = f"{prefix}_axes"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([1], dtype=np.int64), name=starts_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([padded_shape[0]], dtype=np.int64), name=ends_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_name))

        pad_node = helper.make_node(
            "Pad", inputs=[matmul_out, pads_name], outputs=[padded_out],
            name=f"{prefix}_pad")
        slice_node = helper.make_node(
            "Slice", inputs=[padded_out, starts_name, ends_name, axes_name],
            outputs=[sliced_out], name=f"{prefix}_slice")
        add_node = helper.make_node(
            "Add", inputs=[sliced_out, bias_name], outputs=[original_output],
            name=f"{prefix}_add")

        model.graph.value_info.append(
            helper.make_tensor_value_info(matmul_out, TensorProto.BFLOAT16, shape))
        model.graph.value_info.append(
            helper.make_tensor_value_info(padded_out, TensorProto.BFLOAT16, padded_shape))
        model.graph.value_info.append(
            helper.make_tensor_value_info(sliced_out, TensorProto.BFLOAT16, shape))

        new_nodes.append((i + 1, [pad_node, slice_node, add_node]))
        count += 1

    offset = 0
    for idx, nodes in new_nodes:
        for j, node in enumerate(nodes):
            model.graph.node.insert(idx + offset + j, node)
        offset += len(nodes)

    if skipped:
        print(f"Skipped {skipped} Gemm node(s) below threshold")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--names", nargs="+",
                        help="Only process these Gemm node names")
    parser.add_argument("--min-k", type=int, default=None,
                        help="Only extract bias from Gemms with K dimension >= this value")
    parser.add_argument("--min-weight-bytes", type=int, default=None,
                        help="Only extract bias from Gemms with weight tensor >= this many bytes")
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = extract_gemm_bias(
        model,
        target_names=set(args.names) if args.names else None,
        min_k=args.min_k,
        min_weight_bytes=args.min_weight_bytes,
    )
    print(f"Extracted bias from {count} Gemm node(s)")

    out_path = args.output or args.input
    onnx.save(model, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
