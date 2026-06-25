#!/usr/bin/env python3
"""Split large Gemm ops along the output (N) dimension to reduce LRAM pressure.

TileAndFuse loads the full Gemm weight matrix into 512KB LRAM. When the weight
exceeds a threshold, the tiled operation plus carry-over from preceding ops can
overflow LRAM. This script splits each large Gemm into blocked matrix
multiplications where each block's weight fits within max_block_bytes.

  Y = A @ B + C  →  Y = Concat(A @ B_0 + C_0, ..., A @ B_{n-1} + C_{n-1})

where B_i = B[:, i*blk:(i+1)*blk] and C_i = C[i*blk:(i+1)*blk].
"""

import argparse
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def compute_block_count(K, N, elem_size, max_block_bytes):
    max_block_N = max_block_bytes // (K * elem_size)
    if max_block_N <= 0:
        max_block_N = 1
    if max_block_N >= N:
        return 1
    num_blocks = (N + max_block_N - 1) // max_block_N
    return num_blocks


def split_large_gemms(model, min_weight_bytes=262144, max_block_bytes=131072):
    graph = model.graph
    init_map = {i.name: i for i in graph.initializer}

    replacements = {}
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        if "__split" in (node.name or "") and "__split_concat" not in (node.name or ""):
            pass  # allow re-splitting of already-split Gemm blocks

        trans_a = 0
        trans_b = 0
        alpha = 1.0
        beta = 1.0
        for attr in node.attribute:
            if attr.name == "transA":
                trans_a = attr.i
            elif attr.name == "transB":
                trans_b = attr.i
            elif attr.name == "alpha":
                alpha = attr.f
            elif attr.name == "beta":
                beta = attr.f

        # Detect weight location: input[1] (standard) or input[0] (im2col)
        weight_input_idx = None
        if init_map.get(node.input[1]) is not None:
            weight_input_idx = 1
        elif init_map.get(node.input[0]) is not None:
            weight_input_idx = 0

        if weight_input_idx is None:
            continue

        weight_name = node.input[weight_input_idx]
        if f"{weight_name}__split0" in init_map:
            continue
        weight_init = init_map[weight_name]

        weight_array = numpy_helper.to_array(weight_init)
        if weight_array.nbytes < min_weight_bytes:
            continue

        # Determine the output-features dimension (N) to split along
        if weight_input_idx == 1:
            if trans_b:
                K, N = weight_array.shape[1], weight_array.shape[0]
            else:
                K, N = weight_array.shape
            split_dim = 0 if trans_b else 1
            concat_axis = 1
        else:
            # Weight is A: shape [M, K] where M=output features
            if trans_a:
                K, N = weight_array.shape[0], weight_array.shape[1]
                split_dim = 1
            else:
                N, K = weight_array.shape[0], weight_array.shape[1]
                split_dim = 0
            concat_axis = 0

        num_blocks = compute_block_count(K, N, weight_array.dtype.itemsize,
                                         max_block_bytes)
        if num_blocks <= 1:
            continue

        has_bias = len(node.input) > 2 and node.input[2] != ""
        bias_init = init_map.get(node.input[2]) if has_bias else None
        bias_array = numpy_helper.to_array(bias_init) if bias_init else None

        block_N = N // num_blocks
        remainder = N % num_blocks

        block_nodes = []
        block_outputs = []
        new_inits = []
        col = 0

        for i in range(num_blocks):
            bsz = block_N + (1 if i < remainder else 0)

            if split_dim == 0:
                w_block = weight_array[col:col + bsz, :]
            else:
                w_block = weight_array[:, col:col + bsz]

            node_prefix = node.name or node.output[0]
            w_name = f"{node_prefix}__{weight_name}__split{i}"
            new_inits.append(numpy_helper.from_array(w_block, name=w_name))

            if weight_input_idx == 1:
                gemm_inputs = [node.input[0], w_name]
            else:
                gemm_inputs = [w_name, node.input[1]]

            if has_bias and bias_array is not None:
                if bias_array.ndim == 1:
                    b_block = bias_array[col:col + bsz]
                else:
                    b_block = bias_array[col:col + bsz, :]
                b_name = f"{node_prefix}__{node.input[2]}__split{i}"
                new_inits.append(numpy_helper.from_array(b_block, name=b_name))
                gemm_inputs.append(b_name)

            out_name = f"{node.output[0]}__split{i}"
            block_outputs.append(out_name)

            attrs = {"alpha": alpha, "beta": beta}
            if trans_a:
                attrs["transA"] = 1
            if trans_b:
                attrs["transB"] = 1

            block_node = helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[out_name],
                name=f"{node.name}__split{i}",
                **attrs,
            )
            block_nodes.append(block_node)
            col += bsz

        concat_out = f"{node.output[0]}__split_concat_raw"
        concat_node = helper.make_node(
            "Concat",
            inputs=block_outputs,
            outputs=[concat_out],
            name=f"{node.name}__split_concat",
            axis=concat_axis,
        )

        padded_name = f"{node.output[0]}__split_barrier_pad"
        pad_const = f"{node.name}__split_pad_const"
        pad_pads = f"{node.name}__split_pad_pads"
        new_inits.append(numpy_helper.from_array(
            np.zeros([], dtype=weight_array.dtype), name=pad_const))
        ndim = 2
        pads = [0] * ndim + [0] * ndim
        pads[0] = 1
        new_inits.append(numpy_helper.from_array(
            np.array(pads, dtype=np.int64), name=pad_pads))
        pad_node = helper.make_node(
            "Pad",
            inputs=[concat_out, pad_pads, pad_const],
            outputs=[padded_name],
            name=f"{node.name}__split_barrier_pad",
        )

        starts_name = f"{node.name}__split_slice_starts"
        ends_name = f"{node.name}__split_slice_ends"
        axes_name = f"{node.name}__split_slice_axes"
        new_inits.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=starts_name))
        new_inits.append(numpy_helper.from_array(
            np.array([np.iinfo(np.int64).max], dtype=np.int64),
            name=ends_name))
        new_inits.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=axes_name))
        slice_node = helper.make_node(
            "Slice",
            inputs=[padded_name, starts_name, ends_name, axes_name],
            outputs=[node.output[0]],
            name=f"{node.name}__split_barrier_slice",
        )

        barrier_nodes = [pad_node, slice_node]
        replacements[node.name] = (block_nodes, concat_node, barrier_nodes,
                                   new_inits)
        blk_kb = block_N * K * weight_array.dtype.itemsize // 1024
        print(f"  {node.name}: [{K},{N}] -> {num_blocks}x [{K},{block_N}] "
              f"({blk_kb}KB each)")

    if not replacements:
        print("No Gemm ops need splitting")
        return model

    final_nodes = []
    for node in graph.node:
        if node.name in replacements:
            block_nodes, concat_node, barrier_nodes, new_inits = \
                replacements[node.name]
            final_nodes.extend(block_nodes)
            final_nodes.append(concat_node)
            final_nodes.extend(barrier_nodes)
            graph.initializer.extend(new_inits)
        else:
            final_nodes.append(node)

    del graph.node[:]
    graph.node.extend(final_nodes)

    print(f"\nSplit {len(replacements)} Gemm ops into "
          f"{sum(len(r[0]) for r in replacements.values())} blocks + "
          f"{len(replacements)} Concats + {len(replacements)} barriers")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--min-weight-bytes", type=int, default=262144,
                        help="Only split Gemms with weight >= this (default: 256KB)")
    parser.add_argument("--max-block-bytes", type=int, default=131072,
                        help="Max weight bytes per block (default: 128KB)")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    model = onnx.load(args.input)
    print(f"Splitting Gemms with weight >= {args.min_weight_bytes // 1024}KB "
          f"into blocks <= {args.max_block_bytes // 1024}KB:")

    model = split_large_gemms(model, args.min_weight_bytes, args.max_block_bytes)

    print(f"\nSaving to {args.output}")
    onnx.save(model, args.output)
    print("Done")


if __name__ == "__main__":
    main()
