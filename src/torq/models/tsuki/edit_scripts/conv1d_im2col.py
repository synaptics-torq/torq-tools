#!/usr/bin/env python3
"""Replace large Conv1D ops with im2col + Gemm decomposition.

Based on DecomposeStridedConv1D from torq-tools-dev moonshine/_graph.py,
adapted for stride=1 convolutions with padding.

Default im2col approach (transpose-free):
  For each kernel position j in [0, K):
    Slice input[:, :, j : j + Lout] -> [B, Cin, Lout]
  Unsqueeze to [B, Cin, 1, Lout]
  Concat along axis=2 -> [B, Cin, K, Lout]
  Reshape to [Cin*K, Lout]
  Gemm: W[Cout, Cin*K] x patches[Cin*K, Lout] + bias -> [Cout, Lout]
  Reshape to [B, Cout, Lout]

Per-kernel approach (--per-kernel):
  Eliminates the Concat entirely. Each kernel position gets its own Gemm
  with a [Cout, Cin] weight slice, and results are summed:
  For each kernel position j in [0, K):
    Slice input[:, :, j : j + Lout] -> [B, Cin, Lout]
    Reshape to [Cin, Lout]
    Gemm: W_j[Cout, Cin] x pos_j[Cin, Lout] -> [Cout, Lout]
  Sum all K Gemm outputs + bias -> [Cout, Lout]
  Reshape to [B, Cout, Lout]

  Avoids the [B, Cin, K, Lout] Concat tensor which can exceed CSS stack
  limits for large Cin*K*Lout.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def get_conv1d_info(node, initializers):
    """Extract Conv1D attributes and shapes. Returns None if not applicable."""
    if node.op_type != "Conv":
        return None

    attrs = {a.name: a for a in node.attribute}
    kernel_shape = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else None
    if kernel_shape is None or len(kernel_shape) != 1:
        return None

    strides = list(attrs["strides"].ints) if "strides" in attrs else [1]

    group = attrs["group"].i if "group" in attrs else 1
    if group != 1:
        return None

    dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0]

    weight_name = node.input[1]
    if weight_name not in initializers:
        return None
    w = initializers[weight_name]
    w_shape = list(w.dims)
    if len(w_shape) != 3:
        return None

    cout, cin, k = w_shape
    weight_bytes = cout * cin * k * 2  # bf16

    stride = strides[0]
    return {
        "kernel": k,
        "stride": stride,
        "dilation": dilations[0],
        "pads": pads,
        "group": group,
        "cout": cout,
        "cin": cin,
        "weight_bytes": weight_bytes,
        "weight_name": weight_name,
    }


def decompose_conv1d_to_gemm(model, min_weight_bytes=200000, per_kernel=False,
                             max_input_length=None):
    """Replace qualifying Conv1D ops with im2col + Gemm."""
    graph = model.graph

    # Build initializer lookup
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = init

    # Build shape lookup from value_info + inputs + outputs
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.shape.dim:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else -1)
            shapes[vi.name] = dims

    # Also get shapes from initializers
    for name, init in initializers.items():
        shapes[name] = list(init.dims)

    candidates = []
    for node in graph.node:
        info = get_conv1d_info(node, initializers)
        if info is None:
            continue
        if info["weight_bytes"] < min_weight_bytes:
            continue

        # Get input shape
        inp_name = node.input[0]
        if inp_name not in shapes:
            continue
        inp_shape = shapes[inp_name]
        if len(inp_shape) != 3:
            continue

        if max_input_length is not None and inp_shape[2] > max_input_length:
            continue

        info["input_shape"] = inp_shape
        info["node"] = node
        candidates.append(info)

    print(f"Found {len(candidates)} Conv1D ops to decompose (weight >= {min_weight_bytes} bytes)")

    replacement_map = {}  # id(node) -> list of replacement nodes
    new_initializers = []
    new_value_info = []
    nodes_to_remove = set()

    for info in candidates:
        node = info["node"]
        cin = info["cin"]
        cout = info["cout"]
        k = info["kernel"]
        pads = info["pads"]
        pad_left, pad_right = pads[0], pads[1]

        inp_shape = info["input_shape"]
        batch = inp_shape[0]
        l_in = inp_shape[2]
        stride = info["stride"]
        dilation = info["dilation"]
        effective_k = (k - 1) * dilation + 1
        l_out = (l_in + pad_left + pad_right - effective_k) // stride + 1

        prefix = node.name or f"conv1d_im2col_{id(node)}"
        inp_name = node.input[0]
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        out_name = node.output[0]

        dilation_str = f" dilation={dilation}" if dilation > 1 else ""
        print(f"  {prefix}: [{batch},{cin},{l_in}] * [{cout},{cin},{k}] -> [{batch},{cout},{l_out}]"
              f"  pad={pads}{dilation_str} weight={info['weight_bytes']/1024:.0f}KB")

        chain_nodes = []

        # Step 1: Pad input if needed
        if pad_left > 0 or pad_right > 0:
            # Pad along last dimension: pads = [0,0,0, 0,0,0] for [B,C,L] -> [0,0,pad_left, 0,0,pad_right]
            pad_values = np.array([0, 0, pad_left, 0, 0, pad_right], dtype=np.int64)
            pad_const_name = f"{prefix}_pad_vals"
            new_initializers.append(numpy_helper.from_array(pad_values, name=pad_const_name))

            padded_name = f"{prefix}_padded"
            l_padded = l_in + pad_left + pad_right
            chain_nodes.append(helper.make_node(
                "Pad", [inp_name, pad_const_name], [padded_name],
                name=f"{prefix}_pad_input", mode="constant"
            ))
            new_value_info.append(helper.make_tensor_value_info(
                padded_name, TensorProto.BFLOAT16, [batch, cin, l_padded]
            ))
            current_input = padded_name
            current_l = l_padded
        else:
            current_input = inp_name
            current_l = l_in

        # Step 2: Per-kernel-position slices
        axes_const_name = f"{prefix}_axes_2"
        new_initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name=axes_const_name))

        if stride > 1:
            steps_const_name = f"{prefix}_steps_{stride}"
            new_initializers.append(numpy_helper.from_array(np.array([stride], dtype=np.int64), name=steps_const_name))

        slice_outputs = []
        for j in range(k):
            start_name = f"{prefix}_start_{j}"
            end_name = f"{prefix}_end_{j}"
            slice_out_name = f"{prefix}_pos{j}"

            start_pos = j * dilation
            new_initializers.append(numpy_helper.from_array(np.array([start_pos], dtype=np.int64), name=start_name))
            end_val = start_pos + l_out * stride if stride > 1 else start_pos + l_out
            new_initializers.append(numpy_helper.from_array(np.array([end_val], dtype=np.int64), name=end_name))

            slice_inputs = [current_input, start_name, end_name, axes_const_name]
            if stride > 1:
                slice_inputs.append(steps_const_name)

            chain_nodes.append(helper.make_node(
                "Slice", slice_inputs, [slice_out_name],
                name=f"{prefix}_slice_pos{j}"
            ))
            new_value_info.append(helper.make_tensor_value_info(
                slice_out_name, TensorProto.BFLOAT16, [batch, cin, l_out]
            ))
            slice_outputs.append(slice_out_name)

        w_data = numpy_helper.to_array(initializers[weight_name])

        if per_kernel:
            # Per-kernel mode: separate Gemm per kernel position, sum results.
            # No Concat needed — avoids the [B,Cin,K,Lout] tensor.
            flat_shape_name = f"{prefix}_flat_shape"
            new_initializers.append(numpy_helper.from_array(
                np.array([cin, l_out], dtype=np.int64), name=flat_shape_name
            ))

            gemm_outputs = []
            for j in range(k):
                # Reshape [B,Cin,Lout] -> [Cin,Lout]
                pos_flat_name = f"{prefix}_pos{j}_flat"
                chain_nodes.append(helper.make_node(
                    "Reshape", [slice_outputs[j], flat_shape_name], [pos_flat_name],
                    name=f"{prefix}_reshape_pos{j}"
                ))
                new_value_info.append(helper.make_tensor_value_info(
                    pos_flat_name, TensorProto.BFLOAT16, [cin, l_out]
                ))

                # Weight slice: W[:,:,j] -> [Cout,Cin]
                w_j_name = f"{prefix}_w_k{j}"
                w_j_data = w_data[:, :, j].copy()
                new_initializers.append(numpy_helper.from_array(w_j_data, name=w_j_name))

                # Gemm: W_j[Cout,Cin] x pos_j[Cin,Lout] -> [Cout,Lout]
                gemm_out_name = f"{prefix}_gemm_k{j}"
                chain_nodes.append(helper.make_node(
                    "Gemm", [w_j_name, pos_flat_name], [gemm_out_name],
                    name=f"{prefix}_gemm_k{j}_op",
                    transA=0, transB=0, alpha=1.0,
                ))
                new_value_info.append(helper.make_tensor_value_info(
                    gemm_out_name, TensorProto.BFLOAT16, [cout, l_out]
                ))
                gemm_outputs.append(gemm_out_name)

            # Sum all kernel position outputs
            sum_name = gemm_outputs[0]
            for j in range(1, k):
                next_sum = f"{prefix}_sum_k{j}"
                chain_nodes.append(helper.make_node(
                    "Add", [sum_name, gemm_outputs[j]], [next_sum],
                    name=f"{prefix}_add_k{j}"
                ))
                new_value_info.append(helper.make_tensor_value_info(
                    next_sum, TensorProto.BFLOAT16, [cout, l_out]
                ))
                sum_name = next_sum

            final_gemm_out = sum_name

        else:
            # Default mode: Concat all positions, single Gemm
            for j in range(k):
                unsq_axes_name = f"{prefix}_unsq_axes_2"
                if j == 0:
                    new_initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name=unsq_axes_name))
                unsq_out_name = f"{prefix}_pos{j}_4d"
                chain_nodes.append(helper.make_node(
                    "Unsqueeze", [slice_outputs[j], unsq_axes_name], [unsq_out_name],
                    name=f"{prefix}_unsq_pos{j}"
                ))
                new_value_info.append(helper.make_tensor_value_info(
                    unsq_out_name, TensorProto.BFLOAT16, [batch, cin, 1, l_out]
                ))
                slice_outputs[j] = unsq_out_name

            patches_name = f"{prefix}_patches"
            chain_nodes.append(helper.make_node(
                "Concat", slice_outputs, [patches_name],
                name=f"{prefix}_im2col_concat", axis=2
            ))
            new_value_info.append(helper.make_tensor_value_info(
                patches_name, TensorProto.BFLOAT16, [batch, cin, k, l_out]
            ))

            flat_shape_name = f"{prefix}_flat_shape"
            new_initializers.append(numpy_helper.from_array(
                np.array([cin * k, l_out], dtype=np.int64), name=flat_shape_name
            ))
            patches_flat_name = f"{prefix}_patches_flat"
            chain_nodes.append(helper.make_node(
                "Reshape", [patches_name, flat_shape_name], [patches_flat_name],
                name=f"{prefix}_reshape_patches"
            ))
            new_value_info.append(helper.make_tensor_value_info(
                patches_flat_name, TensorProto.BFLOAT16, [cin * k, l_out]
            ))

            w_flat_name = f"{prefix}_w_flat"
            w_flat_data = w_data.reshape(cout, cin * k)
            new_initializers.append(numpy_helper.from_array(w_flat_data, name=w_flat_name))

            gemm_out_name = f"{prefix}_gemm"
            chain_nodes.append(helper.make_node(
                "Gemm", [w_flat_name, patches_flat_name], [gemm_out_name],
                name=f"{prefix}_gemm_op",
                transA=0, transB=0, alpha=1.0,
            ))
            new_value_info.append(helper.make_tensor_value_info(
                gemm_out_name, TensorProto.BFLOAT16, [cout, l_out]
            ))

            final_gemm_out = gemm_out_name

        # Bias Add
        biased_out_name = final_gemm_out
        if bias_name:
            bias_col_name = f"{prefix}_bias_col"
            bias_data = numpy_helper.to_array(initializers[bias_name])
            bias_col = bias_data.reshape(cout, 1)
            new_initializers.append(numpy_helper.from_array(bias_col, name=bias_col_name))

            biased_out_name = f"{prefix}_biased"
            chain_nodes.append(helper.make_node(
                "Add", [final_gemm_out, bias_col_name], [biased_out_name],
                name=f"{prefix}_bias_add"
            ))
            new_value_info.append(helper.make_tensor_value_info(
                biased_out_name, TensorProto.BFLOAT16, [cout, l_out]
            ))

        # Reshape [Cout, Lout] -> [B, Cout, Lout]
        reshape_out_name = f"{prefix}_reshape_out_raw"
        out_3d_shape_name = f"{prefix}_out_3d_shape"
        new_initializers.append(numpy_helper.from_array(
            np.array([batch, cout, l_out], dtype=np.int64), name=out_3d_shape_name
        ))
        chain_nodes.append(helper.make_node(
            "Reshape", [biased_out_name, out_3d_shape_name], [reshape_out_name],
            name=f"{prefix}_reshape_out"
        ))
        new_value_info.append(helper.make_tensor_value_info(
            reshape_out_name, TensorProto.BFLOAT16, [batch, cout, l_out]
        ))

        # Step 8: Pad+Slice fusion barrier (prevents APInt assertion in TileAndFuse)
        barrier_pad_name = f"{prefix}_barrier_pad"
        barrier_pads_name = f"{prefix}_barrier_pads"
        barrier_const_name = f"{prefix}_barrier_const"
        ndim = 3
        pads = [0] * ndim + [0] * ndim
        pads[0] = 1  # pad dim 0 by 1 at the beginning
        new_initializers.append(numpy_helper.from_array(
            np.array(pads, dtype=np.int64), name=barrier_pads_name))
        barrier_const_tensor = TensorProto()
        barrier_const_tensor.name = barrier_const_name
        barrier_const_tensor.data_type = TensorProto.BFLOAT16
        barrier_const_tensor.raw_data = b'\x00\x00'
        new_initializers.append(barrier_const_tensor)
        chain_nodes.append(helper.make_node(
            "Pad", [reshape_out_name, barrier_pads_name, barrier_const_name],
            [barrier_pad_name], name=f"{prefix}_barrier_pad", mode="constant"
        ))
        new_value_info.append(helper.make_tensor_value_info(
            barrier_pad_name, TensorProto.BFLOAT16, [batch + 1, cout, l_out]
        ))

        barrier_starts = f"{prefix}_barrier_starts"
        barrier_ends = f"{prefix}_barrier_ends"
        barrier_axes = f"{prefix}_barrier_axes"
        new_initializers.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=barrier_starts))
        new_initializers.append(numpy_helper.from_array(
            np.array([np.iinfo(np.int64).max], dtype=np.int64), name=barrier_ends))
        new_initializers.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=barrier_axes))
        chain_nodes.append(helper.make_node(
            "Slice", [barrier_pad_name, barrier_starts, barrier_ends, barrier_axes],
            [out_name], name=f"{prefix}_barrier_slice"
        ))

        nodes_to_remove.add(id(node))
        replacement_map[id(node)] = chain_nodes

    if not candidates:
        return model, 0

    all_nodes = []
    for n in graph.node:
        if id(n) in nodes_to_remove:
            if id(n) in replacement_map:
                all_nodes.extend(replacement_map[id(n)])
            continue
        all_nodes.append(n)

    graph.ClearField("node")
    graph.node.extend(all_nodes)

    for init in new_initializers:
        graph.initializer.append(init)

    for vi in new_value_info:
        graph.value_info.append(vi)

    return model, len(candidates)


def main():
    parser = argparse.ArgumentParser(description="Replace large Conv1D with im2col + Gemm")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--min-weight-bytes", type=int, default=200000,
                        help="Minimum weight size in bytes to decompose (default: 200000)")
    parser.add_argument("--per-kernel", action="store_true",
                        help="Use per-kernel-position Gemms instead of Concat (avoids large intermediate tensors)")
    parser.add_argument("--max-input-length", type=int, default=None,
                        help="Only decompose Conv1D with input length <= this value")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    model = onnx.load(args.input)
    print(f"Nodes before: {len(model.graph.node)}")

    model, count = decompose_conv1d_to_gemm(model, min_weight_bytes=args.min_weight_bytes,
                                             per_kernel=args.per_kernel,
                                             max_input_length=args.max_input_length)

    if count > 0:
        print(f"Decomposed {count} Conv1D ops")
        print(f"Nodes after: {len(model.graph.node)}")
        onnx.save(model, args.output)
        print(f"Saved to {args.output}")
    else:
        print("No Conv1D ops matched criteria, saving unchanged")
        onnx.save(model, args.output)


if __name__ == "__main__":
    main()
