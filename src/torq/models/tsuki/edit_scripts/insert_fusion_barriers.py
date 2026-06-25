#!/usr/bin/env python3
"""Insert Pad+Slice fusion barriers before attention blocks.

When TileAndFuse fuses LayerNorm variance ops with attention MatMul+Softmax,
the fused kernel's iteration space can exceed hardware tiling bounds (Bug C).
This script inserts a Pad(1)+Slice barrier on the first Gemm projection output
feeding each attention block, which forces the compiler to materialize the
intermediate tensor and prevents the problematic fusion.

The barrier is semantically a no-op: Pad adds one element on dim 0, Slice
removes it. The intermediate has a different shape so the compiler can't fold it.

Usage:
    python3 scripts/insert_fusion_barriers.py -i model.onnx -o model_barriers.onnx
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    return None


def _get_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    return TensorProto.BFLOAT16


def _node_by_output(nodes, name):
    for n in nodes:
        if name in n.output:
            return n
    return None


def _consumers(nodes, name):
    return [n for n in nodes if name in n.input]


def find_attention_gemm_outputs(model):
    """Find Gemm projection outputs that feed into attention blocks.

    Traces: Softmax <- [Add] <- MatMul <- ... <- Mul <- Split <- Mul <- Unsqueeze <- Gemm
    Returns the Unsqueeze output names (linear_NN) that should get barriers.
    """
    nodes = list(model.graph.node)
    targets = []

    for softmax in nodes:
        if softmax.op_type != "Softmax":
            continue

        softmax_input = softmax.input[0]
        # Find the Q*K^T MatMul feeding into Softmax (possibly via Add)
        add_or_matmul = _node_by_output(nodes, softmax_input)
        if not add_or_matmul:
            continue

        if add_or_matmul.op_type == "Add":
            qk_matmul = None
            for inp in add_or_matmul.input:
                producer = _node_by_output(nodes, inp)
                if producer and producer.op_type == "MatMul":
                    qk_matmul = producer
                    break
            if not qk_matmul:
                continue
        elif add_or_matmul.op_type == "MatMul":
            qk_matmul = add_or_matmul
        else:
            continue

        # Check this is actually an attention pattern (4D, square-ish attention matrix)
        qk_shape = _get_shape(model, qk_matmul.output[0])
        if not qk_shape or len(qk_shape) != 4:
            continue

        # Trace back from Q*K^T MatMul inputs to find the Gemm projection chain
        # Pattern: MatMul <- Mul/Reshape/Transpose <- ... <- Unsqueeze <- Gemm
        found_gemms = set()
        def trace_back(tensor_name, depth=0):
            if depth > 10:
                return
            producer = _node_by_output(nodes, tensor_name)
            if not producer:
                return
            if producer.op_type == "Unsqueeze":
                # Check if this Unsqueeze comes from a Gemm
                gemm_candidate = _node_by_output(nodes, producer.input[0])
                if gemm_candidate and gemm_candidate.op_type == "Gemm":
                    found_gemms.add(producer.output[0])
                    return
            for inp in producer.input:
                if inp:
                    trace_back(inp, depth + 1)

        for inp in qk_matmul.input:
            trace_back(inp)

        if found_gemms:
            # Pick the first one found as the barrier point
            barrier_target = sorted(found_gemms)[0]
            shape = _get_shape(model, barrier_target)
            if shape:
                targets.append({
                    "tensor_name": barrier_target,
                    "shape": shape,
                    "softmax_name": softmax.name,
                    "qk_matmul_name": qk_matmul.name,
                })

    return targets


def insert_pad_slice_barrier(model, target):
    """Insert Pad(1,dim=0)+Slice(1:2,dim=0) barrier on a tensor."""
    graph = model.graph
    nodes = list(graph.node)

    tensor_name = target["tensor_name"]
    shape = target["shape"]
    dtype = _get_dtype(model, tensor_name)

    padded_name = f"{tensor_name}__padded"
    barrier_name = f"{tensor_name}__barrier"

    # Pad: add 1 element on dim 0
    ndim = len(shape)
    pad_values = [1] + [0] * (ndim - 1) + [0] * ndim  # [before_dim0, before_rest..., after_all...]
    pads_init = numpy_helper.from_array(
        np.array(pad_values, dtype=np.int64), name=f"{tensor_name}__barrier_pads")
    graph.initializer.append(pads_init)

    pad_node = helper.make_node(
        "Pad", [tensor_name, f"{tensor_name}__barrier_pads"], [padded_name],
        name=f"{tensor_name}__barrier_pad", mode="constant")

    padded_shape = [shape[0] + 1] + shape[1:]
    graph.value_info.append(helper.make_tensor_value_info(padded_name, dtype, padded_shape))

    # Slice: take [1:shape[0]+1] on dim 0 to get back original shape
    starts_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=f"{tensor_name}__barrier_starts")
    ends_init = numpy_helper.from_array(
        np.array([shape[0] + 1], dtype=np.int64), name=f"{tensor_name}__barrier_ends")
    axes_init = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name=f"{tensor_name}__barrier_axes")
    graph.initializer.append(starts_init)
    graph.initializer.append(ends_init)
    graph.initializer.append(axes_init)

    slice_node = helper.make_node(
        "Slice",
        [padded_name, f"{tensor_name}__barrier_starts",
         f"{tensor_name}__barrier_ends", f"{tensor_name}__barrier_axes"],
        [barrier_name],
        name=f"{tensor_name}__barrier_slice")
    graph.value_info.append(helper.make_tensor_value_info(barrier_name, dtype, shape))

    # Rewire consumers
    for n in nodes:
        for i, inp in enumerate(n.input):
            if inp == tensor_name and tensor_name not in n.output:
                n.input[i] = barrier_name

    # Insert after producer
    for i, n in enumerate(nodes):
        if tensor_name in n.output:
            nodes.insert(i + 1, pad_node)
            nodes.insert(i + 2, slice_node)
            break

    del graph.node[:]
    graph.node.extend(nodes)


def main():
    parser = argparse.ArgumentParser(description="Insert fusion barriers before attention blocks")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument("--dry-run", action="store_true", help="Just print targets")
    args = parser.parse_args()

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    targets = find_attention_gemm_outputs(model)
    print(f"Found {len(targets)} attention block(s) needing barriers:")
    for t in targets:
        print(f"  {t['tensor_name']}{t['shape']} (attn: {t['qk_matmul_name']})")

    if args.dry_run:
        return

    for t in reversed(targets):
        print(f"  Inserting Pad+Slice barrier on {t['tensor_name']}")
        insert_pad_slice_barrier(model, t)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed: {e}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
