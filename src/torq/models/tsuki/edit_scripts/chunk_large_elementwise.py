#!/usr/bin/env python3
"""Chunk elementwise ops on large tensors so they fit in LRAM.

Two strategies:
1. Insert Pad+Slice barriers after Conv/Gemm/MatMul outputs that feed
   elementwise ops — breaks carry-over from fuse-group ops so medium-sized
   elementwise ops can be evaluated in isolation and fit in LRAM → NSS.
2. For elementwise ops whose own I/O > LRAM even in isolation, chunk them:
   Split inputs → N parallel ops → Concat output.
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from collections import defaultdict


LRAM_SIZE = 512 * 1024  # 512KB
BARRIER_THRESHOLD = 128 * 1024  # barrier after Conv/Gemm outputs > this
ELEM_OPS = frozenset({
    'Add', 'Mul', 'Sub', 'Div', 'Exp', 'Sqrt', 'Sigmoid', 'Tanh', 'Relu',
    'Neg', 'Abs', 'Clip', 'Sign', 'Max', 'Min', 'Pow', 'Reciprocal',
    'Floor', 'Ceil', 'Where', 'Sin', 'Cos', 'Erf',
})
REDUCE_OPS = frozenset({'ReduceMean', 'ReduceSum', 'Softmax'})
CHUNKABLE_OPS = ELEM_OPS | REDUCE_OPS
FUSE_OPS = frozenset({'Conv', 'Gemm', 'MatMul', 'ConvTranspose'})


def _get_shape(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            return [d.dim_value for d in item.type.tensor_type.shape.dim]
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _get_dtype(model, name):
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name == name and item.type.HasField("tensor_type"):
            return item.type.tensor_type.elem_type
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return TensorProto.BFLOAT16


def _tensor_bytes(shape):
    if not shape:
        return 2
    r = 2  # bf16
    for d in shape:
        r *= d
    return r


def _broadcast_shape(s1, s2):
    if not s1:
        return list(s2) if s2 else []
    if not s2:
        return list(s1)
    r1, r2 = max(len(s1), len(s2)), max(len(s1), len(s2))
    p1 = [1] * (r1 - len(s1)) + list(s1)
    p2 = [1] * (r1 - len(s2)) + list(s2)
    return [max(a, b) for a, b in zip(p1, p2)]


def insert_post_fuse_barriers(model, barrier_threshold=None):
    """Add Pad+Slice barriers after Conv/Gemm outputs."""
    if barrier_threshold is None:
        barrier_threshold = BARRIER_THRESHOLD
    graph = model.graph
    nodes = list(graph.node)

    consumers = defaultdict(list)
    for node in nodes:
        for inp in node.input:
            consumers[inp].append(node)

    barrier_targets = []
    for node in nodes:
        if node.op_type not in FUSE_OPS:
            continue
        for out in node.output:
            shape = _get_shape(model, out)
            if not shape:
                continue
            if _tensor_bytes(shape) <= barrier_threshold:
                continue
            dtype = _get_dtype(model, out)
            barrier_targets.append((node, out, shape, dtype))

    print(f"  Inserting {len(barrier_targets)} post-Conv/Gemm barriers")

    inserted = 0
    for producer, tname, shape, dtype in barrier_targets:
        ndim = len(shape)
        if ndim == 0:
            continue

        padded_name = f"{tname}__ew_padded"
        barrier_name = f"{tname}__ew_barrier"

        pad_values = [1] + [0] * (ndim - 1) + [0] * ndim
        pads_const = f"{tname}__ew_pads"
        graph.initializer.append(numpy_helper.from_array(
            np.array(pad_values, dtype=np.int64), name=pads_const))

        pad_node = helper.make_node(
            "Pad", [tname, pads_const], [padded_name],
            name=f"{tname}__ew_pad", mode="constant")

        padded_shape = [shape[0] + 1] + shape[1:]
        graph.value_info.append(
            helper.make_tensor_value_info(padded_name, dtype, padded_shape))

        starts_name = f"{tname}__ew_starts"
        ends_name = f"{tname}__ew_ends"
        axes_name = f"{tname}__ew_axes"
        graph.initializer.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=starts_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([shape[0] + 1], dtype=np.int64), name=ends_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=axes_name))

        slice_node = helper.make_node(
            "Slice", [padded_name, starts_name, ends_name, axes_name],
            [barrier_name], name=f"{tname}__ew_slice")

        graph.value_info.append(
            helper.make_tensor_value_info(barrier_name, dtype, shape))

        for n in nodes:
            if n is producer:
                continue
            new_inputs = []
            for inp in n.input:
                new_inputs.append(barrier_name if inp == tname else inp)
            del n.input[:]
            n.input.extend(new_inputs)

        pidx = nodes.index(producer)
        nodes.insert(pidx + 1, pad_node)
        nodes.insert(pidx + 2, slice_node)
        inserted += 1

    del graph.node[:]
    graph.node.extend(nodes)
    return inserted


def _get_reduce_axes(node, ndim):
    """Get reduction axes for ReduceMean/ReduceSum/Softmax."""
    if node.op_type == 'Softmax':
        axis = -1
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = attr.i
        if axis < 0:
            axis += ndim
        return {axis}
    axes = None
    for attr in node.attribute:
        if attr.name == 'axes':
            axes = set(a if a >= 0 else a + ndim for a in attr.ints)
    if axes is None and len(node.input) > 1:
        return None
    return axes or set()


def chunk_large_elementwise(model, max_io=None):
    """Chunk elementwise/reduction ops whose own I/O exceeds LRAM."""
    if max_io is None:
        max_io = LRAM_SIZE
    graph = model.graph
    nodes = list(graph.node)

    targets = []
    for i, node in enumerate(nodes):
        if node.op_type not in CHUNKABLE_OPS:
            continue

        inp_shapes = [_get_shape(model, inp) for inp in node.input]
        out_shape = _get_shape(model, node.output[0]) if node.output else None
        if not out_shape:
            continue

        total_io = _tensor_bytes(out_shape)
        for s in inp_shapes:
            if s:
                total_io += _tensor_bytes(s)

        if total_io > max_io:
            targets.append((i, node, inp_shapes, out_shape))

    print(f"  Chunking {len(targets)} ops with I/O > {max_io // 1024}KB")

    offset = 0
    for orig_idx, node, inp_shapes, out_shape in targets:
        idx = orig_idx + offset

        is_reduce = node.op_type in REDUCE_OPS
        inp0_shape = inp_shapes[0] if inp_shapes else out_shape
        work_shape = inp0_shape if is_reduce else out_shape

        out_bytes = _tensor_bytes(out_shape)
        num_chunks = 2
        while True:
            chunk_io = (out_bytes // num_chunks) * 3
            if chunk_io <= max_io:
                break
            num_chunks += 1
            if num_chunks > 16:
                break

        reduce_axes = _get_reduce_axes(node, len(work_shape)) if is_reduce else set()
        if reduce_axes is None:
            continue

        split_axis = -1
        best_dim = 0
        for a, d in enumerate(work_shape):
            if a in reduce_axes:
                continue
            if d > best_dim:
                best_dim = d
                split_axis = a

        if split_axis < 0 or best_dim < num_chunks:
            continue

        actual_chunks = num_chunks
        chunk_size = best_dim // actual_chunks
        remainder = best_dim % actual_chunks
        if remainder > 0:
            while remainder > 0 and actual_chunks > 2:
                actual_chunks -= 1
                chunk_size = best_dim // actual_chunks
                remainder = best_dim % actual_chunks
            if remainder > 0:
                actual_chunks = num_chunks
                chunk_size = best_dim // actual_chunks

        if best_dim % actual_chunks != 0:
            continue

        chunk_size = best_dim // actual_chunks
        out_dtype = _get_dtype(model, node.output[0])
        name = node.name or f"{node.op_type}_{orig_idx}"
        final_output = node.output[0]

        new_nodes = []
        chunk_outputs = []

        for ci in range(actual_chunks):
            start = ci * chunk_size
            end = start + chunk_size

            starts_name = f"{name}__ew_chunk{ci}_starts"
            ends_name = f"{name}__ew_chunk{ci}_ends"
            axes_name = f"{name}__ew_chunk{ci}_axes"
            graph.initializer.append(numpy_helper.from_array(
                np.array([start], dtype=np.int64), name=starts_name))
            graph.initializer.append(numpy_helper.from_array(
                np.array([end], dtype=np.int64), name=ends_name))
            graph.initializer.append(numpy_helper.from_array(
                np.array([split_axis], dtype=np.int64), name=axes_name))

            chunk_inputs = []
            for j, (inp_name, inp_shape) in enumerate(zip(node.input, inp_shapes)):
                if is_reduce and j > 0:
                    chunk_inputs.append(inp_name)
                    continue
                ref_shape = work_shape if is_reduce else out_shape
                inp_split_axis = split_axis - (len(ref_shape) - len(inp_shape)) if inp_shape else -1
                if inp_shape and inp_split_axis >= 0 and inp_shape[inp_split_axis] > 1:
                    chunk_inp = f"{name}__ew_chunk{ci}_in{j}"
                    if inp_split_axis != split_axis:
                        inp_axes_name = f"{name}__ew_chunk{ci}_axes_in{j}"
                        graph.initializer.append(numpy_helper.from_array(
                            np.array([inp_split_axis], dtype=np.int64), name=inp_axes_name))
                    else:
                        inp_axes_name = axes_name
                    slice_node = helper.make_node(
                        "Slice", [inp_name, starts_name, ends_name, inp_axes_name],
                        [chunk_inp], name=f"{name}__ew_chunk{ci}_slice_in{j}")
                    new_nodes.append(slice_node)
                    inp_chunk_shape = list(inp_shape)
                    inp_chunk_shape[inp_split_axis] = chunk_size
                    inp_dtype = _get_dtype(model, inp_name)
                    graph.value_info.append(
                        helper.make_tensor_value_info(chunk_inp, inp_dtype, inp_chunk_shape))
                    chunk_inputs.append(chunk_inp)
                else:
                    chunk_inputs.append(inp_name)

            chunk_out = f"{name}__ew_chunk{ci}_out"
            op_node = helper.make_node(
                node.op_type, chunk_inputs, [chunk_out],
                name=f"{name}__ew_chunk{ci}")
            for attr in node.attribute:
                op_node.attribute.append(attr)
            new_nodes.append(op_node)

            chunk_out_shape = list(out_shape)
            out_split_axis = split_axis
            if is_reduce and len(out_shape) < len(work_shape):
                removed = sum(1 for a in sorted(reduce_axes) if a < split_axis)
                out_split_axis = split_axis - removed
            if out_split_axis < len(chunk_out_shape):
                chunk_out_shape[out_split_axis] = chunk_size
            graph.value_info.append(
                helper.make_tensor_value_info(chunk_out, out_dtype, chunk_out_shape))
            chunk_outputs.append(chunk_out)

        concat_axis = out_split_axis if is_reduce else split_axis
        barrier_out = f"{name}__ew_concat_raw"
        concat_node = helper.make_node(
            "Concat", chunk_outputs, [barrier_out],
            name=f"{name}__ew_concat", axis=concat_axis)
        new_nodes.append(concat_node)
        graph.value_info.append(
            helper.make_tensor_value_info(barrier_out, out_dtype, list(out_shape)))

        ndim = len(out_shape)
        pads_val = [0] * ndim + [1] + [0] * (ndim - 1)
        pads_name = f"{name}__ew_barrier_pads"
        graph.initializer.append(numpy_helper.from_array(
            np.array(pads_val, dtype=np.int64), pads_name))
        padded_name = f"{name}__ew_barrier_padded"
        padded_shape = list(out_shape)
        padded_shape[0] += 1
        pad_node = helper.make_node(
            "Pad", [barrier_out, pads_name], [padded_name],
            name=f"{name}__ew_barrier_pad")
        new_nodes.append(pad_node)
        graph.value_info.append(
            helper.make_tensor_value_info(padded_name, out_dtype, padded_shape))

        starts_name = f"{name}__ew_barrier_starts"
        ends_name = f"{name}__ew_barrier_ends"
        axes_name = f"{name}__ew_barrier_axes"
        graph.initializer.append(numpy_helper.from_array(
            np.array([1], dtype=np.int64), starts_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([1 + out_shape[0]], dtype=np.int64), ends_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), axes_name))
        slice_node = helper.make_node(
            "Slice", [padded_name, starts_name, ends_name, axes_name],
            [final_output], name=f"{name}__ew_barrier_slice")
        new_nodes.append(slice_node)

        nodes[idx:idx + 1] = new_nodes
        offset += len(new_nodes) - 1

    del graph.node[:]
    graph.node.extend(nodes)
    return len(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--barriers-only", action="store_true",
                        help="Only add post-Conv/Gemm barriers, skip chunking")
    parser.add_argument("--chunk-only", action="store_true",
                        help="Only chunk large ops, skip barriers")
    parser.add_argument("--max-io", type=int, default=LRAM_SIZE,
                        help="Max I/O bytes before chunking (default: 512KB)")
    parser.add_argument("--barrier-threshold", type=int, default=BARRIER_THRESHOLD,
                        help="Min Conv/Gemm output bytes for barrier (default: 128KB)")
    args = parser.parse_args()

    model = onnx.load(args.input)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    if not args.chunk_only:
        insert_post_fuse_barriers(model, args.barrier_threshold)

    if not args.barriers_only:
        chunk_large_elementwise(model, args.max_io)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  Warning: shape inference failed: {e}")

    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
