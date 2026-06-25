#!/usr/bin/env python3
"""Split GatherND ops with large data tensors into per-channel slices.

When GatherND indexes into a data tensor that exceeds LRAM (e.g. [96000,1,9] bf16
= 1.7MB), the compiler can't allocate it. This script splits the data along its
last dimension into individual channels, does a separate GatherND per channel, and
Concats the results. Each channel slice (e.g. [96000,1,1] = 192KB) fits in LRAM.

Only rewrites GatherND ops where:
  - batch_dims == 0
  - indices last dim == 1 (single-axis indexing)
  - data tensor size > --max-bytes (default 498KB = LRAM budget)
  - the data tensor is the oversized one (not the output)
"""
import argparse
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ELEM_SIZES = {
    TensorProto.FLOAT: 4, TensorProto.DOUBLE: 8,
    TensorProto.FLOAT16: 2, TensorProto.BFLOAT16: 2,
    TensorProto.INT8: 1, TensorProto.UINT8: 1,
    TensorProto.INT16: 2, TensorProto.UINT16: 2,
    TensorProto.INT32: 4, TensorProto.UINT32: 4,
    TensorProto.INT64: 8, TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
}


def collect_info(model):
    shapes, dtypes = {}, {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            shapes[vi.name] = [
                d.dim_value if d.dim_value > 0 else None
                for d in tt.shape.dim
            ]
            dtypes[vi.name] = tt.elem_type
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
        dtypes[init.name] = init.data_type
    return shapes, dtypes


def tensor_bytes(shape, dtype):
    if shape is None or any(d is None for d in shape):
        return 0
    elem = ELEM_SIZES.get(dtype, 0)
    return int(np.prod(shape)) * elem


def rewrite(model, max_bytes=498 * 1024, verbose=False):
    model = copy.deepcopy(model)
    shapes, dtypes = collect_info(model)

    existing = set()
    for n in model.graph.node:
        existing.update(n.input)
        existing.update(n.output)
        if n.name:
            existing.add(n.name)
    for x in model.graph.initializer:
        existing.add(x.name)

    def uname(base):
        base = base.replace("/", "_").replace(":", "_")
        name = base
        i = 0
        while name in existing:
            i += 1
            name = f"{base}_{i}"
        existing.add(name)
        return name

    new_nodes = []
    rewrites = 0

    for node in model.graph.node:
        if node.op_type != "GatherND":
            new_nodes.append(node)
            continue

        batch_dims = 0
        for a in node.attribute:
            if a.name == "batch_dims":
                batch_dims = int(a.i)

        data_name = node.input[0]
        idx_name = node.input[1]
        out_name = node.output[0]

        data_shape = shapes.get(data_name)
        idx_shape = shapes.get(idx_name)
        out_shape = shapes.get(out_name)
        data_dtype = dtypes.get(data_name)

        data_size = tensor_bytes(data_shape, data_dtype)
        out_dtype = dtypes.get(out_name, data_dtype)
        out_size = tensor_bytes(out_shape, out_dtype) if out_shape else 0

        if (batch_dims != 0 or data_shape is None or idx_shape is None
                or (data_size <= max_bytes and out_size <= max_bytes)
                or len(data_shape) < 2 or idx_shape[-1] != 1):
            new_nodes.append(node)
            continue

        split_dim = len(data_shape) - 1
        num_channels = data_shape[split_dim]
        if num_channels is None or num_channels < 2:
            new_nodes.append(node)
            continue

        channel_shape = list(data_shape)
        channel_shape[split_dim] = 1
        channel_size = tensor_bytes(channel_shape, data_dtype)
        if channel_size > max_bytes:
            if verbose:
                print(f"  SKIP {node.name}: even single channel {channel_shape} = {channel_size} > {max_bytes}")
            new_nodes.append(node)
            continue

        if verbose:
            trigger = "data" if data_size > max_bytes else "output"
            print(f"  SPLIT {node.name}: {trigger} overflow, data {data_shape} ({data_size}B), "
                  f"out {out_shape} ({out_size}B) -> "
                  f"{num_channels}x {channel_shape} ({channel_size}B each)")

        prefix = node.name or out_name or f"gathernd_split_{rewrites}"
        gather_outputs = []

        for ch in range(num_channels):
            starts_val = [0] * len(data_shape)
            ends_val = list(data_shape)
            starts_val[split_dim] = ch
            ends_val[split_dim] = ch + 1
            axes_val = list(range(len(data_shape)))

            starts_name = uname(f"{prefix}_starts_{ch}")
            ends_name = uname(f"{prefix}_ends_{ch}")
            axes_name = uname(f"{prefix}_axes_{ch}")
            slice_out = uname(f"{prefix}_slice_{ch}")
            gather_out = uname(f"{prefix}_gather_{ch}")

            model.graph.initializer.append(
                numpy_helper.from_array(np.array(starts_val, dtype=np.int64), starts_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(ends_val, dtype=np.int64), ends_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(axes_val, dtype=np.int64), axes_name))

            new_nodes.append(helper.make_node(
                "Slice", [data_name, starts_name, ends_name, axes_name],
                [slice_out], name=uname(f"{prefix}_do_slice_{ch}")))

            new_nodes.append(helper.make_node(
                "GatherND", [slice_out, idx_name], [gather_out],
                name=uname(f"{prefix}_do_gather_{ch}"), batch_dims=batch_dims))

            gather_outputs.append(gather_out)

        new_nodes.append(helper.make_node(
            "Concat", gather_outputs, [out_name],
            name=uname(f"{prefix}_concat"), axis=split_dim))

        rewrites += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx.checker.check_model(model)
    return model, rewrites


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--max-bytes", type=int, default=498 * 1024,
                   help="LRAM budget in bytes (default: 498KB)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    model = onnx.load(args.input)
    new_model, rewrites = rewrite(model, max_bytes=args.max_bytes, verbose=args.verbose)
    onnx.save(new_model, args.output)
    print(f"Rewrote {rewrites} GatherND ops")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
