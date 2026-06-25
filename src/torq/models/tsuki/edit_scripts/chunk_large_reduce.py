#!/usr/bin/env python3
"""Chunk ReduceSum/ReduceMean ops whose reduction dimension exceeds the hardware
NDL descriptor limit (65536). Splits the input along the reduction axis into
chunks <= max_dim, reduces each chunk, then combines (Add for Sum, mean-of-means
weighted by chunk size for Mean).
"""
import argparse
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def collect_shapes(model):
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            shapes[vi.name] = [
                d.dim_value if d.dim_value > 0 else None
                for d in tt.shape.dim
            ]
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def rewrite(model, max_dim=65536):
    model = copy.deepcopy(model)
    shapes = collect_shapes(model)
    inits = {init.name: init for init in model.graph.initializer}

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
        if node.op_type not in ("ReduceSum", "ReduceMean"):
            new_nodes.append(node)
            continue

        in_shape = shapes.get(node.input[0])
        if in_shape is None:
            new_nodes.append(node)
            continue

        keepdims = 1
        for a in node.attribute:
            if a.name == "keepdims":
                keepdims = int(a.i)

        if len(node.input) < 2 or not node.input[1]:
            new_nodes.append(node)
            continue
        axes_init = inits.get(node.input[1])
        if axes_init is None:
            new_nodes.append(node)
            continue
        axes = list(numpy_helper.to_array(axes_init).flatten())

        needs_chunk = False
        for ax in axes:
            real_ax = ax if ax >= 0 else ax + len(in_shape)
            if 0 <= real_ax < len(in_shape) and in_shape[real_ax] and in_shape[real_ax] > max_dim:
                needs_chunk = True
                chunk_axis = real_ax
                break

        if not needs_chunk:
            new_nodes.append(node)
            continue

        total = in_shape[chunk_axis]
        n_chunks = (total + max_dim - 1) // max_dim
        prefix = node.name or node.output[0]
        out_name = node.output[0]

        print(f"  Chunking {node.op_type} {node.name}: dim[{chunk_axis}]={total} -> {n_chunks} chunks of <={max_dim}")

        partial_outputs = []
        chunk_sizes = []
        for c in range(n_chunks):
            start = c * max_dim
            end = min((c + 1) * max_dim, total)
            chunk_sizes.append(end - start)

            s_name = uname(f"{prefix}_chunk_start_{c}")
            e_name = uname(f"{prefix}_chunk_end_{c}")
            ax_name = uname(f"{prefix}_chunk_ax_{c}")
            slice_out = uname(f"{prefix}_chunk_slice_{c}")
            red_axes = uname(f"{prefix}_chunk_red_axes_{c}")
            red_out = uname(f"{prefix}_chunk_red_{c}")

            model.graph.initializer.append(
                numpy_helper.from_array(np.array([start], dtype=np.int64), s_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array([end], dtype=np.int64), e_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array([chunk_axis], dtype=np.int64), ax_name))
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(axes, dtype=np.int64), red_axes))

            new_nodes.append(helper.make_node(
                "Slice", [node.input[0], s_name, e_name, ax_name],
                [slice_out], name=uname(f"{prefix}_do_slice_{c}")))
            new_nodes.append(helper.make_node(
                node.op_type, [slice_out, red_axes], [red_out],
                name=uname(f"{prefix}_do_reduce_{c}"), keepdims=keepdims))

            partial_outputs.append(red_out)

        if node.op_type == "ReduceSum":
            result = partial_outputs[0]
            for i in range(1, len(partial_outputs)):
                add_out = out_name if i == len(partial_outputs) - 1 else uname(f"{prefix}_add_{i}")
                new_nodes.append(helper.make_node(
                    "Add", [result, partial_outputs[i]], [add_out],
                    name=uname(f"{prefix}_do_add_{i}")))
                result = add_out
        else:
            result = partial_outputs[0]
            for i in range(1, len(partial_outputs)):
                add_out = uname(f"{prefix}_sum_{i}")
                new_nodes.append(helper.make_node(
                    "Add", [result, partial_outputs[i]], [add_out],
                    name=uname(f"{prefix}_do_add_{i}")))
                result = add_out
            n_name = uname(f"{prefix}_n_chunks")
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.array(n_chunks, dtype=np.float32).reshape([]),
                    n_name))
            cast_n = uname(f"{prefix}_cast_n")
            new_nodes.append(helper.make_node(
                "Cast", [n_name], [cast_n], to=TensorProto.BFLOAT16))
            new_nodes.append(helper.make_node(
                "Div", [result, cast_n], [out_name],
                name=uname(f"{prefix}_do_div")))

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
    p.add_argument("--max-dim", type=int, default=65536,
                   help="Max reduction dimension (default: 65536 = hardware NDL limit)")
    args = p.parse_args()

    model = onnx.load(args.input)
    new_model, rewrites = rewrite(model, max_dim=args.max_dim)
    onnx.save(new_model, args.output)
    print(f"Rewrote {rewrites} ops")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
