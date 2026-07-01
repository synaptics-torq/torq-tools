#!/usr/bin/env python3
"""Wrap rank-2 ReduceMean ops with Unsqueeze/Squeeze to make them rank-3.

Works around a compiler bug where ReduceMean on rank-2 bf16 tensors produces
wrong results on NSS. Adds Unsqueeze(axis=0) before and Squeeze(axis=0) after.
"""
import argparse
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def fix_rank2_reducemean(model):
    graph = model.graph
    vi_map = {}
    for vi in graph.value_info:
        if vi.type.tensor_type.shape.dim:
            vi_map[vi.name] = (
                [d.dim_value for d in vi.type.tensor_type.shape.dim],
                vi.type.tensor_type.elem_type,
            )
    for inp in graph.input:
        if inp.type.tensor_type.shape.dim:
            vi_map[inp.name] = (
                [d.dim_value for d in inp.type.tensor_type.shape.dim],
                inp.type.tensor_type.elem_type,
            )

    new_nodes = []
    new_inits = []
    new_vis = []
    fixed = 0

    for node in graph.node:
        if node.op_type != "ReduceMean":
            new_nodes.append(node)
            continue

        info = vi_map.get(node.input[0])
        if info is None or len(info[0]) != 2:
            new_nodes.append(node)
            continue

        shape, elem_type = info
        keepdims = 0
        for a in node.attribute:
            if a.name == "keepdims":
                keepdims = a.i

        print(f"  Fixing {node.name}: {shape} (rank-2 -> rank-3)")

        suffix = f"__r2fix_{fixed}"

        # Unsqueeze axes initializer
        axes_name = f"unsqueeze_axes{suffix}"
        new_inits.append(
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_name)
        )

        # Unsqueeze input: [M,N] -> [1,M,N]
        unsq_out = f"{node.input[0]}__unsqueezed{suffix}"
        unsq_node = helper.make_node(
            "Unsqueeze", [node.input[0], axes_name], [unsq_out],
            name=f"{node.name}__unsqueeze{suffix}",
        )
        
        print("New unsq node",unsq_node)
        new_nodes.append(unsq_node)
        new_vis.append(
            helper.make_tensor_value_info(unsq_out, elem_type, [1] + shape)
        )

        # Shift reduction axes by +1 (dim 0 was inserted by Unsqueeze)
        rm_axes = []
        if len(node.input) > 1:
            for init in graph.initializer:
                if init.name == node.input[1]:
                    rm_axes = list(numpy_helper.to_array(init))
                    break

        shifted_axes = []
        for ax in rm_axes:
            ax = int(ax)
            if ax >= 0:
                shifted_axes.append(ax + 1)
            else:
                shifted_axes.append(ax)  # negative axes don't need shifting
        shifted_axes_name = f"rm_axes_shifted{suffix}"
        new_inits.append(
            numpy_helper.from_array(
                np.array(shifted_axes, dtype=np.int64), name=shifted_axes_name
            )
        )
        print(f"    axes {rm_axes} -> {shifted_axes}")

        # ReduceMean on rank-3: [1,M,N] -> depends on axis and keepdims
        rm_out = f"{node.output[0]}__rm3d{suffix}"
        rm_node = helper.make_node(
            "ReduceMean", [unsq_out, shifted_axes_name], [rm_out],
            name=f"{node.name}__rm3d{suffix}",
        )
        for a in node.attribute:
            rm_node.attribute.append(a)
        new_nodes.append(rm_node)

        # Figure out output shape of the rank-3 ReduceMean
        rm3d_shape = list([1] + shape)
        for ax in shifted_axes:
            if ax < 0:
                ax += 3
            if keepdims:
                rm3d_shape[ax] = 1
            else:
                rm3d_shape[ax] = 0
        rm3d_shape = [s for s in rm3d_shape if s != 0]
        new_vis.append(helper.make_tensor_value_info(rm_out, elem_type, rm3d_shape))

        # Squeeze dim=0: remove the batch dim we added
        sq_axes_name = f"squeeze_axes{suffix}"
        new_inits.append(
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=sq_axes_name)
        )
        sq_node = helper.make_node(
            "Squeeze", [rm_out, sq_axes_name], [node.output[0]],
            name=f"{node.name}__squeeze{suffix}",
        )
        new_nodes.append(sq_node)

        fixed += 1

    if fixed == 0:
        print("  No rank-2 ReduceMean ops found")
        return model

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)

    print(f"  Fixed {fixed} rank-2 ReduceMean ops")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = fix_rank2_reducemean(model)
    onnx.save(model, args.output)
    onnx.checker.check_model(args.output)
    print(f"  Saved to {args.output} (ONNX checker passed)")


if __name__ == "__main__":
    main()
