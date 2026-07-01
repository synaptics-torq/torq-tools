#!/usr/bin/env python3
"""Replace ReduceSum with ReduceMean * N.

Works around NSS accuracy bugs on ReduceSum by decomposing into
ReduceMean (which works correctly after rank-2 fix) times the
reduction dimension size.
"""
import argparse
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def replace_reducesum_with_mean(model):
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
    replaced = 0

    for node in graph.node:
        if node.op_type != "ReduceSum":
            new_nodes.append(node)
            continue

        info = vi_map.get(node.input[0])
        if info is None:
            new_nodes.append(node)
            continue

        shape, elem_type = info

        # Find reduction axes
        axes = []
        if len(node.input) > 1:
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = list(numpy_helper.to_array(init))
                    break

        if not axes:
            new_nodes.append(node)
            continue

        # Compute N = product of reduction dimensions
        n = 1
        for ax in axes:
            ax = int(ax)
            if ax < 0:
                ax += len(shape)
            n *= shape[ax]

        if n <= 0:
            new_nodes.append(node)
            continue

        keepdims = 0
        for a in node.attribute:
            if a.name == "keepdims":
                keepdims = a.i

        print(f"  Replacing {node.name}: ReduceSum({shape}, axes={[int(a) for a in axes]}) -> ReduceMean * {n}")

        suffix = f"__rs2rm_{replaced}"

        # ReduceMean with same axes and attributes
        mean_out = f"{node.output[0]}__mean{suffix}"
        mean_node = helper.make_node(
            "ReduceMean", list(node.input), [mean_out],
            name=f"{node.name}__mean{suffix}",
        )
        for a in node.attribute:
            mean_node.attribute.append(a)
        new_nodes.append(mean_node)

        # Compute output shape for value_info
        mean_shape = list(shape)
        for ax in axes:
            ax = int(ax)
            if ax < 0:
                ax += len(shape)
            if keepdims:
                mean_shape[ax] = 1
            else:
                mean_shape[ax] = 0
        if not keepdims:
            mean_shape = [s for s in mean_shape if s != 0]
        new_vis.append(helper.make_tensor_value_info(mean_out, elem_type, mean_shape))

        # N constant in the same dtype as the input
        n_name = f"reducesum_n{suffix}"
        if elem_type == TensorProto.BFLOAT16:
            fp32_val = np.array([n], dtype=np.float32)
            bf16_raw = fp32_val.view(np.uint16)[1::2].copy()
            n_tensor = onnx.TensorProto()
            n_tensor.name = n_name
            n_tensor.data_type = TensorProto.BFLOAT16
            n_tensor.dims[:] = [1]
            n_tensor.raw_data = bf16_raw.tobytes()
        else:
            dtype_map = {
                TensorProto.FLOAT: np.float32,
                TensorProto.DOUBLE: np.float64,
                TensorProto.FLOAT16: np.float16,
            }
            np_dtype = dtype_map.get(elem_type, np.float32)
            n_tensor = numpy_helper.from_array(
                np.array([n], dtype=np_dtype), name=n_name
            )
        new_inits.append(n_tensor)

        # Mul: mean * N = sum
        mul_node = helper.make_node(
            "Mul", [mean_out, n_name], [node.output[0]],
            name=f"{node.name}__mul_n{suffix}",
        )
        new_nodes.append(mul_node)

        replaced += 1

    if replaced == 0:
        print("  No ReduceSum ops replaced")
        return model

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)

    print(f"  Replaced {replaced} ReduceSum ops with ReduceMean * N")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = replace_reducesum_with_mean(model)
    onnx.save(model, args.output)
    onnx.checker.check_model(args.output)
    print(f"  Saved to {args.output} (ONNX checker passed)")


if __name__ == "__main__":
    main()
