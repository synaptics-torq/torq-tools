#!/usr/bin/env python3
"""Decompose unsupported ONNX ops in the tsuki model for executor discovery.

Decomposes:
  - LayerNormalization -> ReduceMean + Sub + Mul + Sqrt + Reciprocal + ...
  - InstanceNormalization -> same pattern with spatial axes + Unsqueeze
  - Resize (nearest) -> Gather with precomputed indices

Usage:
    python3 scripts/decompose_unsupported_ops.py \
        --input=tests/testdata/onnx_models/tsuki_static_bf16.onnx \
        --output=tests/testdata/onnx_models/tsuki_static_bf16_decomposed.onnx

    # Only decompose ops that are unassigned in the discovery JSON:
    python3 scripts/decompose_unsupported_ops.py \
        --input=tests/testdata/onnx_models/tsuki_static_bf16.onnx \
        --output=tests/testdata/onnx_models/tsuki_static_bf16_decomposed.onnx \
        --only-unassigned=executor_assignments_tsuki_static_bf16.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def decompose_layer_normalization(model, target_outputs=None):
    """Decompose LayerNormalization into primitive ops.
    
    If target_outputs is provided, only decompose nodes whose first output name is in the set.
    """
    graph = model.graph
    ln_nodes = [n for n in graph.node if n.op_type == "LayerNormalization"]
    if not ln_nodes:
        return model

    model = copy.deepcopy(model)
    graph = model.graph

    new_nodes = []
    for node in graph.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        if target_outputs is not None and node.output[0] not in target_outputs:
            new_nodes.append(node)
            continue

        axis = -1
        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
            elif attr.name == "epsilon":
                epsilon = attr.f

        x_name = node.input[0]
        scale_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else ""
        y_name = node.output[0]
        prefix = f"_ln_decomp_{y_name}_"

        # Axes tensor (opset 18+ ReduceMean takes axes as input)
        axes_name = prefix + "axes"
        graph.initializer.append(
            numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name)
        )

        # mean = ReduceMean(x, axes)
        mean_name = prefix + "mean"
        new_nodes.append(helper.make_node(
            "ReduceMean", [x_name, axes_name], [mean_name], keepdims=1
        ))

        # diff = x - mean
        diff_name = prefix + "diff"
        new_nodes.append(helper.make_node("Sub", [x_name, mean_name], [diff_name]))

        # diff_sq = diff * diff
        diff_sq_name = prefix + "diff_sq"
        new_nodes.append(helper.make_node("Mul", [diff_name, diff_name], [diff_sq_name]))

        # var = ReduceMean(diff_sq, axes)
        var_name = prefix + "var"
        new_nodes.append(helper.make_node(
            "ReduceMean", [diff_sq_name, axes_name], [var_name], keepdims=1
        ))

        # var_eps = var + epsilon
        eps_name = prefix + "eps"
        graph.initializer.append(
            numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
        )
        var_eps_name = prefix + "var_eps"
        new_nodes.append(helper.make_node("Add", [var_name, eps_name], [var_eps_name]))

        # std = sqrt(var_eps)
        std_name = prefix + "std"
        new_nodes.append(helper.make_node("Sqrt", [var_eps_name], [std_name]))

        # inv_std = 1/std
        inv_std_name = prefix + "inv_std"
        new_nodes.append(helper.make_node("Reciprocal", [std_name], [inv_std_name]))

        # norm = diff * inv_std
        norm_name = prefix + "norm"
        new_nodes.append(helper.make_node("Mul", [diff_name, inv_std_name], [norm_name]))

        # scaled = norm * scale
        scaled_name = prefix + "scaled"
        new_nodes.append(helper.make_node("Mul", [norm_name, scale_name], [scaled_name]))

        # y = scaled + bias (or just identity if no bias)
        if bias_name:
            new_nodes.append(helper.make_node("Add", [scaled_name, bias_name], [y_name]))
        else:
            new_nodes.append(helper.make_node("Identity", [scaled_name], [y_name]))

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def decompose_instance_normalization(model, target_outputs=None):
    """Decompose InstanceNormalization into primitive ops.
    
    If target_outputs is provided, only decompose nodes whose first output name is in the set.
    """
    graph = model.graph
    in_nodes = [n for n in graph.node if n.op_type == "InstanceNormalization"]
    if not in_nodes:
        return model

    model = copy.deepcopy(model)
    graph = model.graph

    def _get_input_rank(input_name):
        for vi in list(graph.input) + list(graph.value_info):
            if vi.name == input_name:
                shape = vi.type.tensor_type.shape
                if shape and shape.dim:
                    return len(shape.dim)
        return None

    new_nodes = []
    for node in graph.node:
        if node.op_type != "InstanceNormalization":
            new_nodes.append(node)
            continue

        if target_outputs is not None and node.output[0] not in target_outputs:
            new_nodes.append(node)
            continue

        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f

        x_name = node.input[0]
        scale_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else ""
        y_name = node.output[0]
        prefix = f"_in_decomp_{y_name}_"

        rank = _get_input_rank(x_name)
        if rank is None:
            rank = 3  # fallback: [N, C, L]

        spatial_axes = list(range(2, rank))

        # Axes tensor
        axes_name = prefix + "axes"
        graph.initializer.append(
            numpy_helper.from_array(np.array(spatial_axes, dtype=np.int64), name=axes_name)
        )

        # mean
        mean_name = prefix + "mean"
        new_nodes.append(helper.make_node(
            "ReduceMean", [x_name, axes_name], [mean_name], keepdims=1
        ))

        # diff
        diff_name = prefix + "diff"
        new_nodes.append(helper.make_node("Sub", [x_name, mean_name], [diff_name]))

        # diff_sq
        diff_sq_name = prefix + "diff_sq"
        new_nodes.append(helper.make_node("Mul", [diff_name, diff_name], [diff_sq_name]))

        # var
        var_name = prefix + "var"
        new_nodes.append(helper.make_node(
            "ReduceMean", [diff_sq_name, axes_name], [var_name], keepdims=1
        ))

        # var + eps
        eps_name = prefix + "eps"
        graph.initializer.append(
            numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
        )
        var_eps_name = prefix + "var_eps"
        new_nodes.append(helper.make_node("Add", [var_name, eps_name], [var_eps_name]))

        # std
        std_name = prefix + "std"
        new_nodes.append(helper.make_node("Sqrt", [var_eps_name], [std_name]))

        # inv_std
        inv_std_name = prefix + "inv_std"
        new_nodes.append(helper.make_node("Reciprocal", [std_name], [inv_std_name]))

        # norm
        norm_name = prefix + "norm"
        new_nodes.append(helper.make_node("Mul", [diff_name, inv_std_name], [norm_name]))

        # Unsqueeze scale from (C,) to (1, C, 1, ..., 1)
        unsqueeze_axes = [0] + spatial_axes
        unsqueeze_axes_name = prefix + "unsqueeze_axes"
        graph.initializer.append(
            numpy_helper.from_array(np.array(unsqueeze_axes, dtype=np.int64), name=unsqueeze_axes_name)
        )

        scale_reshaped_name = prefix + "scale_reshaped"
        new_nodes.append(helper.make_node(
            "Unsqueeze", [scale_name, unsqueeze_axes_name], [scale_reshaped_name]
        ))

        scaled_name = prefix + "scaled"
        new_nodes.append(helper.make_node("Mul", [norm_name, scale_reshaped_name], [scaled_name]))

        if bias_name:
            bias_reshaped_name = prefix + "bias_reshaped"
            new_nodes.append(helper.make_node(
                "Unsqueeze", [bias_name, unsqueeze_axes_name], [bias_reshaped_name]
            ))
            new_nodes.append(helper.make_node("Add", [scaled_name, bias_reshaped_name], [y_name]))
        else:
            new_nodes.append(helper.make_node("Identity", [scaled_name], [y_name]))

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def decompose_resize_nearest(model, target_outputs=None):
    """Decompose nearest-neighbor Resize into Gather with precomputed indices.
    
    If target_outputs is provided, only decompose nodes whose first output name is in the set.
    """
    graph = model.graph
    resize_nodes = [n for n in graph.node if n.op_type == "Resize"]
    if not resize_nodes:
        return model

    model = copy.deepcopy(model)
    graph = model.graph

    def _get_shape(name):
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            if vi.name == name:
                shape = vi.type.tensor_type.shape
                if shape and shape.dim:
                    return [d.dim_value for d in shape.dim]
        return None

    new_nodes = []
    for node in graph.node:
        if node.op_type != "Resize":
            new_nodes.append(node)
            continue

        if target_outputs is not None and node.output[0] not in target_outputs:
            new_nodes.append(node)
            continue

        mode = "nearest"
        for attr in node.attribute:
            if attr.name == "mode":
                mode = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s

        if mode != "nearest":
            new_nodes.append(node)
            continue

        x_name = node.input[0]
        y_name = node.output[0]

        in_shape = _get_shape(x_name)
        out_shape = _get_shape(y_name)

        if not in_shape or not out_shape:
            new_nodes.append(node)
            continue

        resize_axes = [i for i in range(len(in_shape)) if in_shape[i] != out_shape[i]]

        if len(resize_axes) != 1:
            new_nodes.append(node)
            continue

        axis = resize_axes[0]
        in_size = in_shape[axis]
        out_size = out_shape[axis]
        prefix = f"_resize_decomp_{y_name}_"

        # Precompute indices: floor(i * in_size / out_size)
        indices = np.floor(np.arange(out_size) * in_size / out_size).astype(np.int64)
        indices_name = prefix + "indices"
        graph.initializer.append(numpy_helper.from_array(indices, name=indices_name))

        new_nodes.append(helper.make_node(
            "Gather", [x_name, indices_name], [y_name], axis=axis
        ))

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def _load_target_names(only_unassigned):
    if not only_unassigned:
        return None

    with open(only_unassigned) as f:
        data = json.load(f)
    unassigned_keys = {name for name, info in data['ops'].items()
                       if not info.get('recommended_executor')}
    # Discovery JSON keys are like "OpType_outputname" — extract the output name
    # portion after the first underscore for matching against ONNX node outputs.
    target_outputs = set()
    for key in unassigned_keys:
        # e.g. "LayerNormalization_layer_norm_2" -> "layer_norm_2"
        # e.g. "Resize_upsample_nearest1d" -> "upsample_nearest1d"
        parts = key.split("_", 1)
        if len(parts) > 1:
            target_outputs.add(parts[1])
    print(f"Filtering: only decomposing ops with outputs in {len(target_outputs)} unassigned entries")
    return target_outputs


def _count_ops(model):
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return op_counts


def decompose_one_model(input_path, output_path, target_names=None):
    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))

    op_counts_before = _count_ops(model)

    model = decompose_layer_normalization(model, target_names)
    model = decompose_instance_normalization(model, target_names)
    model = decompose_resize_nearest(model, target_names)

    op_counts_after = _count_ops(model)

    print("\nDecompositions applied:")
    summary = {}
    for op in ["LayerNormalization", "InstanceNormalization", "Resize"]:
        before = op_counts_before.get(op, 0)
        after = op_counts_after.get(op, 0)
        decomposed = before - after
        summary[op] = {
            "before": before,
            "after": after,
            "decomposed": decomposed,
        }
        if before > 0:
            print(f"  {op}: {before} -> {after} (decomposed {decomposed})")
        else:
            print(f"  {op}: none found")

    before_total = sum(op_counts_before.values())
    after_total = len(model.graph.node)
    print(f"\nTotal nodes: {after_total} (was {before_total})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"\nSaved: {output_path}")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "ops": summary,
        "nodes_before": before_total,
        "nodes_after": after_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Decompose unsupported ONNX ops")
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path or directory")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ONNX model path, or output directory when --input is a directory. "
            "For directory input, defaults to <input>_decomposed."
        ),
    )
    parser.add_argument("--only-unassigned", metavar="JSON",
                        help="Only decompose ops that are unassigned in this discovery JSON file")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"--input does not exist: {args.input}")
    if args.input.is_file() and args.output is None:
        parser.error("--output is required when --input is a file")
    if args.input.is_dir() and args.output is not None and args.output.suffix == ".onnx":
        parser.error("--output must be a directory when --input is a directory")

    target_names = _load_target_names(args.only_unassigned)

    if args.input.is_file():
        decompose_one_model(args.input, args.output, target_names)
        return

    output_dir = args.output or args.input.parent / f"{args.input.name}_decomposed"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_models = sorted(path for path in args.input.iterdir() if path.is_file() and path.suffix == ".onnx")
    if not input_models:
        parser.error(f"No .onnx files found in directory: {args.input}")

    results = []
    for index, input_path in enumerate(input_models, start=1):
        output_path = output_dir / input_path.name
        print(f"\n[{index}/{len(input_models)}] {input_path}")
        results.append(decompose_one_model(input_path, output_path, target_names))

    totals = {op: 0 for op in ["LayerNormalization", "InstanceNormalization", "Resize"]}
    for result in results:
        for op, stats in result["ops"].items():
            totals[op] += stats["decomposed"]

    print(f"\nWrote decomposed ONNX folder: {output_dir}")
    print("Folder decomposition totals:")
    for op, count in totals.items():
        print(f"  {op}: decomposed {count}")


if __name__ == "__main__":
    main()
