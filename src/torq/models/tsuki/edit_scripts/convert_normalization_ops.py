#!/usr/bin/env python3
"""Rewrite normalization-friendly ONNX patterns.

This targets ONNX graphs that spell rsqrt-normalization as:

    Sqrt(var_eps) -> Reciprocal(std) -> Mul(x, inv_std)

and rewrites the Mul into:

    Div(x, std)

The rewritten Div keeps the original Mul output name, so the graph IO contract
and downstream consumers are unchanged. The Reciprocal node is removed only
when its output is consumed exclusively by rewritten Mul nodes.

It also rewrites constant integer-power Pow nodes into Mul chains:

    Pow(x, 2) -> Mul(x, x)
    Pow(x, 3) -> Mul(Mul(x, x), x)

And rewrites floating-point ReduceSum into ReduceMean plus a scalar multiply:

    ReduceSum(x, axes) -> Mul(ReduceMean(x, axes), product(reduced_dims))

For large Tsuki-style normalization reductions, it can also split:

    ReduceMean([N, C, L], axes=[-1], keepdims=1)

across the channel dimension into smaller Slice+ReduceMean chunks, then Concat
the chunk outputs back to the original output name.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


FLOAT_TENSOR_TYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}

INT_TENSOR_TYPES = {
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
}

FLOAT_ELEM_TYPES = {
    TensorProto.FLOAT16,
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
}

COMPILE_STDERR_TAIL_CHARS = 4000
REDUCEMEAN_WITHIN_CHUNK_REASON = "ReduceMean tile is already within chunk limits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite Sqrt->Reciprocal->Mul rsqrt-normalization patterns into "
            "Sqrt->Div, constant Pow(x, 2/3) into Mul chains, floating-point "
            "ReduceSum into ReduceMean*scale, and large ReduceMean nodes into "
            "chunked ReduceMean, for a single ONNX model or a directory of "
            "ONNX models."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input ONNX model, or a directory of ONNX models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ONNX model, or output directory when --input is a directory. "
            "Required unless --dry-run is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report eligible patterns without writing a model.",
    )
    parser.add_argument(
        "--only-output",
        action="append",
        default=[],
        help=(
            "Only rewrite candidate nodes with this first output name. Can be "
            "passed multiple times."
        ),
    )
    parser.add_argument(
        "--only-node",
        action="append",
        default=[],
        help=(
            "Only rewrite candidate nodes with this node name. Can be passed "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--max-conversions",
        type=int,
        default=None,
        help="Stop after converting this many eligible rewrite candidates per model.",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX checker on the rewritten model.",
    )
    parser.add_argument(
        "--verify-random",
        action="store_true",
        help="Run ONNX Runtime on random inputs and compare original vs rewritten model.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dynamic-dim-value",
        type=int,
        default=1,
        help="Concrete size used for dynamic dimensions during --verify-random.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional JSON report. In folder mode this is the manifest path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print every Reciprocal, Pow, ReduceSum, and ReduceMean "
            "conversion/skip decision."
        ),
    )
    parser.add_argument(
        "--reducemean-channel-chunk",
        type=int,
        default=192,
        help=(
            "Split eligible rank-3 ReduceMean nodes along channel dimension using "
            "this maximum chunk size. Use 0 to disable the ReduceMean split hack. "
            "Default: 192."
        ),
    )
    parser.add_argument(
        "--reducemean-max-chunk-elements",
        type=int,
        default=192 * 1291,
        help=(
            "Maximum channel_chunk * reduced_length tile size for ReduceMean "
            "splitting. This catches long-sequence reductions where the channel "
            "dimension is small but the reduction tile still overflows LRAM. "
            "Default: 192*1291."
        ),
    )
    parser.add_argument(
        "--bf16-reducemean-island",
        action="store_true",
        help=(
            "Wrap rewritten ReduceMean regions in local BF16 Cast islands. "
            "Graph inputs/outputs and surrounding normalization tensors remain "
            "at their original element type."
        ),
    )
    parser.add_argument(
        "--nss-compile-check",
        action="store_true",
        help=(
            "For each converted pattern, compile-check both the isolated old op "
            "and isolated replacement op/chain for NSS. Also sanity-check "
            "ReduceMean nodes skipped because they are already within chunk limits."
        ),
    )
    parser.add_argument(
        "-n",
        "--compile-workers",
        type=int,
        default=1,
        help="Parallel NSS compile-check workers. Default: 1.",
    )
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each import or torq-compile step. Default: 120.",
    )
    parser.add_argument(
        "--compile-artifacts-dir",
        type=Path,
        help="Optional directory to keep per-layer compile-check artifacts.",
    )

    args = parser.parse_args()
    if args.input.is_dir() and args.output is not None and args.output.suffix == ".onnx":
        parser.error("--output must be a directory when --input is a directory")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    if args.max_conversions is not None and args.max_conversions < 0:
        parser.error("--max-conversions must be non-negative")
    if args.compile_workers < 1:
        parser.error("--compile-workers/-n must be at least 1")
    if args.compile_timeout <= 0:
        parser.error("--compile-timeout must be positive")
    if args.reducemean_channel_chunk < 0:
        parser.error("--reducemean-channel-chunk must be non-negative")
    if args.reducemean_max_chunk_elements <= 0:
        parser.error("--reducemean-max-chunk-elements must be positive")
    return args


def safe_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    return safe[:160] or "unnamed"


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int | str | None, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def tensor_elem_type(value_info: onnx.ValueInfoProto) -> int | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    return int(value_info.type.tensor_type.elem_type)


def collect_value_infos(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = {}
    for value_info in (
        list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    ):
        values[value_info.name] = copy.deepcopy(value_info)
    for initializer in model.graph.initializer:
        values.setdefault(
            initializer.name,
            helper.make_tensor_value_info(
                initializer.name,
                initializer.data_type,
                list(initializer.dims),
            ),
        )
    return values


def collect_shapes(model: onnx.ModelProto) -> dict[str, tuple[int | str | None, ...]]:
    shapes = {}
    for name, value_info in collect_value_infos(model).items():
        shape = tensor_shape(value_info)
        if shape is not None:
            shapes[name] = shape
    return shapes


def collect_elem_types(model: onnx.ModelProto) -> dict[str, int]:
    elem_types = {}
    for name, value_info in collect_value_infos(model).items():
        elem_type = tensor_elem_type(value_info)
        if elem_type is not None:
            elem_types[name] = elem_type
    return elem_types


def infer_model_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model)
    except Exception:
        return model


def node_label(node: onnx.NodeProto) -> str:
    if node.name:
        return node.name
    if node.output:
        return f"{node.op_type}_{node.output[0]}"
    return node.op_type


def layer_id(op_type: str, output_name: str) -> str:
    return f"{op_type}_{output_name}"


def non_constant_index_by_node(model: onnx.ModelProto) -> dict[int, int]:
    out = {}
    index = 0
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        out[id(node)] = index
        index += 1
    return out


def build_consumers(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def build_producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }


def graph_output_names(model: onnx.ModelProto) -> set[str]:
    return {output.name for output in model.graph.output}


def graph_tensor_names(model: onnx.ModelProto) -> set[str]:
    names = set()
    for value_info in (
        list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    ):
        names.add(value_info.name)
    for initializer in model.graph.initializer:
        names.add(initializer.name)
    for node in model.graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    return names


def unique_tensor_name(model: onnx.ModelProto, base_name: str) -> str:
    names = graph_tensor_names(model)
    if base_name not in names:
        return base_name
    index = 1
    while f"{base_name}_{index}" in names:
        index += 1
    return f"{base_name}_{index}"


def reserve_unique_name(used_names: set[str], base_name: str) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    index = 1
    while f"{base_name}_{index}" in used_names:
        index += 1
    name = f"{base_name}_{index}"
    used_names.add(name)
    return name


def tensor_from_constant_node(node: onnx.NodeProto) -> onnx.TensorProto | None:
    if node.op_type != "Constant" or len(node.output) != 1:
        return None

    output_name = node.output[0]
    for attr in node.attribute:
        if attr.name == "value" and attr.HasField("t"):
            tensor = copy.deepcopy(attr.t)
            tensor.name = output_name
            return tensor
        if attr.name == "value_float":
            return helper.make_tensor(output_name, TensorProto.FLOAT, [], [attr.f])
        if attr.name == "value_int":
            return helper.make_tensor(output_name, TensorProto.INT64, [], [attr.i])
        if attr.name == "value_floats":
            return helper.make_tensor(
                output_name, TensorProto.FLOAT, [len(attr.floats)], list(attr.floats)
            )
        if attr.name == "value_ints":
            return helper.make_tensor(
                output_name, TensorProto.INT64, [len(attr.ints)], list(attr.ints)
            )
    return None


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    for node in model.graph.node:
        tensor = tensor_from_constant_node(node)
        if tensor is not None:
            initializers.setdefault(tensor.name, tensor)
    return initializers


def constant_tensor_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values = {}
    for name, tensor in initializer_map(model).items():
        try:
            values[name] = numpy_helper.to_array(tensor)
        except Exception:
            continue
    return values


def scalar_float_constant(name: str, constants: dict[str, np.ndarray]) -> float | None:
    value = constants.get(name)
    if value is None or value.size != 1:
        return None
    return float(value.reshape(()))


def supported_integer_power(value: float | None) -> int | None:
    if value is None:
        return None
    if np.isclose(value, 2.0, rtol=0.0, atol=1e-6):
        return 2
    if np.isclose(value, 3.0, rtol=0.0, atol=1e-6):
        return 3
    return None


def constant_int_tuple(name: str, constants: dict[str, np.ndarray]) -> tuple[int, ...] | None:
    value = constants.get(name)
    if value is None:
        return None
    if not np.issubdtype(value.dtype, np.integer):
        return None
    return tuple(int(v) for v in value.reshape(-1))


def node_int_attribute(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def node_ints_attribute(node: onnx.NodeProto, name: str) -> tuple[int, ...] | None:
    for attr in node.attribute:
        if attr.name == name:
            return tuple(int(v) for v in attr.ints)
    return None


def default_opset_version(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 0


def make_i64_initializer(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def make_float_scalar_initializer(
    name: str,
    *,
    value: float,
    elem_type: int,
) -> onnx.TensorProto:
    if elem_type == TensorProto.FLOAT:
        return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)
    if elem_type == TensorProto.FLOAT16:
        return numpy_helper.from_array(np.asarray(value, dtype=np.float16), name=name)
    if elem_type == TensorProto.DOUBLE:
        return numpy_helper.from_array(np.asarray(value, dtype=np.float64), name=name)
    raise ValueError(f"Unsupported floating-point element type {elem_type}")


def make_typed_value_info(
    name: str,
    *,
    elem_type: int,
    shape: tuple[int | str | None, ...] | list[int | str | None],
) -> onnx.ValueInfoProto:
    if not is_fully_static_shape(tuple(shape)):
        raise ValueError(f"Missing fully static shape for tensor {name!r}: {shape}")
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def has_value_info(model: onnx.ModelProto, name: str) -> bool:
    return any(
        value_info.name == name
        for value_info in (
            list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        )
    )


def is_fully_static_shape(shape: tuple[int | str | None, ...] | None) -> bool:
    return shape is not None and all(isinstance(dim, int) and dim >= 0 for dim in shape)


def append_static_value_info(
    model: onnx.ModelProto,
    *,
    name: str,
    elem_type: int | None,
    shape: tuple[int | str | None, ...] | None,
) -> None:
    if has_value_info(model, name):
        return
    if elem_type is None or elem_type == TensorProto.UNDEFINED:
        raise ValueError(f"Missing element type for new tensor {name!r}")
    if not is_fully_static_shape(shape):
        raise ValueError(f"Missing fully static shape for new tensor {name!r}: {shape}")
    model.graph.value_info.append(
        helper.make_tensor_value_info(name, elem_type, list(shape))
    )


def model_opset_imports(model: onnx.ModelProto) -> list[onnx.OperatorSetIdProto]:
    return [copy.deepcopy(opset) for opset in model.opset_import]


def make_value_info(name: str, value_infos: dict[str, onnx.ValueInfoProto]) -> onnx.ValueInfoProto:
    if name not in value_infos:
        raise ValueError(f"Missing static value_info for tensor {name!r}")
    value_info = copy.deepcopy(value_infos[name])
    if tensor_shape(value_info) is None:
        raise ValueError(f"Missing static shape for tensor {name!r}")
    elem_type = tensor_elem_type(value_info)
    if elem_type is None or elem_type == TensorProto.UNDEFINED:
        raise ValueError(f"Missing element type for tensor {name!r}")
    return value_info


def build_single_node_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    name: str,
) -> onnx.ModelProto:
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = []
    for input_name in inputs:
        if input_name in inits:
            graph_initializers.append(copy.deepcopy(inits[input_name]))
        else:
            graph_inputs.append(make_value_info(input_name, value_infos))

    graph_outputs = [make_value_info(output_name, value_infos) for output_name in outputs]
    node = helper.make_node(op_type, inputs, outputs, name=name)
    graph = helper.make_graph(
        [node],
        name,
        graph_inputs,
        graph_outputs,
        initializer=graph_initializers,
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model


def build_copied_node_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    node: onnx.NodeProto,
) -> onnx.ModelProto:
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = []
    for input_name in node.input:
        if not input_name:
            continue
        if input_name in inits:
            graph_initializers.append(copy.deepcopy(inits[input_name]))
        else:
            graph_inputs.append(make_value_info(input_name, value_infos))

    graph_outputs = [make_value_info(output_name, value_infos) for output_name in node.output]
    graph = helper.make_graph(
        [copy.deepcopy(node)],
        node_label(node),
        graph_inputs,
        graph_outputs,
        initializer=graph_initializers,
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model


def build_pow_mul_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    pow_node: onnx.NodeProto,
    exponent: int,
    replacement_mul_outputs: list[str],
) -> onnx.ModelProto:
    input_name = pow_node.input[0]
    output_name = pow_node.output[0]
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = []
    if input_name in inits:
        graph_initializers.append(copy.deepcopy(inits[input_name]))
    else:
        graph_inputs.append(make_value_info(input_name, value_infos))

    nodes = []
    extra_value_infos = []
    if exponent == 2:
        nodes.append(
            helper.make_node(
                "Mul",
                [input_name, input_name],
                [output_name],
                name=f"{node_label(pow_node)}_pow2_mul",
            )
        )
    elif exponent == 3:
        square_output = replacement_mul_outputs[0]
        nodes.append(
            helper.make_node(
                "Mul",
                [input_name, input_name],
                [square_output],
                name=f"{node_label(pow_node)}_pow3_square_mul",
            )
        )
        nodes.append(
            helper.make_node(
                "Mul",
                [square_output, input_name],
                [output_name],
                name=f"{node_label(pow_node)}_pow3_mul",
            )
        )
        extra_value_infos.append(
            helper.make_tensor_value_info(
                square_output,
                elem_types.get(output_name, elem_types.get(input_name, TensorProto.FLOAT)),
                list(shapes.get(output_name, shapes.get(input_name, ()))),
            )
        )
    else:
        raise ValueError(f"Unsupported Pow exponent for Mul rewrite: {exponent}")

    graph_outputs = [make_value_info(output_name, value_infos)]
    graph = helper.make_graph(
        nodes,
        f"{node_label(pow_node)}_pow_mul",
        graph_inputs,
        graph_outputs,
        initializer=graph_initializers,
        value_info=extra_value_infos,
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model


def make_reducesum_mean_mul_components(
    *,
    model_for_names: onnx.ModelProto,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    normalized_axes_value: tuple[int, ...],
    scale_value: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0]
    used_names = graph_tensor_names(model_for_names)
    mean_output = reserve_unique_name(used_names, f"{output_name}_mean")
    scale_name = reserve_unique_name(used_names, f"{output_name}_sum_scale")
    axes_name = reserve_unique_name(used_names, f"{output_name}_mean_axes")
    scale_tensor = make_float_scalar_initializer(
        scale_name, value=float(scale_value), elem_type=elem_type
    )
    if len(normalized_axes_value) != 1:
        raise ValueError("ReduceSum ReduceMean rewrite only handles one axis")

    initializers = [
        scale_tensor,
        make_i64_initializer(axes_name, list(normalized_axes_value)),
    ]
    nodes = [
        helper.make_node(
            "ReduceMean",
            [reduce_node.input[0], axes_name],
            [mean_output],
            name=f"{node_label(reduce_node)}_mean",
            keepdims=node_int_attribute(reduce_node, "keepdims", 1),
            noop_with_empty_axes=node_int_attribute(
                reduce_node, "noop_with_empty_axes", 0
            ),
        ),
        helper.make_node(
            "Mul",
            [mean_output, scale_name],
            [output_name],
            name=f"{node_label(reduce_node)}_sum_scale",
        )
    ]
    value_infos = [
        make_typed_value_info(mean_output, elem_type=elem_type, shape=output_shape)
    ]
    replacement_layer_ids = [
        layer_id("ReduceMean", mean_output),
        layer_id("Mul", output_name),
    ]
    return {
        "nodes": nodes,
        "initializers": initializers,
        "value_infos": value_infos,
        "replacement_layer_ids": replacement_layer_ids,
        "mean_output": mean_output,
        "scale_initializer": scale_name,
    }


def build_reducesum_mean_mul_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    normalized_axes_value: tuple[int, ...],
    scale_value: int,
    bf16_island: bool = False,
) -> tuple[onnx.ModelProto, list[str]]:
    components = make_reducesum_mean_mul_components(
        model_for_names=source_model,
        reduce_node=reduce_node,
        input_shape=input_shape,
        output_shape=output_shape,
        elem_type=elem_type,
        normalized_axes_value=normalized_axes_value,
        scale_value=scale_value,
        bf16_island=bf16_island,
    )
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = list(components["initializers"])
    for input_name in reduce_node.input:
        if not input_name:
            continue
        if input_name in inits:
            graph_initializers.append(copy.deepcopy(inits[input_name]))
        else:
            graph_inputs.append(make_value_info(input_name, value_infos))

    graph = helper.make_graph(
        components["nodes"],
        f"{node_label(reduce_node)}_reducesum_mean_mul",
        graph_inputs,
        [make_value_info(reduce_node.output[0], value_infos)],
        initializer=graph_initializers,
        value_info=components["value_infos"],
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model, components["replacement_layer_ids"]


def reduce_mean_axes(
    reduce_node: onnx.NodeProto,
    constants: dict[str, np.ndarray],
) -> tuple[int, ...] | None:
    if len(reduce_node.input) >= 2 and reduce_node.input[1]:
        return constant_int_tuple(reduce_node.input[1], constants)
    return node_ints_attribute(reduce_node, "axes")


def normalize_axis(axis: int, rank: int) -> int | None:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def normalized_axes(axes: tuple[int, ...], rank: int) -> tuple[int, ...] | None:
    normalized = []
    for axis in axes:
        normalized_axis = normalize_axis(axis, rank)
        if normalized_axis is None:
            return None
        if normalized_axis not in normalized:
            normalized.append(normalized_axis)
    return tuple(normalized)


def reducesum_scale_value(
    *,
    input_shape: tuple[int | str | None, ...],
    axes: tuple[int, ...] | None,
    noop_with_empty_axes: int,
) -> tuple[int | None, tuple[int, ...] | None, str | None]:
    rank = len(input_shape)
    if not is_fully_static_shape(input_shape):
        return None, axes, "missing fully static ReduceSum input shape"

    if axes is None:
        normalized = tuple(range(rank))
    elif len(axes) == 0:
        if noop_with_empty_axes:
            return None, tuple(), "ReduceSum has empty axes with noop enabled"
        normalized = tuple(range(rank))
    else:
        normalized = normalized_axes(axes, rank)
        if normalized is None:
            return None, axes, "ReduceSum axis is out of range"

    scale = 1
    for axis in normalized:
        scale *= int(input_shape[axis])
    return scale, normalized, None


def make_reducemean_split_components(
    *,
    model_for_names: onnx.ModelProto,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    chunk_size: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    input_name = reduce_node.input[0]
    output_name = reduce_node.output[0]
    n_dim, c_dim, l_dim = (int(dim) for dim in input_shape)
    used_names = graph_tensor_names(model_for_names)

    compute_elem_type = TensorProto.BFLOAT16 if bf16_island else elem_type
    compute_input_name = input_name
    final_reduce_output = output_name
    cast_input_name = None
    cast_output_name = None
    if bf16_island:
        cast_input_name = reserve_unique_name(used_names, f"{output_name}_bf16_input")
        cast_output_name = reserve_unique_name(used_names, f"{output_name}_bf16")
        compute_input_name = cast_input_name
        final_reduce_output = cast_output_name

    reduce_axes_name = reserve_unique_name(used_names, f"{output_name}_chunk_reduce_axes")
    initializers = [make_i64_initializer(reduce_axes_name, [-1])]
    nodes = []
    value_infos = []
    reduce_outputs = []
    replacement_layer_ids = []

    if bf16_island:
        assert cast_input_name is not None
        nodes.append(
            helper.make_node(
                "Cast",
                [input_name],
                [cast_input_name],
                name=f"{node_label(reduce_node)}_bf16_input_cast",
                to=TensorProto.BFLOAT16,
            )
        )
        value_infos.append(
            make_typed_value_info(
                cast_input_name,
                elem_type=TensorProto.BFLOAT16,
                shape=input_shape,
            )
        )
        replacement_layer_ids.append(layer_id("Cast", cast_input_name))

    for start in range(0, c_dim, chunk_size):
        end = min(start + chunk_size, c_dim)
        width = end - start
        suffix = f"{start}_{end}"
        starts_name = reserve_unique_name(used_names, f"{output_name}_slice_starts_{suffix}")
        ends_name = reserve_unique_name(used_names, f"{output_name}_slice_ends_{suffix}")
        axes_name = reserve_unique_name(used_names, f"{output_name}_slice_axes_{suffix}")
        steps_name = reserve_unique_name(used_names, f"{output_name}_slice_steps_{suffix}")
        sliced = reserve_unique_name(used_names, f"{output_name}_slice_{suffix}")
        reduced = reserve_unique_name(used_names, f"{output_name}_chunk_mean_{suffix}")

        initializers.extend(
            [
                make_i64_initializer(starts_name, [0, start, 0]),
                make_i64_initializer(ends_name, [n_dim, end, l_dim]),
                make_i64_initializer(axes_name, [0, 1, 2]),
                make_i64_initializer(steps_name, [1, 1, 1]),
            ]
        )
        nodes.append(
            helper.make_node(
                "Slice",
                [compute_input_name, starts_name, ends_name, axes_name, steps_name],
                [sliced],
                name=f"{node_label(reduce_node)}_slice_{suffix}",
            )
        )
        nodes.append(
            helper.make_node(
                "ReduceMean",
                [sliced, reduce_axes_name],
                [reduced],
                name=f"{node_label(reduce_node)}_chunk_mean_{suffix}",
                keepdims=1,
                noop_with_empty_axes=0,
            )
        )
        value_infos.extend(
            [
                helper.make_tensor_value_info(
                    sliced, compute_elem_type, [n_dim, width, l_dim]
                ),
                helper.make_tensor_value_info(
                    reduced, compute_elem_type, [n_dim, width, 1]
                ),
            ]
        )
        reduce_outputs.append(reduced)
        replacement_layer_ids.extend(
            [layer_id("Slice", sliced), layer_id("ReduceMean", reduced)]
        )

    nodes.append(
        helper.make_node(
            "Concat",
            reduce_outputs,
            [final_reduce_output],
            name=f"{node_label(reduce_node)}_chunk_concat",
            axis=1,
        )
    )
    replacement_layer_ids.append(layer_id("Concat", final_reduce_output))

    if bf16_island:
        assert cast_output_name is not None
        value_infos.append(
            make_typed_value_info(
                cast_output_name,
                elem_type=TensorProto.BFLOAT16,
                shape=output_shape,
            )
        )
        nodes.append(
            helper.make_node(
                "Cast",
                [cast_output_name],
                [output_name],
                name=f"{node_label(reduce_node)}_fp32_output_cast",
                to=elem_type,
            )
        )
        replacement_layer_ids.append(layer_id("Cast", output_name))

    if tuple(int(dim) for dim in output_shape) != (n_dim, c_dim, 1):
        raise ValueError(
            f"ReduceMean output shape {output_shape} does not match expected {(n_dim, c_dim, 1)}"
        )

    return {
        "nodes": nodes,
        "initializers": initializers,
        "value_infos": value_infos,
        "replacement_layer_ids": replacement_layer_ids,
        "chunk_count": len(reduce_outputs),
        "chunk_size": chunk_size,
    }


def build_reducemean_split_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    chunk_size: int,
    bf16_island: bool = False,
) -> tuple[onnx.ModelProto, list[str]]:
    components = make_reducemean_split_components(
        model_for_names=source_model,
        reduce_node=reduce_node,
        input_shape=input_shape,
        output_shape=output_shape,
        elem_type=elem_type,
        chunk_size=chunk_size,
        bf16_island=bf16_island,
    )
    input_name = reduce_node.input[0]
    output_name = reduce_node.output[0]
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = list(components["initializers"])
    if input_name in inits:
        graph_initializers.append(copy.deepcopy(inits[input_name]))
    else:
        graph_inputs.append(make_value_info(input_name, value_infos))

    graph = helper.make_graph(
        components["nodes"],
        f"{node_label(reduce_node)}_split_reducemean",
        graph_inputs,
        [make_value_info(output_name, value_infos)],
        initializer=graph_initializers,
        value_info=components["value_infos"],
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model, components["replacement_layer_ids"]


def make_reducemean_transpose_components(
    *,
    model_for_names: onnx.ModelProto,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    axis: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    input_name = reduce_node.input[0]
    output_name = reduce_node.output[0]
    rank = len(input_shape)
    input_dims = [int(dim) for dim in input_shape]
    perm = [dim for dim in range(rank) if dim != axis] + [axis]
    restore_perm = [0] * rank
    for new_index, old_index in enumerate(perm):
        restore_perm[old_index] = new_index

    transposed_shape = [input_dims[dim] for dim in perm]
    reduced_shape = list(transposed_shape)
    reduced_shape[-1] = 1
    expected_output_shape = list(input_dims)
    expected_output_shape[axis] = 1
    if list(int(dim) for dim in output_shape) != expected_output_shape:
        raise ValueError(
            f"ReduceMean output shape {output_shape} does not match expected "
            f"{tuple(expected_output_shape)} for axis {axis}"
        )

    used_names = graph_tensor_names(model_for_names)
    compute_elem_type = TensorProto.BFLOAT16 if bf16_island else elem_type
    compute_input_name = input_name
    final_output_name = output_name
    cast_input_name = None
    cast_output_name = None
    if bf16_island:
        cast_input_name = reserve_unique_name(used_names, f"{output_name}_bf16_input")
        cast_output_name = reserve_unique_name(used_names, f"{output_name}_bf16")
        compute_input_name = cast_input_name
        final_output_name = cast_output_name

    transposed = reserve_unique_name(used_names, f"{output_name}_axis_to_last")
    reduced = reserve_unique_name(used_names, f"{output_name}_last_axis_mean")
    reduce_axes_name = reserve_unique_name(used_names, f"{output_name}_last_axis_reduce_axes")

    nodes = []
    value_infos = []
    replacement_layer_ids = []
    if bf16_island:
        assert cast_input_name is not None
        nodes.append(
            helper.make_node(
                "Cast",
                [input_name],
                [cast_input_name],
                name=f"{node_label(reduce_node)}_bf16_input_cast",
                to=TensorProto.BFLOAT16,
            )
        )
        value_infos.append(
            make_typed_value_info(
                cast_input_name,
                elem_type=TensorProto.BFLOAT16,
                shape=input_shape,
            )
        )
        replacement_layer_ids.append(layer_id("Cast", cast_input_name))

    nodes.extend(
        [
        helper.make_node(
            "Transpose",
            [compute_input_name],
            [transposed],
            name=f"{node_label(reduce_node)}_axis_to_last",
            perm=perm,
        ),
        helper.make_node(
            "ReduceMean",
            [transposed, reduce_axes_name],
            [reduced],
            name=f"{node_label(reduce_node)}_last_axis_mean",
            keepdims=1,
            noop_with_empty_axes=0,
        ),
        helper.make_node(
            "Transpose",
            [reduced],
            [final_output_name],
            name=f"{node_label(reduce_node)}_restore_axis",
            perm=restore_perm,
        ),
        ]
    )
    value_infos.extend(
        [
            helper.make_tensor_value_info(transposed, compute_elem_type, transposed_shape),
            helper.make_tensor_value_info(reduced, compute_elem_type, reduced_shape),
        ]
    )
    replacement_layer_ids.extend(
        [
            layer_id("Transpose", transposed),
            layer_id("ReduceMean", reduced),
            layer_id("Transpose", final_output_name),
        ]
    )

    if bf16_island:
        assert cast_output_name is not None
        value_infos.append(
            make_typed_value_info(
                cast_output_name,
                elem_type=TensorProto.BFLOAT16,
                shape=output_shape,
            )
        )
        nodes.append(
            helper.make_node(
                "Cast",
                [cast_output_name],
                [output_name],
                name=f"{node_label(reduce_node)}_fp32_output_cast",
                to=elem_type,
            )
        )
        replacement_layer_ids.append(layer_id("Cast", output_name))

    return {
        "nodes": nodes,
        "initializers": [make_i64_initializer(reduce_axes_name, [-1])],
        "value_infos": value_infos,
        "replacement_layer_ids": replacement_layer_ids,
        "perm": perm,
        "restore_perm": restore_perm,
        "transposed_shape": transposed_shape,
        "reduced_shape": reduced_shape,
    }


def build_reducemean_transpose_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    axis: int,
    bf16_island: bool = False,
) -> tuple[onnx.ModelProto, list[str], list[int], list[int]]:
    components = make_reducemean_transpose_components(
        model_for_names=source_model,
        reduce_node=reduce_node,
        input_shape=input_shape,
        output_shape=output_shape,
        elem_type=elem_type,
        axis=axis,
        bf16_island=bf16_island,
    )
    input_name = reduce_node.input[0]
    output_name = reduce_node.output[0]
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = list(components["initializers"])
    if input_name in inits:
        graph_initializers.append(copy.deepcopy(inits[input_name]))
    else:
        graph_inputs.append(make_value_info(input_name, value_infos))

    graph = helper.make_graph(
        components["nodes"],
        f"{node_label(reduce_node)}_transpose_reducemean",
        graph_inputs,
        [make_value_info(output_name, value_infos)],
        initializer=graph_initializers,
        value_info=components["value_infos"],
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return (
        model,
        components["replacement_layer_ids"],
        components["perm"],
        components["restore_perm"],
    )


def make_reducemean_bf16_island_components(
    *,
    model_for_names: onnx.ModelProto,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    output_elem_type: int,
) -> dict[str, Any]:
    input_name = reduce_node.input[0]
    output_name = reduce_node.output[0]
    used_names = graph_tensor_names(model_for_names)
    cast_input_name = reserve_unique_name(used_names, f"{output_name}_bf16_input")
    reduce_output = reserve_unique_name(used_names, f"{output_name}_bf16")

    bf16_reduce = copy.deepcopy(reduce_node)
    bf16_reduce.input[0] = cast_input_name
    bf16_reduce.output[0] = reduce_output
    bf16_reduce.name = f"{node_label(reduce_node)}_bf16"

    nodes = [
        helper.make_node(
            "Cast",
            [input_name],
            [cast_input_name],
            name=f"{node_label(reduce_node)}_bf16_input_cast",
            to=TensorProto.BFLOAT16,
        ),
        bf16_reduce,
        helper.make_node(
            "Cast",
            [reduce_output],
            [output_name],
            name=f"{node_label(reduce_node)}_fp32_output_cast",
            to=output_elem_type,
        ),
    ]
    value_infos = [
        make_typed_value_info(
            cast_input_name,
            elem_type=TensorProto.BFLOAT16,
            shape=input_shape,
        ),
        make_typed_value_info(
            reduce_output,
            elem_type=TensorProto.BFLOAT16,
            shape=output_shape,
        ),
    ]
    replacement_layer_ids = [
        layer_id("Cast", cast_input_name),
        layer_id("ReduceMean", reduce_output),
        layer_id("Cast", output_name),
    ]
    return {
        "nodes": nodes,
        "value_infos": value_infos,
        "replacement_layer_ids": replacement_layer_ids,
        "reduce_output": reduce_output,
    }


def build_reducemean_bf16_island_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    output_elem_type: int,
) -> tuple[onnx.ModelProto, list[str], str]:
    components = make_reducemean_bf16_island_components(
        model_for_names=source_model,
        reduce_node=reduce_node,
        input_shape=input_shape,
        output_shape=output_shape,
        output_elem_type=output_elem_type,
    )
    input_name = reduce_node.input[0]
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = []
    if input_name in inits:
        graph_initializers.append(copy.deepcopy(inits[input_name]))
    else:
        graph_inputs.append(make_value_info(input_name, value_infos))

    for input_name in reduce_node.input[1:]:
        if not input_name:
            continue
        if input_name in inits:
            graph_initializers.append(copy.deepcopy(inits[input_name]))
        else:
            graph_inputs.append(make_value_info(input_name, value_infos))

    graph = helper.make_graph(
        components["nodes"],
        f"{node_label(reduce_node)}_bf16_reducemean",
        graph_inputs,
        [make_value_info(reduce_node.output[0], value_infos)],
        initializer=graph_initializers,
        value_info=components["value_infos"],
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model, components["replacement_layer_ids"], components["reduce_output"]


def build_bf16_copied_reducemean_model(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
) -> onnx.ModelProto:
    inits = initializer_map(source_model)
    graph_inputs = []
    graph_initializers = []
    input_name = reduce_node.input[0]
    if input_name in inits:
        graph_initializers.append(copy.deepcopy(inits[input_name]))
    else:
        graph_inputs.append(
            make_typed_value_info(
                input_name,
                elem_type=TensorProto.BFLOAT16,
                shape=input_shape,
            )
        )

    for input_name in reduce_node.input[1:]:
        if not input_name:
            continue
        if input_name in inits:
            graph_initializers.append(copy.deepcopy(inits[input_name]))
        else:
            graph_inputs.append(make_value_info(input_name, value_infos))

    graph_outputs = [
        make_typed_value_info(
            reduce_node.output[0],
            elem_type=TensorProto.BFLOAT16,
            shape=output_shape,
        )
    ]
    graph = helper.make_graph(
        [copy.deepcopy(reduce_node)],
        f"{node_label(reduce_node)}_bf16",
        graph_inputs,
        graph_outputs,
        initializer=graph_initializers,
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model


def build_bf16_reducemean_shapes_model(
    *,
    source_model: onnx.ModelProto,
    graph_name: str,
    output_base: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
) -> tuple[onnx.ModelProto, list[str]]:
    axes_name = f"{safe_name(output_base)}_bf16_reduce_axes"
    initializers = [make_i64_initializer(axes_name, [-1])]
    graph_inputs = []
    graph_outputs = []
    nodes = []
    layer_ids = []
    for index, (input_shape, output_shape) in enumerate(zip(input_shapes, output_shapes)):
        input_name = f"{safe_name(output_base)}_bf16_reduce_input_{index}"
        output_name = f"{safe_name(output_base)}_bf16_reduce_output_{index}"
        nodes.append(
            helper.make_node(
                "ReduceMean",
                [input_name, axes_name],
                [output_name],
                name=f"{safe_name(output_base)}_bf16_reduce_{index}",
                keepdims=1,
                noop_with_empty_axes=0,
            )
        )
        graph_inputs.append(
            make_typed_value_info(
                input_name,
                elem_type=TensorProto.BFLOAT16,
                shape=input_shape,
            )
        )
        graph_outputs.append(
            make_typed_value_info(
                output_name,
                elem_type=TensorProto.BFLOAT16,
                shape=output_shape,
            )
        )
        layer_ids.append(layer_id("ReduceMean", output_name))

    graph = helper.make_graph(
        nodes,
        graph_name,
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=model_opset_imports(source_model))
    model.ir_version = source_model.ir_version
    checker.check_model(model)
    return model, layer_ids


def make_rsqrt_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    non_constant_index: int,
    reciprocal_node: onnx.NodeProto,
    mul_node: onnx.NodeProto,
    numerator_input: str,
    sqrt_output: str,
) -> dict[str, Any]:
    reciprocal_output = reciprocal_node.output[0]
    div_output = mul_node.output[0]
    reciprocal_model = build_single_node_model(
        source_model=source_model,
        value_infos=value_infos,
        op_type="Reciprocal",
        inputs=[reciprocal_node.input[0]],
        outputs=[reciprocal_output],
        name=node_label(reciprocal_node),
    )
    div_model = build_single_node_model(
        source_model=source_model,
        value_infos=value_infos,
        op_type="Div",
        inputs=[numerator_input, sqrt_output],
        outputs=[div_output],
        name=f"{node_label(mul_node)}_rsqrt_div",
    )
    return {
        "kind": "rsqrt_div",
        "job_id": f"{non_constant_index}_{safe_name(reciprocal_output)}_{safe_name(div_output)}",
        "non_constant_index": non_constant_index,
        "reciprocal_output": reciprocal_output,
        "div_output": div_output,
        "reciprocal_layer_id": layer_id("Reciprocal", reciprocal_output),
        "div_layer_id": layer_id("Div", div_output),
        "reciprocal_model": reciprocal_model.SerializeToString(),
        "div_model": div_model.SerializeToString(),
    }


def make_pow_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    non_constant_index: int,
    pow_node: onnx.NodeProto,
    exponent: int,
    replacement_mul_outputs: list[str],
) -> dict[str, Any]:
    pow_output = pow_node.output[0]
    pow_model = build_single_node_model(
        source_model=source_model,
        value_infos=value_infos,
        op_type="Pow",
        inputs=[pow_node.input[0], pow_node.input[1]],
        outputs=[pow_output],
        name=node_label(pow_node),
    )
    mul_model = build_pow_mul_model(
        source_model=source_model,
        value_infos=value_infos,
        shapes=shapes,
        elem_types=elem_types,
        pow_node=pow_node,
        exponent=exponent,
        replacement_mul_outputs=replacement_mul_outputs,
    )
    return {
        "kind": "pow_mul",
        "job_id": f"{non_constant_index}_{safe_name(pow_output)}_pow_mul",
        "non_constant_index": non_constant_index,
        "pow_output": pow_output,
        "mul_output": replacement_mul_outputs[-1],
        "exponent": exponent,
        "pow_layer_id": layer_id("Pow", pow_output),
        "mul_layer_ids": [layer_id("Mul", output) for output in replacement_mul_outputs],
        "pow_model": pow_model.SerializeToString(),
        "mul_model": mul_model.SerializeToString(),
    }


def make_reducesum_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    axes: tuple[int, ...],
    scale_value: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0]
    reducesum_model = build_copied_node_model(
        source_model=source_model,
        value_infos=value_infos,
        node=reduce_node,
    )
    mean_mul_model, replacement_layer_ids = build_reducesum_mean_mul_model(
        source_model=source_model,
        value_infos=value_infos,
        reduce_node=reduce_node,
        input_shape=input_shape,
        output_shape=output_shape,
        elem_type=elem_type,
        normalized_axes_value=axes,
        scale_value=scale_value,
        bf16_island=bf16_island,
    )
    return {
        "kind": "reducesum_mean_mul",
        "job_id": f"{non_constant_index}_{safe_name(output_name)}_reducesum_mean_mul",
        "non_constant_index": non_constant_index,
        "reducesum_output": output_name,
        "mean_mul_output": output_name,
        "axes": list(axes),
        "scale_value": scale_value,
        "bf16_island": bf16_island,
        "reducesum_layer_id": layer_id("ReduceSum", output_name),
        "mean_mul_layer_ids": replacement_layer_ids,
        "reducesum_model": reducesum_model.SerializeToString(),
        "mean_mul_model": mean_mul_model.SerializeToString(),
    }


def make_reducemean_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    chunk_size: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0]
    reduce_model = build_copied_node_model(
        source_model=source_model,
        value_infos=value_infos,
        node=reduce_node,
    )
    if bf16_island:
        n_dim, c_dim, l_dim = (int(dim) for dim in input_shape)
        input_shapes = []
        output_shapes = []
        for start in range(0, c_dim, chunk_size):
            end = min(start + chunk_size, c_dim)
            width = end - start
            input_shapes.append([n_dim, width, l_dim])
            output_shapes.append([n_dim, width, 1])
        split_model, replacement_layer_ids = build_bf16_reducemean_shapes_model(
            source_model=source_model,
            graph_name=f"{node_label(reduce_node)}_bf16_split_reducemean_compile",
            output_base=output_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )
    else:
        split_model, replacement_layer_ids = build_reducemean_split_model(
            source_model=source_model,
            value_infos=value_infos,
            reduce_node=reduce_node,
            input_shape=input_shape,
            output_shape=output_shape,
            elem_type=elem_type,
            chunk_size=chunk_size,
            bf16_island=False,
        )
    return {
        "kind": "reducemean_split",
        "job_id": f"{non_constant_index}_{safe_name(output_name)}_reducemean_split",
        "non_constant_index": non_constant_index,
        "reducemean_output": output_name,
        "split_output": output_name,
        "chunk_size": chunk_size,
        "reducemean_layer_id": layer_id("ReduceMean", output_name),
        "split_layer_ids": replacement_layer_ids,
        "bf16_island": bf16_island,
        "reducemean_model": reduce_model.SerializeToString(),
        "split_model": split_model.SerializeToString(),
    }


def make_reducemean_within_chunk_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    chunk_size: int,
    input_shape: tuple[int | str | None, ...] | None = None,
    output_shape: tuple[int | str | None, ...] | None = None,
    output_elem_type: int | None = None,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0]
    reduce_layer_ids: str | list[str]
    replacement_reduce_output = output_name
    if bf16_island:
        if input_shape is None or output_shape is None or output_elem_type is None:
            raise ValueError("BF16 ReduceMean compile job requires static shape/type")
        reduce_model = build_bf16_copied_reducemean_model(
            source_model=source_model,
            value_infos=value_infos,
            reduce_node=reduce_node,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        reduce_layer_ids = layer_id("ReduceMean", output_name)
        replacement_reduce_output = output_name
    else:
        reduce_model = build_copied_node_model(
            source_model=source_model,
            value_infos=value_infos,
            node=reduce_node,
        )
        reduce_layer_ids = layer_id("ReduceMean", output_name)
    return {
        "kind": "reducemean_within_chunk",
        "job_id": f"{non_constant_index}_{safe_name(output_name)}_reducemean_within_chunk",
        "non_constant_index": non_constant_index,
        "reducemean_output": output_name,
        "chunk_size": chunk_size,
        "chunk_count": 1,
        "reducemean_layer_id": reduce_layer_ids,
        "replacement_reduce_output": replacement_reduce_output,
        "bf16_island": bf16_island,
        "reducemean_model": reduce_model.SerializeToString(),
    }


def make_reducemean_transpose_compile_job(
    *,
    source_model: onnx.ModelProto,
    value_infos: dict[str, onnx.ValueInfoProto],
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    input_shape: tuple[int | str | None, ...],
    output_shape: tuple[int | str | None, ...],
    elem_type: int,
    axis: int,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0]
    reduce_model = build_copied_node_model(
        source_model=source_model,
        value_infos=value_infos,
        node=reduce_node,
    )
    if bf16_island:
        components = make_reducemean_transpose_components(
            model_for_names=source_model,
            reduce_node=reduce_node,
            input_shape=input_shape,
            output_shape=output_shape,
            elem_type=elem_type,
            axis=axis,
            bf16_island=True,
        )
        transpose_model, replacement_layer_ids = build_bf16_reducemean_shapes_model(
            source_model=source_model,
            graph_name=f"{node_label(reduce_node)}_bf16_transpose_reducemean_compile",
            output_base=output_name,
            input_shapes=[components["transposed_shape"]],
            output_shapes=[components["reduced_shape"]],
        )
        perm = components["perm"]
        restore_perm = components["restore_perm"]
    else:
        transpose_model, replacement_layer_ids, perm, restore_perm = (
            build_reducemean_transpose_model(
                source_model=source_model,
                value_infos=value_infos,
                reduce_node=reduce_node,
                input_shape=input_shape,
                output_shape=output_shape,
                elem_type=elem_type,
                axis=axis,
                bf16_island=False,
            )
        )
    return {
        "kind": "reducemean_transpose_axis",
        "job_id": f"{non_constant_index}_{safe_name(output_name)}_reducemean_transpose_axis",
        "non_constant_index": non_constant_index,
        "reducemean_output": output_name,
        "transpose_output": output_name,
        "axis": axis,
        "perm": perm,
        "restore_perm": restore_perm,
        "reducemean_layer_id": layer_id("ReduceMean", output_name),
        "transpose_layer_ids": replacement_layer_ids,
        "bf16_island": bf16_island,
        "reducemean_model": reduce_model.SerializeToString(),
        "transpose_model": transpose_model.SerializeToString(),
    }


def candidate_row(
    *,
    non_constant_index: int,
    reciprocal_node: onnx.NodeProto,
    status: str,
    reason: str,
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    converted_mul_outputs: list[str] | None = None,
) -> dict[str, Any]:
    output_name = reciprocal_node.output[0] if reciprocal_node.output else ""
    input_name = reciprocal_node.input[0] if reciprocal_node.input else ""
    return {
        "status": status,
        "reason": reason,
        "non_constant_index": non_constant_index,
        "node_name": reciprocal_node.name,
        "layer_id": layer_id("Reciprocal", output_name) if output_name else "",
        "input": input_name,
        "output": output_name,
        "input_shape": list(shapes[input_name]) if input_name in shapes else None,
        "output_shape": list(shapes[output_name]) if output_name in shapes else None,
        "input_elem_type": elem_types.get(input_name),
        "output_elem_type": elem_types.get(output_name),
        "converted_mul_outputs": converted_mul_outputs or [],
    }


def pow_candidate_row(
    *,
    non_constant_index: int,
    pow_node: onnx.NodeProto,
    status: str,
    reason: str,
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    exponent_value: float | None = None,
    exponent: int | None = None,
    replacement_mul_outputs: list[str] | None = None,
) -> dict[str, Any]:
    output_name = pow_node.output[0] if pow_node.output else ""
    input_name = pow_node.input[0] if pow_node.input else ""
    exponent_input = pow_node.input[1] if len(pow_node.input) > 1 else ""
    return {
        "status": status,
        "reason": reason,
        "non_constant_index": non_constant_index,
        "node_name": pow_node.name,
        "layer_id": layer_id("Pow", output_name) if output_name else "",
        "input": input_name,
        "exponent_input": exponent_input,
        "output": output_name,
        "input_shape": list(shapes[input_name]) if input_name in shapes else None,
        "output_shape": list(shapes[output_name]) if output_name in shapes else None,
        "input_elem_type": elem_types.get(input_name),
        "output_elem_type": elem_types.get(output_name),
        "exponent_value": exponent_value,
        "exponent": exponent,
        "replacement_mul_outputs": replacement_mul_outputs or [],
    }


def reducesum_candidate_row(
    *,
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    status: str,
    reason: str,
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    axes: tuple[int, ...] | None = None,
    normalized_axes_value: tuple[int, ...] | None = None,
    scale_value: int | None = None,
    mean_output: str | None = None,
    bf16_island: bool = False,
) -> dict[str, Any]:
    output_name = reduce_node.output[0] if reduce_node.output else ""
    input_name = reduce_node.input[0] if reduce_node.input else ""
    return {
        "status": status,
        "reason": reason,
        "non_constant_index": non_constant_index,
        "node_name": reduce_node.name,
        "layer_id": layer_id("ReduceSum", output_name) if output_name else "",
        "input": input_name,
        "output": output_name,
        "axes": list(axes) if axes is not None else None,
        "normalized_axes": (
            list(normalized_axes_value) if normalized_axes_value is not None else None
        ),
        "input_shape": list(shapes[input_name]) if input_name in shapes else None,
        "output_shape": list(shapes[output_name]) if output_name in shapes else None,
        "input_elem_type": elem_types.get(input_name),
        "output_elem_type": elem_types.get(output_name),
        "scale_value": scale_value,
        "mean_output": mean_output,
        "bf16_island": bf16_island,
    }


def reducemean_candidate_row(
    *,
    non_constant_index: int,
    reduce_node: onnx.NodeProto,
    status: str,
    reason: str,
    shapes: dict[str, tuple[int | str | None, ...]],
    elem_types: dict[str, int],
    axes: tuple[int, ...] | None = None,
    chunk_size: int | None = None,
    chunk_count: int | None = None,
    transpose_perm: list[int] | None = None,
    restore_perm: list[int] | None = None,
    bf16_island: bool = False,
    replacement_reduce_output: str | None = None,
) -> dict[str, Any]:
    output_name = reduce_node.output[0] if reduce_node.output else ""
    input_name = reduce_node.input[0] if reduce_node.input else ""
    return {
        "status": status,
        "reason": reason,
        "non_constant_index": non_constant_index,
        "node_name": reduce_node.name,
        "layer_id": layer_id("ReduceMean", output_name) if output_name else "",
        "input": input_name,
        "output": output_name,
        "axes": list(axes) if axes is not None else None,
        "input_shape": list(shapes[input_name]) if input_name in shapes else None,
        "output_shape": list(shapes[output_name]) if output_name in shapes else None,
        "input_elem_type": elem_types.get(input_name),
        "output_elem_type": elem_types.get(output_name),
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "transpose_perm": transpose_perm,
        "restore_perm": restore_perm,
        "bf16_island": bf16_island,
        "replacement_reduce_output": replacement_reduce_output,
    }


def rewrite_model(
    original_model: onnx.ModelProto,
    *,
    only_outputs: set[str],
    only_nodes: set[str],
    max_conversions: int | None,
    reducemean_channel_chunk: int,
    reducemean_max_chunk_elements: int,
    bf16_reducemean_island: bool,
    compile_jobs: list[dict[str, Any]] | None,
) -> tuple[
    onnx.ModelProto,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model = copy.deepcopy(original_model)
    inferred = infer_model_shapes(model)
    value_infos = collect_value_infos(inferred)
    shapes = collect_shapes(inferred)
    elem_types = collect_elem_types(inferred)
    constants = constant_tensor_values(inferred)
    producers = build_producers(model)
    consumers = build_consumers(model)
    output_names = graph_output_names(model)
    non_constant_indices = non_constant_index_by_node(model)

    reciprocal_report = []
    pow_report = []
    reducesum_report = []
    reducemean_report = []
    reciprocal_nodes = [node for node in model.graph.node if node.op_type == "Reciprocal"]
    pow_nodes = [node for node in model.graph.node if node.op_type == "Pow"]
    reducesum_nodes = [node for node in model.graph.node if node.op_type == "ReduceSum"]
    reducemean_nodes = [node for node in model.graph.node if node.op_type == "ReduceMean"]
    remove_node_ids: set[int] = set()
    replacement_nodes_by_id: dict[int, list[onnx.NodeProto]] = {}
    converted_count = 0

    for reciprocal_node in reciprocal_nodes:
        non_constant_index = non_constant_indices[id(reciprocal_node)]
        output_name = reciprocal_node.output[0] if reciprocal_node.output else ""

        if only_outputs and output_name not in only_outputs:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="filtered",
                    reason="output not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        if only_nodes and reciprocal_node.name not in only_nodes:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="filtered",
                    reason="node not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        if max_conversions is not None and converted_count >= max_conversions:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="filtered",
                    reason="max conversions reached",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        if len(reciprocal_node.input) != 1 or len(reciprocal_node.output) != 1:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason="Reciprocal must have one input and one output",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        sqrt_node = producers.get(reciprocal_node.input[0])
        if sqrt_node is None or sqrt_node.op_type != "Sqrt":
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason="input is not produced by Sqrt",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        if output_name in output_names:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason="Reciprocal output is a graph output",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        reciprocal_consumers = consumers.get(output_name, [])
        if not reciprocal_consumers:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason="Reciprocal output has no consumers",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        if any(node.op_type != "Mul" for node in reciprocal_consumers):
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason="Reciprocal output has non-Mul consumers",
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        rewrites = []
        skip_reason = None
        for mul_node in reciprocal_consumers:
            if len(mul_node.input) != 2 or len(mul_node.output) != 1:
                skip_reason = "Mul consumer must have two inputs and one output"
                break
            positions = [i for i, name in enumerate(mul_node.input) if name == output_name]
            if len(positions) != 1:
                skip_reason = "Mul consumer must use Reciprocal output exactly once"
                break
            numerator_input = mul_node.input[1 - positions[0]]
            rewrites.append((mul_node, numerator_input))

        if skip_reason is not None:
            reciprocal_report.append(
                candidate_row(
                    non_constant_index=non_constant_index,
                    reciprocal_node=reciprocal_node,
                    status="skipped",
                    reason=skip_reason,
                    shapes=shapes,
                    elem_types=elem_types,
                )
            )
            continue

        converted_mul_outputs = []
        for mul_node, numerator_input in rewrites:
            if compile_jobs is not None:
                compile_jobs.append(
                    make_rsqrt_compile_job(
                        source_model=inferred,
                        value_infos=value_infos,
                        non_constant_index=non_constant_index,
                        reciprocal_node=reciprocal_node,
                        mul_node=mul_node,
                        numerator_input=numerator_input,
                        sqrt_output=reciprocal_node.input[0],
                    )
                )

            converted_mul_outputs.append(mul_node.output[0])
            mul_node.op_type = "Div"
            mul_node.name = f"{node_label(mul_node)}_rsqrt_div"
            del mul_node.input[:]
            mul_node.input.extend([numerator_input, reciprocal_node.input[0]])
            del mul_node.attribute[:]

        remove_node_ids.add(id(reciprocal_node))
        converted_count += 1
        reciprocal_report.append(
            candidate_row(
                non_constant_index=non_constant_index,
                reciprocal_node=reciprocal_node,
                status="converted",
                reason="rewrote Sqrt->Reciprocal->Mul to Sqrt->Div",
                shapes=shapes,
                elem_types=elem_types,
                converted_mul_outputs=converted_mul_outputs,
            )
        )

    for pow_node in pow_nodes:
        non_constant_index = non_constant_indices[id(pow_node)]
        output_name = pow_node.output[0] if pow_node.output else ""
        exponent_value = (
            scalar_float_constant(pow_node.input[1], constants)
            if len(pow_node.input) > 1
            else None
        )
        exponent = supported_integer_power(exponent_value)

        if only_outputs and output_name not in only_outputs:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="filtered",
                    reason="output not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        if only_nodes and pow_node.name not in only_nodes:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="filtered",
                    reason="node not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        if max_conversions is not None and converted_count >= max_conversions:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="filtered",
                    reason="max conversions reached",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        if len(pow_node.input) != 2 or len(pow_node.output) != 1:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="skipped",
                    reason="Pow must have two inputs and one output",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        if exponent is None:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="skipped",
                    reason="Pow exponent is not a constant scalar 2 or 3",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        input_name = pow_node.input[0]
        output_shape = shapes.get(output_name)
        output_elem_type = elem_types.get(output_name, elem_types.get(input_name))
        if output_elem_type is None or output_elem_type == TensorProto.UNDEFINED:
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="skipped",
                    reason="missing Pow output element type",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue
        if not is_fully_static_shape(output_shape):
            pow_report.append(
                pow_candidate_row(
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    status="skipped",
                    reason="missing fully static Pow output shape",
                    shapes=shapes,
                    elem_types=elem_types,
                    exponent_value=exponent_value,
                    exponent=exponent,
                )
            )
            continue

        if exponent == 2:
            replacement_mul_outputs = [output_name]
            replacement_nodes = [
                helper.make_node(
                    "Mul",
                    [input_name, input_name],
                    [output_name],
                    name=f"{node_label(pow_node)}_pow2_mul",
                )
            ]
        else:
            square_output = unique_tensor_name(model, f"{output_name}_pow3_square")
            replacement_mul_outputs = [square_output, output_name]
            append_static_value_info(
                model,
                name=square_output,
                elem_type=output_elem_type,
                shape=output_shape,
            )
            replacement_nodes = [
                helper.make_node(
                    "Mul",
                    [input_name, input_name],
                    [square_output],
                    name=f"{node_label(pow_node)}_pow3_square_mul",
                ),
                helper.make_node(
                    "Mul",
                    [square_output, input_name],
                    [output_name],
                    name=f"{node_label(pow_node)}_pow3_mul",
                ),
            ]

        if compile_jobs is not None:
            compile_jobs.append(
                make_pow_compile_job(
                    source_model=inferred,
                    value_infos=value_infos,
                    shapes=shapes,
                    elem_types=elem_types,
                    non_constant_index=non_constant_index,
                    pow_node=pow_node,
                    exponent=exponent,
                    replacement_mul_outputs=replacement_mul_outputs,
                )
            )

        replacement_nodes_by_id[id(pow_node)] = replacement_nodes
        converted_count += 1
        pow_report.append(
            pow_candidate_row(
                non_constant_index=non_constant_index,
                pow_node=pow_node,
                status="converted",
                reason=f"rewrote Pow exponent {exponent} to Mul chain",
                shapes=shapes,
                elem_types=elem_types,
                exponent_value=exponent_value,
                exponent=exponent,
                replacement_mul_outputs=replacement_mul_outputs,
            )
        )

    for reduce_node in reducesum_nodes:
        non_constant_index = non_constant_indices[id(reduce_node)]
        output_name = reduce_node.output[0] if reduce_node.output else ""
        input_name = reduce_node.input[0] if reduce_node.input else ""
        axes = reduce_mean_axes(reduce_node, constants)

        if only_outputs and output_name not in only_outputs:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="output not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if only_nodes and reduce_node.name not in only_nodes:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="node not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if max_conversions is not None and converted_count >= max_conversions:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="max conversions reached",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if default_opset_version(model) < 18:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceSum rewrite requires opset >= 18",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if len(reduce_node.input) < 1 or len(reduce_node.output) != 1:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceSum must have at least one input and one output",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        input_shape = shapes.get(input_name)
        output_shape = shapes.get(output_name)
        input_elem_type = elem_types.get(input_name)
        output_elem_type = elem_types.get(output_name, input_elem_type)
        if not is_fully_static_shape(input_shape) or not is_fully_static_shape(output_shape):
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="missing fully static ReduceSum input/output shape",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if output_elem_type not in FLOAT_ELEM_TYPES:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceSum rewrite only handles floating-point tensors",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        assert input_shape is not None
        assert output_shape is not None
        use_bf16_reducesum_island = (
            bf16_reducemean_island
            and input_elem_type == TensorProto.FLOAT
            and output_elem_type == TensorProto.FLOAT
        )
        scale_value, normalized_axes_value, scale_error = reducesum_scale_value(
            input_shape=input_shape,
            axes=axes,
            noop_with_empty_axes=node_int_attribute(
                reduce_node, "noop_with_empty_axes", 0
            ),
        )
        if scale_error is not None or scale_value is None or normalized_axes_value is None:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason=scale_error or "missing ReduceSum scale value",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                    normalized_axes_value=normalized_axes_value,
                )
            )
            continue

        try:
            components = make_reducesum_mean_mul_components(
                model_for_names=model,
                reduce_node=reduce_node,
                input_shape=input_shape,
                output_shape=output_shape,
                elem_type=output_elem_type,
                normalized_axes_value=normalized_axes_value,
                scale_value=scale_value,
                bf16_island=use_bf16_reducesum_island,
            )
        except Exception as exc:
            reducesum_report.append(
                reducesum_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason=str(exc),
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                    normalized_axes_value=normalized_axes_value,
                    scale_value=scale_value,
                )
            )
            continue

        if compile_jobs is not None:
            compile_jobs.append(
                make_reducesum_compile_job(
                    source_model=inferred,
                    value_infos=value_infos,
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    elem_type=output_elem_type,
                    axes=normalized_axes_value,
                    scale_value=scale_value,
                    bf16_island=use_bf16_reducesum_island,
                )
            )

        model.graph.initializer.extend(components["initializers"])
        model.graph.value_info.extend(components["value_infos"])
        replacement_nodes_by_id[id(reduce_node)] = components["nodes"]
        converted_count += 1
        reducesum_report.append(
            reducesum_candidate_row(
                non_constant_index=non_constant_index,
                reduce_node=reduce_node,
                status="converted",
                reason="rewrote ReduceSum to ReduceMean times reduced element count",
                shapes=shapes,
                elem_types=elem_types,
                axes=axes,
                normalized_axes_value=normalized_axes_value,
                scale_value=scale_value,
                mean_output=components["mean_output"],
                bf16_island=use_bf16_reducesum_island,
            )
        )

    for reduce_node in reducemean_nodes:
        non_constant_index = non_constant_indices[id(reduce_node)]
        output_name = reduce_node.output[0] if reduce_node.output else ""
        input_name = reduce_node.input[0] if reduce_node.input else ""
        axes = reduce_mean_axes(reduce_node, constants)

        if only_outputs and output_name not in only_outputs:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="output not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if only_nodes and reduce_node.name not in only_nodes:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="node not selected",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if reducemean_channel_chunk == 0:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="ReduceMean split disabled",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if max_conversions is not None and converted_count >= max_conversions:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="filtered",
                    reason="max conversions reached",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if default_opset_version(model) < 18:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean split requires opset >= 18",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if len(reduce_node.input) < 1 or len(reduce_node.output) != 1:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean must have at least one input and one output",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        input_shape = shapes.get(input_name)
        output_shape = shapes.get(output_name)
        input_elem_type = elem_types.get(input_name)
        output_elem_type = elem_types.get(output_name, input_elem_type)
        if not is_fully_static_shape(input_shape) or not is_fully_static_shape(output_shape):
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="missing fully static ReduceMean input/output shape",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        assert input_shape is not None
        assert output_shape is not None
        if len(input_shape) != 3 or len(output_shape) != 3:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean split only handles rank-3 tensors",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if axes is None or len(axes) != 1:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean axes must be a constant single axis",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        axis = normalize_axis(axes[0], len(input_shape))
        if axis is None:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean axis is out of range",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if node_int_attribute(reduce_node, "keepdims", 1) != 1:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean rewrite requires keepdims=1",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        input_dims = [int(dim) for dim in input_shape]
        expected_output_shape = list(input_dims)
        expected_output_shape[axis] = 1
        if list(int(dim) for dim in output_shape) != expected_output_shape:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="ReduceMean output shape does not match keepdims single-axis mean",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        if output_elem_type is None or output_elem_type == TensorProto.UNDEFINED:
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason="missing ReduceMean output element type",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                )
            )
            continue

        use_bf16_reducemean_island = (
            bf16_reducemean_island
            and input_elem_type == TensorProto.FLOAT
            and output_elem_type == TensorProto.FLOAT
        )

        if axis != len(input_shape) - 1:
            components = make_reducemean_transpose_components(
                model_for_names=model,
                reduce_node=reduce_node,
                input_shape=input_shape,
                output_shape=output_shape,
                elem_type=output_elem_type,
                axis=axis,
                bf16_island=use_bf16_reducemean_island,
            )

            if compile_jobs is not None:
                compile_jobs.append(
                    make_reducemean_transpose_compile_job(
                        source_model=inferred,
                        value_infos=value_infos,
                        non_constant_index=non_constant_index,
                        reduce_node=reduce_node,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        elem_type=output_elem_type,
                        axis=axis,
                        bf16_island=use_bf16_reducemean_island,
                    )
                )

            model.graph.initializer.extend(components["initializers"])
            model.graph.value_info.extend(components["value_infos"])
            replacement_nodes_by_id[id(reduce_node)] = components["nodes"]
            converted_count += 1
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="converted",
                    reason="moved ReduceMean axis to last axis with Transpose",
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                    transpose_perm=components["perm"],
                    restore_perm=components["restore_perm"],
                    bf16_island=use_bf16_reducemean_island,
                )
            )
            continue

        n_dim, c_dim, l_dim = input_dims
        max_chunk_by_elements = max(1, reducemean_max_chunk_elements // max(1, l_dim))
        effective_chunk_size = min(c_dim, reducemean_channel_chunk, max_chunk_by_elements)

        if effective_chunk_size >= c_dim:
            if use_bf16_reducemean_island:
                components = make_reducemean_bf16_island_components(
                    model_for_names=model,
                    reduce_node=reduce_node,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    output_elem_type=output_elem_type,
                )
                if compile_jobs is not None:
                    compile_jobs.append(
                        make_reducemean_within_chunk_compile_job(
                            source_model=inferred,
                            value_infos=value_infos,
                            non_constant_index=non_constant_index,
                            reduce_node=reduce_node,
                            chunk_size=effective_chunk_size,
                            input_shape=input_shape,
                            output_shape=output_shape,
                            output_elem_type=output_elem_type,
                            bf16_island=True,
                        )
                    )
                model.graph.value_info.extend(components["value_infos"])
                replacement_nodes_by_id[id(reduce_node)] = components["nodes"]
                converted_count += 1
                reducemean_report.append(
                    reducemean_candidate_row(
                        non_constant_index=non_constant_index,
                        reduce_node=reduce_node,
                        status="converted",
                        reason="wrapped within-chunk ReduceMean in BF16 island",
                        shapes=shapes,
                        elem_types=elem_types,
                        axes=axes,
                        chunk_size=effective_chunk_size,
                        chunk_count=1,
                        bf16_island=True,
                        replacement_reduce_output=components["reduce_output"],
                    )
                )
                continue

            if (
                compile_jobs is not None
                and output_elem_type is not None
                and output_elem_type != TensorProto.UNDEFINED
            ):
                compile_jobs.append(
                    make_reducemean_within_chunk_compile_job(
                        source_model=inferred,
                        value_infos=value_infos,
                        non_constant_index=non_constant_index,
                        reduce_node=reduce_node,
                        chunk_size=effective_chunk_size,
                    )
                )
            reducemean_report.append(
                reducemean_candidate_row(
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    status="skipped",
                    reason=REDUCEMEAN_WITHIN_CHUNK_REASON,
                    shapes=shapes,
                    elem_types=elem_types,
                    axes=axes,
                    chunk_size=effective_chunk_size,
                    chunk_count=1,
                )
            )
            continue

        components = make_reducemean_split_components(
            model_for_names=model,
            reduce_node=reduce_node,
            input_shape=input_shape,
            output_shape=output_shape,
            elem_type=output_elem_type,
            chunk_size=effective_chunk_size,
            bf16_island=use_bf16_reducemean_island,
        )

        if compile_jobs is not None:
            compile_jobs.append(
                make_reducemean_compile_job(
                    source_model=inferred,
                    value_infos=value_infos,
                    non_constant_index=non_constant_index,
                    reduce_node=reduce_node,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    elem_type=output_elem_type,
                    chunk_size=effective_chunk_size,
                    bf16_island=use_bf16_reducemean_island,
                )
            )

        model.graph.initializer.extend(components["initializers"])
        model.graph.value_info.extend(components["value_infos"])
        replacement_nodes_by_id[id(reduce_node)] = components["nodes"]
        converted_count += 1
        reducemean_report.append(
            reducemean_candidate_row(
                non_constant_index=non_constant_index,
                reduce_node=reduce_node,
                status="converted",
                reason="split large rank-3 last-axis ReduceMean across channel chunks",
                shapes=shapes,
                elem_types=elem_types,
                axes=axes,
                chunk_size=effective_chunk_size,
                chunk_count=components["chunk_count"],
                bf16_island=use_bf16_reducemean_island,
            )
        )

    if remove_node_ids or replacement_nodes_by_id:
        new_nodes = []
        for node in model.graph.node:
            if id(node) in remove_node_ids:
                continue
            replacements = replacement_nodes_by_id.get(id(node))
            if replacements is not None:
                new_nodes.extend(replacements)
            else:
                new_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return model, reciprocal_report, pow_report, reducesum_report, reducemean_report


def ensure_compile_tools_available() -> None:
    if importlib.util.find_spec("iree.compiler.tools.import_onnx") is None:
        raise RuntimeError(
            "--nss-compile-check requires iree.compiler.tools.import_onnx"
        )
    if shutil.which("torq-compile") is None:
        raise RuntimeError("--nss-compile-check requires torq-compile in PATH")


def tail_text(value: str, limit: int = COMPILE_STDERR_TAIL_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def run_command(cmd: list[str], *, cwd: Path, timeout: float) -> tuple[str, float, str]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return "timeout", elapsed, tail_text(stdout + stderr)

    elapsed = time.perf_counter() - start
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        return "success", elapsed, tail_text(output)
    return "error", elapsed, tail_text(output)


def normalize_layer_ids(layer_ids: str | list[str]) -> list[str]:
    if isinstance(layer_ids, str):
        return [layer_ids]
    return list(layer_ids)


def write_executor_map(path: Path, layer_ids: str | list[str]) -> None:
    payload = {
        "op_assignments": {
            layer_id_value: {
                "executor": "nss",
                "recommend_convert_dtypes": True,
            }
            for layer_id_value in normalize_layer_ids(layer_ids)
        }
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compile_one_model(
    *,
    model_bytes: bytes,
    layer_ids: str | list[str],
    workdir: Path,
    timeout: float,
) -> dict[str, Any]:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    onnx_path = workdir / "model.onnx"
    mlir_path = workdir / "model.mlir"
    vmfb_path = workdir / "model.vmfb"
    map_path = workdir / "executor_map.json"
    debug_dir = workdir / "debug"
    phases_dir = workdir / "phases"
    onnx_path.write_bytes(model_bytes)
    write_executor_map(map_path, layer_ids)

    import_cmd = [
        sys.executable,
        "-m",
        "iree.compiler.tools.import_onnx",
        str(onnx_path),
        "-o",
        str(mlir_path),
        "--data-prop",
    ]
    import_status, import_elapsed, import_output = run_command(
        import_cmd, cwd=workdir, timeout=timeout
    )
    if import_status == "timeout":
        return {
            "status": "timeout",
            "step": "import_onnx",
            "elapsed_sec": round(import_elapsed, 3),
            "command": import_cmd,
            "stderr_tail": import_output,
        }
    if import_status != "success":
        return {
            "status": "import_error",
            "step": "import_onnx",
            "elapsed_sec": round(import_elapsed, 3),
            "command": import_cmd,
            "stderr_tail": import_output,
        }

    compile_cmd = [
        "torq-compile",
        str(mlir_path),
        "-o",
        str(vmfb_path),
        "--torq-hw=SL2610",
        "--torq-css-qemu",
        "--torq-target-host-triple=native",
        f"--torq-executor-map={map_path}",
        "--torq-disable-css",
        "--torq-disable-host",
        "--torq-enable-torq-hl-tiling",
        "--torq-convert-dtypes",
        f"--torq-debug-info={debug_dir}",
        f"--dump-compilation-phases-to={phases_dir}",
    ]
    compile_status, compile_elapsed, compile_output = run_command(
        compile_cmd, cwd=workdir, timeout=timeout
    )
    total_elapsed = import_elapsed + compile_elapsed
    if compile_status == "timeout":
        status = "timeout"
    elif compile_status == "success":
        status = "success"
    else:
        status = "compile_error"
    return {
        "status": status,
        "step": "torq_compile",
        "elapsed_sec": round(total_elapsed, 3),
        "command": compile_cmd,
        "stderr_tail": compile_output,
    }


def compile_job_worker(
    job: dict[str, Any],
    *,
    timeout: float,
    artifacts_dir: str | None,
) -> dict[str, Any]:
    if artifacts_dir:
        job_dir = Path(artifacts_dir) / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix=f"{job['kind']}_{job['job_id']}_")
        job_dir = Path(cleanup.name)

    try:
        if job["kind"] == "pow_mul":
            pow_result = compile_one_model(
                model_bytes=job["pow_model"],
                layer_ids=job["pow_layer_id"],
                workdir=job_dir / "pow",
                timeout=timeout,
            )
            mul_result = compile_one_model(
                model_bytes=job["mul_model"],
                layer_ids=job["mul_layer_ids"],
                workdir=job_dir / "mul",
                timeout=timeout,
            )
            return {
                "kind": "pow_mul",
                "job_id": job["job_id"],
                "non_constant_index": job["non_constant_index"],
                "pow_output": job["pow_output"],
                "mul_output": job["mul_output"],
                "exponent": job["exponent"],
                "pow_layer_id": job["pow_layer_id"],
                "mul_layer_ids": job["mul_layer_ids"],
                "pow": pow_result,
                "mul": mul_result,
                "artifacts_dir": str(job_dir) if artifacts_dir else None,
            }

        if job["kind"] == "reducemean_split":
            reducemean_result = compile_one_model(
                model_bytes=job["reducemean_model"],
                layer_ids=job["reducemean_layer_id"],
                workdir=job_dir / "reducemean",
                timeout=timeout,
            )
            split_result = compile_one_model(
                model_bytes=job["split_model"],
                layer_ids=job["split_layer_ids"],
                workdir=job_dir / "split",
                timeout=timeout,
            )
            return {
                "kind": "reducemean_split",
                "job_id": job["job_id"],
                "non_constant_index": job["non_constant_index"],
                "reducemean_output": job["reducemean_output"],
                "split_output": job["split_output"],
                "chunk_size": job["chunk_size"],
                "reducemean_layer_id": job["reducemean_layer_id"],
                "split_layer_ids": job["split_layer_ids"],
                "reducemean": reducemean_result,
                "split": split_result,
                "artifacts_dir": str(job_dir) if artifacts_dir else None,
            }

        if job["kind"] == "reducesum_mean_mul":
            reducesum_result = compile_one_model(
                model_bytes=job["reducesum_model"],
                layer_ids=job["reducesum_layer_id"],
                workdir=job_dir / "reducesum",
                timeout=timeout,
            )
            mean_mul_result = compile_one_model(
                model_bytes=job["mean_mul_model"],
                layer_ids=job["mean_mul_layer_ids"],
                workdir=job_dir / "mean_mul",
                timeout=timeout,
            )
            return {
                "kind": "reducesum_mean_mul",
                "job_id": job["job_id"],
                "non_constant_index": job["non_constant_index"],
                "reducesum_output": job["reducesum_output"],
                "mean_mul_output": job["mean_mul_output"],
                "axes": job["axes"],
                "scale_value": job["scale_value"],
                "bf16_island": job.get("bf16_island", False),
                "reducesum_layer_id": job["reducesum_layer_id"],
                "mean_mul_layer_ids": job["mean_mul_layer_ids"],
                "reducesum": reducesum_result,
                "mean_mul": mean_mul_result,
                "artifacts_dir": str(job_dir) if artifacts_dir else None,
            }

        if job["kind"] == "reducemean_within_chunk":
            reducemean_result = compile_one_model(
                model_bytes=job["reducemean_model"],
                layer_ids=job["reducemean_layer_id"],
                workdir=job_dir / "reducemean",
                timeout=timeout,
            )
            return {
                "kind": "reducemean_within_chunk",
                "job_id": job["job_id"],
                "non_constant_index": job["non_constant_index"],
                "reducemean_output": job["reducemean_output"],
                "chunk_size": job["chunk_size"],
                "chunk_count": job["chunk_count"],
                "reducemean_layer_id": job["reducemean_layer_id"],
                "reducemean": reducemean_result,
                "artifacts_dir": str(job_dir) if artifacts_dir else None,
            }

        if job["kind"] == "reducemean_transpose_axis":
            reducemean_result = compile_one_model(
                model_bytes=job["reducemean_model"],
                layer_ids=job["reducemean_layer_id"],
                workdir=job_dir / "reducemean",
                timeout=timeout,
            )
            transpose_result = compile_one_model(
                model_bytes=job["transpose_model"],
                layer_ids=job["transpose_layer_ids"],
                workdir=job_dir / "transpose",
                timeout=timeout,
            )
            return {
                "kind": "reducemean_transpose_axis",
                "job_id": job["job_id"],
                "non_constant_index": job["non_constant_index"],
                "reducemean_output": job["reducemean_output"],
                "transpose_output": job["transpose_output"],
                "axis": job["axis"],
                "perm": job["perm"],
                "restore_perm": job["restore_perm"],
                "reducemean_layer_id": job["reducemean_layer_id"],
                "transpose_layer_ids": job["transpose_layer_ids"],
                "reducemean": reducemean_result,
                "transpose": transpose_result,
                "artifacts_dir": str(job_dir) if artifacts_dir else None,
            }

        reciprocal_result = compile_one_model(
            model_bytes=job["reciprocal_model"],
            layer_ids=job["reciprocal_layer_id"],
            workdir=job_dir / "reciprocal",
            timeout=timeout,
        )
        div_result = compile_one_model(
            model_bytes=job["div_model"],
            layer_ids=job["div_layer_id"],
            workdir=job_dir / "div",
            timeout=timeout,
        )
        return {
            "kind": "rsqrt_div",
            "job_id": job["job_id"],
            "non_constant_index": job["non_constant_index"],
            "reciprocal_output": job["reciprocal_output"],
            "div_output": job["div_output"],
            "reciprocal_layer_id": job["reciprocal_layer_id"],
            "div_layer_id": job["div_layer_id"],
            "reciprocal": reciprocal_result,
            "div": div_result,
            "artifacts_dir": str(job_dir) if artifacts_dir else None,
        }
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def error_or_success(status: str) -> str:
    return "success" if status == "success" else "error"


def summarize_compile_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rsqrt_rows = [row for row in rows if row.get("kind") == "rsqrt_div"]
    pow_rows = [row for row in rows if row.get("kind") == "pow_mul"]
    reducesum_rows = [row for row in rows if row.get("kind") == "reducesum_mean_mul"]
    reducemean_rows = [row for row in rows if row.get("kind") == "reducemean_split"]
    reducemean_within_chunk_rows = [
        row for row in rows if row.get("kind") == "reducemean_within_chunk"
    ]
    reducemean_transpose_rows = [
        row for row in rows if row.get("kind") == "reducemean_transpose_axis"
    ]
    reciprocal_counts = Counter(row["reciprocal"]["status"] for row in rsqrt_rows)
    div_counts = Counter(row["div"]["status"] for row in rsqrt_rows)
    rsqrt_pair_counts = Counter(
        f"reciprocal_{error_or_success(row['reciprocal']['status'])}"
        f" -> div_{error_or_success(row['div']['status'])}"
        for row in rsqrt_rows
    )
    pow_counts = Counter(row["pow"]["status"] for row in pow_rows)
    mul_counts = Counter(row["mul"]["status"] for row in pow_rows)
    pow_pair_counts = Counter(
        f"pow_{error_or_success(row['pow']['status'])}"
        f" -> mul_{error_or_success(row['mul']['status'])}"
        for row in pow_rows
    )
    reducesum_counts = Counter(row["reducesum"]["status"] for row in reducesum_rows)
    mean_mul_counts = Counter(row["mean_mul"]["status"] for row in reducesum_rows)
    reducesum_pair_counts = Counter(
        f"reducesum_{error_or_success(row['reducesum']['status'])}"
        f" -> mean_mul_{error_or_success(row['mean_mul']['status'])}"
        for row in reducesum_rows
    )
    reducemean_counts = Counter(row["reducemean"]["status"] for row in reducemean_rows)
    split_counts = Counter(row["split"]["status"] for row in reducemean_rows)
    reducemean_pair_counts = Counter(
        f"reducemean_{error_or_success(row['reducemean']['status'])}"
        f" -> split_{error_or_success(row['split']['status'])}"
        for row in reducemean_rows
    )
    reducemean_within_chunk_counts = Counter(
        row["reducemean"]["status"] for row in reducemean_within_chunk_rows
    )
    reducemean_transpose_counts = Counter(
        row["reducemean"]["status"] for row in reducemean_transpose_rows
    )
    transpose_counts = Counter(row["transpose"]["status"] for row in reducemean_transpose_rows)
    reducemean_transpose_pair_counts = Counter(
        f"reducemean_{error_or_success(row['reducemean']['status'])}"
        f" -> transpose_axis_{error_or_success(row['transpose']['status'])}"
        for row in reducemean_transpose_rows
    )
    return {
        "total": len(rows),
        "rsqrt_div_total": len(rsqrt_rows),
        "pow_mul_total": len(pow_rows),
        "reducesum_mean_mul_total": len(reducesum_rows),
        "reducemean_split_total": len(reducemean_rows),
        "reducemean_within_chunk_total": len(reducemean_within_chunk_rows),
        "reducemean_transpose_axis_total": len(reducemean_transpose_rows),
        "reciprocal_status_counts": dict(sorted(reciprocal_counts.items())),
        "div_status_counts": dict(sorted(div_counts.items())),
        "pair_counts": dict(sorted(rsqrt_pair_counts.items())),
        "pow_status_counts": dict(sorted(pow_counts.items())),
        "mul_status_counts": dict(sorted(mul_counts.items())),
        "pow_pair_counts": dict(sorted(pow_pair_counts.items())),
        "reducesum_status_counts": dict(sorted(reducesum_counts.items())),
        "mean_mul_status_counts": dict(sorted(mean_mul_counts.items())),
        "reducesum_pair_counts": dict(sorted(reducesum_pair_counts.items())),
        "reducemean_status_counts": dict(sorted(reducemean_counts.items())),
        "split_reducemean_status_counts": dict(sorted(split_counts.items())),
        "reducemean_pair_counts": dict(sorted(reducemean_pair_counts.items())),
        "within_chunk_reducemean_status_counts": dict(
            sorted(reducemean_within_chunk_counts.items())
        ),
        "transpose_axis_reducemean_status_counts": dict(
            sorted(reducemean_transpose_counts.items())
        ),
        "transpose_axis_replacement_status_counts": dict(sorted(transpose_counts.items())),
        "transpose_axis_pair_counts": dict(sorted(reducemean_transpose_pair_counts.items())),
    }


def run_nss_compile_checks(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
    timeout: float,
    artifacts_dir: Path | None,
) -> dict[str, Any]:
    ensure_compile_tools_available()
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                compile_job_worker,
                job,
                timeout=timeout,
                artifacts_dir=str(artifacts_dir) if artifacts_dir else None,
            )
            for job in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())

    rows.sort(
        key=lambda row: (
            row["non_constant_index"],
            row.get("div_output")
            or row.get("mul_output")
            or row.get("mean_mul_output")
            or row.get("split_output")
            or row.get("transpose_output")
            or row.get("reducemean_output")
            or "",
        )
    )
    return {
        "summary": summarize_compile_results(rows),
        "rows": rows,
    }


def print_nss_compile_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    print()
    print("NSS compile-check summary")
    print(f"NSS compile rows tested: {summary['total']}")

    if summary["rsqrt_div_total"]:
        print()
        print(f"Reciprocal -> Div rows tested: {summary['rsqrt_div_total']}")
        print("Original Reciprocal statuses:")
        for status, count in summary["reciprocal_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Replacement Div statuses:")
        for status, count in summary["div_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Pair outcomes:")
        for status, count in summary["pair_counts"].items():
            print(f"  {status:<34} {count}")

    if summary["pow_mul_total"]:
        print()
        print(f"Pow -> Mul rows tested: {summary['pow_mul_total']}")
        print("Original Pow statuses:")
        for status, count in summary["pow_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Replacement Mul statuses:")
        for status, count in summary["mul_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Pair outcomes:")
        for status, count in summary["pow_pair_counts"].items():
            print(f"  {status:<34} {count}")

    if summary["reducesum_mean_mul_total"]:
        print()
        print(
            "ReduceSum -> ReduceMean*scale rows tested: "
            f"{summary['reducesum_mean_mul_total']}"
        )
        print("Original ReduceSum statuses:")
        for status, count in summary["reducesum_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Replacement ReduceMean*scale statuses:")
        for status, count in summary["mean_mul_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Pair outcomes:")
        for status, count in summary["reducesum_pair_counts"].items():
            print(f"  {status:<34} {count}")

    if summary["reducemean_split_total"]:
        print()
        print(f"ReduceMean -> split ReduceMean rows tested: {summary['reducemean_split_total']}")
        print("Original ReduceMean statuses:")
        for status, count in summary["reducemean_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Replacement split ReduceMean statuses:")
        for status, count in summary["split_reducemean_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Pair outcomes:")
        for status, count in summary["reducemean_pair_counts"].items():
            print(f"  {status:<34} {count}")

    if summary["reducemean_transpose_axis_total"]:
        print()
        print(
            "ReduceMean -> transpose-axis ReduceMean rows tested: "
            f"{summary['reducemean_transpose_axis_total']}"
        )
        print("Original ReduceMean statuses:")
        for status, count in summary["transpose_axis_reducemean_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Replacement transpose-axis statuses:")
        for status, count in summary["transpose_axis_replacement_status_counts"].items():
            print(f"  {status:<14} {count}")
        print("Pair outcomes:")
        for status, count in summary["transpose_axis_pair_counts"].items():
            print(f"  {status:<34} {count}")

    if summary["reducemean_within_chunk_total"]:
        print()
        print(
            "Within-chunk ReduceMean sanity rows tested: "
            f"{summary['reducemean_within_chunk_total']}"
        )
        print("Within-chunk ReduceMean statuses:")
        for status, count in summary["within_chunk_reducemean_status_counts"].items():
            print(f"  {status:<14} {count}")


def ort_dtype(type_name: str) -> np.dtype:
    if type_name in FLOAT_TENSOR_TYPES:
        return np.dtype(FLOAT_TENSOR_TYPES[type_name])
    if type_name in INT_TENSOR_TYPES:
        return np.dtype(INT_TENSOR_TYPES[type_name])
    if type_name == "tensor(bool)":
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported ONNX Runtime input type for random data: {type_name}")


def concrete_shape(shape: list[Any], dynamic_dim_value: int) -> tuple[int, ...]:
    dims = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            dims.append(dynamic_dim_value)
    return tuple(dims)


def random_input(rng: np.random.Generator, *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return rng.integers(0, 2, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 8, size=shape, dtype=dtype)
    raise ValueError(f"Unsupported random dtype: {dtype}")


def verify_random_outputs(
    original_model: onnx.ModelProto,
    rewritten_model: onnx.ModelProto,
    *,
    seed: int,
    dynamic_dim_value: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "--verify-random requires onnxruntime in the current Python environment"
        ) from exc

    providers = ["CPUExecutionProvider"]
    original_session = ort.InferenceSession(
        original_model.SerializeToString(), providers=providers
    )
    rewritten_session = ort.InferenceSession(
        rewritten_model.SerializeToString(), providers=providers
    )

    rng = np.random.default_rng(seed)
    inputs = {}
    for input_info in original_session.get_inputs():
        dtype = ort_dtype(input_info.type)
        shape = concrete_shape(list(input_info.shape), dynamic_dim_value)
        inputs[input_info.name] = random_input(rng, shape=shape, dtype=dtype)

    output_names = [output.name for output in original_session.get_outputs()]
    original_outputs = original_session.run(output_names, inputs)
    rewritten_outputs = rewritten_session.run(output_names, inputs)

    failures = []
    output_stats = []
    for name, original, rewritten in zip(output_names, original_outputs, rewritten_outputs):
        if original.shape != rewritten.shape:
            failures.append(f"{name}: shape {original.shape} != {rewritten.shape}")
            output_stats.append({"name": name, "shape_mismatch": True})
            continue

        if np.issubdtype(original.dtype, np.floating):
            abs_diff = np.abs(original.astype(np.float64) - rewritten.astype(np.float64))
            max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
            close = np.allclose(original, rewritten, atol=atol, rtol=rtol, equal_nan=True)
            output_stats.append(
                {
                    "name": name,
                    "shape": list(original.shape),
                    "dtype": str(original.dtype),
                    "max_abs_diff": max_abs_diff,
                    "allclose": bool(close),
                }
            )
            if not close:
                failures.append(f"{name}: max_abs_diff={max_abs_diff}")
        else:
            equal = np.array_equal(original, rewritten)
            output_stats.append(
                {
                    "name": name,
                    "shape": list(original.shape),
                    "dtype": str(original.dtype),
                    "equal": bool(equal),
                }
            )
            if not equal:
                failures.append(f"{name}: values differ")

    if failures:
        raise RuntimeError("Random verification failed: " + "; ".join(failures))

    return {"inputs": list(inputs), "outputs": output_stats}


def summarize_report(
    reciprocal_report: list[dict[str, Any]],
    pow_report: list[dict[str, Any]],
    reducesum_report: list[dict[str, Any]],
    reducemean_report: list[dict[str, Any]],
) -> dict[str, int]:
    reciprocal_counts = Counter(row["status"] for row in reciprocal_report)
    pow_counts = Counter(row["status"] for row in pow_report)
    reducesum_counts = Counter(row["status"] for row in reducesum_report)
    reducemean_counts = Counter(row["status"] for row in reducemean_report)
    transpose_axis_reducemean_count = sum(
        1
        for row in reducemean_report
        if row["status"] == "converted" and row.get("transpose_perm")
    )
    return {
        "total_reciprocal": len(reciprocal_report),
        "total_pow": len(pow_report),
        "total_reducesum": len(reducesum_report),
        "total_reducemean": len(reducemean_report),
        "converted": (
            reciprocal_counts["converted"]
            + pow_counts["converted"]
            + reducesum_counts["converted"]
            + reducemean_counts["converted"]
        ),
        "skipped": (
            reciprocal_counts["skipped"]
            + pow_counts["skipped"]
            + reducesum_counts["skipped"]
            + reducemean_counts["skipped"]
        ),
        "filtered": (
            reciprocal_counts["filtered"]
            + pow_counts["filtered"]
            + reducesum_counts["filtered"]
            + reducemean_counts["filtered"]
        ),
        "reciprocal_converted": reciprocal_counts["converted"],
        "reciprocal_skipped": reciprocal_counts["skipped"],
        "reciprocal_filtered": reciprocal_counts["filtered"],
        "pow_converted": pow_counts["converted"],
        "pow_skipped": pow_counts["skipped"],
        "pow_filtered": pow_counts["filtered"],
        "reducesum_converted": reducesum_counts["converted"],
        "reducesum_skipped": reducesum_counts["skipped"],
        "reducesum_filtered": reducesum_counts["filtered"],
        "reducemean_converted": reducemean_counts["converted"],
        "reducemean_skipped": reducemean_counts["skipped"],
        "reducemean_filtered": reducemean_counts["filtered"],
        "div_nodes_created": sum(
            len(row.get("converted_mul_outputs", []))
            for row in reciprocal_report
            if row["status"] == "converted"
        ),
        "mul_nodes_created": sum(
            len(row.get("replacement_mul_outputs", []))
            for row in pow_report
            if row["status"] == "converted"
        ),
        "reducesum_scale_mul_nodes_created": sum(
            1 for row in reducesum_report if row["status"] == "converted"
        ),
        "split_reducemean_nodes_created": sum(
            int(row.get("chunk_count") or 0)
            for row in reducemean_report
            if row["status"] == "converted"
        ),
        "transpose_axis_reducemean_converted": transpose_axis_reducemean_count,
        "transpose_nodes_created": 2 * transpose_axis_reducemean_count,
    }


def print_summary(
    reciprocal_report: list[dict[str, Any]],
    pow_report: list[dict[str, Any]],
    reducesum_report: list[dict[str, Any]],
    reducemean_report: list[dict[str, Any]],
    *,
    verbose: bool,
) -> None:
    summary = summarize_report(
        reciprocal_report, pow_report, reducesum_report, reducemean_report
    )
    print(
        "Reciprocal nodes: {total_reciprocal}, converted: {reciprocal_converted}, "
        "skipped: {reciprocal_skipped}, filtered: {reciprocal_filtered}, "
        "Div nodes created: {div_nodes_created}".format(**summary)
    )
    print(
        "Pow nodes: {total_pow}, converted: {pow_converted}, "
        "skipped: {pow_skipped}, filtered: {pow_filtered}, "
        "Mul nodes created: {mul_nodes_created}".format(**summary)
    )
    print(
        "ReduceSum nodes: {total_reducesum}, converted: {reducesum_converted}, "
        "skipped: {reducesum_skipped}, filtered: {reducesum_filtered}, "
        "Mul nodes created: {reducesum_scale_mul_nodes_created}".format(**summary)
    )
    print(
        "ReduceMean nodes: {total_reducemean}, converted: {reducemean_converted}, "
        "skipped: {reducemean_skipped}, filtered: {reducemean_filtered}, "
        "split ReduceMean nodes created: {split_reducemean_nodes_created}, "
        "transpose-axis rewrites: {transpose_axis_reducemean_converted}".format(
            **summary
        )
    )

    skip_reasons = Counter(
        row["reason"] for row in reciprocal_report if row["status"] == "skipped"
    )
    if skip_reasons:
        print("Top Reciprocal skip reasons:")
        for reason, count in skip_reasons.most_common(8):
            print(f"  {count:4d}  {reason}")

    pow_skip_reasons = Counter(
        row["reason"] for row in pow_report if row["status"] == "skipped"
    )
    if pow_skip_reasons:
        print("Top Pow skip reasons:")
        for reason, count in pow_skip_reasons.most_common(8):
            print(f"  {count:4d}  {reason}")

    reducesum_skip_reasons = Counter(
        row["reason"] for row in reducesum_report if row["status"] == "skipped"
    )
    if reducesum_skip_reasons:
        print("Top ReduceSum skip reasons:")
        for reason, count in reducesum_skip_reasons.most_common(8):
            print(f"  {count:4d}  {reason}")

    reducemean_skip_reasons = Counter(
        row["reason"] for row in reducemean_report if row["status"] == "skipped"
    )
    if reducemean_skip_reasons:
        print("Top ReduceMean skip reasons:")
        for reason, count in reducemean_skip_reasons.most_common(8):
            print(f"  {count:4d}  {reason}")

    if verbose:
        for row in reciprocal_report:
            print(
                "Reciprocal {status:9s} idx={non_constant_index:<5d} "
                "node={node_name!r} output={output!r} reason={reason}".format(**row)
            )
        for row in pow_report:
            print(
                "Pow        {status:9s} idx={non_constant_index:<5d} "
                "node={node_name!r} output={output!r} exponent={exponent!r} "
                "reason={reason}".format(**row)
            )
        for row in reducesum_report:
            print(
                "ReduceSum  {status:9s} idx={non_constant_index:<5d} "
                "node={node_name!r} output={output!r} axes={axes!r} "
                "reason={reason}".format(**row)
            )
        for row in reducemean_report:
            print(
                "ReduceMean {status:9s} idx={non_constant_index:<5d} "
                "node={node_name!r} output={output!r} axes={axes!r} "
                "reason={reason}".format(**row)
            )


def write_report_json(
    path: Path,
    reciprocal_report: list[dict[str, Any]],
    pow_report: list[dict[str, Any]],
    reducesum_report: list[dict[str, Any]],
    reducemean_report: list[dict[str, Any]],
    verification: Any,
    *,
    nss_compile_check: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_report(
            reciprocal_report, pow_report, reducesum_report, reducemean_report
        ),
        "verification": verification,
        "reciprocal_nodes": reciprocal_report,
        "pow_nodes": pow_report,
        "reducesum_nodes": reducesum_report,
        "reducemean_nodes": reducemean_report,
    }
    if nss_compile_check is not None:
        payload["nss_compile_check"] = nss_compile_check
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_compile_csv(path: Path, nss_compile_check: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "non_constant_index",
        "source_output",
        "replacement_output",
        "exponent",
        "source_layer_ids",
        "replacement_layer_ids",
        "source_status",
        "replacement_status",
        "source_elapsed_sec",
        "replacement_elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in nss_compile_check.get("rows", []):
            if row.get("kind") == "pow_mul":
                source = row["pow"]
                replacement = row["mul"]
                source_output = row["pow_output"]
                replacement_output = row["mul_output"]
                source_layer_ids = [row["pow_layer_id"]]
                replacement_layer_ids = row["mul_layer_ids"]
                exponent = row["exponent"]
            elif row.get("kind") == "reducemean_split":
                source = row["reducemean"]
                replacement = row["split"]
                source_output = row["reducemean_output"]
                replacement_output = row["split_output"]
                source_layer_ids = [row["reducemean_layer_id"]]
                replacement_layer_ids = row["split_layer_ids"]
                exponent = ""
            elif row.get("kind") == "reducesum_mean_mul":
                source = row["reducesum"]
                replacement = row["mean_mul"]
                source_output = row["reducesum_output"]
                replacement_output = row["mean_mul_output"]
                source_layer_ids = [row["reducesum_layer_id"]]
                replacement_layer_ids = row["mean_mul_layer_ids"]
                exponent = ""
            elif row.get("kind") == "reducemean_within_chunk":
                source = row["reducemean"]
                replacement = row["reducemean"]
                source_output = row["reducemean_output"]
                replacement_output = ""
                source_layer_ids = [row["reducemean_layer_id"]]
                replacement_layer_ids = []
                exponent = ""
            elif row.get("kind") == "reducemean_transpose_axis":
                source = row["reducemean"]
                replacement = row["transpose"]
                source_output = row["reducemean_output"]
                replacement_output = row["transpose_output"]
                source_layer_ids = [row["reducemean_layer_id"]]
                replacement_layer_ids = row["transpose_layer_ids"]
                exponent = ""
            else:
                source = row["reciprocal"]
                replacement = row["div"]
                source_output = row["reciprocal_output"]
                replacement_output = row["div_output"]
                source_layer_ids = [row["reciprocal_layer_id"]]
                replacement_layer_ids = [row["div_layer_id"]]
                exponent = ""
            writer.writerow(
                {
                    "kind": row.get("kind", "rsqrt_div"),
                    "non_constant_index": row["non_constant_index"],
                    "source_output": source_output,
                    "replacement_output": replacement_output,
                    "exponent": exponent,
                    "source_layer_ids": ",".join(source_layer_ids),
                    "replacement_layer_ids": ",".join(replacement_layer_ids),
                    "source_status": source["status"],
                    "replacement_status": replacement["status"],
                    "source_elapsed_sec": source.get("elapsed_sec"),
                    "replacement_elapsed_sec": replacement.get("elapsed_sec"),
                }
            )


def graph_io_signature(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    def value_signature(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
        shape = tensor_shape(value_info)
        return {
            "name": value_info.name,
            "elem_type": tensor_elem_type(value_info),
            "shape": list(shape) if shape is not None else None,
        }

    return {
        "inputs": [value_signature(value_info) for value_info in model.graph.input],
        "outputs": [value_signature(value_info) for value_info in model.graph.output],
    }


def validate_io_contract(
    original_model: onnx.ModelProto,
    rewritten_model: onnx.ModelProto,
    *,
    input_path: Path,
) -> dict[str, Any]:
    original_signature = graph_io_signature(original_model)
    rewritten_signature = graph_io_signature(rewritten_model)
    if original_signature != rewritten_signature:
        raise RuntimeError(
            "Graph input/output contract changed for "
            f"{input_path}. Before: {json.dumps(original_signature, sort_keys=True)} "
            f"After: {json.dumps(rewritten_signature, sort_keys=True)}"
        )
    return {
        "status": "passed",
        "input_count": len(original_signature["inputs"]),
        "output_count": len(original_signature["outputs"]),
    }


def conversion_summary_for_manifest(
    reciprocal_report: list[dict[str, Any]],
    pow_report: list[dict[str, Any]],
    reducesum_report: list[dict[str, Any]],
    reducemean_report: list[dict[str, Any]],
) -> dict[str, int]:
    return summarize_report(
        reciprocal_report, pow_report, reducesum_report, reducemean_report
    )


def convert_one_model(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path | None,
    report_json: Path | None,
    compile_artifacts_dir: Path | None,
) -> dict[str, Any]:
    original_model = onnx.load(str(input_path))
    compile_jobs: list[dict[str, Any]] | None = [] if args.nss_compile_check else None
    (
        rewritten_model,
        reciprocal_report,
        pow_report,
        reducesum_report,
        reducemean_report,
    ) = rewrite_model(
        original_model,
        only_outputs=set(args.only_output),
        only_nodes=set(args.only_node),
        max_conversions=args.max_conversions,
        reducemean_channel_chunk=args.reducemean_channel_chunk,
        reducemean_max_chunk_elements=args.reducemean_max_chunk_elements,
        bf16_reducemean_island=args.bf16_reducemean_island,
        compile_jobs=compile_jobs,
    )

    print_summary(
        reciprocal_report,
        pow_report,
        reducesum_report,
        reducemean_report,
        verbose=args.verbose,
    )

    nss_compile_check = None
    if args.nss_compile_check:
        assert compile_jobs is not None
        nss_compile_check = run_nss_compile_checks(
            compile_jobs,
            workers=args.compile_workers,
            timeout=args.compile_timeout,
            artifacts_dir=compile_artifacts_dir,
        )
        print_nss_compile_summary(nss_compile_check)

    verification = None
    verification_status = "not_run"
    if args.verify_random:
        verification = verify_random_outputs(
            original_model,
            rewritten_model,
            seed=args.seed,
            dynamic_dim_value=args.dynamic_dim_value,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(
            "Random verification passed "
            f"(rtol={args.rtol:g}, atol={args.atol:g}, seed={args.seed})"
        )
        verification_status = "passed"

    io_contract = validate_io_contract(
        original_model, rewritten_model, input_path=input_path
    )
    print("Input/output contract passed")

    if report_json:
        write_report_json(
            report_json,
            reciprocal_report,
            pow_report,
            reducesum_report,
            reducemean_report,
            verification,
            nss_compile_check=nss_compile_check,
        )
        if nss_compile_check is not None:
            write_compile_csv(report_json.with_suffix(".compile.csv"), nss_compile_check)
        print(f"Wrote report: {report_json}")

    if args.dry_run:
        checker_status = "skipped"
        saved_status = "dry_run"
    else:
        checker_status = "skipped"
        if args.check:
            checker.check_model(rewritten_model)
            checker_status = "passed"
            print("ONNX checker passed")

        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(rewritten_model, str(output_path))
        saved_status = "saved"
        print(f"Wrote model: {output_path}")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path is not None else None,
        "report_path": str(report_json) if report_json is not None else None,
        "summary": conversion_summary_for_manifest(
            reciprocal_report,
            pow_report,
            reducesum_report,
            reducemean_report,
        ),
        "verification_status": verification_status,
        "checker_status": checker_status,
        "io_contract": io_contract,
        "saved_status": saved_status,
    }


def write_folder_manifest(
    path: Path,
    *,
    input_dir: Path,
    output_dir: Path | None,
    model_results: list[dict[str, Any]],
) -> None:
    totals = Counter()
    checker_statuses = Counter()
    verification_statuses = Counter()
    io_contract_statuses = Counter()
    for result in model_results:
        totals.update(result["summary"])
        checker_statuses[result["checker_status"]] += 1
        verification_statuses[result["verification_status"]] += 1
        io_contract_statuses[result["io_contract"]["status"]] += 1

    summary_keys = [
        "total_reciprocal",
        "total_pow",
        "total_reducesum",
        "total_reducemean",
        "converted",
        "skipped",
        "filtered",
        "reciprocal_converted",
        "reciprocal_skipped",
        "reciprocal_filtered",
        "pow_converted",
        "pow_skipped",
        "pow_filtered",
        "reducesum_converted",
        "reducesum_skipped",
        "reducesum_filtered",
        "reducemean_converted",
        "reducemean_skipped",
        "reducemean_filtered",
        "div_nodes_created",
        "mul_nodes_created",
        "reducesum_scale_mul_nodes_created",
        "split_reducemean_nodes_created",
        "transpose_axis_reducemean_converted",
        "transpose_nodes_created",
    ]
    summary_payload = {key: totals[key] for key in summary_keys}
    summary_payload.update(
        {
            "checker_status_counts": dict(sorted(checker_statuses.items())),
            "verification_status_counts": dict(sorted(verification_statuses.items())),
            "io_contract_status_counts": dict(sorted(io_contract_statuses.items())),
        }
    )

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "model_count": len(model_results),
        "summary": summary_payload,
        "models": model_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def convert_folder(args: argparse.Namespace) -> int:
    input_dir = args.input
    output_dir = args.output
    model_paths = sorted(input_dir.glob("*.onnx"))
    if not model_paths:
        print(f"No .onnx files found in {input_dir}", file=sys.stderr)
        return 1

    if not args.dry_run:
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    model_results = []
    for index, input_path in enumerate(model_paths, start=1):
        print()
        print(f"[{index}/{len(model_paths)}] {input_path}")
        model_output_path = output_dir / input_path.name if output_dir is not None else None
        model_report_path = (
            output_dir / f"{input_path.stem}.normalization_ops.json"
            if args.report_json and output_dir is not None
            else None
        )
        model_artifacts_dir = (
            args.compile_artifacts_dir / safe_name(input_path.stem)
            if args.compile_artifacts_dir is not None
            else None
        )
        model_results.append(
            convert_one_model(
                args,
                input_path=input_path,
                output_path=model_output_path,
                report_json=model_report_path,
                compile_artifacts_dir=model_artifacts_dir,
            )
        )

    if args.report_json:
        write_folder_manifest(
            args.report_json,
            input_dir=input_dir,
            output_dir=output_dir,
            model_results=model_results,
        )
        print(f"Wrote folder manifest: {args.report_json}")

    return 0


def main() -> int:
    args = parse_args()

    if args.input.is_dir():
        return convert_folder(args)

    convert_one_model(
        args,
        input_path=args.input,
        output_path=args.output,
        report_json=args.report_json,
        compile_artifacts_dir=args.compile_artifacts_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
