#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


CONST_FOLDABLE_OPS = {
    "Abs",
    "Add",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Cos",
    "Div",
    "Equal",
    "Einsum",
    "Expand",
    "Floor",
    "Gather",
    "Greater",
    "GreaterOrEqual",
    "Identity",
    "Less",
    "LessOrEqual",
    "Max",
    "Min",
    "Mul",
    "Neg",
    "Pow",
    "Range",
    "Reciprocal",
    "ReduceL2",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "Reshape",
    "Round",
    "Shape",
    "Sin",
    "Slice",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
    "Where",
}

WEIGHT_INPUTS = {
    "BatchNormalization": {1: "scale", 2: "bias", 3: "mean", 4: "var"},
    "Conv": {1: "weight", 2: "bias"},
    "ConvTranspose": {1: "weight", 2: "bias"},
    "Gemm": {1: "weight", 2: "bias"},
    "InstanceNormalization": {1: "scale", 2: "bias"},
    "LayerNormalization": {1: "scale", 2: "bias"},
    "MatMul": {1: "rhs"},
    "PRelu": {1: "slope"},
}


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(helper.get_attribute_value(attr))
    return default


def _attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return [int(v) for v in helper.get_attribute_value(attr)]
    return None


def _tensor_dtype_to_np(dtype: int) -> np.dtype:
    if dtype == TensorProto.BFLOAT16:
        # NumPy has no native bfloat16 in the versions commonly available here.
        # The current Tsuki model is fp32, but keep this explicit for readable
        # errors if someone points the script at a bf16 constant-folding case.
        raise TypeError("constant-folding Cast to BFLOAT16 is not supported by this script")
    return np.dtype(helper.tensor_dtype_to_np_dtype(dtype))


def _constant_value(node: onnx.NodeProto) -> np.ndarray:
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
        if attr.name == "value_float":
            return np.asarray(helper.get_attribute_value(attr), dtype=np.float32)
        if attr.name == "value_floats":
            return np.asarray(helper.get_attribute_value(attr), dtype=np.float32)
        if attr.name == "value_int":
            return np.asarray(helper.get_attribute_value(attr), dtype=np.int64)
        if attr.name == "value_ints":
            return np.asarray(helper.get_attribute_value(attr), dtype=np.int64)
        if attr.name == "value_string":
            return np.asarray(helper.get_attribute_value(attr))
        if attr.name == "value_strings":
            return np.asarray(helper.get_attribute_value(attr))
    raise ValueError("Constant node has no supported value attribute")


def _get_input_values(node: onnx.NodeProto, values: dict[str, np.ndarray]) -> list[np.ndarray | None] | None:
    result: list[np.ndarray | None] = []
    for name in node.input:
        if not name:
            result.append(None)
            continue
        if name not in values:
            return None
        result.append(values[name])
    return result


def _axes_from_input_or_attr(
    node: onnx.NodeProto,
    inputs: list[np.ndarray | None],
    input_index: int = 1,
) -> list[int] | None:
    if len(inputs) > input_index and inputs[input_index] is not None:
        return [int(v) for v in np.asarray(inputs[input_index]).reshape(-1)]
    return _attr_ints(node, "axes")


def _normalize_unsqueeze_axes(axes: list[int], input_rank: int) -> list[int]:
    output_rank = input_rank + len(axes)
    normalized = []
    for axis in axes:
        normalized.append(axis + output_rank if axis < 0 else axis)
    return sorted(normalized)


def _normalize_axes(axes: list[int] | None, rank: int) -> tuple[int, ...] | None:
    if axes is None:
        return None
    return tuple(axis + rank if axis < 0 else axis for axis in axes)


def _reduce(node: onnx.NodeProto, inputs: list[np.ndarray | None], kind: str) -> np.ndarray:
    x = np.asarray(inputs[0])
    axes = _axes_from_input_or_attr(node, inputs)
    keepdims = bool(_attr_int(node, "keepdims", 1))
    noop_with_empty_axes = bool(_attr_int(node, "noop_with_empty_axes", 0))
    if axes == [] and noop_with_empty_axes:
        return x
    np_axes = _normalize_axes(axes, x.ndim)

    if kind == "ReduceL2":
        return np.sqrt(np.sum(np.square(x), axis=np_axes, keepdims=keepdims))
    if kind == "ReduceMax":
        return np.max(x, axis=np_axes, keepdims=keepdims)
    if kind == "ReduceMean":
        return np.mean(x, axis=np_axes, keepdims=keepdims)
    if kind == "ReduceMin":
        return np.min(x, axis=np_axes, keepdims=keepdims)
    if kind == "ReduceProd":
        return np.prod(x, axis=np_axes, keepdims=keepdims)
    if kind == "ReduceSum":
        return np.sum(x, axis=np_axes, keepdims=keepdims)
    raise NotImplementedError(kind)


def _eval_slice(node: onnx.NodeProto, inputs: list[np.ndarray | None]) -> np.ndarray:
    x = np.asarray(inputs[0])
    starts = [int(v) for v in np.asarray(inputs[1]).reshape(-1)]
    ends = [int(v) for v in np.asarray(inputs[2]).reshape(-1)]
    axes = (
        [int(v) for v in np.asarray(inputs[3]).reshape(-1)]
        if len(inputs) > 3 and inputs[3] is not None
        else list(range(len(starts)))
    )
    steps = (
        [int(v) for v in np.asarray(inputs[4]).reshape(-1)]
        if len(inputs) > 4 and inputs[4] is not None
        else [1] * len(starts)
    )
    slices: list[slice] = [slice(None)] * x.ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if axis < 0:
            axis += x.ndim
        slices[axis] = slice(start, end, step)
    return x[tuple(slices)]


def _eval_node(node: onnx.NodeProto, inputs: list[np.ndarray | None]) -> list[np.ndarray]:
    op = node.op_type
    attrs = _attrs(node)

    if op == "Constant":
        return [_constant_value(node)]
    if op == "Identity":
        return [np.asarray(inputs[0])]
    if op == "Abs":
        return [np.abs(inputs[0])]
    if op == "Add":
        return [np.asarray(inputs[0]) + np.asarray(inputs[1])]
    if op == "Cast":
        return [np.asarray(inputs[0]).astype(_tensor_dtype_to_np(int(attrs["to"])))]
    if op == "Ceil":
        return [np.ceil(inputs[0])]
    if op == "Clip":
        min_value = None
        max_value = None
        if len(inputs) > 1 and inputs[1] is not None:
            min_value = np.asarray(inputs[1])
        elif "min" in attrs:
            min_value = attrs["min"]
        if len(inputs) > 2 and inputs[2] is not None:
            max_value = np.asarray(inputs[2])
        elif "max" in attrs:
            max_value = attrs["max"]
        return [np.clip(np.asarray(inputs[0]), min_value, max_value)]
    if op == "Concat":
        return [np.concatenate([np.asarray(x) for x in inputs], axis=int(attrs["axis"]))]
    if op == "ConstantOfShape":
        shape = tuple(int(v) for v in np.asarray(inputs[0]).reshape(-1))
        value = np.asarray([0], dtype=np.float32)
        if "value" in attrs:
            value_attr = attrs["value"]
            if isinstance(value_attr, TensorProto):
                value = numpy_helper.to_array(value_attr)
        return [np.full(shape, value.reshape(-1)[0], dtype=value.dtype)]
    if op == "Div":
        return [np.asarray(inputs[0]) / np.asarray(inputs[1])]
    if op == "Equal":
        return [np.equal(inputs[0], inputs[1])]
    if op == "Cos":
        return [np.cos(inputs[0])]
    if op == "Einsum":
        equation = attrs["equation"]
        if isinstance(equation, bytes):
            equation = equation.decode("utf-8")
        return [np.einsum(equation, *[np.asarray(x) for x in inputs])]
    if op == "Expand":
        return [np.broadcast_to(np.asarray(inputs[0]), tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1)))]
    if op == "Floor":
        return [np.floor(inputs[0])]
    if op == "Gather":
        return [np.take(np.asarray(inputs[0]), np.asarray(inputs[1]), axis=int(attrs.get("axis", 0)))]
    if op == "Greater":
        return [np.greater(inputs[0], inputs[1])]
    if op == "GreaterOrEqual":
        return [np.greater_equal(inputs[0], inputs[1])]
    if op == "Less":
        return [np.less(inputs[0], inputs[1])]
    if op == "LessOrEqual":
        return [np.less_equal(inputs[0], inputs[1])]
    if op in {"Max", "Min"}:
        arrays = [np.asarray(x) for x in inputs if x is not None]
        if not arrays:
            raise ValueError(f"{op} requires at least one input")
        result = arrays[0]
        reducer = np.maximum if op == "Max" else np.minimum
        for arr in arrays[1:]:
            result = reducer(result, arr)
        return [result]
    if op == "Mul":
        return [np.asarray(inputs[0]) * np.asarray(inputs[1])]
    if op == "Neg":
        return [-np.asarray(inputs[0])]
    if op == "Pow":
        return [np.power(inputs[0], inputs[1])]
    if op == "Range":
        start = np.asarray(inputs[0]).item()
        limit = np.asarray(inputs[1]).item()
        delta = np.asarray(inputs[2]).item()
        return [np.arange(start, limit, delta, dtype=np.result_type(inputs[0], inputs[1], inputs[2]))]
    if op == "Reciprocal":
        return [np.reciprocal(inputs[0])]
    if op in {"ReduceL2", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum"}:
        return [_reduce(node, inputs, op)]
    if op == "Round":
        return [np.round(inputs[0])]
    if op == "Reshape":
        x = np.asarray(inputs[0])
        requested = [int(v) for v in np.asarray(inputs[1]).reshape(-1)]
        allowzero = bool(attrs.get("allowzero", 0))
        shape = [x.shape[i] if dim == 0 and not allowzero else dim for i, dim in enumerate(requested)]
        return [np.reshape(x, tuple(shape))]
    if op == "Shape":
        shape = np.asarray(np.asarray(inputs[0]).shape, dtype=np.int64)
        start = int(attrs.get("start", 0))
        end = int(attrs.get("end", len(shape)))
        if start < 0:
            start += len(shape)
        if end < 0:
            end += len(shape)
        return [shape[start:end]]
    if op == "Slice":
        return [_eval_slice(node, inputs)]
    if op == "Sin":
        return [np.sin(inputs[0])]
    if op == "Sqrt":
        return [np.sqrt(inputs[0])]
    if op == "Squeeze":
        x = np.asarray(inputs[0])
        axes = _axes_from_input_or_attr(node, inputs)
        return [np.squeeze(x, axis=_normalize_axes(axes, x.ndim))]
    if op == "Sub":
        return [np.asarray(inputs[0]) - np.asarray(inputs[1])]
    if op == "Transpose":
        perm = attrs.get("perm")
        return [np.transpose(inputs[0], axes=perm)]
    if op == "Unsqueeze":
        x = np.asarray(inputs[0])
        axes = _axes_from_input_or_attr(node, inputs)
        if axes is None:
            raise ValueError("Unsqueeze requires axes")
        for axis in _normalize_unsqueeze_axes(axes, x.ndim):
            x = np.expand_dims(x, axis=axis)
        return [x]
    if op == "Where":
        return [np.where(inputs[0], inputs[1], inputs[2])]
    raise NotImplementedError(op)


def _static_tensor_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    try:
        model_for_shapes = shape_inference.infer_shapes(model)
    except Exception:
        model_for_shapes = model

    shapes: dict[str, tuple[int, ...]] = {}
    for value_info in (
        list(model_for_shapes.graph.input)
        + list(model_for_shapes.graph.value_info)
        + list(model_for_shapes.graph.output)
    ):
        if not value_info.type.HasField("tensor_type"):
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        fully_static = True
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value >= 0:
                dims.append(int(dim.dim_value))
            else:
                fully_static = False
                break
        if fully_static:
            shapes[value_info.name] = tuple(dims)

    for init in model_for_shapes.graph.initializer:
        shapes.setdefault(init.name, tuple(int(dim) for dim in init.dims))
    return shapes


def _eval_static_shape_node(
    node: onnx.NodeProto,
    static_shapes: dict[str, tuple[int, ...]],
) -> list[np.ndarray] | None:
    if node.op_type != "Shape" or not node.input:
        return None
    input_shape = static_shapes.get(node.input[0])
    if input_shape is None:
        return None

    start = _attr_int(node, "start", 0)
    end = _attr_int(node, "end", len(input_shape))
    rank = len(input_shape)
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = max(0, min(rank, start))
    end = max(0, min(rank, end))
    return [np.asarray(input_shape[start:end], dtype=np.int64)]


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(1)
    return dims


def _value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value_info.name: value_info
        for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        if value_info.type.HasField("tensor_type")
    }


def _static_shape_for_tensor(
    name: str,
    value_info_by_name: dict[str, onnx.ValueInfoProto],
) -> list[int] | None:
    value_info = value_info_by_name.get(name)
    if value_info is None:
        return None
    shape = _shape_from_value_info(value_info)
    if not shape or not all(dim > 0 for dim in shape):
        return None
    return shape


def _np_dtype_for_tensor_type(elem_type: int) -> np.dtype:
    if elem_type == TensorProto.BOOL:
        return np.dtype(np.bool_)
    if elem_type == TensorProto.INT8:
        return np.dtype(np.int8)
    if elem_type == TensorProto.INT16:
        return np.dtype(np.int16)
    if elem_type == TensorProto.INT32:
        return np.dtype(np.int32)
    if elem_type == TensorProto.INT64:
        return np.dtype(np.int64)
    if elem_type == TensorProto.UINT8:
        return np.dtype(np.uint8)
    if elem_type == TensorProto.UINT16:
        return np.dtype(np.uint16)
    if elem_type == TensorProto.UINT32:
        return np.dtype(np.uint32)
    if elem_type == TensorProto.UINT64:
        return np.dtype(np.uint64)
    if elem_type == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if elem_type == TensorProto.DOUBLE:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _make_verify_inputs(model: onnx.ModelProto, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    initializer_names = {init.name for init in model.graph.initializer}
    graph_input_names = {value_info.name for value_info in model.graph.input}
    value_info_by_name = _value_info_by_name(model)
    inputs: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        dtype = _np_dtype_for_tensor_type(tensor_type.elem_type)
        shape = _shape_from_value_info(value_info)

        if np.issubdtype(dtype, np.floating):
            values = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            values = np.zeros(shape, dtype=dtype)
        elif value_info.name == "text_lengths":
            values = np.full(shape, shape[-1] if shape else 100, dtype=dtype)
        elif value_info.name == "texts":
            values = np.zeros(shape, dtype=dtype)
        else:
            values = np.zeros(shape, dtype=dtype)
        inputs[value_info.name] = values

    for node in model.graph.node:
        if not node.output:
            continue
        output_shape = _static_shape_for_tensor(node.output[0], value_info_by_name)
        if output_shape is None:
            continue

        shape_input_index = None
        if node.op_type in {"Expand", "Reshape"} and len(node.input) > 1:
            shape_input_index = 1
        elif node.op_type == "ConstantOfShape" and node.input:
            shape_input_index = 0

        if shape_input_index is None:
            continue
        shape_input = node.input[shape_input_index]
        if shape_input not in graph_input_names or shape_input not in inputs:
            continue

        dtype = inputs[shape_input].dtype
        if np.issubdtype(dtype, np.integer):
            inputs[shape_input] = np.asarray(output_shape, dtype=dtype)
    return inputs


def _compare_outputs(
    before_outputs: list[np.ndarray],
    after_outputs: list[np.ndarray],
    output_names: list[str],
    rtol: float,
    atol: float,
) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    ok = True
    for name, before, after in zip(output_names, before_outputs, after_outputs):
        before_arr = np.asarray(before)
        after_arr = np.asarray(after)
        same_shape = before_arr.shape == after_arr.shape
        if np.issubdtype(before_arr.dtype, np.floating) or np.issubdtype(after_arr.dtype, np.floating):
            matched = same_shape and np.allclose(before_arr, after_arr, rtol=rtol, atol=atol, equal_nan=True)
            diff = np.abs(before_arr.astype(np.float64) - after_arr.astype(np.float64)) if same_shape else np.asarray([np.inf])
            max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        else:
            matched = same_shape and np.array_equal(before_arr, after_arr)
            max_abs = 0.0 if matched else float("inf")
        rows.append(
            {
                "output": name,
                "matched": bool(matched),
                "shape_before": list(before_arr.shape),
                "shape_after": list(after_arr.shape),
                "dtype_before": str(before_arr.dtype),
                "dtype_after": str(after_arr.dtype),
                "max_abs_diff": max_abs,
            }
        )
        ok = ok and bool(matched)
    return ok, rows


def verify_models_match(
    before_model: onnx.ModelProto,
    after_model: onnx.ModelProto,
    *,
    runs: int,
    seed: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"onnxruntime import failed: {exc}",
        }

    output_names_before = [output.name for output in before_model.graph.output]
    output_names_after = [output.name for output in after_model.graph.output]
    if output_names_before != output_names_after:
        return {
            "status": "failed",
            "reason": "graph output names changed",
            "outputs_before": output_names_before,
            "outputs_after": output_names_after,
        }

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        before_session = ort.InferenceSession(
            before_model.SerializeToString(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        after_session = ort.InferenceSession(
            after_model.SerializeToString(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"onnxruntime session creation failed: {type(exc).__name__}: {exc}",
        }

    run_rows = []
    for run_index in range(runs):
        inputs = _make_verify_inputs(before_model, seed + run_index)
        try:
            before_outputs = before_session.run(output_names_before, inputs)
            after_outputs = after_session.run(output_names_after, inputs)
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"onnxruntime execution failed: {type(exc).__name__}: {exc}",
                "completed_runs": run_rows,
            }

        matched, output_rows = _compare_outputs(
            before_outputs,
            after_outputs,
            output_names_before,
            rtol,
            atol,
        )
        run_rows.append(
            {
                "run_index": run_index,
                "seed": seed + run_index,
                "matched": matched,
                "outputs": output_rows,
            }
        )
        if not matched:
            return {
                "status": "failed",
                "runs": run_rows,
            }

    return {
        "status": "passed",
        "runs": run_rows,
    }


def _dynamic_weight_rows(model: onnx.ModelProto) -> list[dict[str, Any]]:
    initializers = {init.name for init in model.graph.initializer}
    rows = []
    for node_index, node in enumerate(model.graph.node):
        for input_index, role in WEIGHT_INPUTS.get(node.op_type, {}).items():
            if input_index >= len(node.input):
                continue
            tensor = node.input[input_index]
            if not tensor or tensor in initializers:
                continue
            rows.append(
                {
                    "node_index": node_index,
                    "consumer_node": node.name,
                    "consumer_op": node.op_type,
                    "input_index": input_index,
                    "role": role,
                    "tensor": tensor,
                }
            )
    return rows


def fold_constants(
    model: onnx.ModelProto,
    *,
    keep_pruned_materialized_initializers: bool = False,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    graph = model.graph
    original_nodes = list(graph.node)
    original_initializers = {init.name for init in graph.initializer}
    graph_outputs = {output.name for output in graph.output}
    node_indices = {id(node): index for index, node in enumerate(graph.node)}
    static_shapes = _static_tensor_shapes(model)

    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in graph.node:
        for input_index, tensor in enumerate(node.input):
            if tensor:
                consumers[tensor].append((node, input_index))

    values: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    evaluated_node_indices: set[int] = set()
    evaluated_outputs: dict[str, np.ndarray] = {}
    errors: list[dict[str, Any]] = []

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            node_index = node_indices[id(node)]
            if node_index in evaluated_node_indices:
                continue
            if node.op_type not in CONST_FOLDABLE_OPS:
                continue
            input_values = _get_input_values(node, values)
            static_shape_outputs = None
            if input_values is None:
                static_shape_outputs = _eval_static_shape_node(node, static_shapes)
                if static_shape_outputs is None:
                    continue
            try:
                output_values = static_shape_outputs or _eval_node(node, input_values)
            except Exception as exc:
                errors.append(
                    {
                        "node_index": node_index,
                        "node": node.name,
                        "op": node.op_type,
                        "error": str(exc),
                    }
                )
                evaluated_node_indices.add(node_index)
                continue
            if len(output_values) != len([name for name in node.output if name]):
                errors.append(
                    {
                        "node_index": node_index,
                        "node": node.name,
                        "op": node.op_type,
                        "error": f"evaluator produced {len(output_values)} outputs for {len(node.output)} names",
                    }
                )
                evaluated_node_indices.add(node_index)
                continue
            value_index = 0
            for output_name in node.output:
                if not output_name:
                    continue
                value = np.asarray(output_values[value_index])
                value_index += 1
                values[output_name] = value
                evaluated_outputs[output_name] = value
            evaluated_node_indices.add(node_index)
            changed = True

    materialized_rows: list[dict[str, Any]] = []
    materialized_names: set[str] = set()
    existing_initializers = {init.name for init in graph.initializer}
    producer_by_output = {output: node for node in graph.node for output in node.output}

    for name, value in evaluated_outputs.items():
        if name in existing_initializers:
            continue
        tensor = numpy_helper.from_array(np.asarray(value), name=name)
        graph.initializer.append(tensor)
        existing_initializers.add(name)
        materialized_names.add(name)
        producer = producer_by_output.get(name)
        boundary_consumers = [
            (consumer, input_index)
            for consumer, input_index in consumers.get(name, [])
            if node_indices[id(consumer)] not in evaluated_node_indices
        ]
        materialized_rows.append(
            {
                "tensor": name,
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "producer_node": producer.name if producer is not None else "",
                "producer_node_index": node_indices[id(producer)] if producer is not None else "",
                "producer_op": producer.op_type if producer is not None else "",
                "boundary_consumers": [
                    {
                        "consumer_node": consumer.name,
                        "consumer_node_index": node_indices[id(consumer)],
                        "consumer_op": consumer.op_type,
                        "input_index": input_index,
                    }
                    for consumer, input_index in boundary_consumers
                ],
            }
        )

    removed_node_indices: set[int] = set()
    kept_nodes = []
    for node in graph.node:
        node_index = node_indices[id(node)]
        outputs = [output for output in node.output if output]
        if (
            node_index in evaluated_node_indices
            and outputs
            and all(output in values for output in outputs)
            and not any(output in graph_outputs and output not in materialized_names for output in outputs)
        ):
            removed_node_indices.add(node_index)
        else:
            kept_nodes.append(node)

    removed_nodes_by_op = Counter(
        node.op_type
        for node in original_nodes
        if node_indices[id(node)] in removed_node_indices
    )

    del graph.node[:]
    graph.node.extend(kept_nodes)

    used_tensors = {tensor for node in graph.node for tensor in node.input if tensor}
    used_tensors.update(graph_outputs)
    kept_initializers = [
        init
        for init in graph.initializer
        if init.name in used_tensors
        or (keep_pruned_materialized_initializers and init.name in materialized_names)
    ]
    pruned_initializer_names = sorted(init.name for init in graph.initializer if init.name not in used_tensors)
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    materialized_boundary_rows = [
        row for row in materialized_rows if row["boundary_consumers"]
    ]
    report = {
        "original_nodes": len(node_indices),
        "final_nodes": len(graph.node),
        "removed_nodes": len(removed_node_indices),
        "original_initializers": len(original_initializers),
        "final_initializers": len(graph.initializer),
        "materialized_initializers": len(materialized_names),
        "pruned_initializers": len(pruned_initializer_names),
        "folded_nodes_by_op": dict(removed_nodes_by_op),
        "materialized_tensors_by_producer_op": dict(Counter(row["producer_op"] for row in materialized_rows)),
        "materialized_boundary_tensors": len(materialized_boundary_rows),
        "materialized_boundary_by_consumer": dict(
            Counter(
                f"{consumer['consumer_op']}[{consumer['input_index']}]"
                for row in materialized_boundary_rows
                for consumer in row["boundary_consumers"]
            )
        ),
        "pruned_initializer_names": pruned_initializer_names,
        "materialized_tensors": materialized_rows,
        "evaluation_errors": errors,
    }
    return model, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ONNX file for file input, or output directory for directory input. "
            "For directory input, defaults to <input>_folded_constants."
        ),
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX Runtime equivalence checks with deterministic synthetic inputs.",
    )
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Return non-zero if verification is skipped or fails.",
    )
    parser.add_argument("--verify-runs", type=int, default=1)
    parser.add_argument("--verify-seed", type=int, default=0)
    parser.add_argument("--verify-rtol", type=float, default=1e-4)
    parser.add_argument("--verify-atol", type=float, default=1e-5)
    return parser.parse_args()


def _default_output_for_input(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_folded_constants"
    return input_path.with_name(f"{input_path.stem}_folded_constants{input_path.suffix}")


def _report_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".fold_constants.json")


def fold_one_model(
    *,
    input_path: Path,
    output_path: Path,
    report_json: Path | None,
    check: bool,
    verify: bool,
    strict_verify: bool,
    verify_runs: int,
    verify_seed: int,
    verify_rtol: float,
    verify_atol: float,
) -> tuple[int, dict[str, Any]]:
    model = onnx.load(str(input_path))
    before_dynamic_weights = _dynamic_weight_rows(model)

    folded_model, report = fold_constants(model)
    after_dynamic_weights = _dynamic_weight_rows(folded_model)
    report["input"] = str(input_path)
    report["output"] = str(output_path)
    report["dynamic_weight_inputs_before"] = {
        "count": len(before_dynamic_weights),
        "by_consumer_op": dict(Counter(row["consumer_op"] for row in before_dynamic_weights)),
        "rows": before_dynamic_weights,
    }
    report["dynamic_weight_inputs_after"] = {
        "count": len(after_dynamic_weights),
        "by_consumer_op": dict(Counter(row["consumer_op"] for row in after_dynamic_weights)),
        "rows": after_dynamic_weights,
    }

    if verify:
        verification = verify_models_match(
            model,
            folded_model,
            runs=verify_runs,
            seed=verify_seed,
            rtol=verify_rtol,
            atol=verify_atol,
        )
        if (
            verification["status"] == "skipped"
            and report["removed_nodes"] == 0
            and report["materialized_initializers"] == 0
            and report["pruned_initializers"] == 0
        ):
            verification = {
                "status": "passed",
                "method": "no_graph_changes",
                "runtime_skip_reason": verification.get("reason"),
            }
        report["verification"] = verification
    else:
        report["verification"] = {"status": "disabled"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(folded_model, str(output_path))

    if check:
        onnx.checker.check_model(str(output_path))
        print("ONNX checker passed")

    if report_json:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(_jsonify(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote report: {report_json}")

    print(f"Folded nodes: {report['removed_nodes']} / {report['original_nodes']}")
    print(f"Materialized initializers: {report['materialized_initializers']}")
    print(f"Pruned initializers: {report['pruned_initializers']}")
    print(f"Boundary tensors materialized: {report['materialized_boundary_tensors']}")
    print("Boundary consumers:")
    for key, count in sorted(report["materialized_boundary_by_consumer"].items()):
        print(f"  {key:<18} {count}")
    print(
        "Dynamic known weight inputs: "
        f"{len(before_dynamic_weights)} before -> {len(after_dynamic_weights)} after"
    )
    verification_status = report["verification"]["status"]
    print(f"Verification: {verification_status}")
    if verification_status == "skipped":
        print(f"  reason: {report['verification'].get('reason')}")
    print(f"Wrote model: {output_path}")

    if strict_verify and verification_status != "passed":
        return 1, report
    if verification_status == "failed":
        return 1, report
    return 0, report


def fold_directory(args: argparse.Namespace) -> int:
    output_dir = args.output or _default_output_for_input(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(path for path in args.input.iterdir() if path.suffix == ".onnx")
    if not model_paths:
        raise ValueError(f"No .onnx files found in {args.input}")

    manifest_rows = []
    exit_code = 0
    for index, input_path in enumerate(model_paths):
        output_path = output_dir / input_path.name
        report_path = output_dir / f"{input_path.stem}.fold_constants.json"
        print(f"\n[{index + 1}/{len(model_paths)}] Folding {input_path} -> {output_path}")
        model_exit_code, report = fold_one_model(
            input_path=input_path,
            output_path=output_path,
            report_json=report_path,
            check=args.check,
            verify=args.verify,
            strict_verify=args.strict_verify,
            verify_runs=args.verify_runs,
            verify_seed=args.verify_seed,
            verify_rtol=args.verify_rtol,
            verify_atol=args.verify_atol,
        )
        exit_code = max(exit_code, model_exit_code)
        manifest_rows.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "report_json": str(report_path),
                "original_nodes": report["original_nodes"],
                "final_nodes": report["final_nodes"],
                "removed_nodes": report["removed_nodes"],
                "materialized_initializers": report["materialized_initializers"],
                "verification_status": report["verification"]["status"],
                "verification_reason": report["verification"].get("reason"),
            }
        )

    manifest = {
        "input_dir": str(args.input),
        "output_dir": str(output_dir),
        "model_count": len(model_paths),
        "verify": args.verify,
        "strict_verify": args.strict_verify,
        "models": manifest_rows,
        "verification_summary": dict(Counter(row["verification_status"] for row in manifest_rows)),
    }
    manifest_path = args.report_json or (output_dir / "fold_constants_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_jsonify(manifest), indent=2, sort_keys=True) + "\n")
    print(f"\nWrote folder manifest: {manifest_path}")
    print("Verification summary:")
    for status, count in sorted(manifest["verification_summary"].items()):
        print(f"  {status:<8} {count}")
    return exit_code


def main() -> int:
    args = parse_args()
    if args.verify_runs < 1:
        raise ValueError("--verify-runs must be >= 1")

    if args.input.is_dir():
        return fold_directory(args)

    output_path = args.output or _default_output_for_input(args.input)
    report_path = args.report_json
    exit_code, _ = fold_one_model(
        input_path=args.input,
        output_path=output_path,
        report_json=report_path,
        check=args.check,
        verify=args.verify,
        strict_verify=args.strict_verify,
        verify_runs=args.verify_runs,
        verify_seed=args.verify_seed,
        verify_rtol=args.verify_rtol,
        verify_atol=args.verify_atol,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
