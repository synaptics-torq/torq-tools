# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from collections import Counter, defaultdict
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
    "Gather",
    "Greater",
    "GreaterOrEqual",
    "Identity",
    "Less",
    "LessOrEqual",
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
