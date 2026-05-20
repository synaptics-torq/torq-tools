# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Remove singleton-axis ``Gather -> unary -> Unsqueeze`` rank shims.

NNNR's exported tail contains this static-shape pattern::

    x [1, 1, 256]
      -> Gather(axis=1, indices=0) [1, 256]
      -> Sigmoid [1, 256]
      -> Unsqueeze(axis=0) [1, 1, 256]

Because the gathered axis has extent 1 and the final ``Unsqueeze`` restores
the original static shape, this is equivalent to applying the elementwise unary
op directly to ``x``. The pass is conservative: it requires a scalar zero
index, a known singleton gathered axis, one shape-preserving unary op, and a
final shape equal to the original input shape.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger(
    "synaptics-audio.passes.eliminate_singleton_gather_unsqueeze"
)

_ELEMENTWISE_UNARY_OPS = frozenset(
    {
        "Abs",
        "Ceil",
        "Cos",
        "Erf",
        "Exp",
        "Floor",
        "Log",
        "Neg",
        "Relu",
        "Sigmoid",
        "Sin",
        "Sqrt",
        "Tanh",
    }
)


def _static_shape(var) -> list[int] | None:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None
    out: list[int] = []
    for dim in shape:
        try:
            val = int(dim)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        out.append(val)
    return out


def _normalize_axis(axis: int, rank: int) -> int | None:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def _const_array(t) -> np.ndarray | None:
    if isinstance(t, gs.Constant):
        return np.asarray(t.values)
    producers = getattr(t, "inputs", None) or []
    if len(producers) != 1 or producers[0].op != "Constant":
        return None
    value = producers[0].attrs.get("value")
    if value is None or not hasattr(value, "values"):
        return None
    return np.asarray(value.values)


def _const_int_list(t) -> list[int] | None:
    arr = _const_array(t)
    if arr is None:
        return None
    return [int(v) for v in arr.reshape(-1)]


def _is_scalar_zero(t) -> bool:
    arr = _const_array(t)
    if arr is None or arr.shape != ():
        return False
    return int(arr.reshape(())) == 0


def _unsqueeze_axes(node: gs.Node, out_rank: int) -> list[int] | None:
    if node.op != "Unsqueeze":
        return None
    if len(node.inputs) >= 2:
        axes = _const_int_list(node.inputs[1])
    else:
        raw = node.attrs.get("axes")
        axes = [int(v) for v in raw] if raw is not None else None
    if axes is None:
        return None

    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += out_rank
        if axis < 0 or axis >= out_rank:
            return None
        normalized.append(axis)
    return normalized


def _consumers_of(graph: gs.Graph, name: str) -> list[gs.Node]:
    out: list[gs.Node] = []
    for node in graph.nodes:
        if any(getattr(inp, "name", None) == name for inp in node.inputs):
            out.append(node)
    return out


class EliminateSingletonGatherUnsqueeze:
    """Pass: fold singleton-axis ``Gather -> unary -> Unsqueeze`` shims."""

    name = "eliminate_singleton_gather_unsqueeze"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"

        removed = 0
        for gather in list(graph.nodes):
            if (
                gather.op != "Gather"
                or len(gather.inputs) < 2
                or len(gather.outputs) != 1
            ):
                continue
            data, indices = gather.inputs[0], gather.inputs[1]
            data_shape = _static_shape(data)
            gather_shape = _static_shape(gather.outputs[0])
            if data_shape is None or gather_shape is None:
                continue

            axis = _normalize_axis(int(gather.attrs.get("axis", 0)), len(data_shape))
            if axis is None or data_shape[axis] != 1 or not _is_scalar_zero(indices):
                continue
            if gather_shape != data_shape[:axis] + data_shape[axis + 1:]:
                continue

            gather_consumers = _consumers_of(graph, gather.outputs[0].name)
            if len(gather_consumers) != 1:
                continue
            unary = gather_consumers[0]
            if (
                unary.op not in _ELEMENTWISE_UNARY_OPS
                or len(unary.inputs) != 1
                or len(unary.outputs) != 1
            ):
                continue

            unary_consumers = _consumers_of(graph, unary.outputs[0].name)
            if len(unary_consumers) != 1:
                continue
            unsqueeze = unary_consumers[0]
            out_shape = (
                _static_shape(unsqueeze.outputs[0])
                if len(unsqueeze.outputs) == 1
                else None
            )
            if out_shape != data_shape:
                continue
            axes = _unsqueeze_axes(unsqueeze, len(out_shape))
            if axes is None:
                continue

            unary.inputs[0] = data
            unary.outputs[0] = unsqueeze.outputs[0]
            gather.inputs.clear()
            gather.outputs.clear()
            unsqueeze.inputs.clear()
            unsqueeze.outputs.clear()
            removed += 1
            logger.info(
                "removed singleton Gather->%s->Unsqueeze shim at %r axis=%d shape=%s",
                unary.op,
                gather.name,
                axis,
                data_shape,
            )

        if not removed:
            return model

        graph.cleanup().toposort()
        out_model = gs.export_onnx(graph)
        try:
            out_model = shape_inference.infer_shapes(out_model)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        return out_model
