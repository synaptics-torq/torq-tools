# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Remove ``Transpose`` nodes that only move singleton dimensions.

Some exported audio graphs start with layout-normalizing transposes that become
identity operations after input shapes are fixed. For example, NNNR has
``[1, 1, 1, 256]`` and ``perm=[0, 2, 1, 3]``; axes 1 and 2 are both singleton,
so replacing the transpose output with its input is value-preserving.

This pass deliberately does *not* remove transposes just because input and
output shapes happen to match. Swapping two non-singleton axes of equal extent
is still a real data permutation.
"""

from __future__ import annotations

import logging

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.eliminate_degenerate_transpose")


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


def _transpose_perm(node: gs.Node, rank: int) -> list[int] | None:
    raw = node.attrs.get("perm")
    if raw is None:
        return list(reversed(range(rank)))
    perm = [int(v) for v in raw]
    if sorted(perm) != list(range(rank)):
        return None
    return perm


def _moves_only_singleton_axes(shape: list[int], perm: list[int]) -> bool:
    if len(shape) != len(perm):
        return False
    moved = False
    for out_axis, in_axis in enumerate(perm):
        if out_axis == in_axis:
            continue
        moved = True
        if shape[out_axis] != 1 or shape[in_axis] != 1:
            return False
    return moved


def _replace_tensor_uses(graph: gs.Graph, old, new) -> None:
    for node in graph.nodes:
        node.inputs = [
            new if getattr(inp, "name", None) == old.name else inp
            for inp in node.inputs
        ]


class EliminateDegenerateTranspose:
    """Pass: bypass transposes that only reorder singleton static dimensions."""

    name = "eliminate_degenerate_transpose"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"
        graph_output_names = {getattr(out, "name", None) for out in graph.outputs}

        removed = 0
        for node in list(graph.nodes):
            if node.op != "Transpose" or len(node.inputs) != 1 or len(node.outputs) != 1:
                continue

            inp = node.inputs[0]
            out = node.outputs[0]
            if getattr(out, "name", None) in graph_output_names:
                continue

            shape = _static_shape(inp)
            if shape is None:
                continue

            perm = _transpose_perm(node, len(shape))
            if perm is None or not _moves_only_singleton_axes(shape, perm):
                continue

            _replace_tensor_uses(graph, out, inp)
            node.inputs.clear()
            node.outputs.clear()
            removed += 1
            logger.info(
                "removed degenerate Transpose %r perm=%s shape=%s",
                node.name,
                perm,
                shape,
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
