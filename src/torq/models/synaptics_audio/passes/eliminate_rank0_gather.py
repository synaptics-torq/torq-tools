# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Eliminate ``Gather`` ops that produce rank-0 (scalar) intermediates.

IREE's codegen ``FoldCollapseShape{,FullSlice}IntoInterfaceTensorStore``
patterns trip a ranks-don't-match verifier error when a tensor of rank > 0
is collapsed all the way down to rank 0 inside a dispatch (the patterns
emit ``sizes`` with rank 0 but ``strides`` with the source rank). The
audio fold pipeline produces such an intermediate via shape arithmetic
of the form::

    Shape(...) -> Gather(axis=0, indices=<scalar>) -> Unsqueeze(axes=[0])

which fetches a single dimension and immediately re-wraps it into a rank-1
tensor for downstream ``Concat`` / ``Reshape`` use. The rank-0 step in the
middle is what hits the upstream bug.

This pass rewrites that pattern at the ONNX level so the rank-0 intermediate
disappears entirely:

* the scalar ``indices`` input is reshaped to rank 1 (``[1]``); when it is
  a ``Constant`` initializer/op we just rebuild it inline, otherwise a
  ``Reshape`` with the constant ``[1]`` shape is inserted;
* the ``Gather`` then produces a rank-1 ``[1]`` output (same numeric value,
  just one extra leading axis);
* if the original consumer was an ``Unsqueeze`` with ``axes=[0]`` (the
  common ``Gather -> Unsqueeze`` shape-arithmetic idiom), it's removed and
  its consumers are rewired to the Gather's new rank-1 output.

The rewrite is value-preserving for downstream consumers: in the common
case the Unsqueeze is the *only* consumer and was about to add the leading
axis we now produce directly. If a Gather -> Unsqueeze fusion is not
applicable (e.g. another consumer wants the value as a real scalar), the
node is left untouched and a warning is logged so we notice the case.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.eliminate_rank0_gather")


def _output_rank(var) -> int | None:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None
    return len(shape)


def _consumers_of(graph: gs.Graph, name: str) -> list[gs.Node]:
    out: list[gs.Node] = []
    for n in graph.nodes:
        for inp in n.inputs:
            if getattr(inp, "name", None) == name:
                out.append(n)
                break
    return out


def _make_rank1_indices(graph: gs.Graph, idx, name_prefix: str):
    """Return a rank-1 tensor of length 1 with the same value as ``idx``."""
    if isinstance(idx, gs.Constant):
        arr = np.asarray(idx.values, dtype=np.int64).reshape(1)
        return gs.Constant(name=f"{name_prefix}_idx1d", values=arr)

    shape_const = gs.Constant(
        name=f"{name_prefix}_idx_shape",
        values=np.array([1], dtype=np.int64),
    )
    out = gs.Variable(
        name=f"{name_prefix}_idx1d",
        dtype=getattr(idx, "dtype", np.int64),
        shape=(1,),
    )
    graph.nodes.append(
        gs.Node(
            op="Reshape",
            name=f"{name_prefix}_idx_reshape",
            inputs=[idx, shape_const],
            outputs=[out],
        )
    )
    return out


def _const_int_list(t) -> list[int] | None:
    """Extract a 1-D int list from a tensor that may be either a ``gs.Constant``
    initializer or a ``gs.Variable`` produced by a ``Constant`` op.

    Returns ``None`` when the value is not a constant we can statically resolve
    (e.g. produced by an arithmetic chain).
    """
    if isinstance(t, gs.Constant):
        return np.asarray(t.values).reshape(-1).tolist()
    producers = getattr(t, "inputs", None) or []
    if not producers:
        return None
    producer = producers[0]
    if producer.op != "Constant":
        return None
    value = producer.attrs.get("value")
    if value is None or not hasattr(value, "values"):
        return None
    return np.asarray(value.values).reshape(-1).tolist()


def _unsqueeze_axes_is_zero_only(node: gs.Node) -> bool:
    """True if ``node`` is an ``Unsqueeze`` adding exactly axis 0.

    Handles all three ONNX representations of the ``axes`` input:

    * opset 13+ with ``axes`` as a ``gs.Constant`` initializer,
    * opset 13+ with ``axes`` as the output of a separate ``Constant`` op
      (the form produced by ``torch.onnx.export``),
    * opset <= 12 with ``axes`` as a node attribute.
    """
    if node.op != "Unsqueeze":
        return False
    if len(node.inputs) >= 2:
        return _const_int_list(node.inputs[1]) == [0]
    axes_attr = node.attrs.get("axes")
    if axes_attr is None:
        return False
    return list(axes_attr) == [0]


def _rewrite_one(graph: gs.Graph, gather: gs.Node) -> bool:
    if len(gather.outputs) != 1 or len(gather.inputs) < 2:
        return False

    data, indices = gather.inputs[0], gather.inputs[1]
    out = gather.outputs[0]

    if _output_rank(out) != 0:
        return False
    if _output_rank(data) != 1 or _output_rank(indices) != 0:
        return False

    consumers = _consumers_of(graph, out.name)
    fusable = [n for n in consumers if _unsqueeze_axes_is_zero_only(n)]
    other = [n for n in consumers if n not in fusable]

    if other:
        logger.warning(
            "rank-0 Gather %r has non-Unsqueeze consumer(s) %s; leaving untouched "
            "(IREE rank-0 collapse bug may still trigger downstream)",
            gather.name,
            [n.name for n in other],
        )
        return False
    if not fusable:
        logger.warning(
            "rank-0 Gather %r has no consumers in the graph; leaving untouched",
            gather.name,
        )
        return False

    base = gather.name or "rank0_gather"
    new_idx = _make_rank1_indices(graph, indices, base)
    new_out = gs.Variable(name=f"{out.name}_rank1", dtype=out.dtype, shape=(1,))
    gather.inputs[1] = new_idx
    gather.outputs = [new_out]

    for unsq in fusable:
        unsq_out = unsq.outputs[0]
        for n in graph.nodes:
            n.inputs = [
                new_out if getattr(i, "name", None) == unsq_out.name else i
                for i in n.inputs
            ]
        for i, go in enumerate(graph.outputs):
            if getattr(go, "name", None) == unsq_out.name:
                graph.outputs[i] = new_out
        unsq.inputs = []
        unsq.outputs = []

    logger.info(
        "rewrote rank-0 Gather %r (consumed by %d Unsqueeze(s)) to rank-1 path",
        gather.name,
        len(fusable),
    )
    return True


class EliminateRank0Gather:
    """Pass: rewrite ``Gather -> Unsqueeze[axes=[0]]`` chains to remove rank-0 intermediates."""

    name = "eliminate_rank0_gather"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"

        rewritten = 0
        for node in list(graph.nodes):
            if node.op != "Gather":
                continue
            if _rewrite_one(graph, node):
                rewritten += 1

        if not rewritten:
            return model

        graph.cleanup().toposort()
        out_model = gs.export_onnx(graph)
        try:
            out_model = shape_inference.infer_shapes(out_model)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        return out_model
