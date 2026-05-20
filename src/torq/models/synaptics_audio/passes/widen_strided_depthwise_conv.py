# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Widen narrow strided-depthwise Convs to dodge the Torq DEDR codegen path.

Why
---
For a depthwise Conv with ``stride[-1] > 1`` and a *small* last spatial output
dim, the Torq compiler picks a DEDR ``G(L)[sgGroups>1]`` SIMD scatter-gather
descriptor. The current precompiled NSS CModel hangs while executing that
descriptor (depthwise stride-2 + ``sgGroups=4`` codegen path), which makes
isolated depthwise-conv simulator runs time out. The DEDR codegen condition is
roughly::

    sg fires iff   out_w * sgGroups <= bus_width_items

where ``bus_width_items = bus_width_bytes / element_size`` and ``sgGroups``
ranges over 1..``sg_groups_max``. Pushing ``out_w`` just past
``bus_width_items // sg_groups_max`` (e.g. ``> 9`` for bf16) makes the
compiler fall back to the dense ``V(H)`` outerGroups path, which the CModel
executes correctly.

Transform
---------
The trailing ``pads`` entry on the matched Conv is bumped by
``extra * stride[-1]`` so the conv naturally produces ``extra`` extra output
positions with zero-padding (numerically equivalent to extending the input
with zeros). A ``Slice`` node is then inserted after the conv that crops
back to the original output width, so all downstream consumers see a
bit-identical tensor.

Defaults match the Torq HW: ``bus_width_bytes=72`` (``iram_seg_width``),
``sg_groups_max=4``.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.widen_strided_depthwise_conv")


def _element_size_bytes(dtype) -> int | None:
    if isinstance(dtype, np.dtype):
        return dtype.itemsize
    if isinstance(dtype, int):
        sizes = {
            onnx.TensorProto.FLOAT: 4,
            onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1,
            onnx.TensorProto.UINT16: 2, onnx.TensorProto.INT16: 2,
            onnx.TensorProto.INT32: 4, onnx.TensorProto.UINT32: 4,
            onnx.TensorProto.INT64: 8, onnx.TensorProto.UINT64: 8,
            onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2,
            onnx.TensorProto.DOUBLE: 8, onnx.TensorProto.BOOL: 1,
        }
        return sizes.get(int(dtype))
    return None


def _is_static_shape(shape) -> bool:
    return shape is not None and all(
        isinstance(d, (int, np.integer)) for d in shape
    )


def _rewire_consumers(consumers: list[gs.Node], orig, new) -> None:
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp is orig:
                consumer.inputs[i] = new


def _matches(node: gs.Node, threshold_for_dtype) -> int | None:
    """Return the threshold if ``node`` is a depthwise strided Conv to widen."""
    if node.op != "Conv" or len(node.inputs) < 2 or not node.outputs:
        return None
    x, w = node.inputs[0], node.inputs[1]
    out = node.outputs[0]
    if not (
        _is_static_shape(x.shape)
        and _is_static_shape(w.shape)
        and _is_static_shape(out.shape)
    ):
        return None
    if len(x.shape) < 3 or len(w.shape) != len(x.shape):
        return None

    in_channels = int(x.shape[1])
    group = int(node.attrs.get("group", 1))
    if group != in_channels or int(w.shape[1]) != 1:
        return None

    rank = len(x.shape) - 2
    strides = list(node.attrs.get("strides", [1] * rank))
    if len(strides) != rank or int(strides[-1]) <= 1:
        return None

    threshold = threshold_for_dtype(out.dtype)
    if threshold is None or threshold <= 0:
        return None
    if int(out.shape[-1]) > threshold:
        return None
    return threshold


def _transform(graph: gs.Graph, node: gs.Node, threshold: int) -> None:
    x = node.inputs[0]
    out = node.outputs[0]
    rank = len(x.shape) - 2
    cur_out_w = int(out.shape[-1])
    target_out_w = threshold + 1
    extra = target_out_w - cur_out_w

    strides = list(node.attrs.get("strides", [1] * rank))
    s_w = int(strides[-1])
    extra_pad = extra * s_w

    pads = list(node.attrs.get("pads", [0] * (2 * rank)))
    if len(pads) != 2 * rank:
        raise ValueError(
            f"Conv {node.name!r} has unexpected 'pads' length {len(pads)} (expected {2 * rank})"
        )

    # ONNX Conv pads layout: [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
    pads[2 * rank - 1] = int(pads[2 * rank - 1]) + extra_pad
    node.attrs["pads"] = pads
    # `auto_pad` would override an explicit `pads` list.
    node.attrs["auto_pad"] = "NOTSET"

    new_out_shape = list(out.shape)
    new_out_shape[-1] = target_out_w
    widened = gs.Variable(
        name=out.name + "_widened", dtype=out.dtype, shape=new_out_shape,
    )

    consumers = list(out.outputs)
    graph_output_indices = [
        i for i, g_out in enumerate(graph.outputs) if g_out is out
    ]
    node.outputs[0] = widened

    last_axis = len(new_out_shape) - 1
    starts = gs.Constant(f"{node.name}_widen_slice_starts", np.array([0], dtype=np.int64))
    ends = gs.Constant(f"{node.name}_widen_slice_ends", np.array([cur_out_w], dtype=np.int64))
    axes = gs.Constant(f"{node.name}_widen_slice_axes", np.array([last_axis], dtype=np.int64))
    steps = gs.Constant(f"{node.name}_widen_slice_steps", np.array([1], dtype=np.int64))

    # Preserve the original output name on the slice so downstream tensor
    # references stay stable; `out` becomes orphan and is dropped by cleanup.
    sliced = gs.Variable(name=out.name, dtype=out.dtype, shape=list(out.shape))
    graph.nodes.append(
        gs.Node(
            op="Slice",
            name=f"{node.name}_widen_slice",
            inputs=[widened, starts, ends, axes, steps],
            outputs=[sliced],
        )
    )

    _rewire_consumers(consumers, out, sliced)
    for i in graph_output_indices:
        graph.outputs[i] = sliced

    logger.info(
        "widened depthwise Conv %s: out_w %d -> %d (pad+%d on axis %d, slice [:%d])",
        node.name, cur_out_w, target_out_w, extra_pad, last_axis, cur_out_w,
    )


class WidenStridedDepthwiseConv:
    """Pass: widen narrow strided-depthwise Convs and slice back to original width."""

    name = "widen_strided_depthwise_conv"

    def __init__(self, bus_width_bytes: int = 72, sg_groups_max: int = 4) -> None:
        self.bus_width_bytes = bus_width_bytes
        self.sg_groups_max = sg_groups_max

    def _threshold_for_dtype(self, dtype) -> int | None:
        elem = _element_size_bytes(dtype)
        if not elem or self.sg_groups_max <= 0:
            return None
        return (self.bus_width_bytes // elem) // self.sg_groups_max

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"

        widened = 0
        for node in list(graph.nodes):
            threshold = _matches(node, self._threshold_for_dtype)
            if threshold is None:
                continue
            _transform(graph, node, threshold)
            widened += 1

        if not widened:
            return model

        graph.cleanup().toposort()
        out_model = gs.export_onnx(graph)
        try:
            out_model = shape_inference.infer_shapes(out_model)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        return out_model
