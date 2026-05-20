# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Fuse non-negative ``Pad`` layers into the following ``Conv``.

Constant-mode Pads with negative paddings are out of scope (use
:class:`RewriteNegativePads` first; that pass converts them to ``Pad+Slice``).

Both the merged-pad layout and the ``auto_pad`` interaction follow the ONNX
spec: ``auto_pad`` would override an explicit ``pads`` list, so a Conv that
sets ``auto_pad`` is left alone.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.absorb_padding")


def _fuse_pad_into_conv(graph: gs.Graph) -> int:
    fused = 0
    for pad in list(graph.nodes):
        if pad.op != "Pad":
            continue
        if not pad.outputs or not pad.inputs:
            continue
        out_var = pad.outputs[0]
        users = list(out_var.outputs)
        if len(users) != 1 or users[0].op != "Conv":
            continue

        pads_inp = pad.inputs[1]
        if not isinstance(pads_inp, gs.Constant):
            continue
        pads_flat = np.asarray(pads_inp.values).astype(np.int64).reshape(-1)
        if np.any(pads_flat < 0):
            continue

        conv = users[0]
        if "pads" not in conv.attrs or "kernel_shape" not in conv.attrs:
            continue
        if "auto_pad" in conv.attrs:
            continue

        rank = len(pads_flat) // 2
        kshape = conv.attrs["kernel_shape"]
        spatial_rank = len(kshape)
        spatial_axes = list(range(rank - spatial_rank, rank))

        conv_pads = list(conv.attrs["pads"])
        if len(conv_pads) != 2 * spatial_rank:
            continue

        pb_half = pads_flat[:rank]
        pa_half = pads_flat[rank:]
        new_conv_pads: list[int] = []
        for j in range(spatial_rank):
            ax = spatial_axes[j]
            new_conv_pads.append(int(conv_pads[j] + pb_half[ax]))
        for j in range(spatial_rank):
            ax = spatial_axes[j]
            new_conv_pads.append(int(conv_pads[j + spatial_rank] + pa_half[ax]))

        conv.attrs["pads"] = new_conv_pads
        conv.inputs[0] = pad.inputs[0]

        pad.outputs.clear()
        pad.inputs.clear()
        fused += 1

    if fused:
        graph.cleanup().toposort()
    return fused


class AbsorbPadding:
    """Pass: fuse non-negative ``Pad -> Conv`` into a single ``Conv`` with merged pads."""

    name = "absorb_padding"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"

        fused = _fuse_pad_into_conv(graph)
        if not fused:
            return model

        out_model = gs.export_onnx(graph)
        try:
            out_model = shape_inference.infer_shapes(out_model)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        logger.info("fused %d Pad->Conv", fused)
        return out_model
