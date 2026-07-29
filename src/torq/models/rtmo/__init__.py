# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""RTMO tiny (real-time multi-person pose) export for Torq.

The upstream ONNX ships with the mmdeploy post-processing baked in (bbox
decode, DCC pose decode, TopK/NonMaxSuppression, dynamic gathers) and outputs
``dets`` / ``keypoints`` with data-dependent shapes. That tail is not
NPU-friendly. :func:`~torq.models.rtmo.export.export_rtmo` cuts the graph at the
eight dense head convolutions, re-targets it to a chosen square input size, and
converts it to bf16; the decode/NMS is expected to run host-side.

This package covers the build pipeline only — export, quantize, and the hybrid
int8/bf16 compile to three NSS-only vmfbs. Deployment (chaining the vmfbs on a
Torq device + host-side decode/draw) lives with the on-device runner, since the
NSS-only vmfbs need the Torq HAL and cannot run on the host.

Eight fp32 outputs (grouped by branch, level ascending — stride 16 then 32),
for an ``S x S`` input:

    cls_scores_s16  [B, 1,   S/16, S/16]   cls_scores_s32  [B, 1,   S/32, S/32]
    bbox_preds_s16  [B, 4,   S/16, S/16]   bbox_preds_s32  [B, 4,   S/32, S/32]
    kpt_vis_s16     [B, 17,  S/16, S/16]   kpt_vis_s32     [B, 17,  S/32, S/32]
    pose_feats_s16  [B, 192, S/16, S/16]   pose_feats_s32  [B, 192, S/32, S/32]
"""

from __future__ import annotations

from ._hybrid import add_rtmo_hybrid_args, compile_hybrid, quantize_hybrid, split_rtmo
from .download_source import download_source
from .export import add_rtmo_export_args, export_rtmo, export_rtmo_from_args
from .quantize import add_rtmo_quantize_args, quantize_rtmo

__all__ = [
    "add_rtmo_export_args",
    "export_rtmo",
    "export_rtmo_from_args",
    "add_rtmo_quantize_args",
    "quantize_rtmo",
    "add_rtmo_hybrid_args",
    "quantize_hybrid",
    "compile_hybrid",
    "split_rtmo",
    "download_source",
]
