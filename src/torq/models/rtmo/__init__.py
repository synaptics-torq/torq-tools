# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""RTMO tiny (real-time multi-person pose) build pipeline for Torq.

The upstream ONNX bakes in the mmdeploy decode/NMS tail (data-dependent shapes,
not NPU-friendly). export_rtmo cuts the graph at the eight dense head convs
(fixed shapes: cls_scores/bbox_preds/kpt_vis/pose_feats at stride 16 + 32);
the decode runs host-side (:mod:`._postprocess`). quantize/_hybrid produce the
deployable int8 + bf16 parts; the NSS-only vmfbs need the Torq HAL, so
deployment lives with the on-device runner.
"""

from __future__ import annotations

from ._hybrid import add_rtmo_hybrid_args, compare_postprocess, compile_hybrid, quantize_hybrid, split_rtmo
from ._postprocess import model_postprocess
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
    "compare_postprocess",
    "model_postprocess",
    "split_rtmo",
    "download_source",
]
