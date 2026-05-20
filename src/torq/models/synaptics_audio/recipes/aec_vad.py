# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics AEC-VAD recipe (HF repo retains the original ``AI-VAD`` name)."""

from __future__ import annotations

from .base import Recipe

AEC_VAD = Recipe(
    key="aec_vad",
    repo_id="Synaptics/AI-VAD",
    source_filename=(
        "standalone_2ch_DT_VAD_python_0909/"
        "2025-02-11_01-38-13_aec_vad_exp12_d4_model_epoch_t710.onnx"
    ),
)
