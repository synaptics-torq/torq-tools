# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics/Voice-Filter recipe."""

from __future__ import annotations

from .base import Recipe

VOICE_FILTER = Recipe(
    key="voice_filter",
    repo_id="Synaptics/Voice-Filter",
    source_filename=(
        "baseline125K/VF_0290_19.onnx",
        "baseline125KReLU6/VF_0075_11.8389.onnx",
        "baseline423K/VF_0126_19.0517.onnx",
    ),
)
