# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics/NNNR3 (Neural Network Noise Reduction) recipe."""

from __future__ import annotations

from .base import Recipe

NNNR = Recipe(
    key="nnnr",
    repo_id="Synaptics/NNNR3",
    source_filename="NNNR3_0079_0.0960.onnx",
)
