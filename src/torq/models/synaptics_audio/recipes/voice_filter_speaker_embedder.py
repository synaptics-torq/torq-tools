# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics/Voice-Filter speaker embedder recipe."""

from __future__ import annotations

from .base import Recipe

VOICE_FILTER_SPEAKER_EMBEDDER = Recipe(
    key="voice_filter_speaker_embedder",
    repo_id="Synaptics/Voice-Filter",
    source_filename="baseline125K/model_epoch_0290_19.0540_speaker_embedder.onnx",
    input_shape_overrides={"feats": (1, 100, 40)},
)
