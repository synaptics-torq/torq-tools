# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Registered Synaptics audio recipes.

Add a recipe by:

1. dropping a new ``<model>.py`` file next to this one with a single
   :class:`Recipe` instance bound to a module-level constant;
2. importing it here and adding it to :data:`ALL`.

The :data:`BY_KEY` registry is computed from :data:`ALL` so the CLI and any
tooling stay in sync automatically. Recipe keys are unique; a HuggingFace repo
may have multiple recipe entries.
"""

from __future__ import annotations

from .aec_vad import AEC_VAD
from .base import Recipe
from .nnnr import NNNR
from .sed import SED
from .voice_filter import VOICE_FILTER
from .voice_filter_speaker_embedder import VOICE_FILTER_SPEAKER_EMBEDDER

ALL: tuple[Recipe, ...] = (
    AEC_VAD,
    NNNR,
    SED,
    VOICE_FILTER,
    VOICE_FILTER_SPEAKER_EMBEDDER,
)

BY_KEY: dict[str, Recipe] = {r.key: r for r in ALL}

__all__ = [
    "AEC_VAD",
    "ALL",
    "BY_KEY",
    "NNNR",
    "Recipe",
    "SED",
    "VOICE_FILTER",
    "VOICE_FILTER_SPEAKER_EMBEDDER",
]
