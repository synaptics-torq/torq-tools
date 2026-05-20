# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics/SED (Sound Event Detection) recipe."""

from __future__ import annotations

from .base import Recipe

SED = Recipe(
    key="sed",
    repo_id="Synaptics/SED",
    source_filename=(
        "Vivint_GB_SED_v3.57.902.150_e365_0.941/"
        "model[Vivint_GB_SED_v3.57.902.150].onnx"
    ),
    input_shape_overrides={"input_131": (1, 70, 45, 1)},
)
