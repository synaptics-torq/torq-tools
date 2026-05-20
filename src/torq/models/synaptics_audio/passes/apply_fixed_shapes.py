# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Stamp explicit input-shape overrides onto the model's ``graph.input``.

:func:`prepare.prepare` auto-discovers input shapes from the source ONNX, so
this pass is a no-op for the common case where the source already has static
inputs. It only does work when ``Recipe.input_shape_overrides`` is non-empty
-- i.e. when the source has a dynamic ``dim_param`` that the user wants to
pin to a concrete value.

Output shapes are intentionally not stamped: ``shape_inference`` (run inside
the downstream simplification passes) resolves them from the now-static input
shapes.
"""

from __future__ import annotations

import copy
import logging
from typing import Mapping, Sequence

import onnx

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.apply_fixed_shapes")


def _replace_value_info_shape(value_info, shape: Sequence[int]) -> None:
    tensor_type = value_info.type.tensor_type
    while len(tensor_type.shape.dim) < len(shape):
        tensor_type.shape.dim.add()
    for i, dim in enumerate(shape):
        d = tensor_type.shape.dim[i]
        d.ClearField("dim_param")
        d.dim_value = int(dim)


def _stamp(value_infos, name_to_shape: Mapping[str, Sequence[int]]) -> int:
    stamped = 0
    for vi in value_infos:
        if vi.name in name_to_shape:
            _replace_value_info_shape(vi, name_to_shape[vi.name])
            logger.info("input %s shape -> %s", vi.name, list(name_to_shape[vi.name]))
            stamped += 1
    return stamped


class ApplyFixedShapes:
    """Pass: stamp ``ctx.input_shapes`` onto ``graph.input`` (no-op if empty)."""

    name = "apply_fixed_shapes"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        if not ctx.input_shapes:
            return model
        out = copy.deepcopy(model)
        _stamp(out.graph.input, ctx.input_shapes)
        return out
