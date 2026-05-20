# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Pass protocol and shared context used by every ``synaptics_audio`` pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import onnx


@dataclass(frozen=True, slots=True)
class PassContext:
    """Read-only model metadata shared across all passes in :data:`PIPELINE`.

    ``input_shapes`` is the resolved ``{input_name: static_shape}`` map for the
    model being prepared (auto-discovered from the source ONNX, optionally
    augmented by ``Recipe.input_shape_overrides``). Most passes ignore it; only
    :class:`apply_fixed_shapes.ApplyFixedShapes` consumes it directly.
    """

    input_shapes: Mapping[str, Sequence[int]] = field(default_factory=dict)


class OnnxPass(Protocol):
    """An ONNX -> ONNX rewrite that must be value-preserving in FP32.

    Implementations must:

    * be idempotent: running the same pass twice produces the same model;
    * be self-skipping: if the pass's pattern doesn't appear in the model,
      return the input unchanged (do not raise);
    * not change graph I/O names or dtypes.
    """

    name: str

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto: ...
