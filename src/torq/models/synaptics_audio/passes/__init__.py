# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""ONNX simplification passes for Synaptics audio models.

:data:`PIPELINE` is the single fixed sequence of passes applied by
:func:`prepare.prepare`. Order matters:

1. shape-resolving and static no-op cleanup passes first
   (``ApplyFixedShapes`` -> ``FreezeShapeSeeds`` ->
   ``EliminateDegenerateTranspose`` -> ``ReplacePadInputsWithConstants``) so
   later rewrites see static shapes;
2. graph rewrites that turn unsupported constructs into supported ones
   (``DecomposeBidirectionalRnn``, ``EliminateRank0Gather``);
3. Pad/Conv shape rewrites
   (``RewriteNegativePads`` -> ``AbsorbPadding``), then final static rank-shim
   cleanup (``EliminateSingletonGatherUnsqueeze``) and
   ``WidenStridedDepthwiseConv``;
4. ``FinalizeTorqReady`` last (caps IR, runs symbolic shape infer, cleans up
   ``value_info``).

Adding a pass = drop a new file, import the class here, and append it to
:data:`PIPELINE`. The runner has no per-model overrides: self-skipping passes
(those whose pattern doesn't match the model) must return the model unchanged
(see :class:`base.OnnxPass`).
"""

from __future__ import annotations

from .absorb_padding import AbsorbPadding
from .apply_fixed_shapes import ApplyFixedShapes
from .base import OnnxPass, PassContext
from .decompose_bidirectional_rnn import DecomposeBidirectionalRnn
from .eliminate_degenerate_transpose import EliminateDegenerateTranspose
from .eliminate_rank0_gather import EliminateRank0Gather
from .eliminate_singleton_gather_unsqueeze import (
    EliminateSingletonGatherUnsqueeze,
)
from .finalize_torq_ready import FinalizeTorqReady
from .freeze_shape_seeds import FreezeShapeSeeds
from .replace_pad_inputs_with_constants import ReplacePadInputsWithConstants
from .rewrite_negative_pads import RewriteNegativePads
from .widen_strided_depthwise_conv import WidenStridedDepthwiseConv

PIPELINE: tuple[OnnxPass, ...] = (
    ApplyFixedShapes(),
    FreezeShapeSeeds(),
    EliminateDegenerateTranspose(),
    ReplacePadInputsWithConstants(),
    DecomposeBidirectionalRnn(),
    EliminateRank0Gather(),
    RewriteNegativePads(),
    AbsorbPadding(),
    EliminateSingletonGatherUnsqueeze(),
    WidenStridedDepthwiseConv(),
    FinalizeTorqReady(),
)

__all__ = [
    "AbsorbPadding",
    "ApplyFixedShapes",
    "DecomposeBidirectionalRnn",
    "EliminateDegenerateTranspose",
    "EliminateRank0Gather",
    "EliminateSingletonGatherUnsqueeze",
    "FinalizeTorqReady",
    "FreezeShapeSeeds",
    "OnnxPass",
    "PIPELINE",
    "PassContext",
    "ReplacePadInputsWithConstants",
    "RewriteNegativePads",
    "WidenStridedDepthwiseConv",
]
