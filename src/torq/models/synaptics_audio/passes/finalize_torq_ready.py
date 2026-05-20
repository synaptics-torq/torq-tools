# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Final shape inference + ``value_info`` cleanup + IR-version cap.

Wraps :func:`torq.utils.onnx.finalize_torq_ready_onnx` (the in-tree implementation)
as the last pass of :data:`PIPELINE`.

What the underlying helper does:

* ORT symbolic shape inference (resolves ``unk__`` dims where possible);
* drops ``graph.value_info`` entries whose names duplicate ``graph.output``
  (avoids rank mismatches in torch-onnx import for isolated subgraphs);
* caps ``ir_version`` for broader onnxruntime / tooling compatibility;
* refreshes standard ``onnx`` shape inference and runs the ONNX checker.
"""

from __future__ import annotations

import onnx

from torq.utils.onnx import finalize_torq_ready_onnx

from .base import PassContext


class FinalizeTorqReady:
    """Pass: prepare the (already-simplified) ONNX for the Torq importer."""

    name = "finalize_torq_ready"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        return finalize_torq_ready_onnx(model)
