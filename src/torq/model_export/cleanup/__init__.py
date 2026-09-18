# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""Model-agnostic ONNX graph cleanup.

Undo common exporter artifacts before dtype conversion
(``torq-convert-dtype`` from torq-compiler) and Torq compilation, by
composing the ``CollapseUnrolledConcat`` and ``FoldConvBatchNorm`` graph
edits with ORT-backed constant folding. See
:mod:`torq.model_export.cleanup.onnx`.

CLI: ``python -m torq.model_export.cleanup onnx IN.onnx -o OUT.onnx`` (or the
``torq-cleanup-model`` console script).
"""

from .onnx import (
    PASSES,
    add_onnx_cleanup_args,
    cleanup_onnx_model,
    onnx_cleanup_from_args,
)

__all__ = [
    "PASSES",
    "add_onnx_cleanup_args",
    "cleanup_onnx_model",
    "onnx_cleanup_from_args",
]
