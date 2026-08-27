# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""Model-agnostic ONNX graph cleanup.

Undo common exporter artifacts before dtype conversion
(``torq.tools.convert_dtype``) and Torq compilation, by composing the
``CollapseUnrolledConcat`` and ``FoldConvBatchNorm`` graph edits with
ORT-backed constant folding. See :mod:`torq.tools.onnx_cleanup.onnx`.

CLI: ``python -m torq.tools.onnx_cleanup IN.onnx -o OUT.onnx`` (or the
``torq-onnx-cleanup`` console script).
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
