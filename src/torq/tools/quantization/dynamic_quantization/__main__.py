# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""CLI for ONNX 8-bit dynamic quantization via onnxruntime.

Usage
-----

Default quantization (per-channel, int8 weights)::

    python -m torq.tools.quantization.dynamic \\
        -i model.onnx -o model_quantized.onnx

Uint8 weights + per-tensor quantization::

    python -m torq.tools.quantization.dynamic \\
        -i model.onnx -o model_quantized.onnx --uint8-weights --per-tensor
"""

from __future__ import annotations

from .quantize import (
    add_dynamic_quantize_args,
    dynamic_quantize_from_args
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ONNX dynamic quantization tool",
    )
    add_dynamic_quantize_args(parser)
    args = parser.parse_args()
    dynamic_quantize_from_args(args)
