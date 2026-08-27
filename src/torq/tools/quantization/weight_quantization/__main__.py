# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""CLI for ONNX weight quantization.

Usage
-----
Quantize all weights uniformly::

    python -m torq.tools.quantization.weight_quantization quantize \\
        -i model.onnx -o model_int8.onnx --bits 8

Quantize with per-layer config from sensitivity analysis::

    python -m torq.tools.quantization.weight_quantization quantize \\
        -i model.onnx -o model_mixed.onnx --config quant_config.json

Produce a bf16 model with quantization error baked in::

    python -m torq.tools.quantization.weight_quantization quantize \\
        -i model.onnx -o model_bf16.onnx --bits 8 --dequantize-weights

Run sensitivity analysis::

    python -m torq.tools.quantization.weight_quantization analyze \\
        -i model.onnx -o sensitivity.json --config-output quant_config.json

Convert to bf16 only::

    python -m torq.tools.quantization.weight_quantization quantize \\
        -i model.onnx -o model_bf16.onnx --bits 16
"""

from __future__ import annotations

import argparse

from . import add_weights_args, weights_from_args


def main():
    parser = argparse.ArgumentParser(
        prog="torq.tools.quantization.weight_quantization",
        description="ONNX weight quantization tool — int4/int8/bf16 with per-layer sensitivity analysis",
    )
    add_weights_args(parser)
    args = parser.parse_args()
    weights_from_args(args)


if __name__ == "__main__":
    main()
