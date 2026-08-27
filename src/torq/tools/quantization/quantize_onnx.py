# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Unified CLI for ONNX quantization.

Usage
-----
Weight-only quantization (int4/int8/bf16, per-layer mixed precision)::

    torq-quantize-model weights quantize -i model.onnx -o model_int8.onnx --bits 8
    torq-quantize-model weights analyze  -i model.onnx -o sensitivity.json --embeddings emb.npy

Dynamic quantization via onnxruntime (int8)::

    torq-quantize-model dynamic quantize -i model.onnx -o model_dynamic.onnx
"""

import argparse

from .weight_quantization import add_weights_args
from .dynamic_quantization import add_dynamic_args


def main():
    parser = argparse.ArgumentParser(
        prog="torq-quantize-model",
        description="Quantize ONNX models",
    )
    method = parser.add_subparsers(dest="method", required=True)

    weights = method.add_parser(
        "weights",
        help="Weight-only quantization (int4/int8/bf16, per-layer mixed precision)",
    )
    add_weights_args(weights)

    dynamic = method.add_parser(
        "dynamic",
        help="Dynamic quantization via onnxruntime (int8)",
    )
    add_dynamic_args(dynamic)

    args = parser.parse_args()

    if args.method == "weights":
        from .weight_quantization import weights_from_args
        weights_from_args(args)
    elif args.method == "dynamic":
        from .dynamic_quantization import dynamic_from_args
        dynamic_from_args(args)


if __name__ == "__main__":
    main()
