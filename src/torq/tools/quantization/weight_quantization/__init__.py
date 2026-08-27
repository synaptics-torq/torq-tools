# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .quantize import add_weight_quantize_args, weight_quantize_from_args
from .sensitivity import add_weight_analyze_args, weight_analyze_from_args

__all__ = [
    "add_weights_args",
    "weights_from_args",
    "add_weight_quantize_args",
    "weight_quantize_from_args",
    "add_weight_analyze_args",
    "weight_analyze_from_args"
]


def add_weights_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``weights`` method's action subparsers to ``parser``."""
    action = parser.add_subparsers(dest="command", required=True)

    quantize = action.add_parser(
        "quantize",
        help="Quantize MatMul weights in an fp32 ONNX model",
    )
    add_weight_quantize_args(quantize)

    analyze = action.add_parser(
        "analyze",
        help="Run per-layer quantization sensitivity analysis",
    )
    add_weight_analyze_args(analyze)


def weights_from_args(args: argparse.Namespace) -> None:
    """Dispatch a parsed ``weights`` namespace to its action handler."""
    if args.command == "quantize":
        weight_quantize_from_args(args)
    elif args.command == "analyze":
        weight_analyze_from_args(args)
