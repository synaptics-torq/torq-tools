# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse

from .quantize import (
    add_dynamic_quantize_args,
    dynamic_quantize_from_args,
    dynamic_quantize_model
)

__all__ = [
    "add_dynamic_args",
    "dynamic_from_args",
    "add_dynamic_quantize_args",
    "dynamic_quantize_from_args",
    "dynamic_quantize_model"
]


def add_dynamic_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``dynamic`` method's action subparsers to ``parser``."""
    action = parser.add_subparsers(dest="command", required=True)
    quantize = action.add_parser(
        "quantize",
        help="Dynamically quantize an fp32 ONNX model to int8 via onnxruntime",
    )
    add_dynamic_quantize_args(quantize)


def dynamic_from_args(args: argparse.Namespace) -> None:
    """Dispatch a parsed ``dynamic`` namespace to its action handler."""
    if args.command == "quantize":
        dynamic_quantize_from_args(args)
