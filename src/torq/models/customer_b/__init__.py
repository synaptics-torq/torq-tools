# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Customer B model support.

This module provides export, inference, and validation for Customer B ONNX models
(all_conv, all_fc, all_lstm).  The pipeline is:

    ONNX  →  (onnx2tf)  →  TFLite (int8)  →  (iree-import-tflite)  →  TOSA  →  MLIR  →  VMFB

Models directory: ``models/customer_b/``
"""

import argparse
from typing import Final

from torq.compile import add_iree_args
from torq.utils.logging import add_logging_args


# Available ONNX model components
MODEL_COMPONENTS: Final[dict[str, str]] = {
    "all_conv": "all_conv.onnx",
    "all_fc": "all_fc.onnx",
    "all_lstm": "all_lstm.onnx",
}

# Default models directory (relative to repo root)
DEFAULT_MODELS_DIR: Final[str] = "models/customer_b"


def add_customer_b_export_args(parser: argparse.ArgumentParser):
    """Add CLI arguments for Customer B model export."""
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        metavar="DIR",
        help="Directory containing Customer B ONNX models (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_customer_b",
        metavar="DIR",
        help="Output directory for converted models (default: %(default)s)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=list(MODEL_COMPONENTS.keys()),
        default=None,
        help="Export only a specific component (default: all)",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        default=True,
        help="Quantize to int8 TFLite (default: True)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=100,
        help="Number of random samples for int8 calibration (default: %(default)d)",
    )
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        default=False,
        help="Skip ONNX → TFLite conversion (use existing TFLite files)",
    )
    parser.add_argument(
        "--skip-iree",
        action="store_true",
        default=False,
        help="Skip TFLite → MLIR → VMFB compilation",
    )
    add_logging_args(parser)
    add_iree_args(parser)


def add_customer_b_infer_args(parser: argparse.ArgumentParser):
    """Add CLI arguments for Customer B model inference."""
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing compiled VMFB or TFLite models",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=list(MODEL_COMPONENTS.keys()),
        default="all_fc",
        help="Which model component to run (default: %(default)s)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Input data file (.bin or .npy). If not provided, uses random data.",
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    add_logging_args(parser)
