# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from torq.compile import add_iree_args
from torq.utils.logging import add_logging_args

from ...utils.demo import add_common_args
from ...utils.onnx import add_onnx_args

ONNX_DTYPES: Final[list[str]] = ["float", "quantized", "quantized_4bit"]
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]

def add_fumi_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--audio-len",
        type=int,
        required=True,
        help="Set a maximum audio length multiplier u0 for audio dim 300*u0.",
    )
    parser.add_argument(
        "--text-len",
        type=int,
        required=True,
        help="Set a maximum text length for Input.",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        required=True,
        help="Path to the input ONNX model to export",
    )
    add_onnx_args(
        parser,
        model_dtypes=ONNX_DTYPES + OPTIMUM_DTYPES,
        convert_dtypes=True,
        allow_no_opt=False,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--extract-embeddings",
        action="store_true",
        default=False,
        help="Extract large embeddings tables into external .npy data"
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export dynamic models for CPU"
    )
    parser.add_argument(
        "--use-optimum",
        action="store_true",
        default=False,
        help="Use optimum-cli to generate ONNX models rather than loading prebuilt ones"
    )
    parser.add_argument(
        "--skip-iree",
        action="store_true",
        default=False,
        help="Skip exporting to IREE"
    )
    parser.add_argument(
        "--replace-int-bf16-cast",
        action="store_true",
        default=False,
        help="Replace int64 -> bf16 casts with a look-up table"
    )
    parser.add_argument(
        "--broadcast-ops",
        type=str,
        metavar="OP",
        nargs="*",
        default=None,
        help="Broadcast op inputs: specify ops or pass with no args to broadcast for all ops",
    )
    add_logging_args(parser)
    add_iree_args(parser)


def add_fumi_infer_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to Fumi model directory",
    )
    add_common_args(parser)
    add_logging_args(parser)
