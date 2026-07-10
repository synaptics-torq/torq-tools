# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_onnx_args

DEFAULT_INPUT_AUDIO_S: Final[int] = 5
DEFAULT_DEC_TOK_PER_SEC: Final[int] = 6
DEFAULT_MODEL_SIZE: Final[str] = "tiny"
ONNX_DTYPES: Final[list[str]] = ["float"]
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]

STATIC_MODEL_COMPONENTS: Final[list[str]] = [
    "encoder",
    "decoder",
]


def add_moonshine_streaming_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input-seconds",
        type=int,
        default=DEFAULT_INPUT_AUDIO_S,
        help="Input audio length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tokens-per-sec",
        type=int,
        default=DEFAULT_DEC_TOK_PER_SEC,
        help="Max number of tokens decoded per second (default: %(default)d)",
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        choices=["tiny", "small", "medium"],
        default=DEFAULT_MODEL_SIZE,
        help="Moonshine Streaming model size to export (default: %(default)s)",
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
        help="Extract large embeddings tables into external .npy data",
    )
    parser.add_argument(
        "--export-attention",
        action="store_true",
        default=False,
        help="Also output the decoder cross-attention weights ('cross_attn', "
             "[depth,1,heads,1,max_memory_len]) for host-side token->frame alignment. "
             "Adds a per-step output (no extra compute).",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        metavar="N",
        help="Audio chunk size in samples for static streaming export (e.g. 1280 = 80ms @ 16kHz).",
    )
    parser.add_argument(
        "--skip-torq",
        type=str,
        nargs="+",
        choices=["all"] + STATIC_MODEL_COMPONENTS,
        default=None,
        help="Skip Torq export/compile: 'all' to skip entirely, or specify components to skip.",
    )
    parser.add_argument(
        "--skip-export",
        type=str,
        nargs="+",
        choices=STATIC_MODEL_COMPONENTS,
        help="Skip export of specific components",
    )
    parser.add_argument(
        "--broadcast-ops",
        type=str,
        metavar="OP",
        nargs="*",
        default=None,
        help="Broadcast op inputs: specify ops or pass with no args to broadcast for all ops",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="Moonshine Streaming model's HuggingFace repo ID (default: UsefulSensors/moonshine-streaming-{model_size})",
    )
    add_logging_args(parser)
    add_torq_args(parser)


def add_moonshine_streaming_infer_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "inputs",
        type=str,
        metavar="WAV",
        nargs="+",
        help="WAV files for inference",
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to Moonshine model directory",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        required=True,
        choices=["tiny", "small", "medium"],
        help="Moonshine Streaming model size",
    )
    add_common_args(parser)
    add_logging_args(parser)
