# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_onnx_args
from ...graph_edit.harness import add_graph_edit_harness_args


DEFAULT_INPUT_AUDIO_S: Final[int] = 5
DEFAULT_DEC_TOK_PER_SEC: Final[int] = 6
DEFAULT_MODEL_SIZE: Final[str] = "tiny"
ONNX_DTYPES: Final[list[str]] = ["float", "quantized", "quantized_4bit"]
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
STATIC_MODEL_COMPONENTS: Final[list[str]] = ["encoder", "decoder"]
STATIC_MODEL_COMPONENTS_UNFOLDED: Final[list[str]] = ["encoder", "gen_encoder_cache", "decoder"]
DEFAULT_CONV_KERNEL_SIZES: Final[list[int]] = [127, 7, 3]
DEFAULT_CONV_STRIDES: Final[list[int]] = [64, 3, 2]


def add_moonshine_export_args(parser: argparse.ArgumentParser):
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
        choices=["base", "tiny"],
        default=DEFAULT_MODEL_SIZE,
        help="Moonshine model size to export (default: %(default)s)",
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
        "--split-encoder",
        action="store_true",
        default=False,
        help="Split merged encoder into preprocessor and encoder models"
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
        "--no-fold-encoder-cache",
        action="store_true",
        default=False,
        help="Keep gen_encoder_cache as a separate model instead of folding into encoder"
    )
    parser.add_argument(
        "--skip-torq",
        type=str,
        nargs="+",
        choices=["all", "encoder", "preprocessor", "gen_encoder_cache", "decoder"],
        default=None,
        help="Skip Torq export/compile: 'all' to skip entirely, or specify components to skip."
    )
    parser.add_argument(
        "--replace-int-bf16-cast",
        action="store_true",
        default=False,
        help="Replace int64 -> bf16 casts with a look-up table"
    )
    parser.add_argument(
        "--skip-export",
        type=str,
        nargs="+",
        choices=["encoder", "gen_encoder_cache", "decoder", "decoder_merged"],
        help="Skip export of specific components"
    )
    parser.add_argument(
        "--combine-kv-io",
        action="store_true",
        default=False,
        help="Keep KV I/O as separate key, value tensors instead of combining"
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
        help="Moonshine model's HuggingFace repo ID (default: UsefulSensors/moonshine-{model_size})"
    )
    add_graph_edit_harness_args(parser)
    add_logging_args(parser)
    add_torq_args(parser)


def add_moonshine_infer_args(parser: argparse.ArgumentParser):
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
        choices=["base", "tiny"],
        help="Moonshine model size"
    )
    parser.add_argument(
        "--max-inp-len",
        type=int,
        help="Maximum input length (required for static VMFB models)",
    )
    parser.add_argument(
        "--max-dec-len",
        type=int,
        help="Maximum decoder length (required for static VMFB models)",
    )
    add_common_args(parser)
    add_logging_args(parser)
