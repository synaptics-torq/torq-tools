# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from torq.compile import add_iree_args
from torq.utils.logging import add_logging_args

from ...utils.demo import add_common_args
from ...utils.onnx import add_onnx_args


HF_MODEL_REPO: Final[str] = "Synaptics/WashingBERT"
HF_TOKENIZER_REPO: Final[str] = "line-corporation/line-distilbert-base-japanese"

HF_MODEL_FILES: Final[list[str]] = [
    "best_multi_task_model_fp16.onnx",
    "intent_classes.json",
    "types_classes.json",
    "sec_types_classes.json",
]

LABEL_FILES: Final[dict[str, str]] = {
    "intents": "intent_classes.json",
    "type1": "types_classes.json",
    "type2": "sec_types_classes.json",
}

MODEL_COMPONENTS: Final[dict[str, str]] = {
    "model": "best_multi_task_model_fp16.onnx",
}

ONNX_DTYPES: Final[list[str]] = ["float", "fp32", "fp16", "bf16"]
DEFAULT_MAX_SEQ_LEN: Final[int] = 128


def add_washingbert_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum input sequence length (default: %(default)d)",
    )
    add_onnx_args(
        parser,
        model_dtypes=ONNX_DTYPES,
        convert_dtypes=True,
        allow_no_opt=True,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-iree",
        action="store_true",
        default=False,
        help="Skip exporting to IREE",
    )
    add_logging_args(parser)
    add_iree_args(parser)


def add_washingbert_infer_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "inputs",
        type=str,
        metavar="TEXT",
        nargs="+",
        help="Japanese text input(s) for classification",
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to WashingBERT model directory",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum input sequence length (default: %(default)d)",
    )
    add_common_args(parser)
    add_logging_args(parser)
