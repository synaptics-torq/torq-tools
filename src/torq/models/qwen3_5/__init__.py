# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.logging import add_logging_args

DEFAULT_HF_REPO: Final[str] = "Qwen/Qwen3.5-0.8B"
DEFAULT_CONTEXT: Final[int] = 256


def add_qwen35_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face source repository (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for the downloaded and exported models (default: %(default)s)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help="Static context length: KV cache slots for prompt and generation (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-quantization",
        choices=["int4", "none"],
        default="int4",
        help=(
            "Round-to-nearest block-32 INT4 DequantizeLinear weights for the transformer and "
            "LM head MatMuls, or bf16 weights (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Do not compare the static decode step with the Hugging Face model",
    )
    add_torq_args(parser, skip=True)
    add_logging_args(parser)
