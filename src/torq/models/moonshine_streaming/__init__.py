# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from torq.utils.logging import add_logging_args


DEFAULT_INPUT_AUDIO_S: Final[int] = 5
DEFAULT_DEC_TOK_PER_SEC: Final[int] = 6
DEFAULT_MODEL_SIZE: Final[str] = "tiny"


def add_moonshine_streaming_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input-seconds",
        type=int,
        default=DEFAULT_INPUT_AUDIO_S,
        help="Input audio length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="New audio per encoder chunk in seconds (default: same as --input-seconds, "
             "i.e. non-incremental). When set, the encoder static shape is sized for "
             "overlap-and-save incremental encoding: overlap + chunk + finalization_delay. "
             "The decoder cross-attends to a fixed buffer sized by --input-seconds.",
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
        choices=["tiny", "small"],
        default=DEFAULT_MODEL_SIZE,
        help="Moonshine streaming model size to export (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export dynamic-shape models only (skip static conversion)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip numerical validation of exported models",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repo ID (default: UsefulSensors/moonshine-streaming-{model_size})",
    )
    add_logging_args(parser)
