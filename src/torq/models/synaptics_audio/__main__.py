# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""CLI: ``python -m torq.models.synaptics_audio <recipe-key> <dst> [--src <src.onnx>]``.

When ``--src`` is omitted, the source FP32 ONNX file(s) are fetched from
HuggingFace using ``recipe.repo_id`` + ``recipe.source_filename`` (cached under
``$HF_HOME``/``~/.cache/huggingface/hub/``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .prepare import prepare
from .recipes import BY_KEY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torq.models.synaptics_audio",
        description="Prepare a Synaptics audio model for the Torq compiler.",
    )
    parser.add_argument(
        "recipe",
        choices=sorted(BY_KEY),
        help="Recipe key (which audio model to prepare).",
    )
    parser.add_argument(
        "dst",
        type=Path,
        help=(
            "Destination Torq-ready BF16 ONNX path, or an output directory. "
            "Directories write <source-stem>_torq_bf16.onnx."
        ),
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Override source FP32 ONNX path (default: fetch from HF "
        "using recipe.repo_id + recipe.source_filename).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    recipe = BY_KEY[args.recipe]
    try:
        dst = prepare(recipe, args.dst, src=args.src)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dst.is_dir() or args.dst.suffix == "":
        if isinstance(dst, list):
            for path in dst:
                print(path)
        else:
            print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
