# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )


def format_answer(
    answer: str,
    infer_time: float,
    agent_name: str = "Agent"
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    return GREEN + f"{agent_name}: {answer}" + RESET + f" ({infer_time * 1000:.3f} ms)"