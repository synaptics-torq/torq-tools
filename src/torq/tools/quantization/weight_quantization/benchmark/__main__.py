# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Benchmark CLI for quantized ONNX models.

Runs a set of questions through a Gemma3 model on the board and collects
token throughput (TPS), time-to-first-token (TTFT), and answer quality.

Usage
-----
Run benchmark on board::

    python -m torq.tools.quantization.weight_quantization.benchmark run \
        -m /path/to/model.vmfb --instruct-model -o results.json

Compare two benchmark results::

    python -m torq.tools.quantization.weight_quantization.benchmark compare \
        -a results_int8.json -b results_hybrid.json -o comparison.md

"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default questions
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    "What is photosynthesis?",
    "Explain quantum mechanics.",
    "What is the capital of India?",
    "What is the capital of Italy?",
    "What is the capital of USA?",
    "Is AI dangerous?",
    "What is gravity?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "What is DNA?",
    "What causes rain?",
    "What is the largest ocean?",
    "Who invented the telephone?",
    "What is the boiling point of water?",
    "What is a black hole?",
    "What is the tallest mountain?",
    "Who painted the Mona Lisa?",
    "What is evolution?",
    "What is the chemical formula for water?",
    "What is the nearest star to Earth?",
    "What is an atom?",
    "What is the speed of sound?",
    "Who discovered penicillin?",
    "What is the largest planet in our solar system?",
]


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Path to model VMFB (or ONNX for ORT inference)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="benchmark_results.json",
        help="Output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--instruct-model", action="store_true", default=False,
        help="Use instruct-model chat template",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Max sequence length (auto-detected if omitted)",
    )
    parser.add_argument(
        "-j", "--threads", type=int, default=None,
        help="Number of inference threads",
    )
    parser.add_argument(
        "--questions-file", type=str, default=None,
        help="JSON file with list of question strings (default: built-in 24 questions)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: %(default)s = greedy)",
    )
    parser.add_argument(
        "--runner-path", type=str, default=None,
        help="Path to directory containing runner.py (default: auto-detect from model dir)",
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    from .run import run_benchmark

    questions = DEFAULT_QUESTIONS
    if args.questions_file:
        questions = json.loads(open(args.questions_file).read())

    results = run_benchmark(
        model_path=args.model,
        questions=questions,
        instruct_model=args.instruct_model,
        max_seq_len=args.max_seq_len,
        n_threads=args.threads,
        temperature=args.temperature,
        runner_path=args.runner_path,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to %s", args.output)

    # Print summary
    tps_vals = [r["tps"] for r in results]
    ttft_vals = [r["ttft_ms"] for r in results]
    tok_vals = [r["tokens"] for r in results]
    total = sum(tok_vals)
    print(f"\n=== Benchmark Summary ===")
    print(f"Questions: {len(results)}")
    print(f"Avg TPS: {sum(tps_vals)/len(tps_vals):.1f} ({min(tps_vals):.1f}-{max(tps_vals):.1f})")
    print(f"Avg TTFT: {sum(ttft_vals)/len(ttft_vals):.0f} ms")
    print(f"Total tokens: {total}, Avg/question: {total/len(results):.1f}")


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------


def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-a", "--model-a", type=str, required=True,
        help="Path to first model's benchmark JSON",
    )
    parser.add_argument(
        "-b", "--model-b", type=str, required=True,
        help="Path to second model's benchmark JSON",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="comparison.md",
        help="Output markdown comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--name-a", type=str, default="Model A",
        help="Display name for first model",
    )
    parser.add_argument(
        "--name-b", type=str, default="Model B",
        help="Display name for second model",
    )


def _run_compare(args: argparse.Namespace) -> None:
    from .compare import generate_comparison

    results_a = json.loads(open(args.model_a).read())
    results_b = json.loads(open(args.model_b).read())

    md = generate_comparison(
        results_a, results_b,
        name_a=args.name_a, name_b=args.name_b,
    )

    with open(args.output, "w") as f:
        f.write(md)
    logger.info("Saved comparison to %s", args.output)
    print(f"Comparison saved to {args.output}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="torq.tools.quantization.weight_quantization.benchmark",
        description="Benchmark quantized models — run inference and compare results",
    )
    parser.add_argument(
        "--logging", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run benchmark on a model")
    _add_run_args(run_parser)

    cmp_parser = sub.add_parser("compare", help="Compare two benchmark results")
    _add_compare_args(cmp_parser)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging.upper()),
        format="[%(levelname)-8s] %(message)s",
    )

    if args.command == "run":
        _run_benchmark(args)
    elif args.command == "compare":
        _run_compare(args)


if __name__ == "__main__":
    main()
