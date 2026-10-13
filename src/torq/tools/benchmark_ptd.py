# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Benchmark ordinary greedy decode vs Progressive Tree Drafting (PTD).

Runs both runners over the same prompts, verifies token-identical output,
and reports the plan's per-round metrics: processed rows, committed tokens,
candidate hit rate, accepted path length, pool warmup (rounds to first hit),
total latency, and the round-cost speedup estimate
``mean committed tokens * ordinary step time / mean PTD round time``.

PTD: Gao et al. 2026, https://arxiv.org/abs/2607.10661 (attribution in
torq.inference.ptd).

Usage:
    python -m torq.tools.benchmark_ptd --model transformer.onnx \
        --ptd-model ptd_transformer.onnx [--instruct-model] [prompts...]
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from ..models.gemma3._inference import Gemma3PTD, Gemma3Static

DEFAULT_PROMPTS = [
    "Hello",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "Explain what a CPU is in two sentences.",
    "Write a python function to reverse a string.",
    "Repeat the word hello five times.",
]


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS
    return ru / 1024 if platform.system() == "Linux" else ru / (1024 * 1024)


def _model_max_tokens(model_path) -> int:
    """The static cache dimension (rows) baked into the exported model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values.") and len(i.shape) == 4 and isinstance(i.shape[2], int):
            return i.shape[2]
    raise ValueError(f"Could not read the KV cache size from {model_path}")


def _decode_step_time(runner, prompt: str, max_gen_tokens: int, n_warm: int = 1) -> tuple[float, float, int]:
    """(prefill_s, decode_s, tokens): a 0-token run isolates the prefill."""
    for _ in range(n_warm):
        runner.run(prompt)
    t0 = time.perf_counter()
    saved = runner._max_gen_tokens
    runner._max_gen_tokens = 0  # _run() would otherwise latch this on later calls
    try:
        runner.run(prompt)
    finally:
        runner._max_gen_tokens = saved
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    runner.run(prompt, max_gen_tokens=max_gen_tokens)
    total = time.perf_counter() - t0
    tokens = runner._n_tokens_gen
    return prefill, max(total - prefill, 0.0), tokens


def _ptd_timings(ptd_runner, prompt: str, max_gen_tokens: int) -> tuple[float, float, int, list[int]]:
    """(ptd_prefill_s, ptd_total_s, tokens, output) with a warmup run first
    and the prefill isolated via a 0-generation run (same trick as the
    static measurement). PTD rounds record their own elapsed time."""
    ptd_runner.run(prompt, max_gen_tokens=max_gen_tokens)
    t0 = time.perf_counter()
    saved = ptd_runner._max_gen_tokens
    ptd_runner._max_gen_tokens = 0
    try:
        ptd_runner.run(prompt)
    finally:
        ptd_runner._max_gen_tokens = saved
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    ptd_out = ptd_runner.run_tokens(prompt, max_gen_tokens=max_gen_tokens)
    total = time.perf_counter() - t0
    return prefill, total, ptd_runner._n_tokens_gen, ptd_out


def _ptd_summary(ptd_runner) -> dict:
    log = ptd_runner.ptd_round_log
    rounds = len(log)
    committed = sum(e["emitted"] for e in log)
    hits = sum(1 for e in log if e["hit"])
    accepted = [e["accepted"] for e in log]
    warmup = next((e["round"] for e in log if e["hit"]), None)
    return {
        "rounds": rounds,
        "committed_tokens": committed,
        "rows_processed": sum(e["rows"] for e in log),
        "active_rows": sum(e["n_candidates"] + e["n_drafts"] + 1 for e in log),
        "hit_rate": hits / rounds if rounds else 0.0,
        "mean_accept_length": sum(accepted) / rounds if rounds else 0.0,
        "max_accept_length": max(accepted, default=0),
        "mean_body_ms": sum(e["body_ms"] for e in log) / rounds if rounds else 0.0,
        "mean_round_ms": sum(e["round_ms"] for e in log) / rounds if rounds else 0.0,
        "pool_warmup_round": warmup,
    }


def run_benchmark(args) -> dict:
    max_gen_tokens = args.max_gen_tokens
    model_max_tokens = args.model_max_tokens or _model_max_tokens(args.model)
    prompts = args.prompts or DEFAULT_PROMPTS

    static_runner = Gemma3Static.from_onnx(
        args.model, model_max_tokens, max_inp_len=args.max_inp_len,
        n_threads=args.n_threads, instruct_model=args.instruct_model,
        repo_id=args.repo_id, lm_head_path=args.lm_head,
    )
    ptd_runner = Gemma3PTD.from_onnx(
        args.model, args.ptd_model, model_max_tokens, max_inp_len=args.max_inp_len,
        n_threads=args.n_threads, instruct_model=args.instruct_model,
        repo_id=args.repo_id, lm_head_path=args.lm_head,
        ptd_lm_head_path=args.ptd_lm_head, ptd_rows=args.ptd_rows,
        ptd_width=args.ptd_width, ptd_depth=args.ptd_depth,
    )

    report = {"prompts": {}, "totals": {}}
    all_step_times = []
    total_committed = 0
    total_tokens = 0
    identical = True

    for prompt in prompts:
        prefill_s, decode_s, n_tok = _decode_step_time(static_runner, prompt, max_gen_tokens)
        step_time = decode_s / max(n_tok - 1, 1)
        ptd_prefill, ptd_total, ptd_n_tok, ptd_out = _ptd_timings(ptd_runner, prompt, max_gen_tokens)
        ptd_summary = _ptd_summary(ptd_runner)
        # Fallback 1-token steps (cache too full for the row budget) are not
        # included in the measured PTD round times.
        ptd_summary["fallback_steps"] = ptd_n_tok - 1 - ptd_summary["committed_tokens"]
        ptd_summary["prefill_s"] = ptd_prefill
        static_out = static_runner.run_tokens(prompt, max_gen_tokens=max_gen_tokens)
        match = ptd_out == static_out
        static_text = static_runner._tokenizer.decode(static_out)
        ptd_text = ptd_runner._tokenizer.decode(ptd_out)
        identical &= match
        all_step_times.append(step_time)
        total_committed += ptd_summary["committed_tokens"]
        total_tokens += n_tok
        report["prompts"][prompt] = {
            "static": {"tokens": n_tok, "step_s": step_time, "output": static_text[:80]},
            "ptd": {**ptd_summary, "total_s": ptd_total, "output": ptd_text[:80]},
            "outputs_match": match,
        }
        print(f"\n=== {prompt!r}")
        print(f"  static: {n_tok} tokens, step {step_time*1e3:.1f} ms")
        print(f"  ptd:    {ptd_summary['rounds']} rounds + {ptd_summary['fallback_steps']} fallback, "
              f"{ptd_summary['committed_tokens']} committed, hit {ptd_summary['hit_rate']:.0%}, "
              f"accept {ptd_summary['mean_accept_length']:.2f}, body {ptd_summary['mean_body_ms']:.1f} ms, "
              f"warmup round {ptd_summary['pool_warmup_round']}")
        print(f"  match:  {match}")
        print(f"  output: {ptd_text[:100]!r}")

    report["totals"] = {
        "outputs_token_identical": bool(identical),
        "mean_step_s": sum(all_step_times) / len(all_step_times),
        "committed_tokens": total_committed,
        "generated_tokens": total_tokens,
        "peak_rss_mb": _peak_rss_mb(),
    }
    # Round-cost speedup (plan formula): mean committed per round * step /
    # round time, with prefill and fallback steps excluded from the round
    # time.
    speedups = []
    for entry in report["prompts"].values():
        p = entry["ptd"]
        if p["rounds"] == 0:
            continue
        mean_committed = p["committed_tokens"] / p["rounds"]
        round_time = p["mean_round_ms"] / 1e3
        speedups.append(mean_committed * entry["static"]["step_s"] / round_time if round_time > 0 else 0.0)
    report["totals"]["speedup_est"] = sum(speedups) / len(speedups) if speedups else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="ordinary static decode model (transformer.onnx)")
    parser.add_argument("--ptd-model", required=True, help="PTD body (ptd_transformer.onnx)")
    parser.add_argument("--lm-head", default=None, help="decode lm_head.onnx (default: auto next to --model)")
    parser.add_argument("--ptd-lm-head", default=None, help="PTD ptd_lm_head.onnx (default: auto next to --ptd-model)")
    parser.add_argument("--ptd-rows", type=int, default=None, help="PTD row count (default: read from model)")
    parser.add_argument("--model-max-tokens", type=int, default=None,
                        help="static cache rows (default: read from --model)")
    parser.add_argument("--ptd-width", type=int, default=4)
    parser.add_argument("--ptd-depth", type=int, default=6)
    parser.add_argument("--max-gen-tokens", type=int, default=64)
    parser.add_argument("--max-inp-len", type=int, default=None)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--instruct-model", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--json", default=None, help="write the full report to this path")
    parser.add_argument("prompts", nargs="*", help="prompts (default: built-in set)")
    args = parser.parse_args()

    report = run_benchmark(args)
    totals = report["totals"]
    speedup = f"{totals['speedup_est']:.2f}x" if totals["speedup_est"] is not None else "n/a"
    print(f"\n=== totals: outputs_token_identical={totals['outputs_token_identical']} "
          f"mean step {totals['mean_step_s']*1e3:.1f} ms, speedup est {speedup} "
          f"(committed*step/round), peak RSS {totals['peak_rss_mb']:.0f} MB")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
