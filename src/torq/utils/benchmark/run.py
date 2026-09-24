# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Run benchmark inference on a Gemma3 model."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_benchmark(
    model_path: str,
    questions: list[str],
    *,
    instruct_model: bool = True,
    max_seq_len: int | None = None,
    n_threads: int | None = None,
    temperature: float = 0.0,
    runner_path: str | None = None,
) -> list[dict]:
    """Run benchmark questions through a Gemma3 model.

    Parameters
    ----------
    model_path : str
        Path to model VMFB or ONNX file.
    questions : list[str]
        List of questions to evaluate.
    instruct_model : bool
        Whether to use instruct-model chat template.
    max_seq_len : int, optional
        Max sequence length (auto-detected if None).
    n_threads : int, optional
        Number of inference threads.
    temperature : float
        Sampling temperature (0.0 = greedy).
    runner_path : str, optional
        Path to directory containing runner.py. If None, searches
        the model directory and common locations.

    Returns
    -------
    list[dict]
        List of result dicts with keys: question, answer, tokens, tps,
        ttft_ms, total_ms.
    """
    # Resolve runner import path
    model_dir = Path(model_path).resolve().parent
    search_paths = [
        Path(runner_path) if runner_path else None,
        model_dir / ".." / ".." / "gemma3" / "src",
        model_dir / ".." / "src",
        model_dir,
    ]

    runner_found = False
    for p in search_paths:
        if p is None:
            continue
        p = p.resolve()
        if (p / "runner.py").exists():
            sys.path.insert(0, str(p))
            runner_found = True
            logger.debug("Using runner from %s", p)
            break

    if not runner_found:
        raise FileNotFoundError(
            f"Cannot find runner.py. Searched: {[str(p) for p in search_paths if p]}. "
            f"Use --runner-path to specify the directory containing runner.py."
        )

    from runner import Gemma3Static  # type: ignore[import]

    logger.info("Loading model: %s", model_path)
    gemma3 = Gemma3Static(
        model_path,
        max_seq_len,
        instruct_model=instruct_model,
        n_threads=n_threads,
        temperature=temperature,
        top_p=1.0,
        top_k=64,
    )

    results = []
    for i, q in enumerate(questions):
        sys.stdout.write(f"\r[{i+1}/{len(questions)}] {q[:50]:50s}")
        sys.stdout.flush()

        answer = gemma3.run(q)
        decode_ms = gemma3.last_infer_time - gemma3.time_to_first_token
        tps = gemma3.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0

        results.append({
            "question": q,
            "answer": answer.strip() if answer else "",
            "tokens": gemma3.generated_tokens,
            "tps": round(tps, 1),
            "ttft_ms": round(gemma3.time_to_first_token),
            "total_ms": round(gemma3.last_infer_time),
        })
        sys.stdout.write(f" -> {gemma3.generated_tokens} tok, {tps:.1f} tok/s\n")

    return results
