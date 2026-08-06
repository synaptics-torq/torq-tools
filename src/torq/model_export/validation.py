# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Functional-equivalence check shared by the decoder-only ONNX exporters."""

import logging
import os
from pathlib import Path
from typing import Final

__all__ = [
    "VALIDATION_PROMPTS",
    "validate_decoder_only_onnx",
]

# Simple dataset to test functional equivalence.
VALIDATION_PROMPTS: Final[tuple[str, ...]] = (
    # very short (position_ids = 0 edge case)
    "Hello",

    # normal medium-length prompt
    "The quick brown fox jumps over the lazy dog.",

    # repetitive tokens (attention accumulation / stability)
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",

    # non-ASCII / multi-token UTF-8
    "こんにちは世界",

    # structured / punctuation-heavy (tokenizer edge cases)
    "def foo(x): return x * 2 # simple test",
)


def validate_decoder_only_onnx(
    logger: logging.Logger,
    static_cls,
    dynamic_cls,
    model_path: str | os.PathLike,
    reference_model_path: str | os.PathLike,
    *,
    static_models: bool,
    max_gen_tokens: int,
    instruct_model: bool = False,
    repo_id: str | None = None,
    n_iters: int = 5,
    n_threads: int | None = None,
):
    """Compare the exported model's output against the unedited source ONNX.

    The reference is always `dynamic_cls` on the source `model.onnx`; the model
    under test is static or dynamic depending on how it was exported.
    """
    n_threads = n_threads or os.cpu_count()
    if static_models:
        runner = static_cls.from_onnx(
            model_path,
            max_gen_tokens,
            n_threads=n_threads,
            instruct_model=instruct_model,
            repo_id=repo_id,
        )
    else:
        runner = dynamic_cls.from_onnx(
            model_path,
            max_gen_tokens=max_gen_tokens,
            n_threads=n_threads,
            instruct_model=instruct_model,
            repo_id=repo_id,
        )
    val_runner = dynamic_cls.from_onnx(
        Path(reference_model_path),
        max_gen_tokens=max_gen_tokens,
        n_threads=n_threads,
        instruct_model=instruct_model,
        repo_id=repo_id,
    )

    for i in range(n_iters):
        if i >= len(VALIDATION_PROMPTS):
            logger.warning("(ONNX-validation) No more samples to validate, stopping")
            break

        input = VALIDATION_PROMPTS[i]
        output = runner.run(input)
        val_output = val_runner.run(input)
        min_len = min(len(output), len(val_output))
        if output[:min_len] != val_output[:min_len]:
            result = f"Warning: Validation failed, mismatched outputs\nExpected:\n{val_output},\nGenerated:\n{output}"
        else:
            result = "Validation successful, identical outputs"
            if len(output) != len(val_output):
                result += f" (output lengths differ: {len(output)} vs {len(val_output)})"
        logger.info(
            "(ONNX-validation) [iter %d, %.3f ms]: %s",
            i,
            runner.last_infer_time / 1e6,
            result
        )
    logger.info(
        "(ONNX-validation) Avg. inference time: %.3f ms",
        runner.avg_infer_time / 1e6
    )
