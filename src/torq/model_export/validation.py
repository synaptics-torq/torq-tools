# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Functional-equivalence check shared by the decoder-only ONNX exporters."""

import logging
import os
from pathlib import Path
from typing import Final

import numpy as np

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
    lm_head_path: str | os.PathLike | None = None,
    prefill_model_path: str | os.PathLike | None = None,
    prefill_size: int | None = None,
    prompts: tuple[str, ...] = VALIDATION_PROMPTS,
    max_inp_len: int | None = None,
    runner_kwargs: dict | None = None,
    sliding_window: int | None = None,
):
    """Compare the exported model's output against the unedited source ONNX.

    The reference is always `dynamic_cls` on the source `model.onnx`; the model
    under test is static or dynamic depending on how it was exported.
    `lm_head_path` chains a split LM head onto the model under test, whose own
    first output is then hidden states rather than logits. `prefill_model_path`
    uses the fixed-size prefill model for complete prompt chunks. `max_inp_len`
    can allow a complete tool prompt larger than one chunk; `runner_kwargs`
    supplies the same tool definitions to both runners. `sliding_window` adds
    a teacher-forced logit comparison across the window boundary for static exports.
    """
    n_threads = n_threads or os.cpu_count()
    max_inp_len = max_inp_len if max_inp_len is not None else prefill_size
    runner_kwargs = runner_kwargs or {}
    if static_models:
        # Only exporters that can split the model accept these component paths.
        split_model_kwargs = {}
        if lm_head_path:
            split_model_kwargs["lm_head_path"] = lm_head_path
        if prefill_model_path:
            split_model_kwargs.update({
                "prefill_model_path": prefill_model_path,
                "prefill_size": prefill_size,
            })
        runner = static_cls.from_onnx(
            model_path,
            max_gen_tokens,
            max_inp_len=max_inp_len,
            n_threads=n_threads,
            instruct_model=instruct_model,
            repo_id=repo_id,
            **split_model_kwargs,
            **runner_kwargs,
        )
    else:
        runner = dynamic_cls.from_onnx(
            model_path,
            max_gen_tokens=max_gen_tokens,
            max_inp_len=max_inp_len,
            n_threads=n_threads,
            instruct_model=instruct_model,
            repo_id=repo_id,
            **runner_kwargs,
        )
    val_runner = dynamic_cls.from_onnx(
        Path(reference_model_path),
        max_inp_len=max_inp_len,
        max_gen_tokens=max_gen_tokens,
        n_threads=n_threads,
        instruct_model=instruct_model,
        repo_id=repo_id,
        **runner_kwargs,
    )

    for i in range(n_iters):
        if i >= len(prompts):
            logger.warning("(ONNX-validation) No more samples to validate, stopping")
            break

        input = prompts[i]
        output = runner.run(input, skip_special_tokens=False)
        val_output = val_runner.run(input, skip_special_tokens=False)
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

    if static_models and sliding_window is not None and max_gen_tokens > sliding_window:
        _validate_long_context(logger, runner, val_runner, sliding_window, max_gen_tokens)


def _validate_long_context(logger, runner, reference, window: int, capacity: int):
    """Compare logits on one long prompt and a reference-forced continuation."""
    # Repeated natural text makes a deterministic prompt without tokenizer-specific IDs.
    target = min(window + 65, capacity)
    prompt = reference._tokenize_input("The quick brown fox jumps over the lazy dog. " * target)
    prompt = prompt[:max(1, target - 32)]
    runner._kv_cache = runner._init_cache()
    reference._kv_cache = reference._init_cache()

    def step(model, token, position):
        # The runners sample after the optional split LM head. Capture that same
        # logit vector without adding a separate model execution path.
        captured = []
        sample = model.sample_next_token

        def capture(logits):
            captured.append(logits)
            return int(logits.argmax())

        model.sample_next_token = capture
        try:
            next_token, cache = model._llm_step(token, position)
        finally:
            model.sample_next_token = sample
        model._update_cache(cache, position)
        return next_token, captured[0]

    token = prompt[0]
    for position in range(target):
        if position < len(prompt):
            token = prompt[position]
        expected, expected_logits = step(reference, token, position)
        _, actual_logits = step(runner, token, position)
        lut = getattr(runner, "_token_id_lut", None)
        if lut is not None:
            expected_logits = expected_logits[lut]
        try:
            np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-3, atol=1e-3)
        except AssertionError as error:
            raise ValueError(f"Long-context validation failed at position {position}: {error}") from error
        token = expected
    logger.info("(ONNX-validation) Long-context teacher-forced validation passed through position %d", target - 1)
