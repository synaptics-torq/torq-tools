# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Functional validation for the fixed-row PTD body variant.

Single-path check (plan step 1): pack a *linear chain* (row i at position
L+i, each row seeing all preceding chain rows) through the PTD body and
verify every row's logits against separate sequential 1-token executions of
the ordinary decode model. This exercises the per-row positions, the
scratch/committed KV split, the visibility mask, and the block head without
any tree semantics.

PTD: Gao et al. 2026, https://arxiv.org/abs/2607.10661 (attribution in
torq.inference.ptd).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ..inference.runners import ORTInferenceRunner


def _load_session(model_path: str | Path, n_threads: int | None = None) -> ORTInferenceRunner:
    return ORTInferenceRunner(model_path, n_threads=n_threads)


def _input_map(sess: ORTInferenceRunner) -> dict[str, list[int]]:
    return {name: list(shape) for name, shape in sess.input_shapes.items()}


def _output_index(sess: ORTInferenceRunner) -> dict[str, int]:
    return {name: i for i, name in enumerate(sess.output_names)}


def _token_feed(sess: ORTInferenceRunner, tokens: np.ndarray,
                token_embeddings: np.ndarray | None) -> dict[str, np.ndarray]:
    if "token_embedding" in _input_map(sess):
        assert token_embeddings is not None, "token_embedding input requires the embeddings LUT"
        return {"token_embedding": token_embeddings[tokens][None, ...]}
    return {"input_ids": tokens.astype(np.int64)[None, ...]}


def _run_logits(body: ORTInferenceRunner, head: ORTInferenceRunner | None,
                feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, list]:
    out = body.infer(feeds)
    out_idx = _output_index(body)
    first = out[out_idx["logits"]] if "logits" in out_idx else out[out_idx["last_hidden_states"]]
    if head is not None:
        first = head.infer({"last_hidden_states": first})[0]
    return first, out


def validate_ptd_onnx(
    logger: logging.Logger,
    ptd_model_path: str | Path,
    decode_model_path: str | Path,
    *,
    ptd_lm_head_path: str | Path | None = None,
    decode_lm_head_path: str | Path | None = None,
    token_embeddings: np.ndarray | None = None,
    token_id_lut: np.ndarray | None = None,
    rows: int,
    vocab_size: int,
    prompt_tokens: list[int] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    n_threads: int | None = None,
) -> None:
    """Compare packed single-chain PTD rows against sequential decode steps."""
    ptd_body = _load_session(ptd_model_path, n_threads)
    dec_body = _load_session(decode_model_path, n_threads)
    ptd_head = _load_session(ptd_lm_head_path, n_threads) if ptd_lm_head_path else None
    dec_head = _load_session(decode_lm_head_path, n_threads) if decode_lm_head_path else None

    dec_inputs = _input_map(dec_body)
    ptd_inputs = _input_map(ptd_body)
    dec_cache_names = sorted(n for n in dec_inputs if n.startswith("past_key_values."))
    assert dec_cache_names, "decode model has no past_key_values inputs"
    assert sorted(n for n in ptd_inputs if n.startswith("past_key_values.")) == dec_cache_names, (
        "PTD and decode bodies must share the committed cache layout"
    )
    for name in dec_cache_names:
        assert all(isinstance(d, int) for d in dec_inputs[name]), (
            f"expected static cache shape for {name}, got {dec_inputs[name]}"
        )

    cache = {n: np.zeros(dec_inputs[n], dtype=np.float32) for n in dec_cache_names}
    capacity = min(dec_inputs[name][2] for name in dec_cache_names)
    if not 2 <= rows <= capacity:
        raise ValueError("PTD rows must be between 2 and the cache capacity")
    if token_id_lut is not None:
        if token_id_lut.ndim != 1 or len(token_id_lut) == 0:
            raise ValueError("token_id_lut must be a non-empty vector")
        vocab_size = len(token_id_lut)
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if prompt_tokens is None:
        prompt_tokens = [i % vocab_size for i in (1, 2, 3, 5, 10, 20, 40, 80)][:capacity - rows]
        if token_id_lut is not None:
            prompt_tokens = [int(token_id_lut[i]) for i in prompt_tokens]
    if len(prompt_tokens) + rows > capacity:
        raise ValueError("Validation prompt and PTD rows exceed the cache capacity")

    def update_cache(out: list) -> None:
        out_idx = _output_index(dec_body)
        for name in dec_cache_names:
            cache[name] = out[out_idx[name.replace("past_key_values", "present")]]

    # Prefill both models identically (1-token decode steps).
    seq_len = 0
    last_logits = None
    for token in prompt_tokens:
        feeds = _token_feed(dec_body, np.array([token]), token_embeddings)
        feeds.update({"position_ids": np.array([[seq_len]], dtype=np.int64), **cache})
        last_logits, out = _run_logits(dec_body, dec_head, feeds)
        update_cache(out)
        seq_len += 1

    # Chain: anchor (greedy next token) + rows-1 arbitrary distinct tokens.
    anchor = int(np.argmax(last_logits.reshape(-1))) if last_logits is not None else 0
    chain = [anchor] + [(anchor + 7 * (i + 1)) % vocab_size for i in range(rows - 1)]
    if token_id_lut is not None:
        chain = [int(token_id_lut[i]) for i in chain]

    # PTD round: linear chain packing (row i sees preceding chain rows), the
    # same committed cache the sequential reference just built.
    ptd_feeds = _token_feed(ptd_body, np.array(chain), token_embeddings)
    ptd_feeds.update({
        "position_ids": np.array([list(range(seq_len, seq_len + rows))], dtype=np.int64),
        "tree_vis": _chain_vis(rows),
        **cache,
    })
    ptd_first, _ = _run_logits(ptd_body, ptd_head, ptd_feeds)
    ptd_row_logits = ptd_first.reshape(rows, -1)

    # Sequential decode over the chain; PTD row i must match step i.
    for i, token in enumerate(chain):
        feeds = _token_feed(dec_body, np.array([token]), token_embeddings)
        feeds.update({"position_ids": np.array([[seq_len + i]], dtype=np.int64), **cache})
        expected, out = _run_logits(dec_body, dec_head, feeds)
        expected = expected.reshape(-1)
        actual = ptd_row_logits[i]
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   err_msg=f"PTD row {i} (token {token}) diverges from decode step")
        assert int(np.argmax(actual)) == int(np.argmax(expected)), (
            f"PTD row {i} argmax {int(np.argmax(actual))} != decode argmax {int(np.argmax(expected))}"
        )
        update_cache(out)

    # Scratch output sanity: K/V may be combined into one tensor per layer.
    n_scratch = len([name for name in _output_index(ptd_body) if name.endswith("_scratch")])
    assert n_scratch >= 1, "PTD body exposes no scratch KV outputs"

    logger.info(
        "(PTD-validation) single-path check passed: %d packed rows match sequential "
        "decode logits (rtol=%g, atol=%g)", rows, rtol, atol
    )


def _chain_vis(rows: int) -> np.ndarray:
    """Linear-chain visibility: row i (scratch slot i) sees slots 1..i."""
    vis = np.zeros((1, 1, rows, rows - 1), dtype=np.int32)
    for i in range(1, rows):
        vis[0, 0, i, :i] = 1
    return vis
