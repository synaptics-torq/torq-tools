# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""PTD artifact checks against an ordinary decode export in a separate directory.

Set TORQ_PTD_TEST_DIR to a standalone PTD export and TORQ_PTD_REFERENCE_DIR to
an ordinary export. Both use transformer.onnx and lm_head.onnx.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from torq.model_export.validation_ptd import (
    _chain_vis,
    _input_map,
    _load_session,
    _output_index,
    _run_logits,
    _token_feed,
)

EXPORT_DIR = Path(os.environ["TORQ_PTD_TEST_DIR"]) if os.environ.get("TORQ_PTD_TEST_DIR") else None
REFERENCE_DIR = Path(os.environ["TORQ_PTD_REFERENCE_DIR"]) if os.environ.get("TORQ_PTD_REFERENCE_DIR") else None
pytestmark = [
    pytest.mark.integration,
    pytest.mark.ort,
    pytest.mark.skipif(
        EXPORT_DIR is None or REFERENCE_DIR is None,
        reason="Set TORQ_PTD_TEST_DIR and TORQ_PTD_REFERENCE_DIR to separate export directories",
    ),
]


def _load_runners():
    from torq.models.gemma3._inference import Gemma3PTD, Gemma3Static

    shapes = _input_map(_load_session(EXPORT_DIR / "transformer.onnx", 2))
    capacity = next(shape[2] for name, shape in shapes.items() if name.startswith("past_key_values."))
    reference_shapes = _input_map(_load_session(REFERENCE_DIR / "transformer.onnx", 2))
    reference_capacity = next(shape[2] for name, shape in reference_shapes.items() if name.startswith("past_key_values."))
    static = Gemma3Static.from_onnx(
        REFERENCE_DIR / "transformer.onnx", reference_capacity, instruct_model=True, n_threads=2,
    )
    ptd = Gemma3PTD.from_onnx(
        EXPORT_DIR / "transformer.onnx",
        capacity, instruct_model=True, n_threads=2,
    )
    return static, ptd


def test_ptd_greedy_output_matches_static_decode():
    static, ptd = _load_runners()
    for prompt in ("Hello", "The quick brown fox jumps over the lazy dog.", "def foo(x): return x * 2"):
        expected = static.run_tokens(prompt, max_gen_tokens=128)
        actual = ptd.run_tokens(prompt, max_gen_tokens=128)
        assert actual == expected, f"PTD output diverges for {prompt!r}:\n{actual!r}\nvs\n{expected!r}"
        # PTD actually used tree rounds; the instruct
        # warmup + chat framing consume ~40 of the 128-token budget.
        assert len(ptd.ptd_round_log) > 0


def test_ptd_accepts_multiple_tokens_with_gemma():
    static, ptd = _load_runners()
    prompt = "Write a python function to reverse a string."
    expected = static.run_tokens(prompt, max_gen_tokens=128)
    actual = ptd.run_tokens(prompt, max_gen_tokens=128)
    assert actual == expected
    assert any(e["accepted"] > 0 for e in ptd.ptd_round_log)


def test_ptd_round_bookkeeping_consistent():
    _, ptd = _load_runners()
    ptd.run("Repeat the word hello five times.", max_gen_tokens=48)
    log = ptd.ptd_round_log
    committed = sum(e["emitted"] for e in log)
    # Every round processes the anchor plus enough candidate input rows to
    # cover its emission (accepted nodes plus the final prediction).
    rows = sum(e["n_candidates"] + e["n_drafts"] + 1 for e in log)
    assert rows >= committed
    assert all(e["emitted"] >= 1 for e in log)
    # hit rounds accept at least the matching root
    for e in log:
        if e["hit"]:
            assert e["accepted"] >= 1


def test_packed_verification_tree_matches_separate_executions():
    """Plan step 2: each packed candidate row must match a separate causal
    execution of its own path (sibling isolation, repeated positions)."""
    body = _load_session(EXPORT_DIR / "transformer.onnx", 2)
    head = _load_session(EXPORT_DIR / "lm_head.onnx", 2)
    dec_body = _load_session(REFERENCE_DIR / "transformer.onnx", 2)
    dec_head = _load_session(REFERENCE_DIR / "lm_head.onnx", 2)
    emb_path = EXPORT_DIR / "token_embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None

    dec_inputs = _input_map(dec_body)
    cache_names = sorted(n for n in dec_inputs if n.startswith("past_key_values."))
    cache = {n: np.zeros(dec_inputs[n], dtype=np.float32) for n in cache_names}
    ptd_inputs = _input_map(body)
    ptd_seq_input = ptd_inputs.get("token_embedding") or ptd_inputs.get("input_ids")
    rows = ptd_seq_input[1]

    # Prefill a real prompt so the context is non-trivial.
    prompt = [1, 2, 3, 5, 10, 20, 40, 80]

    def update_cache(c: dict, out: list) -> None:
        out_idx = _output_index(dec_body)
        for name in cache_names:
            c[name] = out[out_idx[name.replace("past_key_values", "present")]]

    seq_len = 0
    last_logits = None
    for token in prompt:
        feeds = _token_feed(dec_body, np.array([token]), embeddings)
        feeds.update({"position_ids": np.array([[seq_len]], dtype=np.int64), **cache})
        last_logits, out = _run_logits(dec_body, dec_head, feeds)
        update_cache(cache, out)
        seq_len += 1

    # Candidate tree: two roots (same position) each with one child (same
    # position):  r0 -> r0c,  r1 -> r1c
    anchor = int(np.argmax(last_logits.reshape(-1)))
    r0, r1 = (anchor + 3) % 262144, (anchor + 9) % 262144
    r0c, r1c = (anchor + 13) % 262144, (anchor + 17) % 262144
    tokens = np.array([anchor, r0, r0c, r1, r1c] + [0] * (rows - 5), dtype=np.int64)
    positions = np.array([seq_len, seq_len + 1, seq_len + 2, seq_len + 1, seq_len + 2]
                         + [seq_len] * (rows - 5), dtype=np.int64)
    vis = np.zeros((1, 1, rows, rows - 1), dtype=np.int32)
    # slot 1 (r0): self; slot 2 (r0c): slot1 + self; slot 3 (r1): self;
    # slot 4 (r1c): slot3 + self; roots do NOT see each other
    vis[0, 0, 1, 0] = 1
    vis[0, 0, 2, 0] = 1
    vis[0, 0, 2, 1] = 1
    vis[0, 0, 3, 2] = 1
    vis[0, 0, 4, 2] = 1
    vis[0, 0, 4, 3] = 1

    feeds = _token_feed(body, tokens, embeddings)
    feeds.update({
        "position_ids": positions[None, ...],
        "tree_vis": vis,
        **{name: value[:, :, :ptd_inputs[name][2], :] for name, value in cache.items()},
    })
    ptd_first, _ = _run_logits(body, head, feeds)
    ptd_rows_logit = ptd_first.reshape(rows, -1)

    # Separate causal execution of each path: feed [anchor, path...] and check
    # the logits after each token.
    def separate_logits(path):
        c = {n: v.copy() for n, v in cache.items()}
        sl = seq_len
        logits = []
        for token in [anchor] + path:
            f = _token_feed(dec_body, np.array([token]), embeddings)
            f.update({"position_ids": np.array([[sl]], dtype=np.int64), **c})
            lg, out = _run_logits(dec_body, dec_head, f)
            update_cache(c, out)
            sl += 1
            logits.append(lg.reshape(-1))
        return logits  # [anchor, path[0], path[0..1], ...]

    # anchor row
    np.testing.assert_allclose(ptd_rows_logit[0], separate_logits([])[0], rtol=1e-3, atol=1e-3)
    # root rows (shared position, isolated siblings)
    exp = separate_logits([r0])
    np.testing.assert_allclose(ptd_rows_logit[1], exp[1], rtol=1e-3, atol=1e-3)
    exp = separate_logits([r1])
    np.testing.assert_allclose(ptd_rows_logit[3], exp[1], rtol=1e-3, atol=1e-3)
    # child rows (shared position across subtrees)
    exp = separate_logits([r0, r0c])
    np.testing.assert_allclose(ptd_rows_logit[2], exp[2], rtol=1e-3, atol=1e-3)
    exp = separate_logits([r1, r1c])
    np.testing.assert_allclose(ptd_rows_logit[4], exp[2], rtol=1e-3, atol=1e-3)


def test_ptd_near_full_cache_round():
    """A PTD round with the cache nearly full (scratch region at the cache
    edge) must still match sequential decode."""
    body = _load_session(EXPORT_DIR / "transformer.onnx", 2)
    head = _load_session(EXPORT_DIR / "lm_head.onnx", 2)
    dec_body = _load_session(REFERENCE_DIR / "transformer.onnx", 2)
    dec_head = _load_session(REFERENCE_DIR / "lm_head.onnx", 2)
    embeddings = np.load(EXPORT_DIR / "token_embeddings.npy")
    ptd_inputs = _input_map(body)
    rows = (ptd_inputs.get("token_embedding") or ptd_inputs.get("input_ids"))[1]
    dec_inputs = _input_map(dec_body)
    cache_names = sorted(n for n in dec_inputs if n.startswith("past_key_values."))
    cache = {n: np.zeros(dec_inputs[n], dtype=np.float32) for n in cache_names}
    M = ptd_inputs[cache_names[0]][2]

    # Prefill until L = M - rows (the last legal PTD round start).
    L = M - rows
    prompt = [(i * 37) % 262144 for i in range(L)]

    def update_cache(c: dict, out: list) -> None:
        out_idx = _output_index(dec_body)
        for name in cache_names:
            c[name] = out[out_idx[name.replace("past_key_values", "present")]]

    for i, token in enumerate(prompt):
        feeds = _token_feed(dec_body, np.array([token]), embeddings)
        feeds.update({"position_ids": np.array([[i]], dtype=np.int64), **cache})
        _, out = _run_logits(dec_body, dec_head, feeds)
        update_cache(cache, out)

    anchor = 1234
    chain = [anchor] + [(anchor + i * 5) % 262144 for i in range(1, rows)]
    ptd_feeds = _token_feed(body, np.array(chain, dtype=np.int64), embeddings)
    ptd_feeds.update({
        "position_ids": np.array([list(range(L, L + rows))], dtype=np.int64),
        "tree_vis": _chain_vis(rows),
        **{name: value[:, :, :ptd_inputs[name][2], :] for name, value in cache.items()},
    })
    ptd_first, _ = _run_logits(body, head, ptd_feeds)
    ptd_rows_logit = ptd_first.reshape(rows, -1)

    # Sequential reference over the chain from the same cache.
    c = {n: v.copy() for n, v in cache.items()}
    for i, token in enumerate(chain):
        if i == rows - 1:
            # The same PTD body can run an anchor at M-1, with all scratch
            # slots beyond capacity masked padding.
            padded = _token_feed(body, np.full(rows, token, dtype=np.int64), embeddings)
            padded.update({
                "position_ids": np.full((1, rows), M - 1, dtype=np.int64),
                "tree_vis": np.zeros((1, 1, rows, rows - 1), dtype=np.int32),
                **{name: value[:, :, :ptd_inputs[name][2], :] for name, value in c.items()},
            })
            padded_logits, padded_out = _run_logits(body, head, padded)
        f = _token_feed(dec_body, np.array([token]), embeddings)
        f.update({"position_ids": np.array([[L + i]], dtype=np.int64), **c})
        lg, out = _run_logits(dec_body, dec_head, f)
        update_cache(c, out)
        np.testing.assert_allclose(ptd_rows_logit[i], lg.reshape(-1), rtol=1e-3, atol=1e-3)
        if i == rows - 1:
            np.testing.assert_allclose(padded_logits.reshape(rows, -1)[0], lg.reshape(-1), rtol=1e-3, atol=1e-3)
            ptd_out_idx = _output_index(body)
            for name, value in c.items():
                actual = padded_out[ptd_out_idx[name.replace("past_key_values", "present")]]
                np.testing.assert_allclose(actual, value[:, :, :M, :], rtol=1e-3, atol=1e-3)
