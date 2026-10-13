# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Artifact-dependent PTD tests for Gemma 3 (skipped when the export is absent).

Requires an export dir containing both transformer.onnx (ordinary static
decode body, + lm_head.onnx) and ptd_transformer.onnx (+ ptd_lm_head.onnx):
set TORQ_PTD_TEST_DIR or export with --ptd-rows next to the regular model.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIRS = [
    Path(os.environ.get("TORQ_PTD_TEST_DIR", "")),
    REPO_ROOT / "models-ptd/google/gemma-3-270m-it/export/full/split_lm_head/fp32/static",
    REPO_ROOT / "models/google/gemma-3-270m-it/export/full/split_lm_head/fp32/static",
]


def _export_dir() -> Path | None:
    for d in CANDIDATE_DIRS:
        if d and (d / "transformer.onnx").exists() and (d / "ptd_transformer.onnx").exists():
            return d
    return None


EXPORT_DIR = _export_dir()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.ort,
    pytest.mark.skipif(EXPORT_DIR is None, reason="Gemma 3 PTD export artifacts not present"),
]


def _load_runners():
    from torq.models.gemma3._inference import Gemma3PTD, Gemma3Static

    shapes = _input_map(_load_session(EXPORT_DIR / "transformer.onnx"))
    capacity = next(shape[2] for name, shape in shapes.items() if name.startswith("past_key_values."))
    static = Gemma3Static.from_onnx(
        EXPORT_DIR / "transformer.onnx", capacity, instruct_model=True, n_threads=os.cpu_count(),
    )
    ptd = Gemma3PTD.from_onnx(
        EXPORT_DIR / "transformer.onnx",
        EXPORT_DIR / "ptd_transformer.onnx",
        capacity, instruct_model=True, n_threads=os.cpu_count(),
    )
    return static, ptd


def test_ptd_greedy_output_matches_static_decode():
    static, ptd = _load_runners()
    for prompt in ("Hello", "The quick brown fox jumps over the lazy dog.", "def foo(x): return x * 2"):
        expected = static.run_tokens(prompt, max_gen_tokens=128)
        actual = ptd.run_tokens(prompt, max_gen_tokens=128)
        assert actual == expected, f"PTD output diverges for {prompt!r}:\n{actual!r}\nvs\n{expected!r}"
        # PTD actually used rounds (not just the fallback path); the instruct
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
    static, ptd = _load_runners()
    ptd.run("Repeat the word hello five times.", max_gen_tokens=48)
    log = ptd.ptd_round_log
    committed = sum(e["emitted"] for e in log)
    # every committed token came from a round; the anchor row is processed each
    # round, so rows_processed >= committed + rounds
    rows = sum(e["n_candidates"] + e["n_drafts"] + 1 for e in log)
    assert rows >= committed + len(log)
    assert all(e["emitted"] >= 1 for e in log)
    # hit rounds accept at least the matching root
    for e in log:
        if e["hit"]:
            assert e["accepted"] >= 1


def test_packed_verification_tree_matches_separate_executions():
    """Plan step 2: each packed candidate row must match a separate causal
    execution of its own path (sibling isolation, repeated positions)."""
    body = _load_session(EXPORT_DIR / "ptd_transformer.onnx")
    head = _load_session(EXPORT_DIR / "ptd_lm_head.onnx")
    dec_body = _load_session(EXPORT_DIR / "transformer.onnx")
    dec_head = _load_session(EXPORT_DIR / "lm_head.onnx")
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
        **cache,
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
    body = _load_session(EXPORT_DIR / "ptd_transformer.onnx")
    head = _load_session(EXPORT_DIR / "ptd_lm_head.onnx")
    dec_body = _load_session(EXPORT_DIR / "transformer.onnx")
    dec_head = _load_session(EXPORT_DIR / "lm_head.onnx")
    embeddings = np.load(EXPORT_DIR / "token_embeddings.npy")
    ptd_inputs = _input_map(body)
    rows = (ptd_inputs.get("token_embedding") or ptd_inputs.get("input_ids"))[1]
    dec_inputs = _input_map(dec_body)
    cache_names = sorted(n for n in dec_inputs if n.startswith("past_key_values."))
    cache = {n: np.zeros(dec_inputs[n], dtype=np.float32) for n in cache_names}
    M = dec_inputs[cache_names[0]][2]

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
        **cache,
    })
    ptd_first, _ = _run_logits(body, head, ptd_feeds)
    ptd_rows_logit = ptd_first.reshape(rows, -1)

    # Sequential reference over the chain from the same cache.
    c = {n: v.copy() for n, v in cache.items()}
    for i, token in enumerate(chain):
        f = _token_feed(dec_body, np.array([token]), embeddings)
        f.update({"position_ids": np.array([[L + i]], dtype=np.int64), **c})
        lg, out = _run_logits(dec_body, dec_head, f)
        update_cache(c, out)
        np.testing.assert_allclose(ptd_rows_logit[i], lg.reshape(-1), rtol=1e-3, atol=1e-3)
