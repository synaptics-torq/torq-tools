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
    vis = np.zeros((1, 1, rows, rows - 1), dtype=np.float32)
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
