# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Synthetic-graph accuracy tests for the PTD (fixed-row tree) edits."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, run_model
from torq.graph_edit.edits.artifacts import SplitLMHead, TrimLMHeadVocab
from torq.graph_edit.edits.transformer import (
    ConvertToStaticIndex,
    PTDKVCache,
    PTDTreeAttentionMask,
)


pytestmark = [pytest.mark.transformer, pytest.mark.ort]


def test_trimmed_ptd_block_head_preserves_all_rows(tmp_path):
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 4, 2])
    weights = np.arange(12, dtype=np.float32).reshape(2, 6)
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 4, 6])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, gs.Constant("weights", weights)], outputs=[logits])
    body = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])
    lut = np.array([0, 3, 5])
    TrimLMHeadVocab(body, "integration", kept_token_ids=lut).transform(matmul)
    assert body.outputs[0].shape == [1, 4, 3]
    head_path = tmp_path / "lm_head.onnx"
    SplitLMHead(body, "integration", save_to=head_path).transform(matmul)
    inputs = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    actual = run_model(onnx.load(head_path), {"last_hidden_states": inputs})["logits"]
    np.testing.assert_allclose(actual, (inputs @ weights)[..., lut])


# ---------------------------------------------------------------------------
# PTDKVCache
# ---------------------------------------------------------------------------

def _kv_graph(max_tokens: int, rows: int):
    """past + new KV, the dynamic concat (replaced by the edit), and an
    attention-path consumer of the concat output."""
    past = gs.Variable("past_key_values.0.key", dtype=np.float32, shape=[1, 1, max_tokens, 2])
    new = gs.Variable("new_kv", dtype=np.float32, shape=[1, 1, rows, 2])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1])
    present = gs.Variable("present.0.key", dtype=np.float32, shape=[1, 1, max_tokens, 2])
    concat = gs.Node("Concat", "kv_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2})
    attn = gs.Variable("attn_key", dtype=np.float32, shape=[1, 1, max_tokens, 2])
    identity = gs.Node("Identity", "attn_read", inputs=[present], outputs=[attn])
    g = graph(nodes=[concat, identity], inputs=[past, new, cur_len], outputs=[present, attn])
    return g, concat


def test_ptd_kv_cache_commits_anchor_and_exposes_scratch():
    M, N = 8, 4
    g, concat = _kv_graph(M, N)
    PTDKVCache(g, "integration", cur_len=g.inputs[2], max_tokens=M, rows=N).transform(concat)

    past_value = np.arange(1 * 1 * M * 2, dtype=np.float32).reshape(1, 1, M, 2)
    new_value = np.full((1, 1, N, 2), 9.0, dtype=np.float32)
    new_value[:, :, 0, :] = 0.0  # anchor row
    for i in range(1, N):
        new_value[:, :, i, :] = 100.0 + i
    L = 2
    out = run_model(g, {
        "past_key_values.0.key": past_value,
        "new_kv": new_value,
        "cur_len": np.array([L], dtype=np.int64),
    })

    # present (committed): only the anchor row updated
    expected_present = past_value.copy()
    expected_present[:, :, L:L + 1, :] = new_value[:, :, 0:1, :]
    np.testing.assert_array_equal(out["present.0.key"], expected_present)

    # scratch output: raw tree rows (slots 1..N-1)
    np.testing.assert_array_equal(out["present.0.key_scratch"], new_value[:, :, 1:, :])

    # attention keys: committed + anchor, with tree rows placed at L+1..L+N-1
    expected_attn = past_value.copy()
    expected_attn[:, :, L, :] = new_value[:, :, 0, :]
    for j in range(1, N):
        expected_attn[:, :, L + j, :] = new_value[:, :, j, :]
    np.testing.assert_array_equal(out["attn_key"], expected_attn)


def test_ptd_kv_cache_scratch_region_covers_past_data():
    """Stale committed-looking data inside the scratch region is hidden from
    attention by the placement, but retained in the present output."""
    M, N = 8, 4
    g, concat = _kv_graph(M, N)
    PTDKVCache(g, "integration", cur_len=g.inputs[2], max_tokens=M, rows=N).transform(concat)
    past_value = np.ones((1, 1, M, 2), dtype=np.float32) * 7.0
    new_value = np.zeros((1, 1, N, 2), dtype=np.float32)
    L = 1
    out = run_model(g, {
        "past_key_values.0.key": past_value,
        "new_kv": new_value,
        "cur_len": np.array([L], dtype=np.int64),
    })
    # attention: rows 2..4 (L+1..L+N-1) come from scratch (zeros here)
    np.testing.assert_array_equal(out["attn_key"][:, :, 2:5, :], 0.0)
    # present: scratch region untouched (stale data kept, host zeroing is
    # the runner's job; the mask blocks it from visible rows)
    np.testing.assert_array_equal(out["present.0.key"][:, :, 2:5, :], 7.0)
    # anchor blended
    np.testing.assert_array_equal(out["present.0.key"][:, :, L, :], 0.0)


def test_ptd_kv_cache_rejects_wrong_row_count():
    M, N = 8, 4
    g, concat = _kv_graph(M, N)
    cur_len = g.inputs[2]
    with pytest.raises(ValueError, match="batch rows"):
        PTDKVCache(g, "integration", cur_len=cur_len, max_tokens=M, rows=N + 1).transform(concat)


# ---------------------------------------------------------------------------
# PTDTreeAttentionMask
# ---------------------------------------------------------------------------

def _mask_graph(max_tokens: int, rows: int):
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, rows, max_tokens])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1])
    tree_vis = gs.Variable("tree_vis", dtype=np.float32, shape=[1, 1, rows, rows - 1])
    biased = gs.Variable("biased", dtype=np.float32, shape=[1, 1, rows, max_tokens])
    probs = gs.Variable("probs", dtype=np.float32, shape=[1, 1, rows, max_tokens])
    identity = gs.Node("Identity", "scores_id", inputs=[scores], outputs=[biased])
    softmax = gs.Node("Softmax", "layer/self_attn/Softmax", inputs=[biased], outputs=[probs], attrs={"axis": -1})
    g = graph(nodes=[identity, softmax], inputs=[scores, cur_len, tree_vis], outputs=[probs])
    return g, softmax


def _softmax(scores: np.ndarray) -> np.ndarray:
    x = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def test_ptd_mask_committed_plus_ancestor_visibility():
    M, N, L = 8, 4, 2
    g, softmax = _mask_graph(M, N)
    PTDTreeAttentionMask(
        g, "integration", cur_len=g.inputs[1], max_tokens=M,
        export_dtype=onnx.TensorProto.FLOAT, tree_vis=g.inputs[2], rows=N,
    ).transform(softmax)

    matmul = next(node for node in g.nodes if node.op == "MatMul")
    assert all(tensor.dtype == np.float32 for tensor in [*matmul.inputs, *matmul.outputs])
    assert matmul.inputs[0] is g.inputs[2]

    # scratch slots 1..3 live at keys 3,4,5; visibility per row:
    # row 0 (anchor): none | row 1: self | row 2: slot1 + self | row 3: self
    vis = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32).reshape(1, 1, N, N - 1)

    keep = np.zeros((N, M))
    for i in range(N):
        keep[i, : L + 1] = 1.0            # committed keys 0..L
        for j in range(N - 1):
            if vis[0, 0, i, j]:
                keep[i, L + 1 + j] = 1.0  # scratch slot j at key L+1+j
    scores = np.arange(N * M, dtype=np.float32).reshape(1, 1, N, M)
    masked = np.where(keep.reshape(1, 1, N, M) == 1, scores, -1e9)
    expected = _softmax(masked)

    actual = run_model(g, {
        "scores": scores,
        "cur_len": np.array([L], dtype=np.int64),
        "tree_vis": vis,
    })["probs"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_ptd_mask_sibling_isolation_and_cross_tree():
    """Rows sharing a position (siblings / different trees) must not see each
    other, while each sees its own ancestors and itself."""
    M, N, L = 8, 5, 3
    g, softmax = _mask_graph(M, N)
    PTDTreeAttentionMask(
        g, "integration", cur_len=g.inputs[1], max_tokens=M,
        export_dtype=onnx.TensorProto.FLOAT, tree_vis=g.inputs[2], rows=N,
    ).transform(softmax)
    # slots: 1=a(d1) 2=b(d2, child of a) 3=c(d1, other tree) 4=d(d2, child of c)
    vis = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.float32).reshape(1, 1, N, N - 1)

    keep = np.zeros((N, M))
    for i in range(N):
        keep[i, : L + 1] = 1.0
        for j in range(N - 1):
            if vis[0, 0, i, j]:
                keep[i, L + 1 + j] = 1.0
    scores = np.full((1, 1, N, M), 1.0, dtype=np.float32)
    masked = np.where(keep.reshape(1, 1, N, M) == 1, scores, -1e9)
    expected = _softmax(masked)
    # with uniform scores, each row's mass is uniform over exactly its
    # visible keys; blocked keys get exactly zero
    n_visible = keep.sum(axis=1)
    for i in range(N):
        np.testing.assert_allclose(expected[0, 0, i][keep[i] == 1], 1.0 / n_visible[i], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(expected[0, 0, i][keep[i] == 0], 0.0)

    actual = run_model(g, {
        "scores": scores,
        "cur_len": np.array([L], dtype=np.int64),
        "tree_vis": vis,
    })["probs"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    # row 2 (b) must give zero mass to slot 3's key (sibling/cross-tree c)
    assert actual[0, 0, 2, L + 3] == 0
    # row 4 (d) sees c (its ancestor, slot 3) but not a/b (other tree)
    assert actual[0, 0, 4, L + 3] > 0
    assert actual[0, 0, 4, L + 1] == 0


def test_ptd_mask_invalid_row_still_has_committed_visibility():
    """An invalid row (all-zero vis) still sees the committed keys, so its
    softmax cannot degenerate to NaN."""
    M, N, L = 8, 4, 2
    g, softmax = _mask_graph(M, N)
    PTDTreeAttentionMask(
        g, "integration", cur_len=g.inputs[1], max_tokens=M,
        export_dtype=onnx.TensorProto.FLOAT, tree_vis=g.inputs[2], rows=N,
    ).transform(softmax)
    vis = np.zeros((1, 1, N, N - 1), dtype=np.float32)
    vis[0, 0, 0, :] = 0
    vis[0, 0, 1, 0] = 1  # only row 1 valid, sees itself
    scores = np.arange(N * M, dtype=np.float32).reshape(1, 1, N, M)
    actual = run_model(g, {
        "scores": scores,
        "cur_len": np.array([L], dtype=np.int64),
        "tree_vis": vis,
    })["probs"]
    assert np.isfinite(actual).all()
    # invalid row 3: mass only over committed keys 0..L
    assert actual[0, 0, 3, L + 1:].sum() == 0
    assert actual[0, 0, 3, : L + 1].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ConvertToStaticIndex with positions_input
# ---------------------------------------------------------------------------

def _range_graph():
    start = gs.Variable("start", dtype=np.int64, shape=[1])
    limit = gs.Variable("limit", dtype=np.int64, shape=[1])
    limit_add = gs.Node("Add", "limit_add", inputs=[start, gs.Constant("five", np.asarray([5], dtype=np.int64))], outputs=[limit])
    range_node = gs.Node(
        "Range", "positions_range",
        inputs=[start, limit, gs.Constant("one", np.asarray(1, dtype=np.int64))],
        outputs=[gs.Variable("range_out", dtype=np.int64, shape=[5])],
    )
    out = gs.Variable("pos_out", dtype=np.int64, shape=[5])
    identity = gs.Node("Identity", "pos_id", inputs=[range_node.outputs[0]], outputs=[out])
    g = graph(nodes=[limit_add, range_node, identity], inputs=[start], outputs=[out])
    return g, range_node


def test_static_index_positions_input_used_verbatim():
    g, range_node = _range_graph()
    positions = gs.Variable("positions", dtype=np.int64, shape=[5])
    g.inputs.append(positions)
    ConvertToStaticIndex(g, "integration", chunk_len=5, positions_input=positions).transform(range_node)
    actual = run_model(g, {
        "positions": np.array([7, 7, 9, 7, 12], dtype=np.int64),
    })["pos_out"]
    np.testing.assert_array_equal(actual, [7, 7, 9, 7, 12])


def test_static_index_default_behavior_unchanged():
    g, range_node = _range_graph()
    ConvertToStaticIndex(g, "integration", chunk_len=5).transform(range_node)
    actual = run_model(g, {"start": np.array([2], dtype=np.int64)})["pos_out"]
    np.testing.assert_array_equal(actual, [2, 3, 4, 5, 6])


def test_ptd_padded_anchor_at_last_cache_slot_ignores_out_of_range_scratch():
    M, N, L = 8, 4, 7
    g, concat = _kv_graph(M, N)
    PTDKVCache(g, "integration", cur_len=g.inputs[2], max_tokens=M, rows=N).transform(concat)
    past = np.arange(M * 2, dtype=np.float32).reshape(1, 1, M, 2)
    new = np.full((1, 1, N, 2), 999, dtype=np.float32)
    new[:, :, 0, :] = 42
    out = run_model(g, {
        "past_key_values.0.key": past, "new_kv": new,
        "cur_len": np.array([L], dtype=np.int64),
    })
    expected = past.copy()
    expected[:, :, L, :] = 42
    np.testing.assert_array_equal(out["present.0.key"], expected)
    np.testing.assert_array_equal(out["attn_key"], expected)
    mask_graph, softmax = _mask_graph(M, N)
    PTDTreeAttentionMask(
        mask_graph, "integration", cur_len=mask_graph.inputs[1], max_tokens=M,
        export_dtype=onnx.TensorProto.FLOAT, tree_vis=mask_graph.inputs[2], rows=N,
    ).transform(softmax)
    probs = run_model(mask_graph, {
        "scores": np.zeros((1, 1, N, M), dtype=np.float32),
        "cur_len": np.array([L], dtype=np.int64),
        "tree_vis": np.zeros((1, 1, N, N - 1), dtype=np.float32),
    })["probs"]
    np.testing.assert_allclose(probs, 1 / M)
