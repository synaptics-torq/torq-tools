# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Executable export-contract checks for the PTD body and trimmed block head."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, run_model
from torq.graph_edit.edits.artifacts import SplitLMHead, TrimLMHeadVocab
from torq.graph_edit.edits.transformer import PTDKVCache, PTDTreeAttentionMask
from torq.models.gemma3._graph import Gemma3OnnxGraphEditor

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


def test_ptd_attention_and_combined_cache_match_reference_at_cache_boundary():
    rows, capacity, dim = 4, 8, 2
    queries = gs.Variable("queries", dtype=np.float32, shape=[1, 1, rows, dim])
    position = gs.Variable("cur_len", dtype=np.int64, shape=[1])
    visibility = gs.Variable("tree_vis", dtype=np.float32, shape=[1, 1, rows, rows - 1])
    past, new, present, concats = {}, {}, {}, []
    for role in ("key", "value"):
        past[role] = gs.Variable(f"past_key_values.0.{role}", dtype=np.float32, shape=[1, 1, capacity, dim])
        new[role] = gs.Variable(f"new_{role}", dtype=np.float32, shape=[1, 1, rows, dim])
        present[role] = gs.Variable(f"present.0.{role}", dtype=np.float32, shape=[1, 1, capacity, dim])
        concats.append(gs.Node("Concat", f"concat_{role}", inputs=[past[role], new[role]],
                               outputs=[present[role]], attrs={"axis": -2}))
    keys = gs.Variable("keys", dtype=np.float32, shape=[1, 1, dim, capacity])
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, rows, capacity])
    probs = gs.Variable("probs", dtype=np.float32, shape=scores.shape)
    context = gs.Variable("context", dtype=np.float32, shape=queries.shape)
    softmax = gs.Node("Softmax", "layer/self_attn/Softmax", inputs=[scores], outputs=[probs], attrs={"axis": -1})
    g = graph(nodes=[*concats,
                    gs.Node("Transpose", "key_transpose", inputs=[present["key"]], outputs=[keys], attrs={"perm": [0, 1, 3, 2]}),
                    gs.Node("MatMul", "scores", inputs=[queries, keys], outputs=[scores]), softmax,
                    gs.Node("MatMul", "context", inputs=[probs, present["value"]], outputs=[context])],
              inputs=[queries, position, visibility, *past.values(), *new.values()],
              outputs=[context, *present.values()])
    PTDTreeAttentionMask(g, "integration", cur_len=position, max_tokens=capacity,
                         export_dtype=onnx.TensorProto.FLOAT, tree_vis=visibility, rows=rows).transform(softmax)
    for concat in concats:
        PTDKVCache(g, "integration", cur_len=position, max_tokens=capacity, rows=rows).transform(concat)
    editor = Gemma3OnnxGraphEditor(g, onnx.TensorProto.FLOAT)
    editor.combine_kv_io_tensors([1, 1, capacity, dim])
    editor.combine_kv_io_tensors([1, 1, rows - 1, dim],
                                kv_layer_re=r"\.(\d+)\.(key|value)_scratch$", combined_suffix="_scratch")
    # Keep the compiler workaround explicit: tree visibility must use float MatMul.
    visibility_matmul = next(node for node in g.nodes if node.op == "MatMul" and node.inputs[0] is visibility)
    assert all(t.dtype == np.float32 for t in [*visibility_matmul.inputs, *visibility_matmul.outputs])
    rng = np.random.default_rng(7)
    q = rng.normal(size=queries.shape).astype(np.float32)
    old = rng.normal(size=(1, 2, capacity, dim)).astype(np.float32)
    fresh = rng.normal(size=(1, 2, rows, dim)).astype(np.float32)
    tree = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    for start, vis in [(2, tree), (capacity - 1, np.zeros_like(tree))]:
        feeds = {"queries": q, "cur_len": np.array([start]), "tree_vis": vis[None, None],
                 "past_key_values.0.key_value": old, "new_key": fresh[:, :1], "new_value": fresh[:, 1:]}
        actual = run_model(g, feeds)
        committed = old.copy()
        committed[:, :, start:start + 1] = fresh[:, :, :1]
        np.testing.assert_array_equal(actual["present.0.key_value"], committed)
        np.testing.assert_array_equal(actual["present.0.key_value_scratch"], fresh[:, :, 1:])
        attention_cache = committed.copy()
        count = min(rows, capacity - start)
        attention_cache[:, :, start:start + count] = fresh[:, :, :count]
        allowed = np.zeros((rows, capacity), dtype=bool)
        allowed[:, :start + 1] = True
        allowed[:, start + 1:start + count] = vis[:, :count - 1] > 0.5
        scores = q @ attention_cache[:, :1].swapaxes(-1, -2)
        masked = np.where(allowed, scores, -1e9)
        exp = np.exp(masked - masked.max(axis=-1, keepdims=True))
        expected = (exp / exp.sum(axis=-1, keepdims=True)) @ attention_cache[:, 1:]
        np.testing.assert_allclose(actual["context"], expected, rtol=1e-5, atol=1e-6)
