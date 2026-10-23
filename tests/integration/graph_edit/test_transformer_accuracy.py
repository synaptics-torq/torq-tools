# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import clone_graph, graph, run_model
from torq.graph_edit.edits.transformer import MaskFutureAttentionScores, ReplaceDynamicKVCache
from torq.models.qwen._graph import QwenOnnxGraphEditor


pytestmark = [pytest.mark.transformer, pytest.mark.ort]


def test_static_kv_cache_update_matches_expected_position_blend():
    past = gs.Variable("past", dtype=np.float32, shape=[1, 1, 4, 2])
    new = gs.Variable("new", dtype=np.float32, shape=[1, 1, 4, 2])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1])
    present = gs.Variable("present", dtype=np.float32, shape=[1, 1, 4, 2])
    concat = gs.Node("Concat", "present_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2})
    g = graph(nodes=[concat], inputs=[past, new, cur_len], outputs=[present])
    ReplaceDynamicKVCache(g, "integration", cur_len=cur_len, max_tokens=4).transform(concat)

    past_value = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    new_value = np.full((1, 1, 4, 2), 99, dtype=np.float32)
    actual = run_model(g, {"past": past_value, "new": new_value, "cur_len": np.array([[[[2]]]], dtype=np.int64)})["present"]
    expected = past_value.copy()
    expected[:, :, 2:3, :] = new_value[:, :, 2:3, :]
    np.testing.assert_array_equal(actual, expected)


def test_static_kv_cache_chunk_update_places_each_row():
    past = gs.Variable("past", dtype=np.float32, shape=[1, 1, 4, 2])
    new = gs.Variable("new", dtype=np.float32, shape=[1, 1, 2, 2])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1])
    present = gs.Variable("present", dtype=np.float32, shape=[1, 1, 4, 2])
    concat = gs.Node("Concat", "present_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2})
    g = graph(nodes=[concat], inputs=[past, new, cur_len], outputs=[present])
    ReplaceDynamicKVCache(g, "integration", cur_len=cur_len, max_tokens=4, chunk_len=2).transform(concat)

    past_value = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    new_value = np.array([[[[20, 21], [30, 31]]]], dtype=np.float32)
    actual = run_model(
        g,
        {"past": past_value, "new": new_value, "cur_len": np.array([1], dtype=np.int64)},
    )["present"]
    expected = past_value.copy()
    expected[:, :, 1:3, :] = new_value
    np.testing.assert_array_equal(actual, expected)


def test_attention_future_mask_matches_expected_softmax():
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, 1, 4])
    biased_in = gs.Variable("biased_in", dtype=np.float32, shape=[1, 1, 1, 4])
    probs = gs.Variable("probs", dtype=np.float32, shape=[1, 1, 1, 4])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1])
    identity = gs.Node("Identity", "scores_id", inputs=[scores], outputs=[biased_in])
    softmax = gs.Node("Softmax", "layer/self_attn/Softmax", inputs=[biased_in], outputs=[probs], attrs={"axis": -1})
    g = graph(nodes=[identity, softmax], inputs=[scores, cur_len], outputs=[probs])
    MaskFutureAttentionScores(g, "integration", cur_len=cur_len, max_tokens=4, export_dtype=onnx.TensorProto.FLOAT).transform(softmax)

    score_values = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
    actual = run_model(g, {"scores": score_values, "cur_len": np.array([[[[1]]]], dtype=np.int64)})["probs"]
    masked = np.array([1, 2, -1e9, -1e9], dtype=np.float32)
    expected = np.exp(masked - np.max(masked))
    expected = expected / expected.sum()
    np.testing.assert_allclose(actual.reshape(-1), expected, rtol=1e-6, atol=1e-6)


def test_attention_future_mask_advances_for_each_chunk_row():
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, 2, 4])
    biased_in = gs.Variable("biased_in", dtype=np.float32, shape=[1, 1, 2, 4])
    probs = gs.Variable("probs", dtype=np.float32, shape=[1, 1, 2, 4])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1])
    identity = gs.Node("Identity", "scores_id", inputs=[scores], outputs=[biased_in])
    softmax = gs.Node("Softmax", "layer/self_attn/Softmax", inputs=[biased_in], outputs=[probs], attrs={"axis": -1})
    g = graph(nodes=[identity, softmax], inputs=[scores, cur_len], outputs=[probs])
    MaskFutureAttentionScores(
        g, "integration", cur_len=cur_len, max_tokens=4,
        export_dtype=onnx.TensorProto.FLOAT, chunk_len=2,
    ).transform(softmax)

    score_values = np.array([[[[1, 2, 3, 4], [1, 2, 3, 4]]]], dtype=np.float32)
    actual = run_model(
        g,
        {"scores": score_values, "cur_len": np.array([[[[1]]]], dtype=np.int64)},
    )["probs"]
    assert actual[0, 0, 0, 2] == 0
    assert actual[0, 0, 1, 2] > 0
    assert actual[0, 0, 1, 3] == 0


def _gqa_decode_layer(heads: int, kv_heads: int, max_len: int, head_dim: int) -> gs.Graph:
    """One HF-exported GQA decode layer with the projected Q, K and V as inputs."""
    group = heads // kv_heads
    q = gs.Variable("q", dtype=np.float32, shape=[1, heads, 1, head_dim])
    position_ids = gs.Variable("position_ids", dtype=np.int64, shape=[1, 1])
    inputs, nodes, repeated = [q, position_ids], [], {}
    for role in ("key", "value"):
        past = gs.Variable(f"past_key_values.0.{role}", dtype=np.float32, shape=[1, kv_heads, max_len, head_dim])
        new = gs.Variable(f"new_{role}", dtype=np.float32, shape=[1, kv_heads, 1, head_dim])
        present = gs.Variable(f"present.0.{role}", dtype=np.float32, shape=[1, kv_heads, max_len, head_dim])
        unsqueezed = gs.Variable(f"{role}_unsqueezed", dtype=np.float32)
        expanded = gs.Variable(f"{role}_expanded", dtype=np.float32)
        repeated[role] = gs.Variable(f"{role}_repeated", dtype=np.float32, shape=[1, heads, max_len, head_dim])
        nodes += [
            gs.Node("Concat", f"{role}_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2}),
            gs.Node("Unsqueeze", f"{role}_unsqueeze", inputs=[present, gs.Constant(f"{role}_axes", np.array([2]))],
                    outputs=[unsqueezed]),
            gs.Node("Expand", f"{role}_expand", outputs=[expanded], inputs=[
                unsqueezed, gs.Constant(f"{role}_shape", np.array([1, kv_heads, group, max_len, head_dim]))]),
            gs.Node("Reshape", f"{role}_reshape", outputs=[repeated[role]], inputs=[
                expanded, gs.Constant(f"{role}_heads", np.array([1, heads, max_len, head_dim]))]),
        ]
        inputs += [past, new]
    base = "/model/layers.0/self_attn/"
    # Like real exports, the attention intermediates carry no dtype.
    q_scaled, k_t, k_scaled, scores, probs = (gs.Variable(n) for n in ("q_s", "k_t", "k_s", "s", "p"))
    attn = gs.Variable("attn", dtype=np.float32, shape=[1, heads, 1, head_dim])
    nodes += [
        gs.Node("Mul", "q_scale", inputs=[q, gs.Constant("q_scale_value", np.array(0.5, np.float32))], outputs=[q_scaled]),
        gs.Node("Transpose", "k_transpose", inputs=[repeated["key"]], outputs=[k_t], attrs={"perm": [0, 1, 3, 2]}),
        gs.Node("Mul", "k_scale", inputs=[k_t, gs.Constant("k_scale_value", np.array(0.5, np.float32))], outputs=[k_scaled]),
        gs.Node("MatMul", base + "MatMul", inputs=[q_scaled, k_scaled], outputs=[scores]),
        gs.Node("Softmax", base + "Softmax", inputs=[scores], outputs=[probs], attrs={"axis": -1}),
        gs.Node("MatMul", base + "MatMul_1", inputs=[probs, repeated["value"]], outputs=[attn]),
    ]
    return graph(nodes=nodes, inputs=inputs, outputs=[attn, *(n.outputs[0] for n in nodes if n.op == "Concat")])


def _static_decode_layer(heads: int, kv_heads: int, max_len: int, head_dim: int) -> QwenOnnxGraphEditor:
    editor = QwenOnnxGraphEditor(_gqa_decode_layer(heads, kv_heads, max_len, head_dim), onnx.TensorProto.FLOAT)
    cur_len = editor.graph.layer(
        name="current_len_to_1d", op="Squeeze", inputs=[editor.graph.tensors()["position_ids"], [0]],
        outputs=[gs.Variable("position_ids_squeezed", dtype=np.int64, shape=[1])],
    )[0]
    editor.replace_dynamic_kv_cache(cur_len, max_len).mask_future_attn_scores(cur_len, max_len)
    return editor


def test_group_query_heads_matches_repeated_kv():
    heads, kv_heads, max_len, head_dim = 6, 2, 5, 4
    editor = _static_decode_layer(heads, kv_heads, max_len, head_dim)
    repeated = clone_graph(editor.graph)
    editor.group_query_heads(heads, kv_heads, head_dim)
    assert not [n for n in editor.graph.nodes if n.op == "Expand"]

    rng = np.random.default_rng(1)
    feeds = {
        "q": rng.standard_normal((1, heads, 1, head_dim)).astype(np.float32),
        "new_key": rng.standard_normal((1, kv_heads, 1, head_dim)).astype(np.float32),
        "new_value": rng.standard_normal((1, kv_heads, 1, head_dim)).astype(np.float32),
        "past_key_values.0.key": rng.standard_normal((1, kv_heads, max_len, head_dim)).astype(np.float32),
        "past_key_values.0.value": rng.standard_normal((1, kv_heads, max_len, head_dim)).astype(np.float32),
        "position_ids": np.array([[2]], dtype=np.int64),
    }
    expected, actual = run_model(repeated, feeds), run_model(editor.graph, feeds)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6)
