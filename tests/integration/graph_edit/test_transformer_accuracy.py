import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, run_model
from torq.graph_edit.edits.transformer import MaskFutureAttentionScores, ReplaceDynamicKVCache


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
