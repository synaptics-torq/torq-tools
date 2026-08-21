# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, quantized_lm_head_graph, run_model
from torq.graph_edit.edits.artifacts import SplitLMHead, TrimLMHeadVocab


pytestmark = [pytest.mark.artifacts, pytest.mark.ort]


def test_trim_lm_head_vocab_outputs_selected_logits(tmp_path):
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    weight = gs.Constant("weight", np.arange(10, dtype=np.float32).reshape(2, 5))
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 5])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, weight], outputs=[logits])
    original = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])
    edited = gs.import_onnx(gs.export_onnx(original))
    TrimLMHeadVocab(edited, "integration", kept_token_ids=np.array([0, 3, 4]), save_lut=tmp_path / "lut.npy").transform(edited.nodes[0])

    feeds = {"hidden": np.array([[[2.0, -1.0]]], dtype=np.float32)}
    full = run_model(original, feeds)["logits"]
    trimmed = run_model(edited, feeds)["logits"]
    np.testing.assert_allclose(trimmed, full[..., [0, 3, 4]], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.load(tmp_path / "lut.npy"), np.array([0, 3, 4]))


def test_split_lm_head_saved_model_matches_original_logits(tmp_path):
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    weight = gs.Constant("weight", np.arange(6, dtype=np.float32).reshape(2, 3))
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 3])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, weight], outputs=[logits])
    original = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])
    edited = gs.import_onnx(gs.export_onnx(original))
    save_to = tmp_path / "lm_head.onnx"
    SplitLMHead(edited, "integration", save_to=save_to).transform(edited.nodes[0])

    feeds = {"hidden": np.array([[[2.0, -1.0]]], dtype=np.float32)}
    expected = run_model(original, feeds)["logits"]
    lm_head_model = onnx.load(save_to)
    actual = run_model(lm_head_model, {"last_hidden_states": feeds["hidden"]})["logits"]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_quantized_lm_head_edits_preserve_selected_logits(tmp_path):
    original = quantized_lm_head_graph()
    trimmed = gs.import_onnx(gs.export_onnx(original))
    split = gs.import_onnx(gs.export_onnx(original))
    kept_token_ids = np.array([0, 3, 4])

    TrimLMHeadVocab(trimmed, "integration", kept_token_ids).transform(trimmed.outputs[0].inputs[0])
    save_to = tmp_path / "lm_head.onnx"
    SplitLMHead(split, "integration", save_to=save_to).transform(split.outputs[0].inputs[0])

    feeds = {"hidden": np.array([[[2.0, -1.0]]], dtype=np.float32)}
    expected = run_model(original, feeds)["logits"]
    actual_trimmed = run_model(trimmed, feeds)["logits"]
    actual_split = run_model(onnx.load(save_to), {"last_hidden_states": feeds["hidden"]})["logits"]
    np.testing.assert_allclose(actual_trimmed, expected[..., kept_token_ids], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_split, expected, rtol=1e-6, atol=1e-6)
