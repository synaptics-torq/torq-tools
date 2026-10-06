# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.artifacts import ExtractConstantLUT, SplitLMHead, TakeLastToken, TrimLMHeadVocab


pytestmark = pytest.mark.artifacts


def test_extract_constant_lut_saves_array_and_replaces_gather_with_input(tmp_path):
    lut = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = gs.Variable("tokens", dtype=np.int64, shape=[1])
    gathered = gs.Variable("embeddings", dtype=np.float32, shape=[1, 2])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 2])
    gather_node = gs.Node("Gather", "token_gather", inputs=[gs.Constant("lut", lut), indices], outputs=[gathered], attrs={"axis": 0})
    identity = gs.Node("Identity", "use", inputs=[gathered], outputs=[y])
    g = graph(nodes=[gather_node, identity], inputs=[indices], outputs=[y])
    save_to = tmp_path / "lut.npy"

    edit = ExtractConstantLUT(g, "unit", lut_shape=(3, 2), save_to=save_to, inp_name="token_embedding")
    assert edit.match(gather_node)
    edit.transform(gather_node)

    np.testing.assert_array_equal(np.load(save_to), lut)
    assert identity.inputs[0].name == "token_embedding"
    assert g.inputs[-1].name == "token_embedding"
    assert gather_node.outputs == []


def test_extract_constant_lut_replaces_graph_output_when_gather_is_output(tmp_path):
    lut = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = gs.Variable("tokens", dtype=np.int64, shape=[1])
    gathered = gs.Variable("embeddings", dtype=np.float32, shape=[1, 2])
    gather_node = gs.Node("Gather", "token_gather", inputs=[gs.Constant("lut", lut), indices], outputs=[gathered], attrs={"axis": 0})
    g = graph(nodes=[gather_node], inputs=[indices], outputs=[gathered])

    ExtractConstantLUT(g, "unit", lut_shape=(3, 2), save_to=tmp_path / "lut.npy", inp_name="token_embedding").transform(gather_node)

    assert g.outputs[0].name == "token_embedding"
    assert g.inputs[-1].name == "token_embedding"
    assert gather_node.outputs == []


def test_extract_constant_lut_can_skip_writing_duplicate_table(tmp_path):
    lut = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = gs.Variable("tokens", dtype=np.int64, shape=[1])
    gathered = gs.Variable("embeddings", dtype=np.float32, shape=[1, 2])
    gather_node = gs.Node("Gather", "token_gather", inputs=[gs.Constant("lut", lut), indices], outputs=[gathered])
    g = graph(nodes=[gather_node], inputs=[indices], outputs=[gathered])

    ExtractConstantLUT(g, "unit", lut_shape=(3, 2), save_to=None, inp_name="token_embedding").transform(gather_node)

    assert list(tmp_path.iterdir()) == []
    assert g.inputs[-1].name == "token_embedding"


def test_trim_lm_head_vocab_can_append_argmax_output():
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    weight = gs.Constant("weight", np.ones((2, 4), dtype=np.float32))
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 4])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, weight], outputs=[logits])
    g = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])

    TrimLMHeadVocab(g, "unit", kept_token_ids=np.array([1, 2]), include_argmax=True).transform(matmul)

    assert g.outputs[0].name == "compact_token_idx"
    assert any(node.op == "ArgMax" for node in g.nodes)


def test_split_lm_head_saves_model_and_exposes_hidden_states(tmp_path):
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    weight = gs.Constant("weight", np.arange(6, dtype=np.float32).reshape(2, 3))
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 3])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, weight], outputs=[logits])
    g = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])
    save_to = tmp_path / "lm_head.onnx"

    edit = SplitLMHead(g, "unit", save_to=save_to)
    assert edit.match(matmul)
    edit.transform(matmul)

    assert save_to.exists()
    onnx.checker.check_model(onnx.load(save_to))
    assert g.outputs[0].name == "last_hidden_states"
    assert matmul.outputs == []


def test_take_last_token_makes_prefill_share_single_token_lm_head(tmp_path):
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 4, 2])
    weight = gs.Constant("weight", np.arange(6, dtype=np.float32).reshape(2, 3))
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 4, 3])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, weight], outputs=[logits])
    g = graph(nodes=[matmul], inputs=[hidden], outputs=[logits])

    TakeLastToken(g, "unit").transform(matmul)
    SplitLMHead(g, "unit", save_to=tmp_path / "lm_head.onnx").transform(matmul)

    lm_head = onnx.load(tmp_path / "lm_head.onnx")
    input_shape = [dim.dim_value for dim in lm_head.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 1, 2]
    assert g.outputs[0].shape == [1, 1, 2]
