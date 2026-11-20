# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, run_model
from torq.graph_edit import OnnxGraphEditor
from torq.graph_edit.edits.arithmetic import DequantizeProjectionsMatMul
from torq.graph_edit.edits.shape import SplitRotarySlices
from torq.graph_edit.edits.transformer import AddCurrLenInput, residual_layer_boundaries


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("step", [1, 2])
def test_rotary_slice_split_uses_structure_and_preserves_values(axis, step):
    shape = [4, 4]
    x = gs.Variable("x", dtype=np.float32, shape=shape)
    halves = [gs.Variable(f"half{i}", dtype=np.float32) for i in range(2)]
    neg, out = gs.Variable("neg", dtype=np.float32), gs.Variable("out", dtype=np.float32)
    nodes = [gs.Node("Slice", f"arbitrary{i}", inputs=[x, gs.Constant(f"start{i}", np.array([i * 2], np.int64)),
              gs.Constant(f"end{i}", np.array([(i + 1) * 2], np.int64)), gs.Constant(f"axis{i}", np.array([axis], np.int64)),
              gs.Constant(f"step{i}", np.array([step], np.int64))], outputs=[halves[i]]) for i in range(2)]
    nodes += [gs.Node("Neg", "neg", inputs=[halves[1]], outputs=[neg]),
              gs.Node("Concat", "rotate", inputs=[neg, halves[0]], outputs=[out], attrs={"axis": axis})]
    g = graph(nodes=nodes, inputs=[x], outputs=[out])
    values = np.arange(16, dtype=np.float32).reshape(shape)
    expected = run_model(g, {"x": values})["out"]
    editor = OnnxGraphEditor(g, "unit")
    editor.apply_edit(SplitRotarySlices(g, "unit"))
    assert sum(n.op == "Split" for n in g.nodes) == (step == 1)
    np.testing.assert_array_equal(run_model(g, {"x": values})["out"], expected)


def test_gqa_cleanup_preserves_unused_positional_output():
    x = gs.Variable("x", dtype=np.float32, shape=[1])
    outputs = [gs.Variable(f"anonymous{i}", dtype=np.float32, shape=[1]) for i in range(3)]
    node = gs.Node("GroupQueryAttention", "attention", inputs=[x], outputs=outputs, domain="com.microsoft")
    g = graph(nodes=[node], inputs=[x], outputs=[outputs[0], outputs[2]])
    OnnxGraphEditor(g, "unit")._cleanup()
    assert [o.name for o in node.outputs] == [o.name for o in outputs]


def test_residual_boundaries_follow_casts_and_reject_ambiguity():
    x = gs.Variable("token_embedding", dtype=np.float32, shape=[1, 1, 2])
    view, middle, out = [gs.Variable(name, dtype=np.float32, shape=x.shape) for name in ("view", "middle", "boundary")]
    bias = gs.Constant("bias", np.ones(2, np.float32))
    nodes = [gs.Node("Cast", "cast", inputs=[x], outputs=[view], attrs={"to": onnx.TensorProto.FLOAT}),
             gs.Node("Add", "anonymous0", inputs=[view, bias], outputs=[middle]),
             gs.Node("Add", "anonymous1", inputs=[middle, bias], outputs=[out])]
    g = graph(nodes=nodes, inputs=[x], outputs=[out])
    assert residual_layer_boundaries(g, 1)[0] is out
    g.nodes.append(gs.Node("Add", "ambiguous", inputs=[view, bias], outputs=[gs.Variable("other")]))
    with pytest.raises(ValueError, match="Expected one residual Add"):
        residual_layer_boundaries(g, 1)


def test_projection_matches_renamed_output_and_rejects_wrong_vocab():
    x = gs.Variable("hidden", dtype=np.float32, shape=[1, 2])
    w = gs.Variable("weight", dtype=np.float32, shape=[2, 3])
    y = gs.Variable("renamed_output", dtype=np.float32, shape=[1, 3])
    dq = gs.Node("DequantizeLinear", "dq", inputs=[gs.Constant("w", np.ones((2, 3), np.uint8))], outputs=[w])
    mm = gs.Node("MatMul", "projection", inputs=[x, w], outputs=[y])
    g = graph(nodes=[dq, mm], inputs=[x], outputs=[y])
    assert DequantizeProjectionsMatMul(g, "unit", 2, 3, onnx.TensorProto.FLOAT).match(mm)
    assert not DequantizeProjectionsMatMul(g, "unit", 2, 4, onnx.TensorProto.FLOAT).match(mm)


def test_current_length_preserves_other_shape_consumers():
    cache = gs.Variable("renamed_cache", dtype=np.float32, shape=[1, 2, 4, 8])
    present = gs.Variable("updated", dtype=np.float32)
    shape_out = gs.Variable("shape", dtype=np.int64)
    seq, heads = [gs.Variable(name, dtype=np.int64) for name in ("seq", "heads")]
    nodes = [gs.Node("Concat", "update", inputs=[cache, gs.Variable("new")], outputs=[present], attrs={"axis": -2}),
             gs.Node("Shape", "shape", inputs=[cache], outputs=[shape_out]),
             gs.Node("Gather", "heads", inputs=[shape_out, gs.Constant("head_axis", np.array(1, np.int64))], outputs=[heads]),
             gs.Node("Gather", "seq", inputs=[shape_out, gs.Constant("seq_axis", np.array(2, np.int64))], outputs=[seq])]
    position = gs.Variable("position", dtype=np.int64)
    g = graph(nodes=nodes, inputs=[cache, position], outputs=[present, seq, heads])
    edit = AddCurrLenInput(g, "unit", position)
    assert edit.match(nodes[1])
    edit.transform(nodes[1])
    assert g.outputs[1] is position
    assert nodes[2].inputs[0] is shape_out
    assert nodes[1].inputs[0] is cache
