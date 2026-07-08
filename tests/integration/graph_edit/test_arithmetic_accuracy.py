# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.arithmetic import (
    DequantizeProjectionsMatMul,
    FoldScalarMatMul,
    RemoveRedundantCasts,
    ReplaceConstantDivWithMul,
    ReplaceInt64FloatCast,
)


pytestmark = [pytest.mark.arithmetic, pytest.mark.ort]


def test_constant_div_replacement_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 3])
    div = gs.Node("Div", "div", inputs=[x, gs.Constant("denom", np.array([2, 4, 8], dtype=np.float32))], outputs=[y])
    original = graph(nodes=[div], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    ReplaceConstantDivWithMul(edited, "integration", export_dtype=onnx.TensorProto.FLOAT).transform(edited.nodes[0])

    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    assert_model_outputs_close(original, edited, feeds)


def test_scalar_matmul_fold_is_numerically_equivalent():
    a = gs.Variable("a", dtype=np.float32, shape=[1, 4, 1])
    b = gs.Variable("b", dtype=np.float32, shape=[1, 1])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4, 1])
    matmul = gs.Node("MatMul", "scale", inputs=[a, b], outputs=[y])
    original = graph(nodes=[matmul], inputs=[a, b], outputs=[y])
    edited = clone_graph(original)
    FoldScalarMatMul(edited, "integration").transform(edited.nodes[0])

    feeds = {
        "a": np.arange(4, dtype=np.float32).reshape(1, 4, 1),
        "b": np.array([[2.5]], dtype=np.float32),
    }
    assert_model_outputs_close(original, edited, feeds)


def test_redundant_cast_removal_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[2])
    cast_out = gs.Variable("cast_out", dtype=onnx.TensorProto.FLOAT, shape=[2])
    y = gs.Variable("y", dtype=np.float32, shape=[2])
    cast = gs.Node("Cast", "cast", inputs=[x], outputs=[cast_out], attrs={"to": onnx.TensorProto.FLOAT})
    relu = gs.Node("Relu", "relu", inputs=[cast_out], outputs=[y])
    original = graph(nodes=[cast, relu], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    RemoveRedundantCasts(edited, "integration").transform(edited.nodes[0])

    assert_model_outputs_close(original, edited, {"x": np.array([-1, 2], dtype=np.float32)})


def test_int64_to_float_cast_lookup_is_numerically_equivalent():
    idx = gs.Variable("idx", dtype=np.int64, shape=[1, 1])
    cast_out = gs.Variable("cast_out", dtype=onnx.TensorProto.FLOAT, shape=[1, 1])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 1])
    cast = gs.Node("Cast", "cast", inputs=[idx], outputs=[cast_out], attrs={"to": onnx.TensorProto.FLOAT})
    add = gs.Node("Add", "add", inputs=[cast_out, gs.Constant("one", np.array([[1]], dtype=np.float32))], outputs=[y])
    original = graph(nodes=[cast, add], inputs=[idx], outputs=[y])
    edited = clone_graph(original)
    ReplaceInt64FloatCast(edited, "integration", max_int=8).transform(edited.nodes[0])

    assert_model_outputs_close(original, edited, {"idx": np.array([[3]], dtype=np.int64)})


def test_dequantized_projection_matmul_is_numerically_equivalent():
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    q_weight = gs.Constant("q_weight", np.array([[10, 12, 14], [16, 18, 20]], dtype=np.uint8))
    scale = gs.Constant("scale", np.array(0.5, dtype=np.float32))
    zp = gs.Constant("zp", np.array(10, dtype=np.uint8))
    dequant_out = gs.Variable("weight_float", dtype=np.float32, shape=[2, 3])
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 3])
    dequant = gs.Node("DequantizeLinear", "dq", inputs=[q_weight, scale, zp], outputs=[dequant_out])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, dequant_out], outputs=[logits])
    original = graph(nodes=[dequant, matmul], inputs=[hidden], outputs=[logits])
    edited = clone_graph(original)
    DequantizeProjectionsMatMul(
        edited,
        "integration",
        hidden_size=2,
        vocab_size=3,
        export_dtype=onnx.TensorProto.FLOAT,
    ).transform(edited.nodes[1])

    feeds = {"hidden": np.array([[[1.0, 2.0]]], dtype=np.float32)}
    assert_model_outputs_close(original, edited, feeds)
