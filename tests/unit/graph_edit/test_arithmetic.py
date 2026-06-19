import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.arithmetic import (
    DequantizeProjectionsMatMul,
    FoldScalarMatMul,
    RemoveIsNaN,
    RemoveRedundantCasts,
    ReplaceConstantDivWithMul,
    ReplaceInt64FloatCast,
)


pytestmark = pytest.mark.arithmetic


def test_dequantize_projection_matmul_folds_uint8_weight():
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    q_weight = gs.Constant("q_weight", np.array([[10, 12, 14], [16, 18, 20]], dtype=np.uint8))
    dequant_out = gs.Variable("weight_float", dtype=np.float32, shape=[2, 3])
    dequant = gs.Node(
        "DequantizeLinear",
        "dq",
        inputs=[
            q_weight,
            gs.Constant("scale", np.array(0.5, dtype=np.float32)),
            gs.Constant("zp", np.array(10, dtype=np.uint8)),
        ],
        outputs=[dequant_out],
    )
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 3])
    matmul = gs.Node("MatMul", "lm_head", inputs=[hidden, dequant_out], outputs=[logits])
    g = graph(nodes=[dequant, matmul], inputs=[hidden], outputs=[logits])

    edit = DequantizeProjectionsMatMul(g, "unit", hidden_size=2, vocab_size=3, export_dtype=onnx.TensorProto.FLOAT)
    assert edit.match(matmul)
    edit.transform(matmul)

    assert isinstance(matmul.inputs[1], gs.Constant)
    np.testing.assert_allclose(matmul.inputs[1].values, (q_weight.values.astype(np.int32) - 10).astype(np.float32) * 0.5)
    assert dequant.outputs == []


def test_remove_isnan_rewires_where_consumers_to_original_input():
    x = gs.Variable("x", dtype=np.float32, shape=[2])
    isnan_out = gs.Variable("isnan", dtype=onnx.TensorProto.BOOL, shape=[2])
    where_out = gs.Variable("where_out", dtype=np.float32, shape=[2])
    y = gs.Variable("y", dtype=np.float32, shape=[2])
    isnan = gs.Node("IsNaN", "isnan", inputs=[x], outputs=[isnan_out])
    where = gs.Node(
        "Where",
        "where",
        inputs=[isnan_out, gs.Constant("zero", np.zeros(2, dtype=np.float32)), x],
        outputs=[where_out],
    )
    add = gs.Node("Add", "use", inputs=[where_out, gs.Constant("one", np.ones(2, dtype=np.float32))], outputs=[y])
    g = graph(nodes=[isnan, where, add], inputs=[x], outputs=[y])

    edit = RemoveIsNaN(g, "unit")
    assert edit.match(isnan)
    edit.transform(isnan)

    assert add.inputs[0] is x
    assert isnan.inputs == []
    assert where.outputs == []


def test_remove_redundant_cast_rewires_graph_output():
    x = gs.Variable("x", dtype=np.float32, shape=[2])
    y = gs.Variable("y", dtype=onnx.TensorProto.FLOAT, shape=[2])
    cast = gs.Node("Cast", "cast", inputs=[x], outputs=[y], attrs={"to": onnx.TensorProto.FLOAT})
    g = graph(nodes=[cast], inputs=[x], outputs=[y])

    edit = RemoveRedundantCasts(g, "unit")
    assert edit.match(cast)
    edit.transform(cast)

    assert g.outputs[0] is x
    assert cast.outputs == []


def test_fold_scalar_matmul_replaces_node_with_mul():
    a = gs.Variable("a", dtype=np.float32, shape=[1, 4, 1])
    b = gs.Variable("b", dtype=np.float32, shape=[1, 1])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4, 1])
    matmul = gs.Node("MatMul", "scale", inputs=[a, b], outputs=[y])
    g = graph(nodes=[matmul], inputs=[a, b], outputs=[y])

    edit = FoldScalarMatMul(g, "unit")
    assert edit.match(matmul)
    edit.transform(matmul)

    assert matmul.outputs == []
    assert any(node.op == "Mul" and node.outputs[0] is y for node in g.nodes)


def test_replace_constant_div_with_mul_reuses_reciprocal_constant():
    x = gs.Variable("x", dtype=np.float32, shape=[2])
    y = gs.Variable("y", dtype=np.float32, shape=[2])
    div = gs.Node("Div", "div", inputs=[x, gs.Constant("denom", np.array([2, 4], dtype=np.float32))], outputs=[y])
    g = graph(nodes=[div], inputs=[x], outputs=[y])

    edit = ReplaceConstantDivWithMul(g, "unit", export_dtype=onnx.TensorProto.FLOAT)
    assert edit.match(div)
    edit.transform(div)

    assert div.op == "Mul"
    assert div.inputs[1].name == "denom_reciprocal"
    np.testing.assert_allclose(div.inputs[1].values, np.array([0.5, 0.25], dtype=np.float32))


def test_replace_constant_div_with_mul_rejects_dynamic_divisor():
    x = gs.Variable("x", dtype=np.float32, shape=[2])
    denom = gs.Variable("denom", dtype=np.float32, shape=[2])
    y = gs.Variable("y", dtype=np.float32, shape=[2])
    div = gs.Node("Div", "div", inputs=[x, denom], outputs=[y])
    edit = ReplaceConstantDivWithMul(graph(nodes=[div], inputs=[x, denom], outputs=[y]), "unit", export_dtype=onnx.TensorProto.FLOAT)

    with pytest.raises(TypeError, match="second operand"):
        edit.transform(div)


def test_replace_int64_float_cast_builds_lookup_path():
    idx = gs.Variable("idx", dtype=np.dtype(np.int64), shape=[1, 1])
    cast_out = gs.Variable("idx_float", dtype=onnx.TensorProto.FLOAT, shape=[1, 1])
    y = gs.Variable("y", dtype=onnx.TensorProto.FLOAT, shape=[1, 1])
    cast = gs.Node("Cast", "cast", inputs=[idx], outputs=[cast_out], attrs={"to": onnx.TensorProto.FLOAT})
    add = gs.Node("Add", "use", inputs=[cast_out, gs.Constant("one", np.array([[1]], dtype=np.float32))], outputs=[y])
    g = graph(nodes=[cast, add], inputs=[idx], outputs=[y])

    edit = ReplaceInt64FloatCast(g, "unit", max_int=8)
    assert edit.match(cast)
    edit.transform(cast)

    assert cast.outputs == []
    assert add.inputs[0].name == "idx_float_value_batched"
    assert {"Reshape", "Gather"}.issubset({node.op for node in g.nodes})


def test_replace_int64_float_cast_skips_non_static_input_shape():
    idx = gs.Variable("idx", dtype=np.dtype(np.int64), shape=["batch"])
    out = gs.Variable("out", dtype=onnx.TensorProto.FLOAT, shape=["batch"])
    cast = gs.Node("Cast", "cast", inputs=[idx], outputs=[out], attrs={"to": onnx.TensorProto.FLOAT})

    assert not ReplaceInt64FloatCast(graph(nodes=[cast], inputs=[idx], outputs=[out]), "unit", max_int=8).match(cast)
