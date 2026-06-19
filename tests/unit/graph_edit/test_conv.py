import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.conv import DecomposeStridedConv1D, WidenStridedDepthwiseConv


pytestmark = pytest.mark.conv


def test_widen_strided_depthwise_conv_adds_trailing_pad_and_slice():
    x = gs.Variable("x", dtype=np.dtype(np.float32), shape=[1, 2, 5])
    w = gs.Constant("w", np.ones((2, 1, 3), dtype=np.float32))
    y = gs.Variable("y", dtype=np.dtype(np.float32), shape=[1, 2, 2])
    conv = gs.Node(
        "Conv",
        "conv",
        inputs=[x, w],
        outputs=[y],
        attrs={"kernel_shape": [3], "strides": [2], "group": 2, "pads": [0, 0]},
    )
    g = graph(nodes=[conv], inputs=[x], outputs=[y])

    edit = WidenStridedDepthwiseConv(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    assert conv.outputs[0].name == "y_widened"
    assert conv.attrs["pads"] == [0, 6]
    assert g.outputs[0].name == "y"
    assert any(node.op == "Slice" for node in g.nodes)


def test_widen_strided_depthwise_conv_skips_non_depthwise_conv():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2, 5])
    w = gs.Constant("w", np.ones((4, 2, 3), dtype=np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4, 2])
    conv = gs.Node("Conv", "conv", inputs=[x, w], outputs=[y], attrs={"kernel_shape": [3], "strides": [2], "group": 1, "pads": [0, 0]})

    assert not WidenStridedDepthwiseConv(graph(nodes=[conv], inputs=[x], outputs=[y]), "unit").match(conv)


def test_decompose_strided_conv1d_replaces_general_conv_output():
    data = gs.Variable("data", dtype=np.float32, shape=[1, 2, 5])
    weight = gs.Constant("weight", np.ones((3, 2, 3), dtype=np.float32))
    bias = gs.Constant("bias", np.ones((3,), dtype=np.float32))
    out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 3, 2])
    conv = gs.Node(
        "Conv",
        "conv",
        inputs=[data, weight, bias],
        outputs=[out],
        attrs={"kernel_shape": [3], "strides": [2], "group": 1, "pads": [0, 0]},
    )
    g = graph(nodes=[conv], inputs=[data], outputs=[out])

    edit = DecomposeStridedConv1D(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    assert conv.outputs == []
    assert {"Slice", "Concat", "MatMul", "Add", "Transpose"}.issubset({node.op for node in g.nodes if node.outputs})
    assert g.outputs[0].name == "conv_out"


def test_decompose_strided_conv1d_replaces_single_channel_unsqueeze_case():
    raw = gs.Variable("raw", dtype=np.float32, shape=[1, 8])
    data = gs.Variable("data", dtype=np.float32, shape=[1, 1, 8])
    weight = gs.Constant("weight", np.ones((2, 1, 3), dtype=np.float32))
    out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 2, 3])
    unsq = gs.Node("Unsqueeze", "unsq", inputs=[raw, gs.Constant("axes", np.array([1], dtype=np.int64))], outputs=[data])
    conv = gs.Node(
        "Conv",
        "conv",
        inputs=[data, weight],
        outputs=[out],
        attrs={"kernel_shape": [3], "strides": [2], "group": 1, "pads": [0, 0]},
    )
    g = graph(nodes=[unsq, conv], inputs=[raw], outputs=[out])

    edit = DecomposeStridedConv1D(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    assert conv.outputs == []
    assert unsq.outputs == []
    assert any(node.name == "conv_matmul" for node in g.nodes)
