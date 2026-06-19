import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.padding import AbsorbPadding, ReplacePadWithConcat, RewriteNegativePads


pytestmark = [pytest.mark.padding, pytest.mark.ort]


def test_pad_with_concat_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 4])
    pad = gs.Node(
        "Pad",
        "pad",
        inputs=[
            x,
            gs.Constant("pads", np.array([1, 1, 0, 1], dtype=np.int64)),
            gs.Constant("value", np.array(3, dtype=np.float32)),
        ],
        outputs=[y],
        attrs={"mode": "constant"},
    )
    original = graph(nodes=[pad], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    ReplacePadWithConcat(edited, "integration").transform(edited.nodes[0])

    assert_model_outputs_close(original, edited, {"x": np.array([[1, 2]], dtype=np.float32)})


def test_negative_pad_rewrite_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 4])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4])
    pad = gs.Node(
        "Pad",
        "pad",
        inputs=[x, gs.Constant("pads", np.array([0, -1, 0, 1], dtype=np.int64))],
        outputs=[y],
        attrs={"mode": "constant"},
    )
    original = graph(nodes=[pad], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    RewriteNegativePads(edited, "integration").transform(edited.nodes[0])

    assert_model_outputs_close(original, edited, {"x": np.array([[1, 2, 3, 4]], dtype=np.float32)})


def test_absorbed_padding_is_numerically_equivalent_to_pad_conv_chain():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 1, 4])
    padded = gs.Variable("padded", dtype=np.float32, shape=[1, 1, 6])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 1, 6])
    pad = gs.Node("Pad", "pad", inputs=[x, gs.Constant("pads", np.array([0, 0, 1, 0, 0, 1], dtype=np.int64))], outputs=[padded])
    conv = gs.Node(
        "Conv",
        "conv",
        inputs=[padded, gs.Constant("w", np.ones((1, 1, 1), dtype=np.float32))],
        outputs=[y],
        attrs={"kernel_shape": [1], "pads": [0, 0], "strides": [1]},
    )
    original = graph(nodes=[pad, conv], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    AbsorbPadding(edited, "integration").transform(edited.nodes[0])

    assert_model_outputs_close(original, edited, {"x": np.array([[[1, 2, 3, 4]]], dtype=np.float32)})
