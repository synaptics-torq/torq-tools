import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.padding import AbsorbPadding, ReplacePadWithConcat, RewriteNegativePads


pytestmark = pytest.mark.padding


def test_rewrite_negative_pads_creates_slice_and_optional_positive_pad():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 4])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 4])
    pad = gs.Node(
        "Pad",
        "pad",
        inputs=[x, gs.Constant("pads", np.array([0, -1, 0, 1], dtype=np.int64))],
        outputs=[y],
        attrs={"mode": "constant"},
    )
    g = graph(nodes=[pad], inputs=[x], outputs=[y])

    edit = RewriteNegativePads(g, "unit")
    assert edit.match(pad)
    edit.transform(pad)

    assert pad.outputs == []
    assert {"Pad", "Slice"}.issubset({node.op for node in g.nodes if node.outputs})
    assert g.outputs[0].name == "pad_cropped"


def test_absorb_padding_merges_spatial_pads_into_conv():
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
    g = graph(nodes=[pad, conv], inputs=[x], outputs=[y])

    edit = AbsorbPadding(g, "unit")
    assert edit.match(pad)
    edit.transform(pad)

    assert conv.inputs[0] is x
    assert conv.attrs["pads"] == [1, 1]
    assert pad.outputs == []


def test_replace_pad_with_concat_replaces_constant_pad_node():
    data = gs.Variable("data", dtype=np.float32, shape=[1, 2])
    out = gs.Variable("padded", dtype=np.float32, shape=[1, 4])
    pad = gs.Node(
        "Pad",
        "pad",
        inputs=[
            data,
            gs.Constant("pads", np.array([0, 1, 0, 1], dtype=np.int64)),
            gs.Constant("value", np.array(3, dtype=np.float32)),
        ],
        outputs=[out],
        attrs={"mode": "constant"},
    )
    g = graph(nodes=[pad], inputs=[data], outputs=[out])

    edit = ReplacePadWithConcat(g, "unit")
    assert edit.match(pad)
    edit.transform(pad)

    assert pad.outputs == []
    concat = [node for node in g.nodes if node.op == "Concat" and node.outputs]
    assert len(concat) == 1
    assert concat[0].outputs[0] is out
    assert concat[0].attrs["axis"] == 1
    assert concat[0].inputs[0].values.tolist() == [[3]]


def test_replace_pad_with_concat_removes_noop_pad_graph_output():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 2])
    pad = gs.Node("Pad", "pad", inputs=[x, gs.Constant("pads", np.zeros(4, dtype=np.int64))], outputs=[y])
    g = graph(nodes=[pad], inputs=[x], outputs=[y])

    ReplacePadWithConcat(g, "unit").transform(pad)

    assert g.outputs[0] is x
    assert pad.outputs == []


def test_replace_pad_with_concat_rejects_negative_pads():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 1])
    pad = gs.Node("Pad", "pad", inputs=[x, gs.Constant("pads", np.array([0, -1, 0, 0], dtype=np.int64))], outputs=[y])

    with pytest.raises(ValueError, match="negative pads"):
        ReplacePadWithConcat(graph(nodes=[pad], inputs=[x], outputs=[y]), "unit").transform(pad)
