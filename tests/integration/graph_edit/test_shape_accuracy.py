import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.shape import BroadcastOpInputs, ConstantBroadcastPolicy, EliminateTranspose


pytestmark = [pytest.mark.shape, pytest.mark.ort]


def test_data_preserving_transpose_elimination_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 1, 3])
    trans = gs.Node("Transpose", "transpose", inputs=[x], outputs=[y], attrs={"perm": [1, 0, 2]})
    original = graph(nodes=[trans], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    EliminateTranspose(edited, "integration").transform(edited.nodes[0])

    feeds = {"x": np.arange(6, dtype=np.float32).reshape(1, 2, 3)}
    assert_model_outputs_close(original, edited, feeds)


def test_materialized_constant_broadcast_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    bias = gs.Constant("bias", np.array([[1, 2, 3]], dtype=np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[2, 3])
    add = gs.Node("Add", "add", inputs=[x, bias], outputs=[y])
    original = graph(nodes=[add], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    BroadcastOpInputs(
        edited,
        "integration",
        ops=["Add"],
        inp_idx=[1],
        constants_policy=ConstantBroadcastPolicy.MATERIALIZE,
    ).transform(edited.nodes[0])

    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    assert_model_outputs_close(original, edited, feeds)
