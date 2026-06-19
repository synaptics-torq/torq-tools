import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.rnn import DecomposeBidirectionalRnn


pytestmark = [pytest.mark.rnn, pytest.mark.ort]


def _rnn_graph(*, direction: str, seq_len: int = 4):
    directions = 2 if direction == "bidirectional" else 1
    x = gs.Variable("x", dtype=np.float32, shape=[seq_len, 1, 2])
    w = gs.Constant("w", np.arange(directions * 4, dtype=np.float32).reshape(directions, 2, 2) / 10.0)
    r = gs.Constant("r", np.eye(2, dtype=np.float32).reshape(1, 2, 2).repeat(directions, axis=0) / 2.0)
    y = gs.Variable("y", dtype=np.float32, shape=[seq_len, directions, 1, 2])
    y_h = gs.Variable("y_h", dtype=np.float32, shape=[directions, 1, 2])
    rnn = gs.Node("RNN", "rnn", inputs=[x, w, r], outputs=[y, y_h], attrs={"direction": direction, "hidden_size": 2})
    return graph(nodes=[rnn], inputs=[x], outputs=[y, y_h])


def test_bidirectional_rnn_decomposition_is_numerically_equivalent():
    original = _rnn_graph(direction="bidirectional", seq_len=4)
    edited = clone_graph(original)
    DecomposeBidirectionalRnn(edited, "integration").transform(edited.nodes[0])

    feeds = {"x": np.arange(8, dtype=np.float32).reshape(4, 1, 2) / 5.0}
    assert_model_outputs_close(original, edited, feeds, rtol=1e-5, atol=1e-5)


def test_forward_rnn_chunking_is_numerically_equivalent():
    original = _rnn_graph(direction="forward", seq_len=5)
    edited = clone_graph(original)
    DecomposeBidirectionalRnn(edited, "integration", max_chunk_len=2).transform(edited.nodes[0])

    feeds = {"x": np.arange(10, dtype=np.float32).reshape(5, 1, 2) / 5.0}
    assert_model_outputs_close(original, edited, feeds, rtol=1e-5, atol=1e-5)
