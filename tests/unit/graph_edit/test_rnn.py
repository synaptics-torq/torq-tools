import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.rnn import DecomposeBidirectionalRnn


pytestmark = pytest.mark.rnn


def _rnn_node(direction="bidirectional", layout=0):
    x = gs.Variable("x", dtype=np.float32, shape=[3, 1, 2])
    w = gs.Constant("w", np.ones((2 if direction == "bidirectional" else 1, 2, 2), dtype=np.float32))
    r = gs.Constant("r", np.ones((2 if direction == "bidirectional" else 1, 2, 2), dtype=np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[3, 2 if direction == "bidirectional" else 1, 1, 2])
    node = gs.Node("RNN", "rnn", inputs=[x, w, r], outputs=[y], attrs={"direction": direction, "hidden_size": 2, "layout": layout})
    return graph(nodes=[node], inputs=[x], outputs=[y]), node


def test_restore_output_arity_adds_missing_state_outputs():
    g, node = _rnn_node(direction="forward")

    DecomposeBidirectionalRnn.restore_output_arity(g)

    assert len(node.outputs) == 2
    assert node.outputs[1].name == "rnn:Y_h"
    assert node.outputs[1].shape == (1, 1, 2)


def test_max_chunk_len_must_be_positive():
    g, _ = _rnn_node(direction="forward")

    with pytest.raises(ValueError, match="max_chunk_len"):
        DecomposeBidirectionalRnn(g, "unit", max_chunk_len=0)


def test_bidirectional_rnn_decomposition_creates_forward_and_reverse_branches():
    g, node = _rnn_node()

    edit = DecomposeBidirectionalRnn(g, "unit")
    assert edit.match(node)
    edit.transform(node)

    live_ops = {n.op for n in g.nodes if n.outputs}
    assert {"RNN", "Gather", "Concat"}.issubset(live_ops)
    assert node.outputs == []
    assert any(n.name == "rnn_fwd" for n in g.nodes)
    assert any(n.name == "rnn_rev" for n in g.nodes)


def test_bidirectional_layout_one_raises_not_implemented():
    g, node = _rnn_node(layout=1)
    edit = DecomposeBidirectionalRnn(g, "unit")

    with pytest.raises(NotImplementedError, match="layout=0"):
        edit.transform(node)


def test_forward_rnn_chunking_splits_long_sequence():
    g, node = _rnn_node(direction="forward")
    node.inputs.extend([gs.Variable.empty(), gs.Variable.empty(), gs.Variable.empty()])
    node.outputs.append(gs.Variable("y_h", dtype=np.float32, shape=[1, 1, 2]))

    edit = DecomposeBidirectionalRnn(g, "unit", max_chunk_len=2)
    assert edit.match(node)
    edit.transform(node)

    assert node.outputs == []
    assert any(n.op == "Split" and n.name == "rnn_x_split" for n in g.nodes)
    assert [n.name for n in g.nodes if n.name.startswith("rnn_chunk")] == ["rnn_chunk0", "rnn_chunk1"]
    assert any(n.op == "Concat" and n.name == "rnn_y_concat" for n in g.nodes)
