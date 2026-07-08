# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.shape import (
    BroadcastOpInputs,
    CollapseReshapeChain,
    ConstantBroadcastPolicy,
    EliminateExpand,
    EliminateRank0Gather,
    EliminateSingletonGatherUnsqueeze,
    EliminateTranspose,
)


pytestmark = pytest.mark.shape


def test_eliminate_expand_rewires_target_node_input():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3])
    expanded = gs.Variable("x_expanded", dtype=np.float32, shape=[2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 3])
    out = gs.Variable("out", dtype=np.float32, shape=[2, 3])
    expand = gs.Node("Expand", "expand", inputs=[x, gs.Constant("shape", np.array([2, 3], dtype=np.int64))], outputs=[expanded])
    add = gs.Node("Add", "add", inputs=[expanded, y], outputs=[out])
    g = graph(nodes=[expand, add], inputs=[x, y], outputs=[out])

    edit = EliminateExpand(g, "unit", ops=["Add"])
    assert edit.match(add)
    edit.transform(add)

    assert add.inputs[0] is x
    assert expand.outputs == []


def test_eliminate_transpose_identity_bypasses_graph_output():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 3])
    trans = gs.Node("Transpose", "transpose", inputs=[x], outputs=[y], attrs={"perm": [0, 1]})
    g = graph(nodes=[trans], inputs=[x], outputs=[y])

    edit = EliminateTranspose(g, "unit")
    assert edit.match(trans)
    edit.transform(trans)

    assert g.outputs[0] is x
    assert trans.outputs == []


def test_eliminate_transpose_data_preserving_perm_replaces_with_reshape():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 1, 3])
    z = gs.Variable("z", dtype=np.float32, shape=[2, 1, 3])
    trans = gs.Node("Transpose", "transpose", inputs=[x], outputs=[y], attrs={"perm": [1, 0, 2]})
    identity = gs.Node("Identity", "use", inputs=[y], outputs=[z])
    g = graph(nodes=[trans, identity], inputs=[x], outputs=[z])

    edit = EliminateTranspose(g, "unit")
    assert edit.match(trans)
    edit.transform(trans)

    assert trans.outputs == []
    assert identity.inputs[0].name == "y_reshaped"
    assert any(node.op == "Reshape" for node in g.nodes)


def test_eliminate_transpose_rejects_real_reorder():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[3, 2])
    trans = gs.Node("Transpose", "transpose", inputs=[x], outputs=[y], attrs={"perm": [1, 0]})

    assert not EliminateTranspose(graph(nodes=[trans], inputs=[x], outputs=[y]), "unit").match(trans)


def test_collapse_reshape_chain_keeps_final_reshape_output():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    mid = gs.Variable("mid", dtype=np.float32, shape=[6])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 6])
    reshape0 = gs.Node("Reshape", "reshape0", inputs=[x, gs.Constant("s0", np.array([6], dtype=np.int64))], outputs=[mid])
    reshape1 = gs.Node("Reshape", "reshape1", inputs=[mid, gs.Constant("s1", np.array([1, 6], dtype=np.int64))], outputs=[out])
    g = graph(nodes=[reshape0, reshape1], inputs=[x], outputs=[out])

    edit = CollapseReshapeChain(g, "unit")
    assert edit.match(reshape0)
    edit.transform(reshape0)

    assert reshape0.outputs == []
    assert reshape1.inputs[0] is x


def test_broadcast_op_inputs_adds_expand_for_variable_input():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3])
    y = gs.Variable("y", dtype=np.float32, shape=[2, 3])
    out = gs.Variable("out", dtype=np.float32, shape=[2, 3])
    add = gs.Node("Add", "add", inputs=[x, y], outputs=[out])
    g = graph(nodes=[add], inputs=[x, y], outputs=[out])

    edit = BroadcastOpInputs(g, "unit", ops=["Add"], inp_idx=[0])
    assert edit.match(add)
    edit.transform(add)

    assert add.inputs[0].name == "x_expanded"
    assert any(node.op == "Expand" for node in g.nodes)


def test_broadcast_op_inputs_materializes_constant_input():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    bias = gs.Constant("bias", np.array([[1, 2, 3]], dtype=np.float32))
    out = gs.Variable("out", dtype=np.float32, shape=[2, 3])
    add = gs.Node("Add", "add", inputs=[x, bias], outputs=[out])
    g = graph(nodes=[add], inputs=[x], outputs=[out])

    edit = BroadcastOpInputs(g, "unit", ops=["Add"], inp_idx=[1], constants_policy=ConstantBroadcastPolicy.MATERIALIZE)
    assert edit.match(add)
    edit.transform(add)

    assert isinstance(add.inputs[1], gs.Constant)
    assert add.inputs[1].name == "bias_bcast"
    np.testing.assert_array_equal(add.inputs[1].values, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32))


def test_broadcast_op_inputs_defer_runtime_constant_input():
    x = gs.Variable("x", dtype=np.float32, shape=[2, 3])
    bias = gs.Constant("bias", np.array([[1, 2, 3]], dtype=np.float32))
    out = gs.Variable("out", dtype=np.float32, shape=[2, 3])
    add = gs.Node("Add", "add", inputs=[x, bias], outputs=[out])
    g = graph(nodes=[add], inputs=[x], outputs=[out])

    edit = BroadcastOpInputs(g, "unit", ops=["Add"], inp_idx=[1], constants_policy=ConstantBroadcastPolicy.DEFER_RUNTIME)
    edit.transform(add)

    assert add.inputs[1].name == "bias_expanded"
    assert any(node.op == "Expand" for node in g.nodes)


def test_eliminate_rank0_gather_rewrites_unsqueeze_consumers():
    data = gs.Variable("data", dtype=np.float32, shape=[3])
    idx = gs.Constant("idx", np.array(1, dtype=np.int64))
    scalar = gs.Variable("scalar", dtype=np.float32, shape=[])
    out = gs.Variable("out", dtype=np.float32, shape=[1])
    gather = gs.Node("Gather", "gather", inputs=[data, idx], outputs=[scalar])
    unsq = gs.Node("Unsqueeze", "unsq", inputs=[scalar, gs.Constant("axes", np.array([0], dtype=np.int64))], outputs=[out])
    g = graph(nodes=[gather, unsq], inputs=[data], outputs=[out])

    edit = EliminateRank0Gather(g, "unit")
    assert edit.match(gather)
    edit.transform(gather)

    assert gather.outputs[0].shape == (1,)
    assert g.outputs[0] is gather.outputs[0]
    assert unsq.outputs == []


def test_eliminate_singleton_gather_unsqueeze_feeds_unary_from_original_data():
    data = gs.Variable("data", dtype=np.float32, shape=[2, 1, 3])
    gathered = gs.Variable("gathered", dtype=np.float32, shape=[2, 3])
    unary_out = gs.Variable("unary_out", dtype=np.float32, shape=[2, 3])
    out = gs.Variable("out", dtype=np.float32, shape=[2, 1, 3])
    gather_node = gs.Node("Gather", "gather", inputs=[data, gs.Constant("zero", np.array(0, dtype=np.int64))], outputs=[gathered], attrs={"axis": 1})
    relu = gs.Node("Relu", "relu", inputs=[gathered], outputs=[unary_out])
    unsq = gs.Node("Unsqueeze", "unsq", inputs=[unary_out, gs.Constant("axes", np.array([1], dtype=np.int64))], outputs=[out])
    g = graph(nodes=[gather_node, relu, unsq], inputs=[data], outputs=[out])

    edit = EliminateSingletonGatherUnsqueeze(g, "unit")
    assert edit.match(gather_node)
    edit.transform(gather_node)

    assert relu.inputs[0] is data
    assert relu.outputs[0] is out
    assert gather_node.outputs == []
    assert unsq.outputs == []
