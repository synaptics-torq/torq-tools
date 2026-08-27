# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.shape import (
    BroadcastOpInputs,
    CollapseUnrolledConcat,
    ConstantBroadcastPolicy,
    EliminateTranspose,
)


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


def _unrolled_concat_original():
    v = gs.Variable("v", dtype=np.float32, shape=[1, 4, 8])
    tok = gs.Variable("tok", dtype=np.float32, shape=[1, 1, 8])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 5, 8])
    nodes = []
    cat_inputs = [tok]
    for i in range(4):
        sliced = gs.Variable(f"sl{i}", dtype=np.float32, shape=[1, 1, 8])
        squeezed = gs.Variable(f"sq{i}", dtype=np.float32, shape=[1, 8])
        unsqueezed = gs.Variable(f"un{i}", dtype=np.float32, shape=[1, 1, 8])
        nodes += [
            gs.Node("Slice", f"slice{i}", inputs=[
                v,
                gs.Constant(f"s{i}", np.array([i], dtype=np.int64)),
                gs.Constant(f"e{i}", np.array([i + 1], dtype=np.int64)),
                gs.Constant(f"a{i}", np.array([1], dtype=np.int64)),
            ], outputs=[sliced]),
            gs.Node("Squeeze", f"squeeze{i}", inputs=[
                sliced, gs.Constant(f"sqax{i}", np.array([1], dtype=np.int64)),
            ], outputs=[squeezed]),
            gs.Node("Unsqueeze", f"unsqueeze{i}", inputs=[
                squeezed, gs.Constant(f"unax{i}", np.array([1], dtype=np.int64)),
            ], outputs=[unsqueezed]),
        ]
        cat_inputs.append(unsqueezed)
    nodes.append(gs.Node("Concat", "cat", inputs=cat_inputs, outputs=[out], attrs={"axis": 1}))
    return graph(nodes=nodes, inputs=[v, tok], outputs=[out])


def test_collapsed_unrolled_concat_is_numerically_equivalent():
    original = _unrolled_concat_original()
    edited = clone_graph(original)
    edit = CollapseUnrolledConcat(edited, "integration", min_fanin=5)
    concat = next(node for node in edited.nodes if node.op == "Concat")
    assert edit.match(concat)
    edit.transform(concat)

    rng = np.random.default_rng(0)
    feeds = {
        "v": rng.standard_normal((1, 4, 8)).astype(np.float32),
        "tok": rng.standard_normal((1, 1, 8)).astype(np.float32),
    }
    assert_model_outputs_close(original, edited, feeds)
