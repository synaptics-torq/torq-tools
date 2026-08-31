# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, graph, to_model
from torq.tools.onnx_cleanup import cleanup_onnx_model


pytestmark = pytest.mark.ort


def _artifact_model():
    """A model with all three exporter artifacts the cleanup pipeline targets:
    an unrolled Concat, a constant island feeding the BN scale, and an
    unfused Conv -> Mul -> Add BatchNorm chain."""
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3, 4, 4])
    v = gs.Variable("v", dtype=np.float32, shape=[1, 4, 8])

    # constant island: gamma = g0 * g1 (only computable offline)
    g0 = gs.Constant("g0", np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1))
    g1 = gs.Constant("g1", np.array([1.5, 3.0], dtype=np.float32).reshape(1, 2, 1, 1))
    gamma = gs.Variable("gamma", dtype=np.float32, shape=[1, 2, 1, 1])
    island = gs.Node("Mul", "island", inputs=[g0, g1], outputs=[gamma])

    # Conv -> Mul -> Add (eval-mode BatchNorm)
    w = gs.Constant("w", np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 27.0)
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 2, 4, 4])
    mul_out = gs.Variable("mul_out", dtype=np.float32, shape=[1, 2, 4, 4])
    bn_out = gs.Variable("bn_out", dtype=np.float32, shape=[1, 2, 4, 4])
    conv = gs.Node("Conv", "conv", inputs=[x, w], outputs=[conv_out],
                   attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]})
    mul = gs.Node("Mul", "mul", inputs=[conv_out, gamma], outputs=[mul_out])
    add = gs.Node("Add", "add", inputs=[
        mul_out,
        gs.Constant("beta", np.array([10.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1)),
    ], outputs=[bn_out])

    # unrolled stack/unbind Concat over v
    nodes = [island, conv, mul, add]
    cat_inputs = []
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
    cat_out = gs.Variable("cat_out", dtype=np.float32, shape=[1, 4, 8])
    nodes.append(gs.Node("Concat", "cat", inputs=cat_inputs, outputs=[cat_out], attrs={"axis": 1}))

    return to_model(graph(nodes=nodes, inputs=[x, v], outputs=[bn_out, cat_out]))


def test_cleanup_pipeline_removes_all_three_artifact_kinds():
    model = _artifact_model()
    cleaned = cleanup_onnx_model(model, min_fanin=4)

    ops = [node.op_type for node in cleaned.graph.node]
    # BN and the constant island are folded away; the Concat collapsed to
    # its source tensor; the slice/squeeze/unsqueeze chains are dead-pruned.
    assert ops.count("Conv") == 1
    assert "Mul" not in ops and "Add" not in ops
    assert "Slice" not in ops and "Squeeze" not in ops and "Unsqueeze" not in ops
    concat = next(node for node in cleaned.graph.node if node.op_type == "Concat")
    assert list(concat.input) == ["v"]

    rng = np.random.default_rng(0)
    feeds = {
        "x": rng.standard_normal((1, 3, 4, 4)).astype(np.float32),
        "v": rng.standard_normal((1, 4, 8)).astype(np.float32),
    }
    assert_model_outputs_close(model, cleaned, feeds)


def test_cleanup_is_idempotent():
    model = _artifact_model()
    once = cleanup_onnx_model(model, min_fanin=4)
    twice = cleanup_onnx_model(once, min_fanin=4)
    assert once.SerializeToString() == twice.SerializeToString()


def test_cleanup_skip_flags_disable_passes():
    model = _artifact_model()
    cleaned = cleanup_onnx_model(model, min_fanin=4, skip=("fold-conv-bn",))

    ops = [node.op_type for node in cleaned.graph.node]
    assert "Mul" in ops  # BN chain kept
    concat = next(node for node in cleaned.graph.node if node.op_type == "Concat")
    assert list(concat.input) == ["v"]  # concat still collapsed


def test_cleanup_rejects_unknown_pass_name():
    with pytest.raises(ValueError, match="Unknown cleanup pass"):
        cleanup_onnx_model(_artifact_model(), skip=("no-such-pass",))
