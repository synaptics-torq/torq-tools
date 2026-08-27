# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit.edits.conv import (
    DecomposeStridedConv1D,
    FoldConvBatchNorm,
    WidenStridedDepthwiseConv,
)


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


def _conv_bn_graph(w_values, scale_values, shift_values, bias_values=None, extra_consumer=False):
    cin = w_values.shape[1]
    spatial = [4] * (w_values.ndim - 2)
    cout = w_values.shape[0]
    x = gs.Variable("x", dtype=np.float32, shape=[1, cin] + spatial)
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, cout] + spatial)
    mul_out = gs.Variable("mul_out", dtype=np.float32, shape=[1, cout] + spatial)
    add_out = gs.Variable("add_out", dtype=np.float32, shape=[1, cout] + spatial)
    conv_inputs = [x, gs.Constant("w", w_values)]
    if bias_values is not None:
        conv_inputs.append(gs.Constant("c", bias_values))
    conv = gs.Node("Conv", "conv", inputs=conv_inputs, outputs=[conv_out],
                   attrs={"kernel_shape": list(w_values.shape[2:]), "pads": [1] * (2 * len(spatial))})
    mul = gs.Node("Mul", "mul", inputs=[conv_out, gs.Constant("s", scale_values)], outputs=[mul_out])
    add = gs.Node("Add", "add", inputs=[mul_out, gs.Constant("b", shift_values)], outputs=[add_out])
    nodes = [conv, mul, add]
    outputs = [add_out]
    if extra_consumer:
        relu_out = gs.Variable("relu_out", dtype=np.float32, shape=[1, cout] + spatial)
        nodes.append(gs.Node("Relu", "relu", inputs=[conv_out], outputs=[relu_out]))
        outputs.append(relu_out)
    g = graph(nodes=nodes, inputs=[x], outputs=outputs)
    return g, conv, mul, add


def test_fold_conv_batchnorm_folds_scale_and_shift():
    w = np.ones((2, 3, 3, 3), dtype=np.float32)
    g, conv, mul, add = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
    )

    edit = FoldConvBatchNorm(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    np.testing.assert_array_equal(conv.inputs[1].values[0], np.full((3, 3, 3), 2.0))
    np.testing.assert_array_equal(conv.inputs[1].values[1], np.full((3, 3, 3), 3.0))
    np.testing.assert_array_equal(conv.inputs[2].values, [10.0, 20.0])
    assert conv.outputs[0].name == "add_out"
    assert g.outputs[0] is conv.outputs[0]
    assert mul.outputs == []
    assert add.outputs == []


def test_fold_conv_batchnorm_composes_existing_bias():
    w = np.ones((2, 3, 3, 3), dtype=np.float32)
    g, conv, _, _ = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
        bias_values=np.array([1.0, -1.0], dtype=np.float32),
    )

    edit = FoldConvBatchNorm(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    np.testing.assert_array_equal(conv.inputs[2].values, [12.0, 17.0])  # c*s + b


def test_fold_conv_batchnorm_accepts_scalar_and_conv1d():
    w = np.ones((2, 3, 5), dtype=np.float32)
    g, conv, _, _ = _conv_bn_graph(
        w,
        np.array(2.0, dtype=np.float32),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1),
    )

    edit = FoldConvBatchNorm(g, "unit")
    assert edit.match(conv)
    edit.transform(conv)

    np.testing.assert_array_equal(conv.inputs[1].values, np.full((2, 3, 5), 2.0))
    np.testing.assert_array_equal(conv.inputs[2].values, [10.0, 20.0])


def test_fold_conv_batchnorm_rejects_last_axis_broadcast():
    # A rank-1 [C] constant broadcasts onto the LAST axis of the NCHW output,
    # not the channel axis; folding it per-channel would miscompile.
    w = np.ones((2, 3, 3, 3), dtype=np.float32)
    g, conv, _, _ = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
    )

    assert not FoldConvBatchNorm(g, "unit").match(conv)


def test_fold_conv_batchnorm_rejects_second_consumer():
    w = np.ones((2, 3, 3, 3), dtype=np.float32)
    g, conv, _, _ = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
        extra_consumer=True,
    )

    assert not FoldConvBatchNorm(g, "unit").match(conv)


def test_fold_conv_batchnorm_rejects_activation_add_and_nonconst_weight():
    w = np.ones((2, 3, 3, 3), dtype=np.float32)
    g, conv, mul, add = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
    )
    residual = gs.Variable("residual", dtype=np.float32, shape=[1, 2, 4, 4])
    add.inputs = [mul.outputs[0], residual]
    g.inputs.append(residual)
    assert not FoldConvBatchNorm(g, "unit").match(conv)

    g2, conv2, _, _ = _conv_bn_graph(
        w,
        np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, 20.0], dtype=np.float32).reshape(1, 2, 1, 1),
    )
    w_var = gs.Variable("w_var", dtype=np.float32, shape=[2, 3, 3, 3])
    conv2.inputs[1] = w_var
    g2.inputs.append(w_var)
    assert not FoldConvBatchNorm(g2, "unit").match(conv2)
