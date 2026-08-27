# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import assert_model_outputs_close, clone_graph, graph
from torq.graph_edit.edits.conv import (
    DecomposeStridedConv1D,
    FoldConvBatchNorm,
    WidenStridedDepthwiseConv,
)


pytestmark = [pytest.mark.conv, pytest.mark.ort]


def test_decomposed_strided_conv1d_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 2, 5])
    w = gs.Constant("w", np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 10.0)
    b = gs.Constant("b", np.array([0.5, -1.0, 2.0], dtype=np.float32))
    y = gs.Variable("y", dtype=np.float32, shape=[1, 3, 2])
    conv = gs.Node("Conv", "conv", inputs=[x, w, b], outputs=[y], attrs={"kernel_shape": [3], "strides": [2], "pads": [0, 0], "group": 1})
    original = graph(nodes=[conv], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    DecomposeStridedConv1D(edited, "integration").transform(edited.nodes[0])

    feeds = {"x": np.arange(10, dtype=np.float32).reshape(1, 2, 5) / 7.0}
    assert_model_outputs_close(original, edited, feeds, rtol=1e-5, atol=1e-5)


def test_widened_depthwise_conv_is_numerically_equivalent_after_slice():
    x = gs.Variable("x", dtype=np.dtype(np.float32), shape=[1, 2, 5])
    w = gs.Constant("w", np.array([[[1, 0, -1]], [[0.5, 1.0, 0.5]]], dtype=np.float32))
    y = gs.Variable("y", dtype=np.dtype(np.float32), shape=[1, 2, 2])
    conv = gs.Node("Conv", "conv", inputs=[x, w], outputs=[y], attrs={"kernel_shape": [3], "strides": [2], "pads": [0, 0], "group": 2})
    original = graph(nodes=[conv], inputs=[x], outputs=[y])
    edited = clone_graph(original)
    WidenStridedDepthwiseConv(edited, "integration").transform(edited.nodes[0])

    feeds = {"x": np.arange(10, dtype=np.float32).reshape(1, 2, 5) / 5.0}
    assert_model_outputs_close(original, edited, feeds)


def test_folded_conv_batchnorm_is_numerically_equivalent():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3, 4, 4])
    w = gs.Constant("w", np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 27.0)
    c = gs.Constant("c", np.array([0.5, -1.0], dtype=np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 2, 4, 4])
    mul_out = gs.Variable("mul_out", dtype=np.float32, shape=[1, 2, 4, 4])
    add_out = gs.Variable("add_out", dtype=np.float32, shape=[1, 2, 4, 4])
    conv = gs.Node("Conv", "conv", inputs=[x, w, c], outputs=[conv_out],
                   attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]})
    mul = gs.Node("Mul", "mul", inputs=[conv_out, gs.Constant("s", np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1))], outputs=[mul_out])
    add = gs.Node("Add", "add", inputs=[mul_out, gs.Constant("b", np.array([10.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1))], outputs=[add_out])
    original = graph(nodes=[conv, mul, add], inputs=[x], outputs=[add_out])

    edited = clone_graph(original)
    edit = FoldConvBatchNorm(edited, "integration")
    conv_edited = next(node for node in edited.nodes if node.op == "Conv")
    assert edit.match(conv_edited)
    edit.transform(conv_edited)

    feeds = {"x": np.random.default_rng(0).standard_normal((1, 3, 4, 4)).astype(np.float32)}
    assert_model_outputs_close(original, edited, feeds)
