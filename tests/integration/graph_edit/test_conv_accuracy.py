# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import (
    assert_model_outputs_close,
    clone_graph,
    conv_bn_graph,
    graph,
    only_node,
)
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
    original, _, _, _ = conv_bn_graph(
        np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 27.0,
        np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1),
        bias_values=np.array([0.5, -1.0], dtype=np.float32),
    )
    edited = clone_graph(original)
    edit = FoldConvBatchNorm(edited, "integration")
    conv_edited = only_node(edited, "Conv")
    assert edit.match(conv_edited)
    edit.transform(conv_edited)

    feeds = {"x": np.random.default_rng(0).standard_normal((1, 3, 4, 4)).astype(np.float32)}
    assert_model_outputs_close(original, edited, feeds)
