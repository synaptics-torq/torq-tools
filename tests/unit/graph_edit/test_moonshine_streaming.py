# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Regression tests for the moonshine_streaming static-encoder mask sentinels.

HF transformers' eager mask interface materializes blocked attention entries as
``-finfo(min)``; the static encoder export bakes those masks into fp32 constants
that exceed the bf16 converter's ``max_float`` guard (issue #58: "Cannot convert
constant 'where' to bf16"). ``clamp_attention_mask_constants`` must replace the
sentinels with the finite block bias so the converter accepts them.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from torq.lab.model_tools.dtype_conversion.onnx import convert_model
from torq.models.moonshine_streaming._graph import (
    MoonshineStreamingOnnxGraphEditor,
    _ATTN_BLOCK_BIAS,
)


def _make_encoder_like_model() -> onnx.ModelProto:
    """Minimal graph mirroring the baked attention-bias constant + a weight."""
    mask = np.zeros((1, 2, 3), dtype=np.float32)
    mask[0, 0, 0] = np.finfo(np.float32).min
    mask[0, 1, 2] = -np.inf
    weight = np.ones((3,), dtype=np.float32)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3])
    nodes = [
        helper.make_node("Add", ["x", "weight"], ["t"]),
        helper.make_node("Add", ["t", "mask"], ["y"]),
    ]
    inits = [
        helper.make_tensor("mask", TensorProto.FLOAT, mask.shape, mask),
        helper.make_tensor("weight", TensorProto.FLOAT, weight.shape, weight),
    ]
    graph = helper.make_graph(nodes, "encoder", [x], [y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        i.name: onnx.numpy_helper.to_array(i)
        for i in model.graph.initializer
    }


def test_bf16_convert_rejects_unclamped_mask_constant(tmp_path):
    # Guards the premise: without the clamp, the converter raises exactly the
    # ValueError from issue #58.
    src = tmp_path / "unclamped.onnx"
    onnx.save(_make_encoder_like_model(), src)
    with pytest.raises(ValueError, match="Cannot convert constant 'mask' to bf16"):
        convert_model(src, tmp_path / "out.onnx", "bf16")


def test_clamp_attention_mask_constants_replaces_sentinels():
    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
        _make_encoder_like_model(), "encoder", TensorProto.FLOAT
    )
    assert editor.clamp_attention_mask_constants() is editor

    inits = _initializers(editor.to_onnx(override_ir=10))
    mask, weight = inits["mask"], inits["weight"]
    assert (mask == 0.0).sum() == 4
    assert (mask == _ATTN_BLOCK_BIAS).sum() == 2
    assert np.array_equal(weight, np.ones(3, dtype=np.float32))


def test_clamped_model_converts_to_bf16(tmp_path):
    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
        _make_encoder_like_model(), "encoder", TensorProto.FLOAT
    )
    editor.clamp_attention_mask_constants()
    src = tmp_path / "clamped.onnx"
    onnx.save(editor.to_onnx(override_ir=10), src)

    for values in _initializers(onnx.load(src)).values():
        assert np.all(np.abs(values) <= 1e9)
    convert_model(src, tmp_path / "clamped_bf16.onnx", "bf16")


def test_clamp_is_idempotent():
    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
        _make_encoder_like_model(), "encoder", TensorProto.FLOAT
    )
    editor.clamp_attention_mask_constants()
    first = _initializers(editor.to_onnx(override_ir=10))
    editor.clamp_attention_mask_constants()
    second = _initializers(editor.to_onnx(override_ir=10))
    assert set(first) == set(second)
    assert all(np.array_equal(first[k], second[k]) for k in first)
