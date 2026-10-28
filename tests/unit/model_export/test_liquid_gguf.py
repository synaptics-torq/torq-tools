# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

gguf = pytest.importorskip("gguf")

from torq.models.liquid.gguf import apply_gguf_q4_0, gguf_tensor_name, unpack_q4_0, unpack_q6_k  # noqa: E402

N_IN, N_OUT = 64, 48
WEIGHT = "model.layers.0.mlp.down_proj.MatMul.weight"


def _q4_0_bytes(rng) -> np.ndarray:
    """Random ggml block_q4_0 data for a [N_OUT, N_IN] weight: fp16 d + 16 nibble bytes per block."""
    nb = N_IN // 32
    d = rng.uniform(-0.05, 0.05, size=(N_OUT, nb, 1)).astype(np.float16).view(np.uint8)
    qs = rng.integers(0, 256, size=(N_OUT, nb, 16), dtype=np.uint8)
    return np.concatenate([d, qs], axis=-1).reshape(N_OUT, nb * 18)


class _Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = np.array([N_IN, N_OUT], dtype=np.uint64)   # ggml ne order
        self.tensor_type = gguf.GGMLQuantizationType.Q4_0


def test_unpack_q4_0_matches_ggml_dequantize():
    raw = _q4_0_bytes(np.random.default_rng(0))
    codes, scale = unpack_q4_0(_Tensor(raw))
    assert codes.shape == (N_IN, N_OUT) and scale.shape == (N_IN // 32, N_OUT)
    assert codes.min() >= -8 and codes.max() <= 7
    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.Q4_0)          # [N_OUT, N_IN]
    deq = (codes.reshape(-1, 32, N_OUT) * scale[:, None, :]).reshape(N_IN, N_OUT)
    np.testing.assert_array_equal(deq.T, ref)


def test_gguf_tensor_names():
    assert gguf_tensor_name("model.embed_tokens.weight", 16) == "token_embd.weight"
    assert gguf_tensor_name("model.layers.16.final_norm_layernorm.weight", 16) == "token_embd_norm.weight"
    assert gguf_tensor_name("model.layers.3.conv.in_proj.MatMul.weight", 16) == "blk.3.shortconv.in_proj.weight"
    assert gguf_tensor_name("cos_cache", 16) is None


def test_apply_gguf_q4_0_is_lossless(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    raw = _q4_0_bytes(np.random.default_rng(1))
    w = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.Q4_0).T.astype(np.float32)

    writer = gguf.GGUFWriter(str(tmp_path / "m.gguf"), "lfm2")
    writer.add_block_count(1)
    writer.add_tensor("blk.0.ffn_down.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, N_IN])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N_OUT])
    g = helper.make_graph([helper.make_node("MatMul", ["x", WEIGHT], ["y"])], "g", [x], [y],
                          [numpy_helper.from_array(w, WEIGHT)])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)]), tmp_path / "src.onnx")

    assert apply_gguf_q4_0(tmp_path / "m.gguf", tmp_path / "src.onnx", tmp_path / "dst.onnx") == 1
    dst = onnx.load(tmp_path / "dst.onnx")
    dql = [n for n in dst.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dql) == 1 and helper.get_attribute_value(dql[0].attribute[1]) == 32
    assert {t.name: t.data_type for t in dst.graph.initializer}[f"{WEIGHT}/int4"] == TensorProto.INT4

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(tmp_path / "dst.onnx"), opts, providers=["CPUExecutionProvider"])
    xv = np.random.default_rng(2).standard_normal((1, N_IN)).astype(np.float32)
    np.testing.assert_allclose(sess.run(None, {"x": xv})[0], xv @ w, rtol=1e-5, atol=1e-5)


def test_unpack_q6_k_matches_ggml_dequantize():
    rows, n = 8, 512                                  # two Q6_K super-blocks per row
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=(rows, n // 256, 210), dtype=np.uint8)
    d = rng.uniform(-0.01, 0.01, size=(rows, n // 256, 1)).astype(np.float16).view(np.uint8)
    raw[..., 208:210] = d
    raw = raw.reshape(rows, -1)

    class T:
        data = raw
        shape = np.array([n, rows], dtype=np.uint64)

    codes, scales = unpack_q6_k(T())
    assert codes.shape == (rows, n) and scales.shape == (rows, n // 16)
    assert codes.min() >= -32 and codes.max() <= 31
    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.Q6_K)
    np.testing.assert_array_equal((codes.reshape(rows, -1, 16) * scales[..., None]).reshape(rows, n), ref)
