# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""EmbeddingGemma-2 exporter: CLI arguments, NSS program budgets and the int8/int4
weight rewrites for torq-compile, on small synthetic graphs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.models.embedding_gemma2.export import add_embedding_gemma2_export_args, nss_programs_size
from torq.models.embedding_gemma2.q4 import (
    BLOCK,
    avoid_int8_min_zero_point,
    split_int4_matmul_rows,
    split_int8_matmul_cols,
)

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_embedding_gemma2_export_args(parser)
    return parser


def test_cli_defaults_cover_every_network():
    args = _parser().parse_args([])
    assert args.weights == "fp32"
    assert [int(s) for s in args.seq_lens.split(",")] == [128, 512]
    assert args.image_sizes == [(384, 384)]
    assert sorted(int(f) for f in args.audio_frames.split(",")) == [140, 280, 560, 1120]
    assert not args.no_audio and not args.no_compile and not args.skip_validate


@pytest.mark.parametrize("weights", ["fp32", "q4", "q8"])
def test_cli_weights(weights):
    assert _parser().parse_args(["--weights", weights]).weights == weights


def test_cli_rejects_image_size_off_the_patch_grid():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--image-sizes", "384x380"])


def test_export_model_cli_loads_without_pillow():
    """`torq-export-model` imports every exporter's argument parser; the EmbeddingGemma-2
    one must not need Pillow or onnxsim (optional extra) to load."""
    code = ("import sys; sys.modules['PIL'] = None; sys.modules['onnxsim'] = None; "
            "sys.argv = ['torq-export-model', 'embeddinggemma2', '--help']; "
            "from torq.models.export_model import main; main()")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "--weights" in out.stdout


@pytest.mark.parametrize(("name", "weights", "mb"), [
    ("vision_384x384", "q8", 100),
    ("audio_140", "q8", 32),
    ("audio_1120", "q4", 24),
    ("text_body_s128", "q8", 16),
    ("text_body_s512", "q8", 60),
    ("text_body_s128", "fp32", 8),
    ("text_body_s512", "fp32", 16),
])
def test_nss_programs_size(name, weights, mb):
    assert nss_programs_size(name, weights) == mb << 20


def _dql_matmul_model(q: np.ndarray, scale: np.ndarray, zp: np.ndarray, x_shape: list[int],
                      qtype: int = TensorProto.INT8) -> onnx.ModelProto:
    """``y = x @ DequantizeLinear(q, scale, zp, axis=0, block_size=32)``."""
    k, n = q.shape
    if qtype == TensorProto.INT8:
        tensors = [numpy_helper.from_array(q.astype(np.int8), "q"), numpy_helper.from_array(zp.astype(np.int8), "zp")]
    else:
        tensors = [helper.make_tensor("q", TensorProto.INT4, [k, n], q.astype(np.int8).reshape(-1).tolist()),
                   helper.make_tensor("zp", TensorProto.INT4, list(zp.shape), zp.astype(np.int8).reshape(-1).tolist())]
    graph = helper.make_graph(
        [helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["w"], axis=0, block_size=BLOCK),
         helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape[:-1] + [n])],
        initializer=tensors + [numpy_helper.from_array(scale.astype(np.float32), "scale")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _dequantize(model: onnx.ModelProto) -> np.ndarray:
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    q, scale, zp = (inits[n].astype(np.float32) for n in ("q", "scale", "zp"))
    return (q - np.repeat(zp, BLOCK, axis=0)) * np.repeat(scale, BLOCK, axis=0)


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x})[0]


def test_avoid_int8_min_zero_point():
    rng = np.random.default_rng(0)
    k, n = 2 * BLOCK, 3
    q = rng.integers(-128, 127, (k, n))                      # values below 127
    q[BLOCK + 5, 1] = 127                                    # block (1, 1) uses all 256 levels
    scale = rng.uniform(0.01, 0.02, (k // BLOCK, n))
    zp = np.zeros((k // BLOCK, n), np.int64)
    zp[0, 0] = zp[1, 1] = -128
    model = _dql_matmul_model(q, scale, zp, [1, k])
    before = _dequantize(model)

    assert avoid_int8_min_zero_point(model) == (1, 1)
    after = _dequantize(model)
    new_zp = numpy_helper.to_array(next(t for t in model.graph.initializer if t.name == "zp"))
    assert not (new_zp == -128).any()
    assert new_zp[0, 0] == new_zp[1, 1] == -127
    # block (0, 0) shifted exactly, block (1, 1) requantized within half of its widened step
    np.testing.assert_array_equal(after[:BLOCK, 0], before[:BLOCK, 0])
    step = scale[1, 1] * 255.0 / 254.0
    assert np.abs(after[BLOCK:, 1] - before[BLOCK:, 1]).max() <= step / 2 + 1e-6
    # untouched blocks keep their values
    np.testing.assert_array_equal(after[:, 2], before[:, 2])
    np.testing.assert_array_equal(after[BLOCK:, 0], before[BLOCK:, 0])


def test_avoid_int8_min_zero_point_no_match():
    q = np.zeros((BLOCK, 2), np.int64)
    model = _dql_matmul_model(q, np.ones((1, 2)), np.zeros((1, 2), np.int64), [1, BLOCK])
    assert avoid_int8_min_zero_point(model) == (0, 0)


@pytest.mark.ort
def test_split_int8_matmul_cols_matches_unsplit():
    rng = np.random.default_rng(1)
    k, n = 2 * BLOCK, 1024
    q = rng.integers(-127, 128, (k, n))
    scale = rng.uniform(0.01, 0.02, (k // BLOCK, n))
    zp = rng.integers(-8, 8, (k // BLOCK, n))
    model = _dql_matmul_model(q, scale, zp, [1, 4, k])
    x = rng.standard_normal((1, 4, k)).astype(np.float32)
    ref = _run(model, x)

    assert split_int8_matmul_cols(model, cols=512) == 1
    ops = [nd.op_type for nd in model.graph.node]
    assert ops.count("DequantizeLinear") == 2 and ops.count("MatMul") == 2 and "Concat" in ops
    np.testing.assert_allclose(_run(model, x), ref, rtol=1e-5, atol=1e-5)


def test_split_int8_matmul_cols_keeps_narrow_weights():
    q = np.zeros((BLOCK, 512), np.int64)
    model = _dql_matmul_model(q, np.ones((1, 512)), np.zeros((1, 512), np.int64), [1, 4, BLOCK])
    assert split_int8_matmul_cols(model, cols=512) == 0


@pytest.mark.ort
def test_split_int4_matmul_rows_matches_unsplit():
    rng = np.random.default_rng(2)
    k, n, m = 2 * BLOCK, 8, 256
    q = rng.integers(-8, 8, (k, n))
    scale = rng.uniform(0.01, 0.02, (k // BLOCK, n))
    zp = rng.integers(-2, 2, (k // BLOCK, n))
    model = _dql_matmul_model(q, scale, zp, [1, m, k], qtype=TensorProto.INT4)
    x = rng.standard_normal((1, m, k)).astype(np.float32)
    ref = _run(model, x)

    assert split_int4_matmul_rows(model, rows=128) == 1
    ops = [nd.op_type for nd in model.graph.node]
    assert ops.count("MatMul") == 2 and ops.count("DequantizeLinear") == 2 and "Split" in ops
    np.testing.assert_allclose(_run(model, x), ref, rtol=1e-5, atol=1e-5)
