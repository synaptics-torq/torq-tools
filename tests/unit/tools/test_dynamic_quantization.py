# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import json
import shutil

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.tools.quantization.dynamic_quantization import (
    add_dynamic_analyze_args,
    add_dynamic_quantize_args,
    analyze_dynamic_quantization,
    dynamic_analyze_from_args,
    dynamic_quantize_from_args,
    dynamic_quantize_model,
)


def _make_matmul_model(path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(
        np.arange(12, dtype=np.float32).reshape(4, 3), name="w"
    )
    matmul = helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul")
    graph = helper.make_graph([matmul], "matmul_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def _initializer_by_name(model, name: str):
    return next(init for init in model.graph.initializer if init.name == name)


@pytest.fixture
def input_path(tmp_path):
    path = tmp_path / "model.onnx"
    _make_matmul_model(path)
    return path


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "quantized.onnx"


def test_dynamic_quantize_model_default_produces_int8_per_channel_chain(input_path, output_path):
    dynamic_quantize_model(input_path, output_path, skip_preprocess=True)

    quantized = onnx.load(output_path)

    assert [node.op_type for node in quantized.graph.node] == [
        "DynamicQuantizeLinear", "Mul", "MatMulInteger", "Cast", "Mul",
    ]
    assert _initializer_by_name(quantized, "w_quantized").data_type == TensorProto.INT8
    assert list(_initializer_by_name(quantized, "w_scale").dims) == [3]


def test_dynamic_quantize_model_uint8_weights(input_path, output_path):
    dynamic_quantize_model(input_path, output_path, skip_preprocess=True, uint8_weights=True)

    w_quantized = _initializer_by_name(onnx.load(output_path), "w_quantized")
    assert w_quantized.data_type == TensorProto.UINT8


def test_dynamic_quantize_model_per_tensor_uses_scalar_scale_and_zero_point(input_path, output_path):
    dynamic_quantize_model(input_path, output_path, skip_preprocess=True, per_tensor=True)

    quantized = onnx.load(output_path)
    assert list(_initializer_by_name(quantized, "w_scale").dims) == []
    assert list(_initializer_by_name(quantized, "w_zero_point").dims) == []


def test_dynamic_quantize_model_quantize_only_ops_restricts_quantization(input_path, output_path):
    dynamic_quantize_model(input_path, output_path, skip_preprocess=True, quantize_only_ops=["Add"])

    quantized = onnx.load(output_path)
    assert [node.op_type for node in quantized.graph.node] == ["MatMul"]


def test_dynamic_quantize_model_quantize_only_nodes_restricts_quantization(input_path, output_path):
    dynamic_quantize_model(input_path, output_path, skip_preprocess=True, quantize_only_nodes=["not_matmul"])

    quantized = onnx.load(output_path)
    assert [node.op_type for node in quantized.graph.node] == ["MatMul"]


def test_dynamic_quantize_model_skip_preprocess_true_skips_quant_pre_process(monkeypatch, input_path, output_path):
    calls = []
    monkeypatch.setattr(
        "torq.tools.quantization.dynamic_quantization.quantize.quant_pre_process",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    dynamic_quantize_model(input_path, output_path, skip_preprocess=True)

    assert calls == []
    assert output_path.exists()


def test_dynamic_quantize_model_runs_quant_pre_process_by_default(monkeypatch, input_path, output_path):
    calls = []

    def fake_pre_process(model_input, model_output):
        calls.append((model_input, model_output))
        shutil.copy(model_input, model_output)

    monkeypatch.setattr(
        "torq.tools.quantization.dynamic_quantization.quantize.quant_pre_process",
        fake_pre_process,
    )

    dynamic_quantize_model(input_path, output_path)

    assert calls == [(input_path, output_path)]


def test_dynamic_quantize_model_forwards_extra_kwargs_to_quantize_dynamic(input_path, output_path):
    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        dynamic_quantize_model(
            input_path, output_path, skip_preprocess=True, not_a_real_kwarg=True
        )


def test_add_dynamic_quantize_args_registers_expected_defaults():
    parser = argparse.ArgumentParser()
    add_dynamic_quantize_args(parser)

    args = parser.parse_args(["-i", "in.onnx", "-o", "out.onnx"])

    assert args.input == "in.onnx"
    assert args.output == "out.onnx"
    assert args.quantize_only_ops is None
    assert args.quantize_only_nodes is None
    assert args.skip_preprocess is False
    assert args.uint8_weights is False
    assert args.per_tensor is False
    assert args.extra_quant_args is None
    assert args.logging == "INFO"


def test_dynamic_quantize_from_args_invokes_dynamic_quantize_model(input_path, output_path):
    parser = argparse.ArgumentParser()
    add_dynamic_quantize_args(parser)
    args = parser.parse_args(
        ["-i", str(input_path), "-o", str(output_path), "--skip-preprocess"]
    )

    dynamic_quantize_from_args(args)

    quantized = onnx.load(output_path)
    assert "DynamicQuantizeLinear" in [node.op_type for node in quantized.graph.node]


def test_dynamic_quantize_from_args_forwards_extra_quant_args(input_path, output_path):
    parser = argparse.ArgumentParser()
    add_dynamic_quantize_args(parser)
    args = parser.parse_args(
        [
            "-i", str(input_path), "-o", str(output_path),
            "--skip-preprocess",
            "--extra-quant-args", "reduce_range", "True",
        ]
    )

    dynamic_quantize_from_args(args)

    assert output_path.exists()


def _make_two_matmul_model(path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w1 = numpy_helper.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4), name="w1"
    )
    w2 = numpy_helper.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4)[::-1].copy(), name="w2"
    )
    mm1 = helper.make_node("MatMul", ["x", "w1"], ["h"], name="MM1")
    mm2 = helper.make_node("MatMul", ["h", "w2"], ["y"], name="MM2")
    graph = helper.make_graph([mm1, mm2], "two_matmul", [x], [y], [w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture
def two_matmul_path(tmp_path):
    path = tmp_path / "two_matmul.onnx"
    _make_two_matmul_model(path)
    return path


def test_analyze_dynamic_quantization_ranks_all_candidate_nodes(two_matmul_path):
    results = analyze_dynamic_quantization(two_matmul_path, skip_preprocess=True)

    assert {r["node"] for r in results} == {"MM1", "MM2"}
    for r in results:
        assert set(r) == {"node", "op_type", "kl", "cosine", "max_abs_error", "classification"}
        assert r["op_type"] == "MatMul"
        assert r["classification"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    # sorted most-sensitive (highest KL) first
    assert results[0]["kl"] >= results[1]["kl"]


def test_analyze_dynamic_quantization_skip_nodes_and_op_types(two_matmul_path):
    only_mm2 = analyze_dynamic_quantization(
        two_matmul_path, skip_preprocess=True, skip_nodes=["MM1"]
    )
    assert [r["node"] for r in only_mm2] == ["MM2"]

    # no Gemm nodes exist, so nothing is analyzed
    assert analyze_dynamic_quantization(
        two_matmul_path, skip_preprocess=True, op_types=["Gemm"]
    ) == []


def test_dynamic_analyze_from_args_writes_report_and_exclude(two_matmul_path, tmp_path):
    report = tmp_path / "report.json"
    exclude = tmp_path / "exclude.json"
    parser = argparse.ArgumentParser()
    add_dynamic_analyze_args(parser)
    args = parser.parse_args(
        [
            "-i", str(two_matmul_path), "-o", str(report),
            "--skip-preprocess",
            "--exclude-output", str(exclude), "--exclude-class", "MEDIUM",
        ]
    )

    dynamic_analyze_from_args(args)

    data = json.loads(report.read_text())
    assert isinstance(data, list) and len(data) == 2
    excluded = json.loads(exclude.read_text())
    assert isinstance(excluded, list) and set(excluded) <= {"MM1", "MM2"}


def test_dynamic_quantize_model_exclude_nodes_keeps_node_float(two_matmul_path, output_path):
    dynamic_quantize_model(
        two_matmul_path, output_path, skip_preprocess=True, exclude_nodes=["MM1"]
    )

    op_types = [node.op_type for node in onnx.load(output_path).graph.node]
    assert "MatMul" in op_types          # MM1 stayed float
    assert "MatMulInteger" in op_types   # MM2 was quantized
