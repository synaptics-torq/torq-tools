# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
from pathlib import Path

import flatbuffers
import numpy as np
import pytest

from torq.tools.convert_static import schema_py_generated as schema_fb
from torq.tools.convert_static.tflite import (
    add_tflite_static_convert_args,
    convert_model,
    tflite_static_convert_from_args,
)


def _make_tensor(name: str, shape_signature) -> schema_fb.TensorT:
    tensor = schema_fb.TensorT()
    tensor.shape = [1]
    tensor.type = schema_fb.TensorType.FLOAT32
    tensor.buffer = 0
    tensor.name = name
    tensor.shapeSignature = shape_signature
    return tensor


def _make_subgraph(tensors) -> schema_fb.SubGraphT:
    subgraph = schema_fb.SubGraphT()
    subgraph.tensors = list(tensors)
    subgraph.inputs = []
    subgraph.outputs = []
    subgraph.operators = []
    return subgraph


def _write_model(path: Path, subgraphs) -> None:
    model_t = schema_fb.ModelT()
    model_t.version = 3
    model_t.operatorCodes = []
    model_t.buffers = [schema_fb.BufferT()]
    model_t.subgraphs = list(subgraphs)

    builder = flatbuffers.Builder(1024)
    builder.Finish(model_t.Pack(builder), b"TFL3")
    path.write_bytes(builder.Output())


def _load_model(path: Path) -> schema_fb.ModelT:
    buf = path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    return schema_fb.ModelT.InitFromObj(model)


def _tensors_by_name(model_t: schema_fb.ModelT) -> dict:
    return {
        tensor.name.decode()
        if isinstance(tensor.name, bytes)
        else tensor.name: tensor
        for subgraph in model_t.subgraphs
        for tensor in subgraph.tensors
    }


@pytest.fixture
def input_path(tmp_path) -> Path:
    return tmp_path / "input.tflite"


@pytest.fixture
def output_path(tmp_path) -> Path:
    return tmp_path / "output.tflite"


def test_clears_dynamic_shape_signature(input_path, output_path):
    tensor = _make_tensor("dynamic", [-1, 3])
    _write_model(input_path, [_make_subgraph([tensor])])

    convert_model(input_path, output_path)

    converted = _tensors_by_name(_load_model(output_path))["dynamic"]
    assert converted.shapeSignature is None


def test_leaves_static_shape_signature_untouched(input_path, output_path):
    tensor = _make_tensor("static", [1, 3])
    _write_model(input_path, [_make_subgraph([tensor])])

    convert_model(input_path, output_path)

    converted = _tensors_by_name(_load_model(output_path))["static"]
    np.testing.assert_array_equal(converted.shapeSignature, [1, 3])


def test_leaves_missing_shape_signature_as_none(input_path, output_path):
    tensor = _make_tensor("no_signature", None)
    _write_model(input_path, [_make_subgraph([tensor])])

    convert_model(input_path, output_path)

    converted = _tensors_by_name(_load_model(output_path))["no_signature"]
    assert converted.shapeSignature is None


def test_handles_empty_shape_signature_without_crashing(input_path, output_path):
    tensor = _make_tensor("empty_signature", [])
    _write_model(input_path, [_make_subgraph([tensor])])

    convert_model(input_path, output_path)

    converted = _tensors_by_name(_load_model(output_path))["empty_signature"]
    assert converted.shapeSignature is None or len(converted.shapeSignature) == 0


def test_only_converts_dynamic_tensors_and_reports_accurate_count(
    input_path, output_path, capsys
):
    tensors = [
        _make_tensor("dynamic_a", [-1, 3]),
        _make_tensor("dynamic_b", [1, -1, 4]),
        _make_tensor("static", [1, 3]),
        _make_tensor("missing", None),
        _make_tensor("empty", []),
    ]
    _write_model(input_path, [_make_subgraph(tensors)])

    convert_model(input_path, output_path)

    by_name = _tensors_by_name(_load_model(output_path))
    assert by_name["dynamic_a"].shapeSignature is None
    assert by_name["dynamic_b"].shapeSignature is None
    np.testing.assert_array_equal(by_name["static"].shapeSignature, [1, 3])
    assert by_name["missing"].shapeSignature is None
    assert by_name["empty"].shapeSignature is None or len(by_name["empty"].shapeSignature) == 0

    assert "Removed dynamic shape signatures from 2 tensor(s)" in capsys.readouterr().out


def test_converts_dynamic_tensors_across_multiple_subgraphs(input_path, output_path):
    first_subgraph = _make_subgraph([_make_tensor("first_dynamic", [-1])])
    second_subgraph = _make_subgraph([_make_tensor("second_dynamic", [-1, 2])])
    _write_model(input_path, [first_subgraph, second_subgraph])

    convert_model(input_path, output_path)

    by_name = _tensors_by_name(_load_model(output_path))
    assert by_name["first_dynamic"].shapeSignature is None
    assert by_name["second_dynamic"].shapeSignature is None


def test_output_is_a_valid_tflite_flatbuffer(input_path, output_path):
    _write_model(input_path, [_make_subgraph([_make_tensor("t", [-1, 3])])])

    convert_model(input_path, output_path)

    assert output_path.read_bytes()[4:8] == b"TFL3"


def test_raises_for_missing_input_file(tmp_path, output_path):
    missing = tmp_path / "does_not_exist.tflite"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        convert_model(missing, output_path)


def test_raises_for_directory_input(tmp_path, output_path):
    directory = tmp_path / "a_directory.tflite"
    directory.mkdir()

    with pytest.raises(ValueError, match="Path is not a file"):
        convert_model(directory, output_path)


def test_raises_for_wrong_extension(tmp_path, output_path):
    wrong_extension = tmp_path / "input.onnx"
    wrong_extension.write_bytes(b"not a real model")

    with pytest.raises(ValueError, match="Expected a .tflite file"):
        convert_model(wrong_extension, output_path)


def test_add_tflite_static_convert_args_registers_input_and_output():
    parser = argparse.ArgumentParser()
    add_tflite_static_convert_args(parser)

    args = parser.parse_args(["-i", "in.tflite", "-o", "out.tflite"])

    assert args.input == "in.tflite"
    assert args.output == "out.tflite"


def test_tflite_static_convert_from_args_invokes_convert_model(input_path, output_path):
    _write_model(input_path, [_make_subgraph([_make_tensor("t", [-1])])])
    args = argparse.Namespace(input=str(input_path), output=str(output_path))

    tflite_static_convert_from_args(args)

    assert output_path.exists()
