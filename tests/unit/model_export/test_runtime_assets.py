# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import ml_dtypes
import numpy as np
import pytest
from onnx import TensorProto, helper

import torq.model_export.onnx as me
from support.model_export import StubExporter


ASSETS = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "processor_config.json", "added_tokens.json", "vocab.json",
    "chat_template.jinja", "tokenizer.model", "tokenizer.tiktoken", "vocab.txt", "merges.txt",
    "tokens.txt", "sentencepiece.spm", "sentencepiece.vocab",
)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("convert_io", [False, True])
def test_runtime_assets_survive_all_export_stages(tmp_path, monkeypatch, static, convert_io):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    source = tmp_path / "source"
    source.mkdir()
    for name in ASSETS:
        (source / name).write_text(name)
    (source / "compile_flags.json").write_text("source-only flags")
    (source / "weights.safetensors").write_bytes(b"not a runtime sidecar")
    (source / "nested.json").mkdir()
    np.save(source / "token_embeddings.npy", np.ones((2, 4), dtype=np.float32))
    np.save(source / "token_id_lut.npy", np.arange(2, dtype=np.int64))
    exporter = StubExporter(tmp_path, {"model": model}, dynamic_quantize=True, convert_dtypes=True)
    exporter._static_models = static
    monkeypatch.setattr(exporter, "make_static", lambda: None)

    def write_flags(directory, **kwargs):
        (directory / "compile_flags.json").write_text(directory.name)

    monkeypatch.setattr(exporter, "_write_compile_flags_snapshot", write_flags)
    monkeypatch.setattr(me, "export_torq", lambda *args, **kwargs: None)
    exporter.export_onnx(validate=False)
    exporter.dynamic_quantize_models(skip=["model"])
    exporter.convert_models(
        skip=["model"],
        external_data=[(tmp_path / "quantize" / "token_embeddings.npy", np.dtype(ml_dtypes.bfloat16))],
    )
    flags = ["--torq-convert-io-dtype"] if convert_io else []
    exporter.export_torq(torq_compile_args=flags)

    for directory in (tmp_path / "export", tmp_path / "quantize", tmp_path / "convert", tmp_path / "convert/compiled"):
        for name in ASSETS:
            assert (directory / name).read_text() == name
        assert not (directory / "weights.safetensors").exists()
        assert not (directory / "nested.json").exists()
        assert np.load(directory / "token_id_lut.npy").dtype == (np.int32 if convert_io and directory.name == "compiled" else np.int64)
        if directory.name == "compiled":
            assert not (directory / "compile_flags.json").exists()
        else:
            assert (directory / "compile_flags.json").read_text() == directory.name
    # The source float table must not overwrite the converted table at compilation.
    expected = tmp_path / "expected.npy"
    np.save(expected, np.ones((2, 4), dtype=ml_dtypes.bfloat16))
    assert (tmp_path / "convert/compiled/token_embeddings.npy").read_bytes() == expected.read_bytes()


def test_copy_runtime_assets_can_skip_tables_and_copy_to_itself(tmp_path):
    exporter = StubExporter(tmp_path, {})
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    np.save(source / "table.npy", np.zeros(1))
    destination = tmp_path / "destination"
    exporter._copy_runtime_assets(destination, source, include_npy_data=False)
    assert (destination / "config.json").read_text() == "{}"
    assert not (destination / "table.npy").exists()
    exporter._copy_runtime_assets(source, source)
    assert (source / "config.json").read_text() == "{}"
