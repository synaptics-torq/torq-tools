# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Dynamic quantization of >2 GiB models.

ORT's in-memory ``quant_pre_process`` raises ``EncodeError`` for models
whose in-memory proto exceeds 2 GiB — its symbolic shape inference
serializes the model before any file is written. The exporter must fall
back to the large-model path: preprocess with the external-data flags + no
symbolic inference, then quantize with external-data output.
"""

from pathlib import Path

import google.protobuf.message
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

import torq.model_export.onnx as me
from support.model_export import StubExporter

pytestmark = pytest.mark.unit


def _tiny_model() -> onnx.ModelProto:
    inp = gs.Variable("x", dtype=np.float32, shape=[2])
    out = gs.Variable("y", dtype=np.float32, shape=[2])
    graph = gs.Graph(
        nodes=[gs.Node("Identity", "id", inputs=[inp], outputs=[out])],
        inputs=[inp],
        outputs=[out],
        opset=17,
    )
    return gs.export_onnx(graph)


def _stub_summary(*args, **kwargs):
    return {"kl": 0.0, "cosine": 1.0, "max_abs_error": 0.0, "classification": "ok"}


def test_large_model_quantize_uses_external_data_fallback(tmp_path, monkeypatch):
    exporter = StubExporter(tmp_path, {"model": _tiny_model()}, dynamic_quantize=True)
    exporter.export_onnx(validate=False, cleanup=False)

    quantize_calls = []
    preprocess_calls = []
    import onnxruntime.quantization.preprocess as pp

    def fake_quantize_file(model_input, model_output, **kwargs):
        quantize_calls.append((str(model_input), str(model_output), kwargs))
        if kwargs.get("skip_preprocess"):
            onnx.save(_tiny_model(), str(model_output))
            # The quantizer's final single-file re-save orphans this.
            Path(str(model_output) + ".data").write_bytes(b"orphan")
            return Path(model_output)
        raise google.protobuf.message.EncodeError("Failed to serialize proto")

    def fake_preprocess(model_input, model_output, **kwargs):
        preprocess_calls.append((str(model_input), str(model_output), kwargs))
        onnx.save(_tiny_model(), str(model_output))

    monkeypatch.setattr(me, "onnx_dynamic_quantize_file", fake_quantize_file)
    monkeypatch.setattr(me, "summarize_dynamic_quantization", _stub_summary)
    monkeypatch.setattr(pp, "quant_pre_process", fake_preprocess)

    exporter.dynamic_quantize_models()

    # First attempt died at the serialization wall; the fallback ran.
    assert len(quantize_calls) == 2
    src, dst, kwargs = quantize_calls[1]
    assert kwargs["skip_preprocess"] is True
    assert kwargs["use_external_data_format"] is True
    assert len(preprocess_calls) == 1
    pp_src, pp_dst, pp_kwargs = preprocess_calls[0]
    assert pp_src == str(tmp_path / "export" / "model.onnx")
    assert pp_kwargs == {
        "skip_symbolic_shape": True,
        "save_as_external_data": True,
        "all_tensors_to_one_file": True,
        "external_data_location": "model_preprocess.onnx.data",
    }
    # Temp preprocess files and the orphaned external data are cleaned up.
    quantize_dir = tmp_path / "quantize"
    assert not list(quantize_dir.glob("*_preprocess*"))
    assert (quantize_dir / "model.onnx").exists()
    assert not (quantize_dir / "model.onnx.data").exists()
    assert exporter._export_paths["model"] == quantize_dir / "model.onnx"


def test_small_model_quantize_keeps_the_plain_path(tmp_path, monkeypatch):
    exporter = StubExporter(tmp_path, {"model": _tiny_model()}, dynamic_quantize=True)
    exporter.export_onnx(validate=False, cleanup=False)

    calls = []
    import onnxruntime.quantization.preprocess as pp

    def fake_quantize_file(model_input, model_output, **kwargs):
        calls.append((str(model_input), str(model_output), kwargs))
        onnx.save(_tiny_model(), str(model_output))
        return Path(model_output)

    def fake_preprocess(*a, **k):
        raise AssertionError("quant_pre_process must not run for small models")

    monkeypatch.setattr(me, "onnx_dynamic_quantize_file", fake_quantize_file)
    monkeypatch.setattr(me, "summarize_dynamic_quantization", _stub_summary)
    monkeypatch.setattr(pp, "quant_pre_process", fake_preprocess)

    exporter.dynamic_quantize_models()

    assert len(calls) == 1
    assert "skip_preprocess" not in calls[0][2]
    assert "use_external_data_format" not in calls[0][2]
