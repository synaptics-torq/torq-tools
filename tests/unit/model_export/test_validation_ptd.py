# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from torq.model_export import validation_ptd
from torq.model_export.onnx import OnnxModelExporterBase
from torq.models.gemma3 import export as gemma3_export


@pytest.mark.parametrize("capacity, rows, trimmed", [(2, 2, False), (4, 2, True), (8, 8, True), (16, 4, False)])
@pytest.mark.parametrize("combined_scratch", [True, False])
def test_validation_respects_capacity_and_token_id_lut(monkeypatch, capacity, rows, trimmed, combined_scratch):
    lut = np.array([0, 5, 10]) if trimmed else None
    calls = []

    class FakeRunner:
        def __init__(self, path):
            self.rows = rows if path == "ptd" else 1
            self.input_shapes = {
                "input_ids": [1, self.rows],
                "past_key_values.0.key_value": [1, 2, capacity, 1],
            }
            self.output_names = ["logits", "present.0.key_value"]
            if self.rows > 1:
                self.output_names += (["present.0.key_value_scratch"] if combined_scratch else
                                      ["present.0.key_scratch", "present.0.value_scratch"])

        def infer(self, feeds):
            tokens = feeds["input_ids"].reshape(-1)
            positions = feeds["position_ids"].reshape(-1)
            assert positions.max() < capacity
            if lut is not None:
                assert set(tokens).issubset(set(lut))
            calls.append((self.rows, tokens.copy(), positions.copy()))
            logits = np.zeros((1, self.rows, 3), dtype=np.float32)
            logits[:, :, 2] = 1
            cache = feeds["past_key_values.0.key_value"].copy()
            out = [logits, cache]
            if self.rows > 1:
                out += ([np.zeros((1, 2, self.rows - 1, 1), dtype=np.float32)] if combined_scratch else
                        [np.zeros((1, 1, self.rows - 1, 1), dtype=np.float32)] * 2)
            return out

    monkeypatch.setattr(validation_ptd, "_load_session", lambda path, _: FakeRunner(path))
    validation_ptd.validate_ptd_onnx(logging.getLogger(__name__), "ptd", "decode",
                                     rows=rows, vocab_size=3, token_id_lut=lut)
    ptd_call = next(call for call in calls if call[0] == rows)
    # Compact argmax 2 must be fed back as model token 10.
    anchor = 0 if capacity == rows else 2
    assert ptd_call[1][0] == (lut[anchor] if trimmed else anchor)
    assert len(calls) == min(8, capacity - rows) + 1 + rows


def test_exporter_accepts_trimmed_ptd(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    config = SimpleNamespace(hidden_size=8, vocab_size=16, num_attention_heads=2, sliding_window=512)
    monkeypatch.setattr(gemma3_export.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(OnnxModelExporterBase, "__init__", lambda *args, **kwargs: None)
    exporter = gemma3_export.Gemma3ModelExporter(onnx_source_dir=source, trim_vocab=True, ptd_rows=8)
    assert exporter._trim_vocab
    assert exporter._ptd_rows == 8


def test_exporter_passes_lut_and_model_vocabulary_to_validation(monkeypatch, tmp_path):
    lut = np.array([0, 5, 10], dtype=np.int32)
    np.save(tmp_path / "token_id_lut.npy", lut)
    exporter = object.__new__(gemma3_export.Gemma3ModelExporter)
    exporter._export_paths = {"model": tmp_path / "model.onnx", "model_ptd": tmp_path / "ptd.onnx"}
    exporter._trim_vocab = True
    exporter._vocab_size = 16
    exporter._ptd_rows = 8
    exporter._logger = logging.getLogger(__name__)
    calls = []
    monkeypatch.setattr(validation_ptd, "validate_ptd_onnx", lambda *args, **kwargs: calls.append(kwargs))
    exporter._validate_ptd_onnx()
    np.testing.assert_array_equal(calls[0]["token_id_lut"], lut)
    assert calls[0]["vocab_size"] == 16


@pytest.mark.parametrize("component", ["model", "model_prefill", "model_ptd"])
@pytest.mark.parametrize("individual_kv", [False, True])
def test_static_patch_combines_scratch_only_for_ptd(monkeypatch, tmp_path, component, individual_kv):
    exporter = object.__new__(gemma3_export.Gemma3ModelExporter)
    exporter._onnx_export_dtype = 1
    exporter._config = SimpleNamespace(num_key_value_heads=2, head_dim=4)
    exporter._max_gen_tokens = 16
    exporter._ptd_rows = 8 if component == "model_ptd" else None
    exporter._keep_individual_kv_io = individual_kv
    exporter._extract_embeddings = False
    exporter._trim_vocab = False
    exporter._split_lm_head = False
    exporter._harness = None
    exporter._editor_dump_kwargs = Mock(return_value={})
    exporter.graph_edit_blocks = Mock(return_value={"model.patch": [], "model.patch (batch prefill)": []})
    exporter._save_component_model = Mock()
    editor = Mock()
    monkeypatch.setattr(gemma3_export, "load_onnx_lazy", lambda _: SimpleNamespace(ir_version=10))
    monkeypatch.setattr(gemma3_export.Gemma3OnnxGraphEditor, "from_onnx", lambda *args, **kwargs: editor)

    model_path = tmp_path / f"{component}.onnx"
    exporter._patch_static_model(model_path, component)

    expected = [] if individual_kv else [call([1, 2, 16, 4])]
    if component == "model_ptd" and not individual_kv:
        expected.append(call([1, 2, 7, 4], kv_layer_re=r"\.(\d+)\.(key|value)_scratch$", combined_suffix="_scratch"))
    assert editor.combine_kv_io_tensors.call_args_list == expected
    exporter._save_component_model.assert_called_once_with(editor.to_onnx.return_value, model_path)
