# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from torq.model_export import validation_ptd
from torq.model_export.onnx import OnnxModelExporterBase
from torq.models.gemma3 import export as gemma3_export


def test_exporter_selects_ptd_or_separate_prefill(monkeypatch, tmp_path, caplog):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    config = SimpleNamespace(hidden_size=8, vocab_size=16, num_attention_heads=2, sliding_window=512)
    monkeypatch.setattr(gemma3_export.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(OnnxModelExporterBase, "__init__", lambda *args, **kwargs: None)
    caplog.set_level(logging.INFO)
    # PTD wins over the CLI's default prefill size, even with a smaller cache.
    ptd = gemma3_export.Gemma3ModelExporter(
        onnx_source_dir=tmp_path, ptd_rows=8, batch_prefill=64, max_gen_tokens=16, trim_vocab=True,
    )
    assert ptd._ptd_rows == 8 and ptd._batch_prefill is None
    assert ptd._split_lm_head and ptd._trim_vocab
    assert "disabling separate batch prefill" in caplog.text
    ptd._export_dir = tmp_path
    assert ptd._export_path_for_component("model").name == "transformer.onnx"
    assert ptd._export_path_for_component("lm_head").name == "lm_head.onnx"
    ptd._logger = logging.getLogger(__name__)
    source = object()
    ptd._components = {"model": source}
    ptd.check_model = lambda model: model
    ptd._make_model_static = Mock()
    ptd.make_static()
    ptd._make_model_static.assert_called_once_with(source, chunk_len=8)
    assert "model_prefill" not in ptd._components
    ordinary = gemma3_export.Gemma3ModelExporter(
        onnx_source_dir=tmp_path, ptd_rows=0, batch_prefill=8, split_lm_head=True,
    )
    assert ordinary._ptd_rows is None and ordinary._batch_prefill == 8


def test_validation_respects_capacity_and_token_id_lut(monkeypatch):
    capacity, rows = 8, 4
    lut = np.array([0, 5, 10])
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
                self.input_shapes["tree_vis"] = [1, 1, self.rows, self.rows - 1]
                self.output_names.append("present.0.key_value_scratch")

        def infer(self, feeds):
            tokens = feeds["input_ids"].reshape(-1)
            positions = feeds["position_ids"].reshape(-1)
            assert positions.max() < capacity
            assert set(tokens).issubset(set(lut))
            calls.append((self.rows, tokens.copy(), positions.copy(), feeds.get("tree_vis")))
            logits = np.zeros((1, self.rows, 3), dtype=np.float32)
            logits[:, :, 2] = 1
            cache = feeds["past_key_values.0.key_value"].copy()
            out = [logits, cache]
            if self.rows > 1:
                out.append(np.zeros((1, 2, self.rows - 1, 1), dtype=np.float32))
            return out

    monkeypatch.setattr(validation_ptd, "_load_session", lambda path, _: FakeRunner(path))
    validation_ptd.validate_ptd_onnx(logging.getLogger(__name__), "ptd",
                                     rows=rows, vocab_size=3, token_id_lut=lut)
    ptd_call = next(call for call in calls if call[0] == rows and np.any(call[3]))
    # Compact argmax 2 must be fed back as model token 10.
    assert ptd_call[1][0] == lut[2]
    assert len(calls) == min(8, capacity - rows) + 1 + rows
