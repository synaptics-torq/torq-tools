# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import json
import tempfile
import unittest
from pathlib import Path

from torq.models.liquid.export import LiquidModelExporter
from torq.models.liquid.export_vl import DECODER, DECODER_PREFILL, LiquidVLModelExporter


_CONFIG = {
    "hidden_size": 1024,
    "vocab_size": 65536,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "conv_dim": 1024,
    "conv_L_cache": 3,
    "num_hidden_layers": 16,
    "layer_types": ["conv", "full_attention"],
    "bos_token_id": 1,
    "eos_token_id": 7,
}


def _write_config(directory: Path) -> Path:
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text(json.dumps(_CONFIG))
    return source_dir


class LiquidExportFilenameTests(unittest.TestCase):
    def test_text_export_uses_model_filename(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/onnx/fp32/static/model.onnx"),
        )

    def test_batch_prefill_uses_distinct_prefill_filename(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component("model_prefill"),
            Path("export/onnx/fp32/static/model_prefill.onnx"),
        )

    def test_vl_batch_prefill_uses_distinct_decoder_filename(self):
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component(DECODER),
            Path("export/onnx/fp32/static/decoder_model_merged.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component(DECODER_PREFILL),
            Path("export/onnx/fp32/static/decoder_model_merged_prefill.onnx"),
        )


class LiquidBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidModelExporter:
        source_dir = _write_config(tmp_path)
        return LiquidModelExporter(onnx_source_dir=source_dir, **kwargs)

    def test_batch_prefill_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._create_exporter(self.tmp, batch_prefill=0)

    def test_batch_prefill_cannot_exceed_max_gen_tokens(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._create_exporter(self.tmp, batch_prefill=9, max_gen_tokens=8)

    def test_batch_prefill_requires_static_models(self):
        with self.assertRaisesRegex(ValueError, "static LFM exports"):
            self._create_exporter(self.tmp, batch_prefill=8, static_models=False)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)

    def test_batch_prefill_block_present_only_when_enabled(self):
        enabled = self._create_exporter(self.tmp, batch_prefill=8)
        disabled = self._create_exporter(self.tmp)

        self.assertIn("model.patch (batch prefill)", enabled.graph_edit_blocks())
        self.assertNotIn("model.patch (batch prefill)", disabled.graph_edit_blocks())


class LiquidVLBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidVLModelExporter:
        source_dir = _write_config(tmp_path)
        return LiquidVLModelExporter(onnx_source_dir=source_dir, **kwargs)

    def test_batch_prefill_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._create_exporter(self.tmp, batch_prefill=0)

    def test_batch_prefill_cannot_exceed_max_gen_tokens(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._create_exporter(self.tmp, batch_prefill=9, max_gen_tokens=8)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)


if __name__ == "__main__":
    unittest.main()
