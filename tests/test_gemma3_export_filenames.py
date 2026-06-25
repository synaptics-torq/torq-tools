# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import unittest
from pathlib import Path

from torq.models.gemma3.export import (
    Gemma3ModelExporter,
    _GEMMA3_MODEL_FILENAMES,
)


class Gemma3ExportFilenameTests(unittest.TestCase):
    def _exporter(self, split_lm_head: bool) -> Gemma3ModelExporter:
        exporter = Gemma3ModelExporter.__new__(Gemma3ModelExporter)
        exporter._export_dir = Path("export")
        exporter._export_model_filenames = _GEMMA3_MODEL_FILENAMES[0 if split_lm_head else 1]
        return exporter

    def test_unified_export_uses_model_filename(self):
        exporter = self._exporter(split_lm_head=False)

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/model.onnx"),
        )

    def test_split_export_uses_transformer_and_lm_head_filenames(self):
        exporter = self._exporter(split_lm_head=True)

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/transformer.onnx"),
        )
        self.assertEqual(
            exporter._export_model_filenames,
            ("transformer.onnx", "lm_head.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
