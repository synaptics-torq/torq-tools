# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from pathlib import Path

from torq.model_export.onnx import OnnxModelExporterBase


class StubExporter(OnnxModelExporterBase):
    """Minimal concrete exporter serving in-memory components.

    ``setup_calls`` / ``load_calls`` stand in for the real exporters' source
    download and multi-GB ONNX read.
    """

    def __init__(self, root, components, dynamic_quantize=False, convert_dtypes=False, split_weights=False):
        self._root = Path(root)
        self._stub_components = dict(components)
        self.setup_calls = 0
        self.load_calls = 0
        super().__init__(
            "fp32", False, {}, self._root,
            dynamic_quantize=dynamic_quantize, convert_dtypes=convert_dtypes, opt_configs={},
            split_weights=split_weights,
        )

    def _setup_dirs(self):
        self.setup_calls += 1
        export_dir = self._root / "export"
        quantize_dir = self._root / "quantize"
        convert_dir = self._root / "convert"
        variant_dir = convert_dir if self._convert_dtypes else (
            quantize_dir if self._dynamic_quantize else export_dir)
        return (
            self._root / "source",
            export_dir,
            quantize_dir,
            convert_dir,
            variant_dir / "compiled",
        )

    def _load_onnx(self):
        self.load_calls += 1
        return dict(self._stub_components)

    def make_static(self):
        raise AssertionError("static export not exercised by these tests")

    def apply_post_static_patches(self, model_path, component):
        pass

    def validate_onnx(self, n_iters: int = 5):
        pass
