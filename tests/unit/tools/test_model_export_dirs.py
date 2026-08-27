# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export-directory lifecycle for :class:`OnnxModelExporterBase`.

The export dir must only be reset by ``export_onnx``, never by merely
constructing an exporter -- otherwise read-only commands like
``--view-graph-edits`` destroy a previous export's artifacts.
"""

import logging
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.model_export import StubExporter
from torq.model_export.onnx import OnnxModelExporterBase


pytestmark = pytest.mark.unit


def _identity_model() -> onnx.ModelProto:
    inp = gs.Variable("x", dtype=np.float32, shape=[2])
    out = gs.Variable("y", dtype=np.float32, shape=[2])
    graph = gs.Graph(
        nodes=[gs.Node("Identity", "id", inputs=[inp], outputs=[out])],
        inputs=[inp],
        outputs=[out],
        opset=17,
    )
    return gs.export_onnx(graph)


class _StubExporter(OnnxModelExporterBase):
    """Minimal concrete exporter writing one trivial component.

    ``setup_calls`` / ``load_calls`` stand in for the real exporters' source
    download and multi-GB ONNX read.
    """

    def __init__(self, root: Path, dynamic_quantize: bool = False, convert_dtypes: bool = False):
        self._root = Path(root)
        self.setup_calls = 0
        self.load_calls = 0
        super().__init__(
            "fp32", False, {}, self._root,
            dynamic_quantize=dynamic_quantize, convert_dtypes=convert_dtypes, opt_configs={},
        )

    def _setup_dirs(self):
        self.setup_calls += 1
        return (
            self._root / "source",
            self._root / "export",
            self._root / "quantize",
            self._root / "convert",
            self._root / "torq",
        )

    def _load_onnx(self):
        self.load_calls += 1
        return {"model": _identity_model()}

    def make_static(self):
        raise AssertionError("static export not exercised by this test")

    def apply_post_static_patches(self, model_path, component):
        pass

    def validate_onnx(self, n_iters: int = 5):
        pass


def test_construction_leaves_existing_export_artifacts_untouched(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True)
    artifact = export_dir / "model.onnx"
    artifact.write_bytes(b"previous export")

    StubExporter(tmp_path, {"model": _identity_model()})

    assert artifact.read_bytes() == b"previous export"


def test_construction_does_not_create_the_export_dir(tmp_path):
    StubExporter(tmp_path, {"model": _identity_model()})

    assert not (tmp_path / "export").exists()


def test_export_onnx_resets_stale_artifacts(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True)
    stale = export_dir / "stale_lm_head.onnx"
    stale.write_bytes(b"stale")

    exporter = StubExporter(tmp_path, {"model": _identity_model()})
    exporter.export_onnx(validate=False)

    assert not stale.exists()
    assert (export_dir / "model.onnx").exists()


def test_construction_does_not_download_or_load_the_source_model(tmp_path):
    """`--view-graph-edits` builds an exporter purely to render its edit plan."""
    exporter = StubExporter(tmp_path, {"model": _identity_model()})

    assert (exporter.setup_calls, exporter.load_calls) == (0, 0)
    assert exporter.describe_graph_edits() == {}


def test_export_onnx_prepares_once(tmp_path):
    exporter = StubExporter(tmp_path, {"model": _identity_model()})

    exporter.export_onnx(validate=False)
    exporter.export_onnx(validate=False)

    assert (exporter.setup_calls, exporter.load_calls) == (1, 1)


def test_dynamic_quantize_models_skips_when_disabled(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="_StubExporter")
    exporter = _StubExporter(tmp_path)

    exporter.dynamic_quantize_models()

    assert (exporter.setup_calls, exporter.load_calls) == (0, 0)
    assert not (tmp_path / "quantize").exists()
    assert "Skipping dynamic quantization" in caplog.text


def test_dynamic_quantize_models_quantizes_and_updates_export_paths(tmp_path):
    exporter = _StubExporter(tmp_path, dynamic_quantize=True)
    exporter.export_onnx(validate=False)

    exporter.dynamic_quantize_models(skip_preprocess=True)

    quantized_path = tmp_path / "quantize" / "model.onnx"
    assert quantized_path.exists()
    assert exporter._export_paths["model"] == quantized_path


def test_dynamic_quantize_models_resets_stale_artifacts(tmp_path):
    quantize_dir = tmp_path / "quantize"
    quantize_dir.mkdir(parents=True)
    stale = quantize_dir / "stale.onnx"
    stale.write_bytes(b"stale")

    exporter = _StubExporter(tmp_path, dynamic_quantize=True)
    exporter.export_onnx(validate=False)
    exporter.dynamic_quantize_models(skip_preprocess=True)

    assert not stale.exists()


def test_dynamic_quantize_models_skip_list_copies_without_quantizing(tmp_path):
    exporter = _StubExporter(tmp_path, dynamic_quantize=True)
    exporter.export_onnx(validate=False)
    original_bytes = (tmp_path / "export" / "model.onnx").read_bytes()

    exporter.dynamic_quantize_models(skip=["model"], skip_preprocess=True)

    quantized_path = tmp_path / "quantize" / "model.onnx"
    assert quantized_path.read_bytes() == original_bytes
    assert exporter._export_paths["model"] == quantized_path


def test_dynamic_quantize_models_copies_runtime_assets_from_export_dir(tmp_path):
    exporter = _StubExporter(tmp_path, dynamic_quantize=True)
    exporter.export_onnx(validate=False)
    export_dir = tmp_path / "export"
    (export_dir / "config.json").write_text("{}")
    (export_dir / "extra_data.npy").write_bytes(b"npy-bytes")

    exporter.dynamic_quantize_models(skip_preprocess=True)

    quantize_dir = tmp_path / "quantize"
    assert (quantize_dir / "config.json").read_text() == "{}"
    assert (quantize_dir / "extra_data.npy").read_bytes() == b"npy-bytes"


def test_convert_models_skip_list_updates_export_paths(tmp_path):
    """A skipped component is copied into convert_dir; later stages (e.g. export_torq) must pick up that copy, not the stale pre-conversion path."""
    exporter = _StubExporter(tmp_path, convert_dtypes=True)
    exporter.export_onnx(validate=False)
    original_bytes = (tmp_path / "export" / "model.onnx").read_bytes()

    exporter.convert_models(skip=["model"])

    converted_path = tmp_path / "convert" / "model.onnx"
    assert converted_path.read_bytes() == original_bytes
    assert exporter._export_paths["model"] == converted_path
