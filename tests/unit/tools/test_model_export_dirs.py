# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export-directory lifecycle for :class:`OnnxModelExporterBase`.

The export dir must only be reset by ``export_onnx``, never by merely
constructing an exporter -- otherwise read-only commands like
``--view-graph-edits`` destroy a previous export's artifacts.
"""

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.model_export import StubExporter


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
