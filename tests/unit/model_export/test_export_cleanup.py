# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The base exporter's default-on ONNX cleanup hook (export_onnx cleanup=...)."""

import numpy as np
import onnx
import pytest

from support.graph_edit import conv_bn_graph, to_model
from support.model_export import StubExporter


pytestmark = pytest.mark.ort


def _conv_bn_model():
    g, _, _, _ = conv_bn_graph(
        np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 27.0,
        np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1),
        np.array([10.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1),
    )
    return to_model(g)


def _exported_ops(tmp_path, **export_kwargs):
    exporter = StubExporter(tmp_path, {"model": _conv_bn_model()})
    exporter.export_onnx(validate=False, **export_kwargs)
    exported = onnx.load(exporter.export_dir / "model.onnx")
    return [node.op_type for node in exported.graph.node]


@pytest.mark.parametrize("cleanup", [None, False])
def test_export_onnx_cleanup_flag(tmp_path, cleanup):
    kwargs = {} if cleanup is None else {"cleanup": cleanup}
    ops = _exported_ops(tmp_path, **kwargs)
    if cleanup is False:
        assert ops == ["Conv", "Mul", "Add"]
    else:  # cleanup is the default
        assert ops == ["Conv"]  # BN chain folded into the conv


def test_export_onnx_survives_cleanup_failure(tmp_path, monkeypatch):
    """Cleanup is an optimization: a failure must not break the export."""
    import torq.model_export.onnx as me

    def boom(model, **kwargs):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(me, "cleanup_onnx_model", boom)
    ops = _exported_ops(tmp_path)
    assert ops == ["Conv", "Mul", "Add"]  # uncleaned model exported
