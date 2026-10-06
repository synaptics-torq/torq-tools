# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""E-2: ``--batch-prefill`` auto-raises the NSS program-space budget.

16-layer batched prefill emits ~10.2 MB of NSS programs vs the compiler's
8 MB default, so every exporter with ``--batch-prefill`` (gemma3, liquid,
liquid-vl — all funnel through the base ``export_torq``) must add
``--torq-max-nss-programs-size`` without the user having to.
"""

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


def _fake_compile(monkeypatch) -> list:
    calls = []

    def fake_export_torq(input_model, output_dir, **kwargs):
        calls.append(kwargs.get("compiler_args"))

    monkeypatch.setattr(me, "export_torq", fake_export_torq)
    return calls


def _exporter(tmp_path, batch_prefill=None):
    exporter = StubExporter(tmp_path, {"model": _tiny_model()})
    exporter.export_onnx(validate=False, cleanup=False)
    exporter._batch_prefill = batch_prefill
    return exporter


def test_batch_prefill_adds_nss_programs_size_flag(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    _exporter(tmp_path, batch_prefill=64).export_torq()

    assert calls == [["--torq-max-nss-programs-size", "402653184"]]


def test_without_batch_prefill_compile_args_are_untouched(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    _exporter(tmp_path).export_torq(torq_compile_args=["--foo"])

    assert calls == [["--foo"]]


def test_existing_nss_flag_is_not_duplicated(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    _exporter(tmp_path, batch_prefill=64).export_torq(
        torq_compile_args=["--torq-max-nss-programs-size", "999"]
    )

    assert calls == [["--torq-max-nss-programs-size", "999"]]
