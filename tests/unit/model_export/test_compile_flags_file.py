# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Model-local ``compile_flags.json``: the validated torq-compile flag set for
a model lives in a visible data file next to its exporter. The exporter
applies it to every compile (so no ``--compile-flags`` are needed for an
optimal export) and records it in each export variant dir, so a standalone
compile of a pre-exported model — possibly in an environment without
torq-tools — picks the flags up from the dir instead of the user having to
know them."""

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import torq.model_export.onnx as me
from torq.utils.compile import COMPILE_FLAGS_FILE
from support.model_export import StubExporter

pytestmark = pytest.mark.unit


def _flags_file(tmp_path: Path, flags: list[str], w8a8: list[str] | None = None) -> Path:
    data = {"flags": flags}
    if w8a8 is not None:
        data["w8a8"] = w8a8
    path = tmp_path / "model" / COMPILE_FLAGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _tiny_matmul_model() -> onnx.ModelProto:
    """Small model with a MatMul so the quantize/convert stages have work."""
    weight = np.ones((4, 8), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "W"], ["y"]),
        ],
        "main",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def _exporter(tmp_path: Path, **kwargs) -> StubExporter:
    exporter = StubExporter(tmp_path, {"model": _tiny_matmul_model()}, **kwargs)
    exporter.export_onnx(validate=False, cleanup=False)
    return exporter


def test_source_tree_model_flag_files_are_complete():
    """The chip-compiled exporters ship a validated flag file (guards against
    drift)."""
    models_dir = Path(me.__file__).parent.parent / "models"
    for model in ("gemma3", "smollm2", "liquid", "moonshine", "moonshine_streaming"):
        path = models_dir / model / COMPILE_FLAGS_FILE
        assert path.is_file(), f"missing validated flags for {model}"
        data = json.loads(path.read_text())
        assert data.get("flags"), f"{model}: empty flags"
        assert data.get("w8a8") == ["--torq-disable-host", "--torq-disable-css"], (
            f"{model}: w8a8 workaround (torq-compile 2.2.1 host/CSS bug) missing"
        )
    # moonshine is validated against the LLM flag set, not its own.
    llm = json.loads((models_dir / "gemma3" / COMPILE_FLAGS_FILE).read_text())
    for model in ("moonshine", "moonshine_streaming"):
        data = json.loads((models_dir / model / COMPILE_FLAGS_FILE).read_text())
        assert data.get("flags") == llm["flags"], f"{model}: flags drifted from the LLM set"
        assert data.get("w8a8") == llm["w8a8"], f"{model}: w8a8 drifted from the LLM set"


@pytest.mark.ci
def test_installed_compiler_matches_shipped_flag_versions():
    """CI-only guard: the installed torq-compiler must be listed in the
    ``version`` of every shipped ``compile_flags.json``. Bumping the compiler
    without re-validating a model's flag set fails CI here instead of
    silently shipping stale flags. Skipped outside CI (``CI`` unset): a
    deliberate local compiler/flag mismatch must not break regular use."""
    if not os.environ.get("CI"):
        pytest.skip("runs in CI only; local compiler/flag mismatches may be deliberate")
    try:
        installed = version("torq-compiler")
    except PackageNotFoundError as e:
        pytest.fail(f"torq-compiler not installed: {e!r}")
    models_dir = Path(me.__file__).parent.parent / "models"
    for path in sorted(models_dir.rglob(COMPILE_FLAGS_FILE)):
        rel = path.relative_to(models_dir)
        data = json.loads(path.read_text())
        assert installed in data.get("version") or [], (
            f"{rel}: installed torq-compiler {installed} not in "
            f"version {data.get('version')!r}; re-validate the flags for "
            f"{installed} and update the file alongside the compiler requirement"
        )


def test_export_onnx_writes_variant_snapshot(tmp_path):
    exporter = _exporter(tmp_path)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter._batch_prefill = 64
    exporter.export_onnx(validate=False, cleanup=False)

    snapshot = json.loads((exporter.export_dir / COMPILE_FLAGS_FILE).read_text())
    assert snapshot["flags"] == [
        "--a",
        "--torq-max-nss-programs-size", me.TORQ_MAX_NSS_PROGRAMS_SIZE,
    ]


def test_quantized_variant_snapshot_includes_w8a8(tmp_path):
    exporter = _exporter(tmp_path, dynamic_quantize=True)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter.dynamic_quantize_models()

    snapshot = json.loads(
        (Path(exporter._quantize_dir) / COMPILE_FLAGS_FILE).read_text()
    )
    assert snapshot["flags"] == ["--a", "--b"]


def test_converted_variant_snapshot_has_base_flags_only(tmp_path):
    exporter = _exporter(tmp_path, convert_dtypes=True)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter.convert_models()

    snapshot = json.loads(
        (Path(exporter._convert_dir) / COMPILE_FLAGS_FILE).read_text()
    )
    assert snapshot["flags"] == ["--a"]


def _fake_compile(monkeypatch) -> list:
    calls = []

    def fake_export_torq(input_model, output_dir, **kwargs):
        calls.append(kwargs.get("compiler_args"))

    monkeypatch.setattr(me, "export_torq", fake_export_torq)
    return calls


def test_export_torq_applies_model_flags_before_user_flags(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    exporter = _exporter(tmp_path)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter.export_torq(torq_compile_args=["--user"])

    assert calls == [["--a", "--user"]]


def test_export_torq_quantized_variant_gets_w8a8_flags(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    exporter = _exporter(tmp_path, dynamic_quantize=True)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter.export_torq()

    assert calls == [["--a", "--b"]]


def test_compile_driver_loads_flags_from_model_dir(tmp_path, monkeypatch):
    """Standalone `python -m torq.utils.compile <dir>/model.onnx` picks the up
    variant dir's recorded flags; explicit --compile-flags come after (win)."""
    import torq.utils.compile as tc

    model_dir = tmp_path / "variant"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"")
    (model_dir / COMPILE_FLAGS_FILE).write_text(json.dumps({"flags": ["--a"]}))

    compile_calls = []

    def fake_export_onnx_to_mlir(model, mlir, **kwargs):
        pass

    def fake_compile(mlir, out, target, compile_args, *args):
        compile_calls.append(list(compile_args))

    monkeypatch.setattr(tc, "export_onnx_to_mlir", fake_export_onnx_to_mlir)
    monkeypatch.setattr(tc, "compile_mlir_for_vm", fake_compile)
    monkeypatch.chdir(tmp_path)

    import sys
    monkeypatch.setattr(sys, "argv", ["compile.py", str(model_dir / "model.onnx")])
    tc.main()
    assert compile_calls == [["--a"]]

    monkeypatch.setattr(sys, "argv", ["compile.py", str(model_dir / "model.onnx"),
                                      "--compile-flags", "--a", "--user"])
    tc.main()
    assert compile_calls[1] == ["--a", "--a", "--user"]
