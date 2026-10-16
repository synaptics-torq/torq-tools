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
from torq.utils.compile import COMPILE_FLAGS_FILE, select_compile_flags_entry
from support.model_export import StubExporter

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pin_compiler_version(monkeypatch):
    """Exports resolve their flag set against a pinned compiler version
    (matching ``_flags_file``'s default) rather than the environment's.
    """
    monkeypatch.setattr(me, "get_iree_version", lambda compiler_path=None: "2.2.1")


def _flags_file(
    tmp_path: Path,
    flags: list[str],
    dynamic_quantize: list[str] | None = None,
    version: str = "2.2.1",
) -> Path:
    entry = {"common": flags}
    if dynamic_quantize is not None:
        entry["dynamic_quantize"] = dynamic_quantize
    path = tmp_path / "model" / COMPILE_FLAGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({version: entry}, indent=2) + "\n")
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
        for version, entry in data.items():
            assert entry.get("common"), f"{model} {version}: empty common flags"
        assert data["2.2.1"].get("dynamic_quantize") == [
            "--torq-disable-host",
            "--torq-disable-css",
        ], f"{model}: 2.2.1 dynamic_quantize workaround (torq-compile 2.2.1 host/CSS bug) missing"
    # moonshine is validated against the LLM flag set, not its own.
    llm = json.loads((models_dir / "gemma3" / COMPILE_FLAGS_FILE).read_text())
    for model in ("moonshine", "moonshine_streaming"):
        data = json.loads((models_dir / model / COMPILE_FLAGS_FILE).read_text())
        assert data == llm, f"{model}: flags drifted from the LLM set"


@pytest.mark.ci
def test_installed_compiler_matches_shipped_flag_versions():
    """CI-only guard: the installed torq-compiler must be a top-level key in
    every shipped ``compile_flags.json``. Bumping the compiler without
    re-validating a model's flag set fails CI here instead of silently
    shipping stale flags. Skipped outside CI (``CI`` unset): a deliberate
    local compiler/flag mismatch must not break regular use."""
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
        assert installed in data, (
            f"{rel}: installed torq-compiler {installed} not in validated "
            f"versions {sorted(data)!r}; re-validate the flags for {installed} "
            "and add an entry alongside the compiler requirement"
        )


def test_pinned_compiler_has_entries_in_all_flag_files():
    """The torq-compiler pinned in pyproject (what a package install pulls
    in) must have an entry in every shipped ``compile_flags.json``, so a
    plain ``pip install torq-tools`` + export resolves each model's
    validated flags instead of warning and compiling without them."""
    import tomllib

    with open(Path(me.__file__).parents[3] / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    pin = next((d for d in deps if d.startswith("torq-compiler==")), None)
    assert pin, "pyproject.toml must pin torq-compiler==<version> exactly"
    pinned = pin.split("==", 1)[1]

    models_dir = Path(me.__file__).parent.parent / "models"
    for path in sorted(models_dir.rglob(COMPILE_FLAGS_FILE)):
        rel = path.relative_to(models_dir)
        data = json.loads(path.read_text())
        assert pinned in data, (
            f"{rel}: pinned torq-compiler {pinned} not in validated "
            f"versions {sorted(data)!r}; add a flag set validated for "
            f"{pinned} so package installs can export this model"
        )


def test_select_compile_flags_entry():
    data = {
        "2.2.0": {"common": ["--old"]},
        "2.2.1": {"common": ["--new"], "dynamic_quantize": ["--w8a8"]},
    }
    # Exact match only.
    assert select_compile_flags_entry(data, "2.2.1") == {
        "common": ["--new"], "dynamic_quantize": ["--w8a8"],
    }
    assert select_compile_flags_entry(data, "2.2.0") == {"common": ["--old"]}
    # Newer, older, dev-build, unknown, and unparseable versions get none.
    assert select_compile_flags_entry(data, "2.3.0") == {}
    assert select_compile_flags_entry(data, "1.5") == {}
    assert select_compile_flags_entry(data, "2.1.0") == {}
    assert select_compile_flags_entry(data, "0.dev0+20261015t152220z.g639d1ed") == {}
    assert select_compile_flags_entry(data, None) == {}
    assert select_compile_flags_entry(data, "not-a-version") == {}
    assert select_compile_flags_entry({}, "2.2.1") == {}


def test_get_iree_version_from_binary(tmp_path):
    """An explicit compiler path is versioned from its --version output."""
    import torq.utils.compile as tc

    script = tmp_path / "torq-compile"
    script.write_text("#!/bin/sh\necho 'IREE compiler version 2.3.0'\n")
    script.chmod(0o755)
    assert tc.get_iree_version(script) == "2.3.0"


def test_get_iree_version_falls_back_to_pip_package(monkeypatch, tmp_path):
    """The torq-compiler pip wheel's console script reports '(unknown)', so
    the installed package version is the fallback; a bare binary path with
    no package is unknown."""
    import torq.utils.compile as tc

    wheel_script = tmp_path / "torq-compile"
    wheel_script.write_text("#!/bin/sh\necho 'IREE compiler version (unknown)'\n")
    wheel_script.chmod(0o755)

    monkeypatch.setattr(tc, "version", lambda name: "2.2.1")
    assert tc.get_iree_version(wheel_script) == "2.2.1"
    assert tc.get_iree_version(tmp_path / "missing") == "2.2.1"

    def missing(name):
        raise tc.PackageNotFoundError(name)

    monkeypatch.setattr(tc, "version", missing)
    assert tc.get_iree_version(wheel_script) is None
    assert tc.get_iree_version(tmp_path / "missing") is None


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


def test_quantized_variant_snapshot_includes_dynamic_quantize_flags(tmp_path):
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


def test_export_torq_quantized_variant_gets_dynamic_quantize_flags(tmp_path, monkeypatch):
    calls = _fake_compile(monkeypatch)
    exporter = _exporter(tmp_path, dynamic_quantize=True)
    exporter._compile_flags_file = lambda: _flags_file(tmp_path, ["--a"], ["--b"])
    exporter.export_torq()

    assert calls == [["--a", "--b"]]


def test_export_torq_selects_flags_for_compiler_version(tmp_path, monkeypatch):
    """A per-version flag file resolves against the compiling compiler."""
    calls = _fake_compile(monkeypatch)
    path = tmp_path / "model" / COMPILE_FLAGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "2.2.0": {"common": ["--v220"]},
        "2.2.1": {"common": ["--v221"], "dynamic_quantize": ["--b"]},
    }, indent=2) + "\n")
    for compiler_version, expected in (("2.2.0", ["--v220"]), ("2.2.1", ["--v221"])):
        monkeypatch.setattr(me, "get_iree_version", lambda compiler_path=None: compiler_version)
        exporter = _exporter(tmp_path)
        exporter._compile_flags_file = lambda: path
        exporter.export_torq()
        assert calls[-1] == expected

    # Quantized: the dynamic_quantize group of the matching version is added.
    monkeypatch.setattr(me, "get_iree_version", lambda compiler_path=None: "2.2.1")
    exporter = _exporter(tmp_path, dynamic_quantize=True)
    exporter._compile_flags_file = lambda: path
    exporter.export_torq()
    assert calls[-1] == ["--v221", "--b"]


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
