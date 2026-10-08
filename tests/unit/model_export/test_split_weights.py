# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Weight storage: --split-weights exports, dump kwargs, and lazy (external
data on demand) loading and saving."""

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper, numpy_helper

from support.model_export import StubExporter
from torq.graph_edit.harness import GraphEditHarness
from torq.model_export.cleanup import cleanup_onnx_model
from torq.utils.onnx import external_data_cwd, load_onnx_lazy, save_onnx


def _two_tensor_model() -> tuple[onnx.ModelProto, np.ndarray]:
    """MatMul + Add model: 1200-byte weight (externalized) + 120-byte bias (inline)."""
    weight = np.arange(300, dtype=np.float32).reshape(10, 30)
    bias = np.arange(30, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 30])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["mm"]),
            helper.make_node("Add", ["mm", "bias"], ["y"]),
        ],
        "g",
        [x],
        [y],
        initializer=[
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), weight


def _inits(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {i.name: i for i in model.graph.initializer}


def test_export_onnx_split_weights_writes_data_file(tmp_path):
    model, weight = _two_tensor_model()
    exporter = StubExporter(tmp_path, {"model": model}, split_weights=True)
    exporter.export_onnx(validate=False, cleanup=False)

    onnx_path = exporter.export_dir / "model.onnx"
    data_path = exporter.export_dir / "model.onnx.data"
    assert onnx_path.exists()
    assert data_path.exists()

    m = onnx.load(str(onnx_path), load_external_data=False)
    inits = _inits(m)
    # The > 1024-byte weight is externalized...
    assert not inits["weight"].raw_data
    assert dict((e.key, e.value) for e in inits["weight"].external_data)["location"] == "model.onnx.data"
    # ...while the small constant stays inline (onnxruntime shape-op requirement).
    assert len(inits["bias"].raw_data) == 120
    # Standard loading transparently restores the weight from the .data file.
    loaded = _inits(onnx.load(str(onnx_path)))
    assert np.array_equal(numpy_helper.to_array(loaded["weight"]), weight)


def test_editor_dump_kwargs_reads_harness(tmp_path):
    exporter = StubExporter(tmp_path, {"model": _two_tensor_model()[0]})

    # No harness (or no --dump-after-edit) -> no dump kwargs.
    assert exporter._editor_dump_kwargs(Path("export/model.onnx")) == {}
    exporter.set_graph_edit_harness(GraphEditHarness())
    assert exporter._editor_dump_kwargs(Path("export/model.onnx")) == {}

    exporter.set_graph_edit_harness(GraphEditHarness(dump_after_edit="all"))
    assert exporter._editor_dump_kwargs(Path("export/model.onnx")) == {
        "dump_path": Path("export/intermediates/model.onnx"),
        "dump_after_edit": "all",
    }


def _chain_model(weights: dict[str, np.ndarray], unused: np.ndarray) -> onnx.ModelProto:
    """x -> MatMul(w1) -> MatMul(w2) -> y, plus an unreferenced initializer."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"]),
            helper.make_node("MatMul", ["h", "w2"], ["y"]),
        ],
        "g",
        [x],
        [y],
        initializer=[numpy_helper.from_array(v, name=k) for k, v in weights.items()]
        + [numpy_helper.from_array(unused, name="unused")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_lazy_edit_and_in_place_save(tmp_path, monkeypatch):
    """Weights load on demand from the model's directory (not the CWD), and an
    in-place save keeps edited, untouched and drops removed weights."""
    rng = np.random.default_rng(0)
    w1, w2 = (rng.standard_normal((16, 16)).astype(np.float32) for _ in range(2))
    path = tmp_path / "src" / "model.onnx"
    path.parent.mkdir()
    save_onnx(_chain_model({"w1": w1, "w2": w2}, np.ones(512, np.float32)), path, split=True)
    monkeypatch.chdir(tmp_path)

    model = load_onnx_lazy(path)
    assert not any(t.raw_data for t in model.graph.initializer if t.name != "unused")
    with external_data_cwd(model):
        onnx.checker.check_model(model, full_check=True)
    graph = gs.import_onnx(model)
    consts = graph.tensors()
    consts["w1"].values = consts["w1"].values * 2
    graph.cleanup(remove_unused_graph_inputs=True)
    save_onnx(gs.export_onnx(graph), path, split=True)

    data_path = path.with_name("model.onnx.data")
    assert data_path.stat().st_size == w1.nbytes + w2.nbytes
    inits = _inits(onnx.load(str(path)))
    assert inits.keys() == {"w1", "w2"}
    assert np.array_equal(numpy_helper.to_array(inits["w1"]), w1 * 2)
    assert np.array_equal(numpy_helper.to_array(inits["w2"]), w2)

    save_onnx(load_onnx_lazy(path), path)
    assert not data_path.exists()
    assert np.array_equal(numpy_helper.to_array(_inits(onnx.load(str(path)))["w2"]), w2)


def test_lazy_load_rejects_data_outside_the_model_dir(tmp_path):
    w = np.ones((16, 16), np.float32)
    path = tmp_path / "model.onnx"
    save_onnx(_chain_model({"w1": w, "w2": w}, w), path, split=True)
    model = load_onnx_lazy(path)
    for entry in model.graph.initializer[0].external_data:
        if entry.key == "location":
            entry.value = "../model.onnx.data"
    with pytest.raises(ValueError, match="outside"):
        gs.import_onnx(model).tensors()["w1"].values


def test_cleanup_folds_lazy_constants(tmp_path):
    """fold_constants evaluates in ORT from a serialized model: a lazy
    model's weights must stay reachable, or the fold is silently skipped."""
    w = np.arange(256, dtype=np.float32).reshape(16, 16)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
            helper.make_node("MatMul", ["x", "wt"], ["y"]),
        ],
        "g", [x], [y], initializer=[numpy_helper.from_array(w, name="w")],
    )
    path = tmp_path / "model.onnx"
    save_onnx(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path, split=True)

    cleaned = cleanup_onnx_model(load_onnx_lazy(path))

    assert [n.op_type for n in cleaned.graph.node] == ["MatMul"]
    (folded,) = cleaned.graph.initializer
    assert np.array_equal(numpy_helper.to_array(folded), w.T)
