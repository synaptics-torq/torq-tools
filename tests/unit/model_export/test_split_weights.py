# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Base exporter --split-weights handling: split final exports and dump kwargs."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from support.model_export import StubExporter
from torq.graph_edit.harness import GraphEditHarness


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


def test_export_onnx_inline_without_split_weights(tmp_path):
    model, weight = _two_tensor_model()
    exporter = StubExporter(tmp_path, {"model": model})
    exporter.export_onnx(validate=False, cleanup=False)

    assert not (exporter.export_dir / "model.onnx.data").exists()
    m = onnx.load(str(exporter.export_dir / "model.onnx"))
    assert np.array_equal(numpy_helper.to_array(_inits(m)["weight"]), weight)


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

    split = StubExporter(tmp_path, {"model": _two_tensor_model()[0]}, split_weights=True)
    split.set_graph_edit_harness(GraphEditHarness(dump_after_edit="EliminateExpand,FoldScalarMatMul"))
    assert split._editor_dump_kwargs(Path("export/model.onnx")) == {
        "dump_path": Path("export/intermediates/model.onnx"),
        "dump_after_edit": "EliminateExpand,FoldScalarMatMul",
    }
