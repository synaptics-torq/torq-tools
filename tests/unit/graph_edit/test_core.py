# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import dataclasses
import inspect

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEdit,
    OnnxGraphEditor,
    rewire_consumers,
)


pytestmark = pytest.mark.core


@dataclasses.dataclass
class RenameIdentityOutput(OnnxGraphEdit):
    def match(self, node: gs.Node) -> bool:
        return node.op == "Identity"

    def transform(self, node: gs.Node):
        node.outputs[0].name = "renamed"


def test_edit_import_surface_reexports_dataclass_edits():
    import torq.graph_edit.edits as edits

    assert "CommonGraphEditsMixin" in edits.__all__
    missing = []
    for name in edits.__all__:
        obj = getattr(edits, name)
        if inspect.isclass(obj) and issubclass(obj, OnnxGraphEdit):
            if not dataclasses.is_dataclass(obj):
                missing.append(name)
    assert missing == []


def test_rewire_consumers_replaces_only_identity_matched_input():
    orig = gs.Variable("orig", dtype=np.float32, shape=[1])
    replacement = gs.Variable("replacement", dtype=np.float32, shape=[1])
    other = gs.Variable("other", dtype=np.float32, shape=[1])
    out = gs.Variable("out", dtype=np.float32, shape=[1])
    consumer = gs.Node("Add", "consumer", inputs=[orig, other], outputs=[out])

    rewire_consumers([consumer], orig, replacement)

    assert consumer.inputs == [replacement, other]


def test_editor_registers_and_applies_named_edit():
    x = gs.Variable("x", dtype=np.float32, shape=[1])
    y = gs.Variable("y", dtype=np.float32, shape=[1])
    identity = gs.Node("Identity", "id", inputs=[x], outputs=[y])
    g = graph(nodes=[identity], inputs=[x], outputs=[y])
    edit = RenameIdentityOutput(g, "unit")

    editor = OnnxGraphEditor(g, "unit")
    editor.register_edit(edit, "rename")
    editor.apply_edit("rename")

    assert editor.edits == ["rename"]
    assert editor.graph.outputs[0].name == "renamed"


def test_editor_context_restores_backup_on_exception():
    x = gs.Variable("x", dtype=np.float32, shape=[1])
    y = gs.Variable("y", dtype=np.float32, shape=[1])
    g = graph(nodes=[gs.Node("Identity", "id", inputs=[x], outputs=[y])], inputs=[x], outputs=[y])
    editor = OnnxGraphEditor(g, "unit")

    with pytest.raises(RuntimeError):
        with editor:
            editor._graph.outputs[0].name = "mutated"
            raise RuntimeError("boom")

    assert editor.graph.outputs[0].name == "y"


def test_fix_io_dims_supports_digits_exact_and_contains_matches():
    inp = gs.Variable("inp", dtype=np.float32, shape=["1", "batch", "past_seq"])
    out = gs.Variable("out", dtype=np.float32, shape=["hidden"])
    g = graph(nodes=[], inputs=[inp], outputs=[out])
    editor = OnnxGraphEditor(g, "unit")

    editor.fix_io_dims(
        [
            FixedDimMapping("batch", DimMatchType.EXACT, 2),
            FixedDimMapping("seq", DimMatchType.CONTAINS, 5),
            FixedDimMapping("hidden", DimMatchType.EXACT, 7),
        ]
    )

    assert inp.shape == [1, 2, 5]
    assert out.shape == [7]


def test_fix_io_dims_raises_for_unexpected_dynamic_dim():
    inp = gs.Variable("inp", dtype=np.float32, shape=["mystery"])
    editor = OnnxGraphEditor(graph(nodes=[], inputs=[inp], outputs=[]), "unit")

    with pytest.raises(ValueError, match="Unexpected dynamic dimension"):
        editor.fix_io_dims([])


def test_reorder_graph_input_and_output_clamps_negative_positions():
    a = gs.Variable("a", dtype=np.float32, shape=[1])
    b = gs.Variable("b", dtype=np.float32, shape=[1])
    c = gs.Variable("c", dtype=np.float32, shape=[1])
    out0 = gs.Variable("out0", dtype=np.float32, shape=[1])
    out1 = gs.Variable("out1", dtype=np.float32, shape=[1])
    editor = OnnxGraphEditor(graph(nodes=[], inputs=[a, b, c], outputs=[out0, out1]), "unit")

    editor.reorder_graph_input("a", -1)
    editor.reorder_graph_output("out1", 0)

    assert [i.name for i in editor.graph.inputs] == ["b", "c", "a"]
    assert [o.name for o in editor.graph.outputs] == ["out1", "out0"]


# -----------------------------------------------------------------------------
# Per-edit intermediate dumps (OnnxGraphEditor.dump_path / dump_after_edit)
# -----------------------------------------------------------------------------

class RenameFinalOutput(OnnxGraphEdit):
    def match(self, node: gs.Node) -> bool:
        return node.name == "id2"

    def transform(self, node: gs.Node):
        node.outputs[0].name = "y2_renamed"


class AppendIdentity(OnnxGraphEdit):
    def match(self, node: gs.Node) -> bool:
        return node.name == "id2"

    def transform(self, node: gs.Node):
        out = gs.Variable("y3", dtype=np.float32, shape=[1])
        self.graph.layer(name="id3", op="Identity", inputs=[node.outputs[0]], outputs=[out])
        self.graph.outputs[self.graph.outputs.index(node.outputs[0])] = out


class RenameMatMulOutput(OnnxGraphEdit):
    def match(self, node: gs.Node) -> bool:
        return node.op == "MatMul"

    def transform(self, node: gs.Node):
        node.outputs[0].name = "y_renamed"


def _identity_chain_graph() -> gs.Graph:
    x = gs.Variable("x", dtype=np.float32, shape=[1])
    y1 = gs.Variable("y1", dtype=np.float32, shape=[1])
    y2 = gs.Variable("y2", dtype=np.float32, shape=[1])
    id1 = gs.Node("Identity", "id1", inputs=[x], outputs=[y1])
    id2 = gs.Node("Identity", "id2", inputs=[y1], outputs=[y2])
    return graph(nodes=[id1, id2], inputs=[x], outputs=[y2])


def test_editor_dump_after_every_edit_uses_numbered_paths(tmp_path):
    dump = tmp_path / "model.onnx"
    editor = OnnxGraphEditor(
        _identity_chain_graph(), "unit", dump_path=dump, dump_after_edit="all"
    )

    editor.apply_edit(RenameFinalOutput(editor.graph, "unit"))
    first_dump = tmp_path / "0001_RenameFinalOutput_model.onnx"
    assert onnx.load(first_dump).graph.output[0].name == "y2_renamed"

    editor.apply_edit(AppendIdentity(editor.graph, "unit"))
    second_dump = tmp_path / "0002_AppendIdentity_model.onnx"
    dumped = onnx.load(second_dump)
    assert [n.name for n in dumped.graph.node] == ["id1", "id2", "id3"]
    assert dumped.graph.output[0].name == "y3"
    assert onnx.load(first_dump).graph.output[0].name == "y2_renamed"


def test_editor_dump_continues_numbering_across_editor_instances(tmp_path):
    dump = tmp_path / "model.onnx"
    first = OnnxGraphEditor(
        _identity_chain_graph(), "unit", dump_path=dump, dump_after_edit="all"
    )
    first.apply_edit(RenameFinalOutput(first.graph, "unit"))

    second = OnnxGraphEditor(
        _identity_chain_graph(), "unit", dump_path=dump, dump_after_edit="all"
    )
    second.apply_edit(AppendIdentity(second.graph, "unit"))

    assert (tmp_path / "0001_RenameFinalOutput_model.onnx").exists()
    assert (tmp_path / "0002_AppendIdentity_model.onnx").exists()


def test_editor_dump_only_for_named_edits(tmp_path):
    dump = tmp_path / "model.onnx"
    editor = OnnxGraphEditor(
        _identity_chain_graph(), "unit", dump_path=dump,
        dump_after_edit="RenameFinalOutput",
    )

    editor.apply_edit(RenameFinalOutput(editor.graph, "unit"))
    editor.apply_edit(AppendIdentity(editor.graph, "unit"))

    dumped = onnx.load(tmp_path / "0001_RenameFinalOutput_model.onnx")
    assert [n.name for n in dumped.graph.node] == ["id1", "id2"]
    assert dumped.graph.output[0].name == "y2_renamed"


def _matmul_graph(weight: np.ndarray) -> gs.Graph:
    x = gs.Variable("x", dtype=np.float32, shape=[1, weight.shape[0]])
    y = gs.Variable("y", dtype=np.float32, shape=[1, weight.shape[1]])
    mm = gs.Node("MatMul", "mm", inputs=[x, gs.Constant("w", weight)], outputs=[y])
    return graph(nodes=[mm], inputs=[x], outputs=[y])


@pytest.mark.parametrize("split_weights", [False, True])
def test_editor_dump_externalizes_and_discards_weights(tmp_path, split_weights):
    dump = tmp_path / "model.onnx"
    weight = np.arange(16, dtype=np.float32).reshape(4, 4)
    editor = OnnxGraphEditor(
        _matmul_graph(weight), "unit",
        dump_path=dump, dump_after_edit="all", split_weights=split_weights,
    )

    editor.apply_edit(RenameMatMulOutput(editor.graph, "unit"))
    editor.apply_edit(RenameMatMulOutput(editor.graph, "unit"))

    first_dump = tmp_path / "0001_RenameMatMulOutput_model.onnx"
    second_dump = tmp_path / "0002_RenameMatMulOutput_model.onnx"
    assert first_dump.exists()
    assert second_dump.exists()
    assert not list(tmp_path.glob("*.data"))
    # The .onnx carries no tensor payload, only a reference to discarded data.
    unloaded = onnx.load(first_dump, load_external_data=False)
    init = unloaded.graph.initializer[0]
    assert not init.raw_data
    assert dict((e.key, e.value) for e in init.external_data) == {
        "location": "0001_RenameMatMulOutput_model.onnx.data",
        "offset": "0",
        "length": str(weight.nbytes),
    }


def test_freeze_shape_seeds_folds_static_shapes_but_not_input_values():
    x = gs.Variable("x", np.float32, [2, 3, 4])
    pos = gs.Variable("pos", np.int64, [1])
    weight = gs.Constant("weight", np.ones((2048,), np.float32))
    g = gs.Graph(inputs=[x, pos], outputs=[], opset=17)
    shape = g.layer(op="Shape", inputs=[x], outputs=[gs.Variable("shape", np.int64)], attrs={"start": 0, "end": 2})[0]
    target = g.layer(op="Concat", inputs=[shape, np.array([-1], np.int64)], outputs=[gs.Variable("target", np.int64)], attrs={"axis": 0})[0]
    reshaped = g.layer(op="Reshape", inputs=[x, target], outputs=[gs.Variable("reshaped", np.float32)])[0]
    end = g.layer(op="Add", inputs=[pos, np.array([1], np.int64)], outputs=[gs.Variable("end", np.int64)])[0]
    sliced = g.layer(op="Slice", inputs=[weight, pos, end], outputs=[gs.Variable("sliced", np.float32)])[0]
    g.outputs = [reshaped, sliced]

    editor = OnnxGraphEditor(g, "test")
    editor.freeze_shape_seeds()

    nodes = {node.op: node for node in editor.graph.nodes}
    assert set(nodes) == {"Reshape", "Add", "Slice"}
    assert np.array_equal(nodes["Reshape"].inputs[1].values, [2, 3, -1])
    assert nodes["Slice"].inputs[1].name == "pos"
    assert np.array_equal(nodes["Slice"].inputs[0].values, weight.values)


def test_infer_shapes_detaches_and_restores_large_initializers():
    from torq.utils.onnx import infer_shapes

    weight = np.arange(1024, dtype=np.float32)
    model = onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "w"], ["y"])],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1024])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
        [onnx.numpy_helper.from_array(weight, "w")],
    ))

    inferred = infer_shapes(model)

    assert inferred.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 1024
    for m in (model, inferred):
        init = m.graph.initializer[0]
        assert init.data_location == onnx.TensorProto.DEFAULT and not init.external_data
        assert np.array_equal(onnx.numpy_helper.to_array(init), weight)
