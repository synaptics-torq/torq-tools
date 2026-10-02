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
from torq.graph_edit.edits import CommonGraphEditsMixin


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


def test_apply_fixed_input_shapes_updates_only_named_inputs():
    a = gs.Variable("a", dtype=np.float32, shape=["batch", 3])
    b = gs.Variable("b", dtype=np.float32, shape=["batch", 3])
    editor = OnnxGraphEditor(graph(nodes=[], inputs=[a, b], outputs=[]), "unit")

    editor.apply_fixed_input_shapes({"a": [2, 3]})

    assert a.shape == [2, 3]
    assert b.shape == ["batch", 3]


class DelegatingEditor(OnnxGraphEditor, CommonGraphEditsMixin):
    def __init__(self, tmp_path):
        super().__init__(graph(nodes=[], inputs=[], outputs=[]), "delegating", export_dtype=onnx.TensorProto.FLOAT)
        self.applied = []
        self.tmp_path = tmp_path

    def apply_edit(self, edit):
        self.applied.append(edit)
        return self


@pytest.mark.parametrize(
    ("method_name", "args", "expected_type"),
    [
        ("replace_dynamic_kv_cache", (gs.Variable("cur", dtype=np.int64, shape=[1]), 8), "ReplaceDynamicKVCache"),
        ("mask_future_attn_scores", (gs.Variable("cur", dtype=np.int64, shape=[1]), 8), "MaskFutureAttentionScores"),
        ("add_curr_len_input", (gs.Variable("cur", dtype=np.int64, shape=[1]),), "AddCurrLenInput"),
        ("decompose_layer_normalization", (), "DecomposeLayerNormalization"),
        ("convert_to_static_index", (), "ConvertToStaticIndex"),
        ("dequantize_projections_matmul", (2, 4), "DequantizeProjectionsMatMul"),
        ("remove_isNaN", (), "RemoveIsNaN"),
        ("remove_redundant_casts", (), "RemoveRedundantCasts"),
        ("fold_scalar_matmul", (), "FoldScalarMatMul"),
        ("replace_constant_div_with_mul", (), "ReplaceConstantDivWithMul"),
        ("replace_int64_float_cast", (8,), "ReplaceInt64FloatCast"),
        ("broadcast_op_inputs", (["Add"],), "BroadcastOpInputs"),
        ("eliminate_expands", (["Add"],), "EliminateExpand"),
        ("eliminate_transposes", (), "EliminateTranspose"),
        ("collapse_reshape_chains", (), "CollapseReshapeChain"),
        ("retarget_cross_attn_key_layout", (), "RetargetCrossAttnKeyLayout"),
        ("collapse_gqa_broadcast", (), "CollapseGQABroadcast"),
        ("trim_lm_head_vocab", ([0, 1],), "TrimLMHeadVocab"),
        ("eliminate_rank0_gather", (), "EliminateRank0Gather"),
        ("eliminate_singleton_gather_unsqueeze", (), "EliminateSingletonGatherUnsqueeze"),
        ("rewrite_negative_pads", (), "RewriteNegativePads"),
        ("absorb_padding", (), "AbsorbPadding"),
        ("replace_pad_with_concat", (), "ReplacePadWithConcat"),
        ("widen_strided_depthwise_conv", (), "WidenStridedDepthwiseConv"),
        ("decompose_strided_conv1d", (), "DecomposeStridedConv1D"),
        ("decompose_bidirectional_rnn", (), "DecomposeBidirectionalRnn"),
    ],
)
def test_common_graph_edit_mixin_methods_delegate_to_expected_edit(tmp_path, method_name, args, expected_type):
    editor = DelegatingEditor(tmp_path)

    result = getattr(editor, method_name)(*args)

    assert result is editor
    assert editor.applied[-1].__class__.__name__ == expected_type


def test_common_graph_edit_mixin_artifact_methods_delegate(tmp_path):
    editor = DelegatingEditor(tmp_path)

    editor.extract_token_embeddings(2, 4, tmp_path / "lut.npy")
    editor.split_lm_head(tmp_path / "lm_head.onnx")

    assert [edit.__class__.__name__ for edit in editor.applied] == [
        "ExtractConstantLUT",
        "SplitLMHead",
    ]


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


def test_editor_does_not_dump_without_dump_path(tmp_path):
    editor = OnnxGraphEditor(_identity_chain_graph(), "unit")

    editor.apply_edit(RenameFinalOutput(editor.graph, "unit"))

    assert list(tmp_path.iterdir()) == []


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


def test_editor_dump_never_writes_for_unmatched_name(tmp_path):
    dump = tmp_path / "model.onnx"
    editor = OnnxGraphEditor(
        _identity_chain_graph(), "unit", dump_path=dump, dump_after_edit="NoSuchEdit"
    )

    editor.apply_edit(RenameFinalOutput(editor.graph, "unit"))

    assert list(tmp_path.iterdir()) == []


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
