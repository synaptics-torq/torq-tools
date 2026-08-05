# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit import OnnxGraphEditor
from torq.graph_edit.harness import (
    ContextRef,
    EditSpec,
    GraphEditHarness,
    ctx,
    edit_registry,
    load_edit_specs_file,
    load_exclude_names_file,
    parse_edit_flag,
    render_graph_edit_plan,
    resolve_args,
)


pytestmark = pytest.mark.core


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def test_parse_edit_flag_bare_name():
    assert parse_edit_flag("RemoveIsNaN") == EditSpec("RemoveIsNaN", ())


def test_parse_edit_flag_with_yaml_list_args():
    spec = parse_edit_flag("EliminateExpand:[Add, Mul]")
    assert spec == EditSpec("EliminateExpand", ("Add", "Mul"))


def test_parse_edit_flag_scalar_and_typed_args():
    assert parse_edit_flag("WidenStridedDepthwiseConv:[64, 4]") == EditSpec(
        "WidenStridedDepthwiseConv", (64, 4)
    )
    assert parse_edit_flag("Foo:5") == EditSpec("Foo", (5,))


def test_parse_edit_flag_empty_name_raises():
    with pytest.raises(Exception):
        parse_edit_flag(":[1, 2]")


def test_load_edit_and_exclude_files(tmp_path):
    p = tmp_path / "edits.yaml"
    p.write_text(
        "edits:\n"
        "  - RemoveIsNaN\n"
        "  - name: EliminateExpand\n"
        "    args: [[Add, Mul]]\n"
    )
    specs = load_edit_specs_file(p)
    assert specs == [
        EditSpec("RemoveIsNaN", ()),
        EditSpec("EliminateExpand", (["Add", "Mul"],)),
    ]

    ex = tmp_path / "ex.yaml"
    ex.write_text("- RemoveIsNaN\n- FoldScalarMatMul\n")
    assert load_exclude_names_file(ex) == ["RemoveIsNaN", "FoldScalarMatMul"]


# -----------------------------------------------------------------------------
# Registry / context resolution
# -----------------------------------------------------------------------------

def test_registry_contains_known_edits_and_no_mixins():
    reg = edit_registry()
    assert "RemoveIsNaN" in reg
    assert "EliminateExpand" in reg
    assert "CommonGraphEditsMixin" not in reg
    assert "OnnxGraphEdit" not in reg


def test_resolve_args_substitutes_context_refs():
    out = resolve_args((ctx("cur_len"), 5, "x"), {"cur_len": 42})
    assert out == [42, 5, "x"]


def test_resolve_args_missing_context_raises():
    with pytest.raises(KeyError):
        resolve_args((ContextRef("missing"),), {})


# -----------------------------------------------------------------------------
# Merge / precedence semantics
# -----------------------------------------------------------------------------

def _names(specs):
    return [s.name for s in specs]


def test_finalize_defaults_passthrough():
    defaults = [EditSpec("RemoveIsNaN"), EditSpec("EliminateTranspose")]
    assert GraphEditHarness().finalize(defaults) == defaults


def test_finalize_override_keeps_position_and_replaces_args():
    defaults = [
        EditSpec("EliminateTranspose"),
        EditSpec("BroadcastOpInputs", (["Mul"],)),
        EditSpec("FoldScalarMatMul"),
    ]
    h = GraphEditHarness(apply_flag=[EditSpec("BroadcastOpInputs", (["Add"],))])
    out = h.finalize(defaults)
    assert _names(out) == ["EliminateTranspose", "BroadcastOpInputs", "FoldScalarMatMul"]
    assert out[1].args == (["Add"],)


def test_finalize_new_edit_is_appended():
    defaults = [EditSpec("EliminateTranspose")]
    h = GraphEditHarness(apply_flag=[EditSpec("FoldScalarMatMul")])
    assert _names(h.finalize(defaults)) == ["EliminateTranspose", "FoldScalarMatMul"]


def test_finalize_flag_beats_file_on_override():
    defaults = [EditSpec("BroadcastOpInputs", (["Mul"],))]
    h = GraphEditHarness(
        apply_file=[EditSpec("BroadcastOpInputs", (["File"],))],
        apply_flag=[EditSpec("BroadcastOpInputs", (["Flag"],))],
    )
    out = h.finalize(defaults)
    assert out[0].args == (["Flag"],)


def test_finalize_exclude_removes_default_and_extra():
    defaults = [EditSpec("RemoveIsNaN"), EditSpec("EliminateTranspose")]
    h = GraphEditHarness(
        apply_flag=[EditSpec("FoldScalarMatMul")],
        exclude={"RemoveIsNaN", "FoldScalarMatMul"},
    )
    assert _names(h.finalize(defaults)) == ["EliminateTranspose"]


def test_finalize_bare_name_does_not_wipe_default_args():
    defaults = [EditSpec("BroadcastOpInputs", (["Mul"],))]
    h = GraphEditHarness(apply_flag=[EditSpec("BroadcastOpInputs", ())])
    assert h.finalize(defaults)[0].args == (["Mul"],)


def test_finalize_unknown_edit_raises():
    with pytest.raises(ValueError):
        GraphEditHarness(apply_flag=[EditSpec("NoSuchEdit")]).finalize([])


# -----------------------------------------------------------------------------
# from_args + rendering
# -----------------------------------------------------------------------------

class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_from_args_collects_flags_files_and_excludes(tmp_path):
    edits_file = tmp_path / "e.yaml"
    edits_file.write_text("- FoldScalarMatMul\n")
    ex_file = tmp_path / "x.yaml"
    ex_file.write_text("- RemoveIsNaN\n")
    args = _NS(
        apply_graph_edit=[EditSpec("EliminateTranspose")],
        apply_graph_edits_from_file=str(edits_file),
        exclude_graph_edit=["CollapseReshapeChain"],
        exclude_graph_edits_from_file=str(ex_file),
        view_graph_edits=True,
    )
    h = GraphEditHarness.from_args(args)
    assert h.apply_flag == [EditSpec("EliminateTranspose")]
    assert h.apply_file == [EditSpec("FoldScalarMatMul")]
    assert h.exclude == {"CollapseReshapeChain", "RemoveIsNaN"}
    assert h.view is True


def test_render_graph_edit_plan_shows_context_refs():
    blocks = {"model": [EditSpec("ReplaceDynamicKVCache", (ctx("cur_len"), 8))]}
    text = render_graph_edit_plan(blocks)
    assert "[model]" in text
    assert "$cur_len" in text
    assert "ReplaceDynamicKVCache" in text


# -----------------------------------------------------------------------------
# Editor integration
# -----------------------------------------------------------------------------

def _identity_graph():
    inp = gs.Variable("x", dtype=np.float32, shape=[2])
    mid = gs.Variable("m", dtype=np.float32, shape=[2])
    out = gs.Variable("y", dtype=np.float32, shape=[2])
    n1 = gs.Node("Identity", "id1", inputs=[inp], outputs=[mid])
    n2 = gs.Node("Identity", "id2", inputs=[mid], outputs=[out])
    return graph(nodes=[n1, n2], inputs=[inp], outputs=[out])


def test_apply_specs_excludes_default_edit():
    editor = OnnxGraphEditor(_identity_graph(), "g")
    editor.apply_specs(
        [EditSpec("RemoveIsNaN"), EditSpec("EliminateTranspose")],
        GraphEditHarness(exclude={"RemoveIsNaN"}),
    )
    # Nothing to assert on graph content here beyond it not raising; the
    # excluded RemoveIsNaN simply must not be constructed/applied.
    assert isinstance(editor.graph, gs.Graph)


def test_apply_specs_novel_edit_applied_once_per_editor():
    editor = OnnxGraphEditor(_identity_graph(), "g")
    calls = []
    orig = editor.apply_edit

    def _spy(edit):
        calls.append(edit.name)
        return orig(edit)

    editor.apply_edit = _spy
    h = GraphEditHarness(apply_flag=[EditSpec("FoldScalarMatMul")])
    editor.apply_specs([EditSpec("RemoveIsNaN")], h)
    editor.apply_specs([EditSpec("EliminateTranspose")], h)
    assert calls.count("FoldScalarMatMul") == 1
