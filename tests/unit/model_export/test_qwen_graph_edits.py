# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from torq.graph_edit.harness import GraphEditHarness
from torq.models.qwen import add_qwen_export_args
from torq.models.qwen import export as qwen


def _args(*argv):
    parser = argparse.ArgumentParser()
    add_qwen_export_args(parser)
    return parser.parse_args(argv)


def _cast_model():
    position = gs.Variable('position_ids', np.int64, [1, 1])
    cast = gs.Variable('cast', np.int64, [1, 1])
    output = gs.Variable('output', np.int64, [1, 1])
    graph = gs.Graph(nodes=[
        gs.Node('Cast', 'redundant', inputs=[position], outputs=[cast], attrs={'to': onnx.TensorProto.INT64}),
        gs.Node('Identity', 'output', inputs=[cast], outputs=[output]),
    ], inputs=[position], outputs=[output], opset=17)
    return gs.export_onnx(graph)


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen.AutoConfig, 'from_pretrained', Mock(return_value=SimpleNamespace(
        hidden_size=8, vocab_size=16, num_attention_heads=2, num_key_value_heads=1,
    )))
    result = qwen.QwenModelExporter(models_dir=tmp_path, split_lm_head=True)
    result._export_dir = tmp_path
    return result


def test_view_plan_merges_overrides_and_skips_export(exporter, monkeypatch, capsys):
    monkeypatch.setattr(qwen, 'QwenModelExporter', Mock(return_value=exporter))
    download = Mock(side_effect=AssertionError('View must not download GPTQ'))
    monkeypatch.setattr(qwen, '_resolve_gptq_checkpoint', download)
    export = Mock(side_effect=AssertionError('View must not export'))
    monkeypatch.setattr(exporter, 'export_onnx', export)
    qwen.export_qwen_from_args(_args(
        '--view-graph-edits', '--apply-graph-edit', 'PrecomputeRotaryTables:512',
        '--exclude-graph-edit', 'RemoveRedundantCasts',
    ))
    plan = capsys.readouterr().out
    assert 'PrecomputeRotaryTables:[512]' in plan
    assert 'RemoveRedundantCasts' not in plan
    assert 'SplitRotarySlices' in plan
    assert 'ExtractConstantLUT' in plan
    download.assert_not_called()
    export.assert_not_called()


@pytest.mark.parametrize('exclude', [False, True])
def test_static_exclusion_changes_graph_and_dumps(exporter, tmp_path, exclude):
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(_args(
        '--dump-after-edit', *(['--exclude-graph-edit', 'RemoveRedundantCasts'] if exclude else []),
    )))
    result = exporter._make_model_static(_cast_model())
    assert any(node.name == 'redundant' for node in result.graph.node) == exclude
    dumps = list((tmp_path / 'intermediates').glob('*RemoveRedundantCasts*.onnx'))
    assert bool(dumps) != exclude


@pytest.mark.parametrize("from_file", [False, True])
def test_apply_changes_post_static_graph(exporter, tmp_path, from_file):
    # The Qwen patch block does not normally remove redundant casts.
    edits = tmp_path / 'edits.yaml'
    edits.write_text('- RemoveRedundantCasts\n')
    exporter._split_lm_head = False
    exporter._read_only_kv_cache = False
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(_args(
        *(['--apply-graph-edits-from-file', str(edits)] if from_file else ['--apply-graph-edit', 'RemoveRedundantCasts']),
    )))
    path = tmp_path / 'model.onnx'
    onnx.save(_cast_model(), path)
    exporter._patch_static_model(path)
    assert all(node.name != 'redundant' for node in onnx.load(path).graph.node)


def test_plan_respects_export_modes_and_head_dim(exporter):
    exporter._config.head_dim = 16
    blocks = exporter.graph_edit_blocks()
    assert blocks['model.patch (attention)'][-1].args == (2, 1, 16)
    exporter._split_lm_head = False
    exporter._read_only_kv_cache = False
    exporter._static_models = False
    blocks = exporter.graph_edit_blocks()
    assert list(blocks) == ['model.patch', 'model.patch (attention)']
    assert [spec.name for spec in blocks['model.patch (attention)']] == ['RemoveIsNaN']


@pytest.mark.parametrize("exclude", [False, True])
def test_converted_rotary_block_honors_exclusion(exporter, exclude):
    x = gs.Variable("x", np.float32, [1, 4])
    halves = [gs.Variable(f"half{i}", np.float32, [1, 2]) for i in range(2)]
    neg = gs.Variable("neg", np.float32, [1, 2])
    out = gs.Variable("out", np.float32, [1, 4])
    nodes = [gs.Node("Slice", f"slice{i}", inputs=[
        x, gs.Constant(f"start{i}", np.array([i * 2], np.int64)),
        gs.Constant(f"end{i}", np.array([(i + 1) * 2], np.int64)),
        gs.Constant(f"axis{i}", np.array([1], np.int64)),
    ], outputs=[halves[i]]) for i in range(2)]
    nodes += [gs.Node("Neg", inputs=[halves[1]], outputs=[neg]),
              gs.Node("Concat", inputs=[neg, halves[0]], outputs=[out], attrs={"axis": 1})]
    editor = qwen.DecoderGraphEditor(gs.Graph(nodes=nodes, inputs=[x], outputs=[out], opset=17))
    harness = GraphEditHarness(exclude={"SplitRotarySlices"} if exclude else set())
    editor.apply_specs(exporter.graph_edit_blocks()["transformer.convert"], harness)
    assert sum(node.op == "Split" for node in editor.graph.nodes) == (0 if exclude else 1)


def test_static_context_uses_graph_after_shape_inference(exporter, monkeypatch):
    from torq.graph_edit.edits import ReplaceDynamicKVCache

    original_match = ReplaceDynamicKVCache.match

    def match(edit, node):
        position = edit.cur_len.inputs[0].inputs[0]
        assert any(position is inp for inp in edit.graph.inputs)
        return original_match(edit, node)

    monkeypatch.setattr(ReplaceDynamicKVCache, "match", match)
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(_args(
        '--apply-graph-edit', 'EliminateExpand:[[]]',
    )))
    exporter._make_model_static(_cast_model())
