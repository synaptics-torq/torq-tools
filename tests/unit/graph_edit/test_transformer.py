# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import clone_graph, graph, run_model
from torq.graph_edit import OnnxGraphEditor
from torq.graph_edit.edits.transformer import (
    AddCurrLenInput,
    CollapseGQABroadcast,
    CombineKVCacheMixin,
    ConvertToStaticIndex,
    MaskFutureAttentionScores,
    PrecomputeRotaryTables,
    ReplaceDynamicKVCache,
    ReplaceLegacyRotaryCache,
    RetargetCrossAttnKeyLayout,
)


pytestmark = pytest.mark.transformer


def test_replace_dynamic_kv_cache_replaces_output_concat_with_where():
    past = gs.Variable("past", dtype=np.float32, shape=[1, 1, 4, 2])
    new = gs.Variable("new", dtype=np.float32, shape=[1, 1, 4, 2])
    present = gs.Variable("present.0.key", dtype=np.float32, shape=[1, 1, 4, 2])
    concat = gs.Node("Concat", "present_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2})
    g = graph(nodes=[concat], inputs=[past, new], outputs=[present])

    edit = ReplaceDynamicKVCache(g, "unit", cur_len=gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1]), max_tokens=4)
    assert edit.match(concat)
    edit.transform(concat)

    assert concat.outputs == []
    assert {"Equal", "Where"}.issubset({node.op for node in g.nodes if node.outputs})


def test_replace_dynamic_kv_cache_uses_fixed_shape_chunk_placement():
    past = gs.Variable("past", dtype=np.float32, shape=[1, 1, 4, 2])
    new = gs.Variable("new", dtype=np.float32, shape=[1, 1, 2, 2])
    present = gs.Variable("present.0.key", dtype=np.float32, shape=[1, 1, 4, 2])
    concat = gs.Node("Concat", "present_concat", inputs=[past, new], outputs=[present], attrs={"axis": -2})
    g = graph(nodes=[concat], inputs=[past, new], outputs=[present])

    ReplaceDynamicKVCache(
        g, "unit", cur_len=gs.Variable("cur_len", dtype=np.int64, shape=[1]), max_tokens=4, chunk_len=2
    ).transform(concat)

    live_ops = {node.op for node in g.nodes if node.outputs}
    assert {"Add", "And", "Equal", "GreaterOrEqual", "Less", "MatMul", "Where"}.issubset(live_ops)
    assert "Pad" not in live_ops
    scatter_mask = next(node.outputs[0] for node in g.nodes if node.name == "present.0.key_scatter_mask")
    assert scatter_mask.dtype == onnx.TensorProto.BOOL
    assert scatter_mask.shape == [1, 1, 4, 2]
    placed = next(node.outputs[0] for node in g.nodes if node.name == "present.0.key_place_new_kv")
    assert placed.shape == [1, 1, 4, 2]


def test_mask_future_attention_scores_inserts_bias_add_when_producer_is_not_add():
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, 1, 4])
    pre_softmax = gs.Variable("pre_softmax", dtype=np.float32, shape=[1, 1, 1, 4])
    probs = gs.Variable("probs", dtype=np.float32, shape=[1, 1, 1, 4])
    identity = gs.Node("Identity", "scores_id", inputs=[scores], outputs=[pre_softmax])
    softmax = gs.Node("Softmax", "layer/self_attn/Softmax", inputs=[pre_softmax], outputs=[probs])
    g = graph(nodes=[identity, softmax], inputs=[scores], outputs=[probs])

    edit = MaskFutureAttentionScores(
        g,
        "unit",
        cur_len=gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1]),
        max_tokens=4,
        export_dtype=onnx.TensorProto.FLOAT,
    )
    assert edit.match(softmax)
    edit.transform(softmax)

    assert softmax.inputs[0].name == "layer/self_attn/Softmax_biased"
    assert {"LessOrEqual", "Where", "Add"}.issubset({node.op for node in g.nodes if node.outputs})


def test_mask_future_attention_scores_shape_fallback_requires_opt_in():
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, 1, 1, 4])
    probs = gs.Variable("probs", dtype=np.float32, shape=[1, 1, 1, 4])
    identity = gs.Node("Identity", "scores_id", inputs=[scores], outputs=[gs.Variable("pre", dtype=np.float32, shape=[1, 1, 1, 4])])
    softmax = gs.Node("Softmax", "decoder/layer0/Softmax_5", inputs=[identity.outputs[0]], outputs=[probs])
    g = graph(nodes=[identity, softmax], inputs=[scores], outputs=[probs])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[1, 1, 1, 1])

    default_edit = MaskFutureAttentionScores(g, "unit", cur_len=cur_len, max_tokens=4, export_dtype=onnx.TensorProto.FLOAT)
    assert not default_edit.match(softmax)

    opted_in_edit = MaskFutureAttentionScores(
        g, "unit", cur_len=cur_len, max_tokens=4, export_dtype=onnx.TensorProto.FLOAT, match_shape_fallback=True,
    )
    assert opted_in_edit.match(softmax)
    opted_in_edit.transform(softmax)
    assert {"LessOrEqual", "Where", "Add"}.issubset({node.op for node in g.nodes if node.outputs})


def test_add_curr_len_input_rewires_shape_gather_consumers():
    pkv = gs.Variable("past_key_values.0.key", dtype=np.float32, shape=[1, 1, 4, 2])
    shape_out = gs.Variable("shape_out", dtype=np.int64, shape=[4])
    gathered = gs.Variable("seq_len", dtype=np.int64, shape=[])
    out = gs.Variable("out", dtype=np.int64, shape=[])
    shape = gs.Node("Shape", "shape", inputs=[pkv], outputs=[shape_out])
    gather_node = gs.Node("Gather", "gather", inputs=[shape_out, gs.Constant("axis", np.array(2, dtype=np.int64))], outputs=[gathered])
    add = gs.Node("Add", "use", inputs=[gathered, gs.Constant("zero", np.array(0, dtype=np.int64))], outputs=[out])
    cur_len = gs.Variable("cur_len", dtype=np.int64, shape=[])
    g = graph(nodes=[shape, gather_node, add], inputs=[pkv, cur_len], outputs=[out])

    edit = AddCurrLenInput(g, "unit", cur_len=cur_len)
    assert edit.match(shape)
    edit.transform(shape)

    assert add.inputs[0] is cur_len
    assert shape.inputs == []
    assert gather_node.outputs == []


def test_convert_to_static_index_rewires_range_consumers_to_start():
    start = gs.Variable("start", dtype=np.int64, shape=[])
    limit = gs.Variable("limit", dtype=np.int64, shape=[])
    rng = gs.Variable("range", dtype=np.int64, shape=[1])
    out = gs.Variable("out", dtype=np.int64, shape=[1])
    add = gs.Node("Add", "limit_add", inputs=[start, gs.Constant("one", np.array(1, dtype=np.int64))], outputs=[limit])
    range_node = gs.Node("Range", "range", inputs=[start, limit, gs.Constant("delta", np.array(1, dtype=np.int64))], outputs=[rng])
    identity = gs.Node("Identity", "use", inputs=[rng], outputs=[out])
    g = graph(nodes=[add, range_node, identity], inputs=[start], outputs=[out])

    edit = ConvertToStaticIndex(g, "unit")
    assert edit.match(range_node)
    edit.transform(range_node)

    assert identity.inputs[0] is start
    assert range_node.outputs == []


def test_convert_to_static_index_builds_fixed_chunk_positions():
    start = gs.Variable("start", dtype=np.int64, shape=[])
    limit = gs.Variable("limit", dtype=np.int64, shape=[])
    rng = gs.Variable("range", dtype=np.int64, shape=[3])
    out = gs.Variable("out", dtype=np.int64, shape=[3])
    add = gs.Node("Add", "limit_add", inputs=[start, gs.Constant("three", np.array(3, dtype=np.int64))], outputs=[limit])
    range_node = gs.Node("Range", "range", inputs=[start, limit, gs.Constant("delta", np.array(1, dtype=np.int64))], outputs=[rng])
    identity = gs.Node("Identity", "use", inputs=[rng], outputs=[out])
    g = graph(nodes=[add, range_node, identity], inputs=[start], outputs=[out])

    ConvertToStaticIndex(g, "unit", chunk_len=3).transform(range_node)

    assert identity.inputs[0].shape == [3]
    assert range_node.outputs == []
    assert all(node.op != "Range" for node in g.nodes if node.outputs)


def test_retarget_cross_attention_key_layout_updates_producer_permutation_and_shape():
    inp = gs.Variable("pre_key", dtype=np.float32, shape=[1, 3, 2, 4])
    out = gs.Variable("layer.encoder.key", dtype=np.float32, shape=[1, 2, 3, 4])
    trans = gs.Node("Transpose", "key_transpose", inputs=[inp], outputs=[out], attrs={"perm": [0, 2, 1, 3]})
    g = graph(nodes=[trans], inputs=[inp], outputs=[out])

    edit = RetargetCrossAttnKeyLayout(g, "unit")
    assert edit.match(trans)
    edit.transform(trans)

    assert trans.attrs["perm"] == [0, 2, 3, 1]
    assert out.shape == [1, 2, 4, 3]


def test_retarget_cross_attention_key_layout_drops_consumer_transpose():
    key = gs.Variable("layer.encoder.key", dtype=np.float32, shape=[1, 2, 3, 4])
    transposed = gs.Variable("key_t", dtype=np.float32, shape=[1, 2, 4, 3])
    out = gs.Variable("out", dtype=np.float32, shape=[1, 2, 4, 3])
    trans = gs.Node("Transpose", "key_transpose", inputs=[key], outputs=[transposed], attrs={"perm": [0, 1, 3, 2]})
    identity = gs.Node("Identity", "use", inputs=[transposed], outputs=[out])
    g = graph(nodes=[trans, identity], inputs=[key], outputs=[out])

    edit = RetargetCrossAttnKeyLayout(g, "unit")
    assert edit.match(trans)
    edit.transform(trans)

    assert identity.inputs[0] is key
    assert key.shape == [1, 2, 4, 3]
    assert trans.outputs == []


def test_collapse_gqa_broadcast_replaces_unsqueeze_expand_reshape_chain():
    x = gs.Variable("x", dtype=np.float32, shape=[1, 1, 3, 4])
    unsq_out = gs.Variable("unsq_out", dtype=np.float32, shape=[1, 1, 1, 3, 4])
    exp_out = gs.Variable("exp_out", dtype=np.float32, shape=[1, 2, 1, 3, 4])
    final = gs.Variable("final", dtype=np.float32, shape=[1, 2, 3, 4])
    y = gs.Variable("y", dtype=np.float32, shape=[1, 2, 3, 4])
    unsq = gs.Node("Unsqueeze", "unsq", inputs=[x, gs.Constant("axes", np.array([1], dtype=np.int64))], outputs=[unsq_out])
    expand = gs.Node("Expand", "expand", inputs=[unsq_out, gs.Constant("exp_shape", np.array([1, 2, 1, 3, 4], dtype=np.int64))], outputs=[exp_out])
    reshape = gs.Node("Reshape", "reshape", inputs=[exp_out, gs.Constant("final_shape", np.array([1, 2, 3, 4], dtype=np.int64))], outputs=[final])
    identity = gs.Node("Identity", "use", inputs=[final], outputs=[y])
    g = graph(nodes=[unsq, expand, reshape, identity], inputs=[x], outputs=[y])

    edit = CollapseGQABroadcast(g, "unit")
    assert edit.match(unsq)
    edit.transform(unsq)

    assert identity.inputs[0].name == "final_expanded"
    assert unsq.outputs == []
    assert expand.outputs == []
    assert reshape.outputs == []


class KVEditor(OnnxGraphEditor, CombineKVCacheMixin):
    pass


def test_combine_kv_io_tensors_replaces_key_value_inputs_with_combined_input():
    key = gs.Variable("past_key_values.0.key", dtype=np.float32, shape=[1, 1, 2, 3])
    value = gs.Variable("past_key_values.0.value", dtype=np.float32, shape=[1, 1, 2, 3])
    present_key = gs.Variable("present.0.key", dtype=np.float32, shape=[1, 1, 2, 3])
    present_value = gs.Variable("present.0.value", dtype=np.float32, shape=[1, 1, 2, 3])
    key_use = gs.Node("Identity", "key_use", inputs=[key], outputs=[present_key])
    value_use = gs.Node("Identity", "value_use", inputs=[value], outputs=[present_value])
    g = graph(nodes=[key_use, value_use], inputs=[key, value], outputs=[present_key, present_value])
    editor = KVEditor(g, "unit")

    editor.combine_kv_io_tensors([1, 1, 2, 3])

    assert [i.name for i in editor.graph.inputs] == ["past_key_values.0.key_value"]
    assert key_use.inputs[0].name == "past_key_values.0.key_from_combined"
    assert value_use.inputs[0].name == "past_key_values.0.value_from_combined"
    assert [o.name for o in editor.graph.outputs] == ["present.0.key_value"]


def test_split_transformer_layers_chains_to_full_model():
    hidden, layers = 2, 3
    token = gs.Variable("token_embedding", dtype=np.float32, shape=[1, 1, hidden])
    positions = gs.Variable("position_ids", dtype=np.int64, shape=[1, 1])
    position = gs.Variable("position_f", dtype=np.float32)
    nodes = [gs.Node("Cast", "position_cast", attrs={"to": onnx.TensorProto.FLOAT}, inputs=[positions], outputs=[position])]
    caches, presents, residual = [], [], token
    for i in range(layers):
        past = gs.Variable(f"past_key_values.{i}.key_value", dtype=np.float32, shape=[1, 2, 4, hidden])
        present = gs.Variable(f"present.{i}.key_value", dtype=np.float32, shape=[1, 2, 4, hidden])
        mixed, out = gs.Variable(f"/model/layers.{i}/Add_output_0"), gs.Variable(f"/model/layers.{i}/Add_1_output_0")
        nodes += [
            gs.Node("Add", f"/model/layers.{i}/Add", inputs=[residual, position], outputs=[mixed]),
            gs.Node("Add", f"/model/layers.{i}/Add_1", inputs=[mixed, gs.Constant(f"bias_{i}", np.full(hidden, i, np.float32))], outputs=[out]),
            gs.Node("Add", f"/model/layers.{i}/cache_update", inputs=[past, out], outputs=[present]),
        ]
        caches.append(past)
        presents.append(present)
        residual = out
    last = gs.Variable("last_hidden_states", dtype=np.float32, shape=[1, 1, hidden])
    nodes.append(gs.Node("Identity", "final", inputs=[residual], outputs=[last]))
    editor = KVEditor(graph(nodes=nodes, inputs=[token, positions, *caches], outputs=[*presents, last]), "unit")

    part_a, part_b = editor.split_transformer_layers(split_at=1, num_layers=layers, hidden_size=hidden)

    rng = np.random.default_rng(0)
    feeds = {t.name: rng.standard_normal(t.shape).astype(np.float32) for t in [token, *caches]}
    feeds["position_ids"] = np.array([[5]], dtype=np.int64)
    full = run_model(editor.graph, feeds)
    a = run_model(part_a, {name: feeds[name] for name in ("token_embedding", "position_ids", "past_key_values.0.key_value")})
    b = run_model(part_b, {
        "/model/layers.0/Add_1_output_0": a["/model/layers.0/Add_1_output_0"],
        "position_ids": feeds["position_ids"],
        **{f"past_key_values.{i}.key_value": feeds[f"past_key_values.{i}.key_value"] for i in range(1, layers)},
    })
    for name, value in {**a, **b}.items():
        if name in full:
            np.testing.assert_allclose(value, full[name])
    assert set(full) <= set(a) | set(b)


def _rotary_graph(chunk: int) -> gs.Graph:
    """Two layers computing the same RoPE from shared int positions, plus a Sin of a float input."""
    positions = gs.Variable("position_ids", dtype=np.int64, shape=[1, chunk])
    pos3d = gs.Variable("pos3d", dtype=np.int64, shape=[1, 1, chunk])
    x = gs.Variable("x", dtype=np.float32, shape=[1, chunk, 8])
    inv_freq = np.array([1.0, 0.1, 0.01, 0.001], dtype=np.float32).reshape(1, 4, 1)
    nodes = [gs.Node("Unsqueeze", "Unsqueeze", inputs=[positions, gs.Constant("axis", np.array([1]))], outputs=[pos3d])]
    outputs = []
    for layer in range(2):
        p = f"/model/layers.{layer}/rope"
        t = [gs.Variable(f"{p}/{name}") for name in ("posf", "angles", "angles_t", "full")]
        nodes += [
            gs.Node("Cast", f"{p}/Cast", attrs={"to": onnx.TensorProto.FLOAT}, inputs=[pos3d], outputs=[t[0]]),
            gs.Node("MatMul", f"{p}/MatMul", inputs=[gs.Constant(f"{p}/inv_freq", inv_freq), t[0]], outputs=[t[1]]),
            gs.Node("Transpose", f"{p}/Transpose", attrs={"perm": [0, 2, 1]}, inputs=[t[1]], outputs=[t[2]]),
            gs.Node("Concat", f"{p}/Concat", attrs={"axis": -1}, inputs=[t[2], t[2]], outputs=[t[3]]),
        ]
        for op in ("Sin", "Cos"):
            raw, out = gs.Variable(f"{p}/{op}_out", dtype=np.float32), gs.Variable(f"{p}/{op}_x", dtype=np.float32, shape=[1, chunk, 8])
            nodes += [gs.Node(op, f"{p}/{op}", inputs=[t[3]], outputs=[raw]), gs.Node("Mul", f"{p}/{op}_mul", inputs=[raw, x], outputs=[out])]
            outputs.append(out)
    float_sin = gs.Variable("x_sin", dtype=np.float32, shape=[1, chunk, 8])
    nodes.append(gs.Node("Sin", "x_sin", inputs=[x], outputs=[float_sin]))
    return graph(nodes=nodes, inputs=[positions, x], outputs=outputs + [float_sin])


@pytest.mark.parametrize("chunk", [1, 3])
def test_precompute_rotary_tables_matches_computed_rope(chunk):
    g = _rotary_graph(chunk)
    before = clone_graph(g)
    edit = PrecomputeRotaryTables(g, "unit", max_positions=1024)
    for node in list(g.nodes):
        edit(node)
    g.cleanup().toposort()

    assert [node.name for node in g.nodes if node.op in ("Sin", "Cos")] == ["x_sin"]
    tables = [node.inputs[1].values for node in g.nodes if PrecomputeRotaryTables.TABLE_NAME in node.name]
    assert [table.shape for table in tables] == [(32, 8)] * 4  # cos/sin of 32 high and 32 low digits, shared by both layers
    x = np.random.default_rng(0).standard_normal((1, chunk, 8)).astype(np.float32)
    for start in (0, 300, 1024 - chunk):
        feeds = {"position_ids": np.arange(start, start + chunk).reshape(1, chunk), "x": x}
        for expected, actual in zip(run_model(before, feeds).values(), run_model(g, feeds).values()):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_precompute_rotary_tables_keeps_exact_bf16_capacities():
    g = _rotary_graph(1)
    edit = PrecomputeRotaryTables(g, "unit", max_positions=PrecomputeRotaryTables.EXACT_BF16_POSITIONS)
    assert not any(edit.match(node) for node in g.nodes)


def test_replace_legacy_rotary_cache_rejects_short_table():
    positions = gs.Variable("position_ids", dtype=np.int64, shape=[1, 1])
    sliced, out = gs.Variable("sliced", dtype=np.float32), gs.Variable("cos", dtype=np.float32)
    slice_node = gs.Node("Slice", "/rotary_emb/Slice", inputs=[
        gs.Constant("cos_cached", np.zeros((4, 2), np.float32)), gs.Constant("starts", np.array([0])), gs.Constant("ends", np.array([4])),
    ], outputs=[sliced])
    g = graph(nodes=[slice_node, gs.Node("Gather", "gather", inputs=[sliced, positions], outputs=[out])], inputs=[positions], outputs=[out])

    edit = ReplaceLegacyRotaryCache(g, "unit", max_len=8)
    assert edit.match(slice_node)
    with pytest.raises(ValueError, match="fewer than max_len=8"):
        edit.transform(slice_node)
