# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph
from torq.graph_edit import OnnxGraphEditor
from torq.graph_edit.edits.transformer import (
    AddCurrLenInput,
    CollapseGQABroadcast,
    CombineKVCacheMixin,
    ConvertToStaticIndex,
    MaskFutureAttentionScores,
    ReplaceDynamicKVCache,
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
