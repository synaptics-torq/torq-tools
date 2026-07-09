# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Numeric-equivalence tests for the LFM2.5 custom-op replacements.

Each edit rewrites an ORT ``com.microsoft`` fused op into standard ONNX ops the
Torq compiler can lower.  These tests build the fused op, apply the edit, run the
rewritten graph under ONNX Runtime, and check the output against the original op
(when ORT can execute it) or against a numpy reference of the op's definition.

ORT 1.26 CPU coverage of the three fused ops differs:
  * SkipSimplifiedLayerNormalization — has a CPU kernel, so we compare the
    rewritten graph directly against the original op (true original-vs-replaced).
  * SimplifiedLayerNormalization — no CPU kernel; we compare against the op's
    mathematical definition (RMS norm), which is what the rewrite must reproduce.
  * GroupQueryAttention — the CPU kernel is not drop-in runnable here and the
    decomposition is intentionally an *expanded* form that matches the fp32
    reference model (runtime-recomputed RoPE), so we validate the rewritten graph
    end-to-end against a numpy reference of that intended attention math.

The rewrites emit opset-18 ops (e.g. ReduceMean with axes as an input), matching
the LFM2.5 export, so these graphs are built and executed at opset 18 rather than
through the opset-17 ``support.run_model`` helper.
"""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import pytest

from support.graph_edit import graph
from torq.utils.ort import make_cpu_session
from torq.graph_edit.edits.custom_ops import (
    ReplaceGroupQueryAttention,
    ReplaceSimplifiedLayerNorm,
    ReplaceSkipSimplifiedLayerNorm,
)


pytestmark = [pytest.mark.custom_ops, pytest.mark.ort]

OPSET = 18
COM_MICROSOFT = [onnx.helper.make_opsetid("com.microsoft", 1)]


def _run(model_or_graph, feeds):
    """Export (if needed), shape-infer and run under ORT with no fusion."""
    if isinstance(model_or_graph, gs.Graph):
        model = gs.export_onnx(
            model_or_graph.copy()
            .cleanup(remove_unused_graph_inputs=True, remove_unused_node_outputs=True)
            .toposort()
        )
    else:
        model = model_or_graph
    if model.ir_version > 11:
        model.ir_version = 11
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = make_cpu_session(model.SerializeToString(), sess_options=so)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, dict(feeds))))


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * weight).astype(np.float32)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# SimplifiedLayerNormalization  ->  RMS norm decomposition
# ---------------------------------------------------------------------------
def test_simplified_layernorm_matches_rms_reference():
    rng = np.random.default_rng(0)
    shape = [1, 3, 8]
    eps = 1e-5
    x_val = rng.standard_normal(shape).astype(np.float32)
    w_val = rng.standard_normal(shape[-1]).astype(np.float32)

    x = gs.Variable("x", np.float32, shape)
    w = gs.Constant("gamma", w_val)
    y = gs.Variable("y", np.float32, shape)
    node = gs.Node(
        "SimplifiedLayerNormalization", "sln",
        inputs=[x, w], outputs=[y],
        attrs={"epsilon": eps, "axis": -1}, domain="com.microsoft",
    )
    # Route the norm output through a downstream consumer (as in a real model,
    # where it feeds attention/FFN) so the edit rewires consumers rather than
    # repointing a bare graph output.
    y_out = gs.Variable("y_out", np.float32, shape)
    ident = gs.Node("Identity", "y_identity", inputs=[y], outputs=[y_out])
    g = graph(nodes=[node, ident], inputs=[x], outputs=[y_out], opset=OPSET)

    ReplaceSimplifiedLayerNorm(g, "test").transform(node)

    actual = _run(g, {"x": x_val})
    expected = _rms_norm(x_val, w_val, eps)
    np.testing.assert_allclose(list(actual.values())[0], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# SkipSimplifiedLayerNormalization  ->  (input+skip) then RMS norm
# ---------------------------------------------------------------------------
def _skip_original_model(shape, w_val, eps) -> onnx.ModelProto:
    x = gs.Variable("input", np.float32, shape)
    skip = gs.Variable("skip", np.float32, shape)
    w = gs.Constant("gamma", w_val)
    out = gs.Variable("out", np.float32, shape)
    mean = gs.Variable("mean")
    inv = gs.Variable("inv_std")
    skip_sum = gs.Variable("skip_sum", np.float32, shape)
    node = gs.Node(
        "SkipSimplifiedLayerNormalization", "ssln",
        inputs=[x, skip, w], outputs=[out, mean, inv, skip_sum],
        attrs={"epsilon": eps}, domain="com.microsoft",
    )
    g = gs.Graph(
        nodes=[node], inputs=[x, skip], outputs=[out, skip_sum],
        opset=OPSET, import_domains=COM_MICROSOFT,
    )
    return gs.export_onnx(g)


def test_skip_simplified_layernorm_matches_original_op():
    rng = np.random.default_rng(1)
    shape = [1, 4, 8]
    eps = 1e-5
    x_val = rng.standard_normal(shape).astype(np.float32)
    skip_val = rng.standard_normal(shape).astype(np.float32)
    w_val = rng.standard_normal(shape[-1]).astype(np.float32)
    feeds = {"input": x_val, "skip": skip_val}

    # Original fused op (ORT has a CPU kernel for this one).
    original = _run(_skip_original_model(shape, w_val, eps), feeds)
    orig_out, orig_skip_sum = list(original.values())

    # Rewritten graph.  Both fused-op outputs feed downstream consumers (as in a
    # real model) so the edit rewires consumers rather than repointing bare
    # graph outputs.
    x = gs.Variable("input", np.float32, shape)
    skip = gs.Variable("skip", np.float32, shape)
    w = gs.Constant("gamma", w_val)
    out = gs.Variable("out", np.float32, shape)
    mean = gs.Variable("mean")
    inv = gs.Variable("inv_std")
    skip_sum = gs.Variable("skip_sum", np.float32, shape)
    node = gs.Node(
        "SkipSimplifiedLayerNormalization", "ssln",
        inputs=[x, skip, w], outputs=[out, mean, inv, skip_sum],
        attrs={"epsilon": eps}, domain="com.microsoft",
    )
    out_final = gs.Variable("out_final", np.float32, shape)
    sum_final = gs.Variable("sum_final", np.float32, shape)
    id_out = gs.Node("Identity", "out_identity", inputs=[out], outputs=[out_final])
    id_sum = gs.Node("Identity", "sum_identity", inputs=[skip_sum], outputs=[sum_final])
    g = graph(nodes=[node, id_out, id_sum], inputs=[x, skip],
              outputs=[out_final, sum_final], opset=OPSET)
    ReplaceSkipSimplifiedLayerNorm(g, "test").transform(node)
    rewritten = _run(g, feeds)
    rw_out, rw_skip_sum = list(rewritten.values())

    # Rewritten == original op, and == the RMS-norm definition.
    np.testing.assert_allclose(rw_out, orig_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(rw_skip_sum, orig_skip_sum, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        rw_out, _rms_norm(x_val + skip_val, w_val, eps), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# GroupQueryAttention  ->  expanded RoPE + KV-concat + scaled-dot-product attn
# ---------------------------------------------------------------------------
def _gqa_reference(q, k, v, past_k, past_v, cos_cache, sin_cache, pos,
                   nh, kvh, hd, scale):
    B, S = q.shape[0], q.shape[1]
    # RoPE angles are recomputed from inv_freq = atan2(sin[1], cos[1]).
    inv_freq = np.arctan2(sin_cache[1], cos_cache[1]).astype(np.float32)
    angles = (inv_freq * np.float32(pos)).astype(np.float32)
    angles_full = np.concatenate([angles, angles]).astype(np.float32)
    cos = np.cos(angles_full).astype(np.float32)
    sin = np.sin(angles_full).astype(np.float32)

    def split_heads(t, heads):
        return t.reshape(B, S, heads, hd).transpose(0, 2, 1, 3)

    def rope(x):
        mul_cos = x * cos
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        return mul_cos + rotated * sin

    Q = rope(split_heads(q, nh))
    K = rope(split_heads(k, kvh))
    V = split_heads(v, kvh)

    present_k = np.concatenate([past_k, K], axis=2)
    present_v = np.concatenate([past_v, V], axis=2)

    rf = nh // kvh
    Kx = np.repeat(present_k, rf, axis=1)
    Vx = np.repeat(present_v, rf, axis=1)

    scores = (Q * scale) @ np.swapaxes(Kx, -1, -2)
    weights = _softmax(scores, axis=-1)
    out = weights @ Vx
    out = out.transpose(0, 2, 1, 3).reshape(B, S, nh * hd)
    return (out.astype(np.float32),
            present_k.astype(np.float32),
            present_v.astype(np.float32))


def test_group_query_attention_matches_reference():
    rng = np.random.default_rng(2)
    B, S = 1, 1
    nh, kvh, hd = 4, 2, 4
    past_len = 2
    pos = past_len + S            # seqlen_k value fed at runtime
    scale = 1.0 / (hd ** 0.5)

    # cos/sin caches built from a known inv_freq so the rewrite's
    # atan2(sin[1], cos[1]) recovers exactly that inv_freq.
    max_seq = 4
    inv_freq = np.array([0.5, 0.2], dtype=np.float32)      # length hd//2
    j = np.arange(max_seq, dtype=np.float32)[:, None]
    cos_cache = np.cos(j * inv_freq).astype(np.float32)     # (max_seq, hd//2)
    sin_cache = np.sin(j * inv_freq).astype(np.float32)

    q_val = rng.standard_normal((B, S, nh * hd)).astype(np.float32)
    k_val = rng.standard_normal((B, S, kvh * hd)).astype(np.float32)
    v_val = rng.standard_normal((B, S, kvh * hd)).astype(np.float32)
    pk_val = rng.standard_normal((B, kvh, past_len, hd)).astype(np.float32)
    pv_val = rng.standard_normal((B, kvh, past_len, hd)).astype(np.float32)
    slk_val = np.array([pos], dtype=np.int32)

    q = gs.Variable("q", np.float32, [B, S, nh * hd])
    k = gs.Variable("k", np.float32, [B, S, kvh * hd])
    v = gs.Variable("v", np.float32, [B, S, kvh * hd])
    pk = gs.Variable("past_key", np.float32, [B, kvh, past_len, hd])
    pv = gs.Variable("past_value", np.float32, [B, kvh, past_len, hd])
    slk = gs.Variable("seqlen_k", np.int32, [1])
    tsl = gs.Constant("total_seq_len", np.array([pos], dtype=np.int32))
    cosc = gs.Constant("cos_cache", cos_cache)
    sinc = gs.Constant("sin_cache", sin_cache)

    attn = gs.Variable("attn", np.float32)
    present_key = gs.Variable("present_key", np.float32)
    present_value = gs.Variable("present_value", np.float32)
    gqa = gs.Node(
        "GroupQueryAttention", "layer/attn/GroupQueryAttention",
        inputs=[q, k, v, pk, pv, slk, tsl, cosc, sinc],
        outputs=[attn, present_key, present_value],
        attrs={"num_heads": nh, "kv_num_heads": kvh}, domain="com.microsoft",
    )
    # The attn output must have a downstream consumer: the edit rewires attn
    # consumers but does not itself repoint graph outputs (real models feed it
    # into o_proj).  An Identity stands in for that projection.
    attn_out = gs.Variable("attn_out", np.float32)
    ident = gs.Node("Identity", "attn_identity", inputs=[attn], outputs=[attn_out])

    g = graph(
        nodes=[gqa, ident],
        inputs=[q, k, v, pk, pv, slk],
        outputs=[attn_out, present_key, present_value],
        opset=OPSET,
    )
    ReplaceGroupQueryAttention(
        g, "test", num_heads=nh, kv_num_heads=kvh, head_dim=hd
    ).transform(gqa)

    feeds = {
        "q": q_val, "k": k_val, "v": v_val,
        "past_key": pk_val, "past_value": pv_val, "seqlen_k": slk_val,
    }
    actual = _run(g, feeds)
    ref_attn, ref_pk, ref_pv = _gqa_reference(
        q_val, k_val, v_val, pk_val, pv_val,
        cos_cache, sin_cache, pos, nh, kvh, hd, scale,
    )

    # Match outputs by graph-output order (attn, present_key, present_value).
    got_attn, got_pk, got_pv = list(actual.values())
    np.testing.assert_allclose(got_attn, ref_attn, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_pk, ref_pk, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_pv, ref_pv, rtol=1e-4, atol=1e-5)
