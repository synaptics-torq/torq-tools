# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Static, Torq-compilable rewrites of the onnx-community EmbeddingGemma-2 graphs.

Vision (``vision_encoder.onnx``):
    pixel_patches [1, N, 768] -> image_embeds [1, N/9, 512]
    for one fixed patch grid. The (x, y) patch positions become a constant, so the
    position-table lookups, the axial RoPE tables, the 3x3 pooling matrix and the
    padding strip all constant-fold.

Text (``model.onnx``):
    inputs_embeds [1, S, 512] + attention_mask [1, S] -> last_hidden_state [1, S, 768]
    The graph is cut after the token-embedding gather and the multimodal merge (done
    on the host) and before the pooling tail (also done on the host).
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ...graph_edit.edits.custom_ops import ReplaceSimplifiedLayerNorm, ReplaceSkipSimplifiedLayerNorm

logger = logging.getLogger("embedding_gemma2.static")

VISION_HEADS, VISION_HEAD_DIM = 12, 64


def _apply(edit_cls, graph: gs.Graph, name: str) -> int:
    edit = edit_cls(graph, name)
    nodes = [n for n in graph.nodes if edit.match(n)]
    for n in nodes:
        edit(n)
    graph.cleanup().toposort()
    return len(nodes)


def add_unit_norm_weights(graph: gs.Graph) -> int:
    """Give weight-less (``with_scale=False``) SimplifiedLayerNormalization nodes an
    explicit ones weight so the generic replacement edit applies; the Mul by ones is
    removed again by :func:`remove_mul_by_one`."""
    n = 0
    for node in graph.nodes:
        if node.op == "SimplifiedLayerNormalization" and len(node.inputs) < 2:
            dim = node.inputs[0].shape[-1] if node.inputs[0].shape else None
            if not isinstance(dim, int):
                raise ValueError(f"{node.name}: unknown norm width")
            node.inputs.append(gs.Constant(f"{node.name}/unit_w", np.ones(dim, np.float32)))
            n += 1
    return n


def remove_mul_by_one(graph: gs.Graph) -> int:
    """Drop ``Mul(x, c)`` where ``c`` is all ones and does not broadcast ``x`` up."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Mul" or len(node.inputs) != 2:
            continue
        for ci in (0, 1):
            c, x = node.inputs[ci], node.inputs[1 - ci]
            if not isinstance(c, gs.Constant) or not np.all(c.values == 1):
                continue
            if c.values.ndim > len(x.shape or []) or not x.shape:
                continue
            out = node.outputs[0]
            for consumer in list(out.outputs):
                consumer.inputs = [x if i is out else i for i in consumer.inputs]
            graph.outputs = [x if o is out else o for o in graph.outputs]
            node.outputs.clear()
            n += 1
            break
    graph.cleanup().toposort()
    return n


def gemm_transb_to_matmul(graph: gs.Graph) -> int:
    """Gemm(A, B^T [+C]) with a constant B -> MatMul(A, B_t) [+ Add(C)]."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Gemm" or not isinstance(node.inputs[1], gs.Constant):
            continue
        if float(node.attrs.get("alpha", 1.0)) != 1.0 or int(node.attrs.get("transA", 0)):
            continue
        w = node.inputs[1].values
        if int(node.attrs.get("transB", 0)):
            w = np.ascontiguousarray(w.T)
        out = node.outputs[0]
        mm = gs.Variable(f"{node.name}/mm", dtype=out.dtype)
        graph.layer(op="MatMul", name=f"{node.name}/matmul",
                    inputs=[node.inputs[0], gs.Constant(f"{node.name}/w", w)], outputs=[mm])
        y = mm
        if len(node.inputs) > 2 and node.inputs[2] is not None and getattr(node.inputs[2], "name", ""):
            y = gs.Variable(f"{node.name}/biased", dtype=out.dtype)
            graph.layer(op="Add", name=f"{node.name}/add", inputs=[mm, node.inputs[2]], outputs=[y])
        _replace_output(graph, node, y)
        n += 1
    graph.cleanup().toposort()
    return n


def _replace_output(graph: gs.Graph, node: gs.Node, new: gs.Variable):
    out = node.outputs[0]
    for consumer in list(out.outputs):
        consumer.inputs = [new if i is out else i for i in consumer.inputs]
    graph.outputs = [new if o is out else o for o in graph.outputs]
    node.outputs.clear()


def fold_const_where(graph: gs.Graph) -> int:
    """Where(c, a, b) with a constant all-True / all-False condition -> a / b, when the
    chosen branch already has the output shape."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Where" or not isinstance(node.inputs[0], gs.Constant):
            continue
        c = node.inputs[0].values
        pick = node.inputs[1] if c.all() else node.inputs[2] if not c.any() else None
        if pick is None or not isinstance(pick, gs.Variable) or pick.shape != node.outputs[0].shape:
            continue
        _replace_output(graph, node, pick)
        n += 1
    graph.cleanup().toposort()
    return n


def drop_identity_gathernd(graph: gs.Graph) -> int:
    """GatherND(x [1, N, D], [[0, 0], [0, 1], ... [0, N-1]]) selects every row in order:
    bypass it so the tail stays rank-3 ([1, N, D]); rank-2 norms are fragile on Torq."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "GatherND" or not isinstance(node.inputs[1], gs.Constant):
            continue
        x, idx = node.inputs[0], node.inputs[1].values
        if not x.shape or len(x.shape) != 3 or x.shape[0] != 1 or int(node.attrs.get("batch_dims", 0)):
            continue
        N = x.shape[1]
        if idx.shape != (N, 2) or not np.array_equal(idx, np.stack([np.zeros(N, int), np.arange(N)], 1)):
            continue
        _replace_output(graph, node, x)
        n += 1
    graph.cleanup().toposort()
    return n


def reinfer(graph: gs.Graph) -> gs.Graph:
    """Drop all intermediate shapes and re-run shape inference."""
    for t in graph.tensors().values():
        if isinstance(t, gs.Variable) and t not in graph.inputs:
            t.shape = None
    for o in graph.outputs:
        o.shape = None
    return gs.import_onnx(onnx.shape_inference.infer_shapes(gs.export_onnx(graph)))


def gather_perm_to_slices(graph: gs.Graph) -> int:
    """Gather(x, perm, axis=last) with a constant index made of contiguous runs
    (e.g. the RoPE rotate-half permutation) -> Concat of Slices."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Gather" or not isinstance(node.inputs[1], gs.Constant):
            continue
        x, idx = node.inputs[0], node.inputs[1].values
        if not x.shape or idx.ndim != 1:
            continue
        axis = int(node.attrs.get("axis", 0)) % len(x.shape)
        if axis != len(x.shape) - 1 or sorted(idx.tolist()) != list(range(x.shape[axis])):
            continue
        runs, start = [], 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] != idx[i - 1] + 1:
                runs.append((int(idx[start]), int(idx[i - 1]) + 1))
                start = i
        parts = []
        for ri, (a, b) in enumerate(runs):
            o = gs.Variable(f"{node.name}/run{ri}", dtype=node.outputs[0].dtype)
            graph.layer(op="Slice", name=f"{node.name}/slice{ri}",
                        inputs=[x, gs.Constant(f"{node.name}/s{ri}", np.array([a], np.int64)),
                                gs.Constant(f"{node.name}/e{ri}", np.array([b], np.int64)),
                                gs.Constant(f"{node.name}/a{ri}", np.array([axis], np.int64))],
                        outputs=[o])
            parts.append(o)
        cat = gs.Variable(f"{node.name}/perm", dtype=node.outputs[0].dtype, shape=node.outputs[0].shape)
        graph.layer(op="Concat", name=f"{node.name}/concat", inputs=parts, outputs=[cat], attrs={"axis": axis})
        _replace_output(graph, node, cat)
        n += 1
    graph.cleanup().toposort()
    return n


def simplify(model: onnx.ModelProto, input_shapes: dict[str, list[int]] | None = None) -> onnx.ModelProto:
    import onnxsim

    model, ok = onnxsim.simplify(model, overwrite_input_shapes=input_shapes or None)
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    return model


def op_histogram(model: onnx.ModelProto) -> dict[str, int]:
    return dict(Counter((f"{n.domain}::" if n.domain else "") + n.op_type
                        for n in model.graph.node).most_common())


def vision_positions(grid: tuple[int, int]) -> np.ndarray:
    gh, gw = grid
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.int64)[None]


def _producer(t):
    return t.inputs[0] if t.inputs else None


def decompose_padded_mha(graph: gs.Graph) -> int:
    """Decompose the exporter's padded ``com.microsoft.MultiHeadAttention``.

    The onnx-community vision export widens every head from 64 to 68 dims to fold the
    key-padding mask into Q.K^T: q is Pad(q, value=1), k is Concat(k, mask_term), v is
    Pad(v, 0), and the output is Reshape -> Slice(:64) -> Reshape. Every score therefore
    gets ``sum(mask_term)`` of its key added. Rewrite to plain 64-wide heads
    (MatMul/Softmax/MatMul) and keep the mask term only if it is non-zero (as an
    additive [1, H, 1, S] key bias).
    """
    n = 0
    for node in list(graph.nodes):
        if node.op != "MultiHeadAttention":
            continue
        heads = int(node.attrs["num_heads"])
        scale = float(node.attrs.get("scale", 1.0))
        q_rs, k_rs, v_rs = (_producer(t) for t in node.inputs[:3])
        q_pad, k_cat, v_pad = (_producer(r.inputs[0]) for r in (q_rs, k_rs, v_rs))
        if not (q_pad.op == "Pad" and k_cat.op == "Concat" and v_pad.op == "Pad"):
            raise NotImplementedError(f"{node.name}: unexpected MHA input pattern")
        q4, k4, v4 = q_pad.inputs[0], k_cat.inputs[0], v_pad.inputs[0]  # [1, S, H, D]
        mask_term = k_cat.inputs[1]
        if not isinstance(mask_term, gs.Constant):
            raise NotImplementedError(f"{node.name}: mask term is not constant ({mask_term.name})")
        key_bias = mask_term.values.astype(np.float32).sum(-1)  # [1, S, H]
        if not np.all(np.asarray(q_pad.inputs[2].values if len(q_pad.inputs) > 2 else 0) == 1):
            raise NotImplementedError(f"{node.name}: q pad value is not 1")

        out_rs = next(iter(node.outputs[0].outputs))
        sl = next(iter(out_rs.outputs[0].outputs))
        merge = next(iter(sl.outputs[0].outputs))
        if not (out_rs.op == "Reshape" and sl.op == "Slice" and merge.op == "Reshape"):
            raise NotImplementedError(f"{node.name}: unexpected MHA output pattern")
        final = merge.outputs[0]

        b = node.name
        f32 = np.float32

        def layer(op, ins, nm, **attrs):
            o = gs.Variable(f"{b}/{nm}", dtype=f32)
            graph.layer(op=op, name=f"{b}/{nm}", inputs=ins, outputs=[o], attrs=attrs)
            return o

        qh = layer("Transpose", [q4], "q_t", perm=[0, 2, 1, 3])
        kT = layer("Transpose", [k4], "k_t", perm=[0, 2, 3, 1])
        vh = layer("Transpose", [v4], "v_t", perm=[0, 2, 1, 3])
        s = layer("MatMul", [qh, kT], "qk")
        if scale != 1.0:
            s = layer("Mul", [s, gs.Constant(f"{b}/scale", np.array(scale, f32))], "scaled")
        if np.any(key_bias != 0):
            bias = np.ascontiguousarray(key_bias.transpose(0, 2, 1)[:, :, None, :])  # [1, H, 1, S]
            s = layer("Add", [s, gs.Constant(f"{b}/key_bias", bias)], "masked")
        p = layer("Softmax", [s], "softmax", axis=-1)
        av = layer("MatMul", [p, vh], "av")
        avt = layer("Transpose", [av], "av_t", perm=[0, 2, 1, 3])
        d = final.shape[-1] if final.shape else None
        merged = layer("Reshape", [avt, gs.Constant(f"{b}/merge_shape", np.array([1, -1, d], np.int64))], "merged")
        for consumer in list(final.outputs):
            consumer.inputs = [merged if i is final else i for i in consumer.inputs]
        graph.outputs = [merged if o is final else o for o in graph.outputs]
        merge.outputs.clear()
        node.outputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def build_static_vision(src: str | Path, grid: tuple[int, int]) -> onnx.ModelProto:
    """Static vision encoder for one patch grid (rows, cols); both must be multiples of 3."""
    gh, gw = grid
    if gh % 3 or gw % 3:
        raise ValueError(f"patch grid {grid} must be a multiple of the 3x3 pooling kernel")
    n_patches = gh * gw
    graph = gs.import_onnx(onnx.load(str(src)))

    pos_in = next(i for i in graph.inputs if i.name == "pixel_position_ids")
    pos_const = gs.Constant("pixel_position_ids_const", vision_positions(grid))
    for node in list(pos_in.outputs):
        node.inputs = [pos_const if i is pos_in else i for i in node.inputs]
    pix = next(i for i in graph.inputs if i.name == "pixel_values")
    pix.shape = [1, n_patches, 768]
    graph.inputs = [pix]
    graph.cleanup().toposort()

    model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    graph = gs.import_onnx(model)
    logger.info("vision: unit norm weights added: %d", add_unit_norm_weights(graph))
    logger.info("vision: SimplifiedLayerNorm replaced: %d",
                _apply(ReplaceSimplifiedLayerNorm, graph, "vision"))

    # Fold everything that only depends on the (now constant) positions, then strip the
    # padded-head attention trick and fold again.
    model = simplify(gs.export_onnx(graph), {"pixel_values": [1, n_patches, 768]})
    graph = gs.import_onnx(model)
    logger.info("vision: MHA decomposed: %d", decompose_padded_mha(graph))
    model = simplify(gs.export_onnx(graph), {"pixel_values": [1, n_patches, 768]})
    graph = gs.import_onnx(model)
    logger.info("vision: Gemm -> MatMul: %d", gemm_transb_to_matmul(graph))
    logger.info("vision: Mul-by-one removed: %d", remove_mul_by_one(graph))
    logger.info("vision: const-cond Where folded: %d", fold_const_where(graph))
    logger.info("vision: identity GatherND dropped: %d", drop_identity_gathernd(graph))
    logger.info("vision: Gather permutation -> Slice/Concat: %d", gather_perm_to_slices(graph))

    graph = reinfer(graph)
    graph.inputs[0].name = "pixel_patches"
    out = graph.outputs[0]
    if out.shape and len(out.shape) == 2:
        rs = gs.Variable("image_embeds", dtype=out.dtype, shape=[1] + list(out.shape))
        graph.layer(op="Unsqueeze", name="image_embeds/unsqueeze",
                    inputs=[out, gs.Constant("image_embeds/axes", np.array([0], np.int64))], outputs=[rs])
        graph.outputs = [rs]
    else:
        out.name = "image_embeds"
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


TEXT_HIDDEN = 512
EMBEDS_TENSOR = "/model/multimodal_merge/Reshape_embeds/output_0"


def _find_var(graph: gs.Graph, name: str) -> gs.Variable:
    t = graph.tensors().get(name)
    if t is None:
        raise KeyError(f"tensor {name!r} not found")
    return t


def expand_rotary_embedding(graph: gs.Graph, seq_len: int, inv_freq: dict[str, np.ndarray]) -> int:
    """``com.microsoft::RotaryEmbedding`` (non-interleaved, positions 0..S-1) ->
    ``x * C + rotate_half(x) * S`` with constant C = [cos, cos], S = [-sin, sin].

    ``inv_freq`` maps a cos-cache tensor name to its inverse frequencies [half]; the head
    size is ``2 * half`` and the head count follows from the input width."""
    n = 0
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    for node in list(graph.nodes):
        if node.op != "RotaryEmbedding":
            continue
        if int(node.attrs.get("interleaved", 0)):
            raise NotImplementedError(f"{node.name}: interleaved RoPE")
        x, _pos_ids, cos_in = node.inputs[0], node.inputs[1], node.inputs[2]
        freqs = inv_freq[cos_in.name].reshape(-1).astype(np.float32)
        half = freqs.size
        head = 2 * half
        width = x.shape[-1]
        heads = width // head
        ang = pos * freqs[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        c = np.concatenate([cos, cos], -1)[None, :, None, :].astype(np.float32)   # [1, S, 1, D]
        s = np.concatenate([-sin, sin], -1)[None, :, None, :].astype(np.float32)
        b = node.name
        f32 = np.float32

        def layer(op, ins, nm, **attrs):
            o = gs.Variable(f"{b}/{nm}", dtype=f32)
            graph.layer(op=op, name=f"{b}/{nm}", inputs=ins, outputs=[o], attrs=attrs)
            return o

        x4 = layer("Reshape", [x, gs.Constant(f"{b}/s4", np.array([1, seq_len, heads, head], np.int64))], "x4")
        x1 = layer("Slice", [x4, gs.Constant(f"{b}/z", np.array([0], np.int64)),
                             gs.Constant(f"{b}/h", np.array([half], np.int64)),
                             gs.Constant(f"{b}/ax", np.array([3], np.int64))], "x1")
        x2 = layer("Slice", [x4, gs.Constant(f"{b}/h2", np.array([half], np.int64)),
                             gs.Constant(f"{b}/d", np.array([head], np.int64)),
                             gs.Constant(f"{b}/ax2", np.array([3], np.int64))], "x2")
        rot = layer("Concat", [x2, x1], "rot", axis=3)
        xc = layer("Mul", [x4, gs.Constant(f"{b}/cos", c)], "xc")
        rs = layer("Mul", [rot, gs.Constant(f"{b}/sin", s)], "rs")
        y4 = layer("Add", [xc, rs], "y4")
        y = layer("Reshape", [y4, gs.Constant(f"{b}/s3", np.array([1, seq_len, width], np.int64))], "y")
        _replace_output(graph, node, y)
        n += 1
    graph.cleanup().toposort()
    return n


def flatten_grouped_scores(graph: gs.Graph, bias: gs.Variable) -> int:
    """GQA scores are reshaped to 5D [B, kv, g, S, S] just to add the mask bias and
    softmax, then reshaped back to [B, kv, g*S, S]. With a key-only bias [1, 1, 1, S]
    add it to the 4D scores directly and drop both reshapes."""
    n = 0
    for sm in [x for x in graph.nodes if x.op == "Softmax"]:
        add = sm.inputs[0].inputs[0] if sm.inputs[0].inputs else None
        if add is None or add.op != "Add":
            continue
        rs5 = add.inputs[0].inputs[0] if add.inputs[0].inputs else None
        back = next(iter(sm.outputs[0].outputs), None)
        if rs5 is None or rs5.op != "Reshape" or back is None or back.op != "Reshape":
            continue
        scores4 = rs5.inputs[0]
        b = sm.name
        added = gs.Variable(f"{b}/biased", dtype=np.float32)
        graph.layer(op="Add", name=f"{b}/add_bias", inputs=[scores4, bias], outputs=[added])
        probs = gs.Variable(f"{b}/probs", dtype=np.float32)
        graph.layer(op="Softmax", name=f"{b}/softmax", inputs=[added], outputs=[probs], attrs={"axis": -1})
        _replace_output(graph, back, probs)
        n += 1
    graph.cleanup().toposort()
    return n


def build_static_text(src: str | Path, seq_len: int) -> tuple[onnx.ModelProto, np.ndarray]:
    """Static text body: inputs_embeds [1, S, 512] + attention_bias [1, 1, 1, S] ->
    last_hidden_state [1, S, 768]. Returns (model, token-embedding LUT [V, 512] fp32)."""
    if seq_len > 513:
        raise NotImplementedError("S > 513 needs the banded sliding-window mask")
    graph = gs.import_onnx(onnx.load(str(src)))
    table = graph.tensors().get("model.embed_tokens.weight")
    if table is None:  # dequantized q4 graph: the table is the Gather's constant input
        gather = next(n for n in graph.nodes if n.op == "Gather" and n.name.startswith("/model/embed_tokens/"))
        table = gather.inputs[0]
    lut = table.values.astype(np.float32)
    inv_freq = {}
    for kind in ("full_attention", "sliding_attention"):
        inv_freq[f"/model/rotary_emb/{kind}/Cos/output_0"] = \
            graph.tensors()[f"model.rotary_emb.{kind}_inv_freq"].values

    # New inputs: merged embeddings (host-side gather + media scatter) and the additive
    # key-padding bias (host-side from the attention mask).
    embeds_old = _find_var(graph, EMBEDS_TENSOR)
    embeds = gs.Variable("inputs_embeds", dtype=np.float32, shape=[1, seq_len, TEXT_HIDDEN])
    for consumer in list(embeds_old.outputs):
        consumer.inputs = [embeds if i is embeds_old else i for i in consumer.inputs]
    bias = gs.Variable("attention_bias", dtype=np.float32, shape=[1, 1, 1, seq_len])
    bias5 = gs.Variable("attention_bias_5d", dtype=np.float32)
    graph.layer(op="Unsqueeze", name="attention_bias/unsqueeze",
                inputs=[bias, gs.Constant("attention_bias/axes", np.array([1], np.int64))], outputs=[bias5])
    for kind in ("full", "sliding"):  # identical for S <= 513: every key is within the band
        old = _find_var(graph, f"/model/attention_bias/{kind}/Unsqueeze_5d/output_0")
        for consumer in list(old.outputs):
            consumer.inputs = [bias5 if i is old else i for i in consumer.inputs]

    hidden = _find_var(graph, "last_hidden_state")
    graph.inputs = [embeds, bias]
    graph.outputs = [hidden]
    graph.cleanup().toposort()

    # Remaining shape glue only depends on S: pin the score-shape Concat input.
    seq_1d = graph.tensors().get("/model/shared_dims/attention_mask_seq_len_1d/output_0")
    if seq_1d is not None:
        const = gs.Constant("seq_len_1d", np.array([seq_len], np.int64))
        for consumer in list(seq_1d.outputs):
            consumer.inputs = [const if i is seq_1d else i for i in consumer.inputs]
        graph.cleanup().toposort()

    logger.info("text: RotaryEmbedding expanded: %d", expand_rotary_embedding(graph, seq_len, inv_freq))
    logger.info("text: SkipSimplifiedLayerNorm replaced: %d",
                _apply(ReplaceSkipSimplifiedLayerNorm, graph, "text"))
    logger.info("text: SimplifiedLayerNorm replaced: %d", _apply(ReplaceSimplifiedLayerNorm, graph, "text"))
    model = simplify(gs.export_onnx(graph))
    graph = gs.import_onnx(model)
    bias_in = next(i for i in graph.inputs if i.name == "attention_bias")
    logger.info("text: 5D grouped scores flattened: %d", flatten_grouped_scores(graph, bias_in))
    logger.info("text: Mul-by-one removed: %d", remove_mul_by_one(graph))
    graph = reinfer(graph)
    model = simplify(gs.export_onnx(graph))
    return model, lut


def where_const_mask_to_bias(graph: gs.Graph) -> int:
    """Where(c, v, x) with constant condition c and scalar fill v -> x + bias, where
    bias = v at c and 0 elsewhere (attention masking with a static mask)."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Where" or not isinstance(node.inputs[0], gs.Constant):
            continue
        cond, fill, x = node.inputs
        if not isinstance(fill, gs.Constant) or fill.values.size != 1 or isinstance(x, gs.Constant):
            continue
        if cond.values.all():
            continue
        bias = np.where(cond.values, np.float32(fill.values.reshape(())), np.float32(0)).astype(np.float32)
        out = gs.Variable(f"{node.name}/biased", dtype=np.float32)
        graph.layer(op="Add", name=f"{node.name}/add_bias",
                    inputs=[x, gs.Constant(f"{node.name}/bias", bias)], outputs=[out])
        _replace_output(graph, node, out)
        n += 1
    graph.cleanup().toposort()
    return n


def gathernd_sliding_windows(graph: gs.Graph) -> int:
    """GatherND(x [T, ...], idx [B, W, 1]) with idx[b, w] = b * stride + w and W = 2 * stride
    (overlapping windows, as torch.unfold exports) -> Concat of two block reshapes:
    rows [0, B*stride) and [stride, (B+1)*stride), each viewed as [B, stride, ...]."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "GatherND" or not isinstance(node.inputs[1], gs.Constant):
            continue
        x, idx = node.inputs[0], node.inputs[1].values
        if idx.ndim != 3 or idx.shape[-1] != 1 or int(node.attrs.get("batch_dims", 0)) or not x.shape:
            continue
        blocks, window = idx.shape[:2]
        stride = window // 2
        expected = np.arange(blocks)[:, None] * stride + np.arange(window)[None, :]
        if window % 2 or not np.array_equal(idx[..., 0], expected) or (blocks + 1) * stride > x.shape[0]:
            continue
        rest = list(x.shape[1:])
        b = node.name
        parts = []
        for k, start in enumerate((0, stride)):
            sl = gs.Variable(f"{b}/rows{k}", dtype=node.outputs[0].dtype)
            graph.layer(op="Slice", name=f"{b}/slice{k}",
                        inputs=[x, gs.Constant(f"{b}/s{k}", np.array([start], np.int64)),
                                gs.Constant(f"{b}/e{k}", np.array([start + blocks * stride], np.int64)),
                                gs.Constant(f"{b}/a{k}", np.array([0], np.int64))], outputs=[sl])
            rs = gs.Variable(f"{b}/blocks{k}", dtype=node.outputs[0].dtype)
            graph.layer(op="Reshape", name=f"{b}/reshape{k}",
                        inputs=[sl, gs.Constant(f"{b}/shape{k}", np.array([blocks, stride] + rest, np.int64))],
                        outputs=[rs])
            parts.append(rs)
        cat = gs.Variable(f"{b}/windows", dtype=node.outputs[0].dtype)
        graph.layer(op="Concat", name=f"{b}/concat", inputs=parts, outputs=[cat], attrs={"axis": 1})
        _replace_output(graph, node, cat)
        n += 1
    graph.cleanup().toposort()
    return n


def depthwise_conv1d_to_shift_mac(graph: gs.Graph) -> int:
    """Depthwise Conv1d (group == C, kernel K, stride/dilation 1) on [1, C, T] ->
    Pad + sum_k Slice(x, k) * w[:, k]. Depthwise convolutions have hung the NPU; the
    shifted multiply-accumulate is plain elementwise work."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Conv" or not isinstance(node.inputs[1], gs.Constant):
            continue
        x, w = node.inputs[0], node.inputs[1].values
        if w.ndim != 3 or w.shape[1] != 1 or int(node.attrs.get("group", 1)) != w.shape[0]:
            continue
        if list(node.attrs.get("strides", [1])) != [1] or list(node.attrs.get("dilations", [1])) != [1]:
            continue
        if not x.shape or len(x.shape) != 3:
            continue
        pads = list(node.attrs.get("pads", [0, 0]))
        C, K, T = w.shape[0], w.shape[2], x.shape[2]
        t_out = T + pads[0] + pads[1] - K + 1
        b = node.name
        xp = gs.Variable(f"{b}/padded", dtype=np.float32)
        graph.layer(op="Pad", name=f"{b}/pad",
                    inputs=[x, gs.Constant(f"{b}/pads", np.array([0, 0, pads[0], 0, 0, pads[1]], np.int64))],
                    outputs=[xp], attrs={"mode": "constant"})
        acc = None
        for k in range(K):
            sl = gs.Variable(f"{b}/tap{k}", dtype=np.float32)
            graph.layer(op="Slice", name=f"{b}/slice{k}",
                        inputs=[xp, gs.Constant(f"{b}/s{k}", np.array([k], np.int64)),
                                gs.Constant(f"{b}/e{k}", np.array([k + t_out], np.int64)),
                                gs.Constant(f"{b}/a{k}", np.array([2], np.int64))], outputs=[sl])
            prod = gs.Variable(f"{b}/mul{k}", dtype=np.float32)
            graph.layer(op="Mul", name=f"{b}/mul{k}",
                        inputs=[sl, gs.Constant(f"{b}/w{k}", w[:, 0, k].reshape(1, C, 1).astype(np.float32))],
                        outputs=[prod])
            if acc is None:
                acc = prod
            else:
                nxt = gs.Variable(f"{b}/acc{k}", dtype=np.float32)
                graph.layer(op="Add", name=f"{b}/add{k}", inputs=[acc, prod], outputs=[nxt])
                acc = nxt
        if len(node.inputs) > 2 and isinstance(node.inputs[2], gs.Constant):
            biased = gs.Variable(f"{b}/bias_out", dtype=np.float32)
            graph.layer(op="Add", name=f"{b}/bias",
                        inputs=[acc, gs.Constant(f"{b}/bias_c", node.inputs[2].values.reshape(1, C, 1))],
                        outputs=[biased])
            acc = biased
        _replace_output(graph, node, acc)
        n += 1
    graph.cleanup().toposort()
    return n


def drop_unread_conv_end_pads(graph: gs.Graph) -> int:
    """Lower a strided Conv's end padding by the trailing padded elements no window reads.

    A 3x3 stride-2 conv with pads 1/1 over an even extent never reads its last padded
    row/column, so pads [1, 1, 1, 1] -> [1, 1, 0, 0] gives the same output. It keeps
    torq-compile off a channel-tiled conv whose tile skips the end pad, which it
    miscompiled (EmbeddingGemma-2 audio subsampler: the last mel column was dropped)."""
    count = 0
    for node in graph.nodes:
        if node.op != "Conv" or node.attrs.get("auto_pad", "NOTSET") != "NOTSET":
            continue
        x, w = node.inputs[0], node.inputs[1]
        if not x.shape or any(not isinstance(d, int) for d in x.shape) or not isinstance(w, gs.Constant):
            continue
        spatial = len(x.shape) - 2
        pads = list(node.attrs.get("pads", [0] * 2 * spatial))
        strides = list(node.attrs.get("strides", [1] * spatial))
        dilations = list(node.attrs.get("dilations", [1] * spatial))
        kernel = list(w.values.shape[2:])
        changed = False
        for i in range(spatial):
            span = dilations[i] * (kernel[i] - 1) + 1
            unread = (x.shape[2 + i] + pads[i] + pads[spatial + i] - span) % strides[i]
            drop = min(unread, pads[spatial + i])
            if drop:
                pads[spatial + i] -= drop
                changed = True
        if changed:
            node.attrs["pads"] = pads
            count += 1
    return count


AUDIO_FRAMES = 1120  # 11.2 s of 10 ms mel frames -> 280 tokens (the processor's audio_seq_length)


def build_static_audio(src: str | Path, num_frames: int = AUDIO_FRAMES) -> onnx.ModelProto:
    """Static audio encoder: log-mel features [1, T, 128] -> audio_embeds [1, T/4, 512].

    Shorter clips are zero-padded to T frames on the host and only the first
    ``audio_num_tokens`` outputs are used: with right padding the valid tokens equal the
    unpadded ones whether or not the padded frames are masked (measured: token cos 1.0),
    so the frame mask is fixed to all-valid and every mask-dependent op folds away."""
    graph = gs.import_onnx(onnx.load(str(src)))
    feats = next(i for i in graph.inputs if i.name == "input_features")
    mask_in = next(i for i in graph.inputs if i.name == "input_features_mask")
    feats.shape = [1, num_frames, 128]
    mask_const = gs.Constant("input_features_mask_const", np.ones((1, num_frames), dtype=bool))
    for node in list(mask_in.outputs):
        node.inputs = [mask_const if i is mask_in else i for i in node.inputs]
    graph.inputs = [feats]
    graph.cleanup().toposort()
    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(gs.export_onnx(graph)))
    logger.info("audio: SimplifiedLayerNorm replaced: %d", _apply(ReplaceSimplifiedLayerNorm, graph, "audio"))
    model = simplify(gs.export_onnx(graph), {"input_features": [1, num_frames, 128]})
    graph = gs.import_onnx(model)
    logger.info("audio: Mul-by-one removed: %d", remove_mul_by_one(graph))
    logger.info("audio: const-cond Where folded: %d", fold_const_where(graph))
    logger.info("audio: identity GatherND dropped: %d", drop_identity_gathernd(graph))
    graph = reinfer(graph)
    logger.info("audio: static-mask Where -> bias: %d", where_const_mask_to_bias(graph))
    logger.info("audio: sliding-window GatherND -> slices: %d", gathernd_sliding_windows(graph))
    logger.info("audio: depthwise Conv1d -> shift-MAC: %d", depthwise_conv1d_to_shift_mac(graph))
    logger.info("audio: Gemm -> MatMul: %d", gemm_transb_to_matmul(graph))
    logger.info("audio: unread conv end pads dropped: %d", drop_unread_conv_end_pads(graph))
    from ..liquid._vision_static import decompose_ln
    logger.info("audio: LayerNorm decomposed: %d",
                decompose_ln(graph, lambda name: graph.tensors()[name].shape, materialize=False))
    graph = reinfer(graph)
    graph.inputs[0].name = "input_features"
    out = graph.outputs[0]
    if out.shape and len(out.shape) == 2:
        rs = gs.Variable("audio_embeds", dtype=out.dtype, shape=[1] + list(out.shape))
        graph.layer(op="Unsqueeze", name="audio_embeds/unsqueeze",
                    inputs=[out, gs.Constant("audio_embeds/axes", np.array([0], np.int64))], outputs=[rs])
        graph.outputs = [rs]
    else:
        out.name = "audio_embeds"
    graph.cleanup().toposort()
    return onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
