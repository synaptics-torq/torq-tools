# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Support for the onnx-community ``*_q4`` EmbeddingGemma-2 graphs.

Those graphs carry 4-bit block-quantized weights in onnxruntime contrib ops:

* ``com.microsoft::MatMulNBits`` (bits=4, block_size=32): B uint8 [N, K/32, 16] with two
  nibbles per byte (low nibble first), scales fp32 [N, K/32], zero points uint8
  [N, ceil(K/64)] packed the same way; W[k, n] = (q - zp) * scale.
* ``com.microsoft::GatherBlockQuantized`` (bits=4, block_size=32, quantize_axis=1): the
  text token-embedding table and the vision position tables, same packing per row.

Torq compiles 4-bit weights in the standard ONNX form
``DequantizeLinear(INT4 [K, N], axis=0, block_size=32, scale [K/32, N], zero_point INT4
[K/32, N]) -> MatMul``. The static builders constant-fold with onnxsim, which would fold
a constant DequantizeLinear back to float, so the conversion runs in two steps:

1. :func:`dequantize_q4_graph` rewrites every MatMulNBits into a plain MatMul whose fp32
   weight is the exact dequantized q4 weight (and every GatherBlockQuantized into a Gather
   of the exact dequantized table), and returns a registry of the int4 data per weight.
2. After the static builder ran, :func:`requantize_matmuls` finds those weights again by
   content and turns each MatMul back into ``DequantizeLinear(INT4) -> MatMul``.

Moving uint4 [0, 15] to int4 [-8, 7] (q - 8, zp - 8) leaves q - zp unchanged.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

logger = logging.getLogger(__name__)

BLOCK = 32


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """uint8 [..., n] -> uint8 [..., 2n], low nibble first."""
    lo = packed & 0x0F
    hi = packed >> 4
    return np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@dataclass
class Int4Weight:
    """A [K, N] weight as signed int4 values with per-(block, column) scale and zero point."""

    q: np.ndarray      # int8 [K, N], values in [-8, 7]
    scale: np.ndarray  # fp32 [K/32, N]
    zp: np.ndarray     # int8 [K/32, N], values in [-8, 7]

    def dequantize(self) -> np.ndarray:
        k, n = self.q.shape
        q = self.q.astype(np.float32).reshape(k // BLOCK, BLOCK, n)
        return ((q - self.zp[:, None, :].astype(np.float32)) * self.scale[:, None, :]).reshape(k, n)

    def transpose_free(self) -> "Int4Weight":
        return self


def _matmulnbits_weight(node: gs.Node) -> Int4Weight:
    k, n = node.attrs["K"], node.attrs["N"]
    if node.attrs.get("bits", 4) != 4 or node.attrs.get("block_size", BLOCK) != BLOCK:
        raise NotImplementedError(f"{node.name}: only bits=4, block_size=32 is supported")
    if k % BLOCK:
        raise NotImplementedError(f"{node.name}: K={k} is not a multiple of {BLOCK}")
    nb = k // BLOCK
    b = node.inputs[1].values                        # uint8 [N, nb, 16]
    q = _unpack_nibbles(b.reshape(n, nb, BLOCK // 2)).reshape(n, k).astype(np.int8) - 8
    scale = node.inputs[2].values.reshape(n, nb).astype(np.float32)
    if len(node.inputs) > 3 and node.inputs[3] is not None and not _is_empty(node.inputs[3]):
        zp_t = node.inputs[3].values
        if zp_t.dtype != np.uint8:
            raise NotImplementedError(f"{node.name}: float zero points are not supported")
        zp = _unpack_nibbles(zp_t.reshape(n, -1))[:, :nb].astype(np.int8) - 8
    else:
        zp = np.zeros((n, nb), np.int8)              # default uint4 zero point 8 -> int4 0
    return Int4Weight(q=np.ascontiguousarray(q.T), scale=np.ascontiguousarray(scale.T),
                      zp=np.ascontiguousarray(zp.T))


def _gbq_table(node: gs.Node) -> np.ndarray:
    a = node.attrs
    if a.get("bits", 4) != 4 or a.get("block_size", BLOCK) != BLOCK or a.get("quantize_axis") != 1 \
            or a.get("gather_axis", 0) != 0:
        raise NotImplementedError(f"{node.name}: unsupported GatherBlockQuantized attrs {a}")
    data = node.inputs[0].values                     # uint8 [V, D/2]
    v = data.shape[0]
    q = _unpack_nibbles(data).astype(np.float32)     # [V, D]
    d = q.shape[1]
    scale = node.inputs[2].values.reshape(v, d // BLOCK).astype(np.float32)
    if len(node.inputs) > 3 and node.inputs[3] is not None and not _is_empty(node.inputs[3]):
        zp = _unpack_nibbles(node.inputs[3].values.reshape(v, -1))[:, : d // BLOCK].astype(np.float32)
    else:
        zp = np.full((v, d // BLOCK), 8, np.float32)
    q = q.reshape(v, d // BLOCK, BLOCK)
    return ((q - zp[:, :, None]) * scale[:, :, None]).reshape(v, d)


def _is_empty(t) -> bool:
    return isinstance(t, gs.Variable) and not t.name


def _digest(w: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest() + str(w.shape)


def dequantize_q4_graph(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Int4Weight]]:
    """Exact float rewrite of a q4 graph (see module docstring). Returns the rewritten
    model and a registry {content digest of the fp32 [K, N] weight: Int4Weight}."""
    graph = gs.import_onnx(model)
    registry: dict[str, Int4Weight] = {}
    n_mm = n_gather = 0
    for node in list(graph.nodes):
        if node.op == "MatMulNBits":
            w4 = _matmulnbits_weight(node)
            w = w4.dequantize()
            registry[_digest(w)] = w4
            out = node.outputs[0]
            bias = node.inputs[5] if len(node.inputs) > 5 and not _is_empty(node.inputs[5]) else None
            node.outputs = []
            const = gs.Constant(f"{node.name}/q4_weight", w)
            if bias is None:
                graph.layer(op="MatMul", name=node.name, inputs=[node.inputs[0], const], outputs=[out])
            else:
                mid = gs.Variable(f"{node.name}/matmul_out", dtype=np.float32)
                graph.layer(op="MatMul", name=node.name, inputs=[node.inputs[0], const], outputs=[mid])
                graph.layer(op="Add", name=f"{node.name}/bias", inputs=[mid, bias], outputs=[out])
            n_mm += 1
        elif node.op == "GatherBlockQuantized":
            table = gs.Constant(f"{node.name}/table", _gbq_table(node))
            out = node.outputs[0]
            idx = node.inputs[1]
            node.outputs = []
            graph.layer(op="Gather", name=node.name, inputs=[table, idx], outputs=[out], attrs={"axis": 0})
            n_gather += 1
    graph.cleanup().toposort()
    logger.info("q4: MatMulNBits -> MatMul: %d, GatherBlockQuantized -> Gather: %d", n_mm, n_gather)
    out = gs.export_onnx(graph)
    # the contrib domain may no longer be needed; keep it if other contrib ops remain
    return out, registry


def _match_concat(w: np.ndarray, by_k: dict[int, dict[int, dict[str, Int4Weight]]]) -> Int4Weight | None:
    """A weight that is a concatenation of registered q4 weights along N (e.g. fused Q|K|V
    projections) is still block-quantized along K: rebuild it column segment by segment."""
    k, n = w.shape
    widths = by_k.get(k)
    if not widths:
        return None
    parts, off = [], 0
    while off < n:
        for width, table in sorted(widths.items(), reverse=True):
            if off + width <= n:
                p = table.get(_digest(w[:, off:off + width]))
                if p is not None:
                    parts.append(p)
                    off += width
                    break
        else:
            return None
    if len(parts) < 2:
        return None
    return Int4Weight(q=np.concatenate([p.q for p in parts], 1), scale=np.concatenate([p.scale for p in parts], 1),
                      zp=np.concatenate([p.zp for p in parts], 1))


def requantize_matmuls(model: onnx.ModelProto, registry: dict[str, Int4Weight]) -> tuple[onnx.ModelProto, int, int]:
    """Turn every MatMul whose constant weight is a registered q4 weight (or a concatenation
    of them along N) into ``DequantizeLinear(INT4, axis=0, block_size=32) -> MatMul``.
    Returns (model, requantized, float matmuls left with a constant weight)."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    by_k: dict[int, dict[int, dict[str, Int4Weight]]] = {}
    for d, w4 in registry.items():
        k, n = w4.q.shape
        by_k.setdefault(k, {}).setdefault(n, {})[d] = w4
    new_nodes, new_inits = [], []
    hit = miss = 0
    used: set[str] = set()
    for node in g.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            t = inits[node.input[1]]
            w = numpy_helper.to_array(t)
            w4 = None
            if w.ndim == 2:
                w4 = registry.get(_digest(w)) or _match_concat(w, by_k)
                if w4 is None and registry.get(_digest(w.T)) is not None:
                    raise NotImplementedError(f"{node.name}: transposed q4 weight (blocks would run along N)")
            if w4 is not None:
                base = node.input[1]
                qn, sn, zn = f"{base}/int4", f"{base}/scale", f"{base}/zero_point"
                if base not in used:
                    new_inits += [
                        numpy_helper.from_array(w4.q.astype(ml_dtypes.int4), qn),  # packed raw data
                        numpy_helper.from_array(w4.scale.astype(np.float32), sn),
                        numpy_helper.from_array(w4.zp.astype(ml_dtypes.int4), zn),
                    ]
                    new_nodes.append(helper.make_node(
                        "DequantizeLinear", [qn, sn, zn], [f"{base}/dequantized"],
                        name=f"{base}/DequantizeLinear", axis=0, block_size=BLOCK))
                    used.add(base)
                node.input[1] = f"{base}/dequantized"
                hit += 1
            else:
                miss += 1
        new_nodes.append(node)
    # drop the fp32 copies that are now unused
    still_used = {i for n in new_nodes for i in n.input}
    keep = [t for t in g.initializer if t.name in still_used]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes)
    # DequantizeLinear with INT4 + block_size needs opset 21
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx") and o.version < 21:
            o.version = 21
    lifted = _lift_rank2_int4_matmuls(model)
    logger.info("q4: MatMul -> DequantizeLinear(INT4) + MatMul: %d (float MatMuls with constant weight left: %d,"
                " rank-2 lifted to rank 3: %d)", hit, miss, lifted)
    return model, hit, miss


SPLIT_ROWS = 128


def split_int4_matmul_rows(model: onnx.ModelProto, rows: int = SPLIT_ROWS) -> int:
    """For torq-compile: run every ``[1,M,K] x DequantizeLinear(INT4)`` MatMul with M > rows as
    M/rows row chunks.

    torq-compile fuses the int4 dequant into the matmul tile. With a long M (text S=512) its
    tile search ends at one row per tile, so the weight is dequantized again for every row
    (S=512: ~10 s per pass, over the NPU's 5 s job limit). At 128 rows it keeps whole tiles.
    Each chunk gets its own DequantizeLinear so the dequant stays local to its matmul."""
    g = model.graph
    shapes = onnx.shape_inference.infer_shapes(model)
    shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
             for v in list(shapes.graph.value_info) + list(shapes.graph.input)
             if v.type.tensor_type.HasField("shape")}
    dq = {n.output[0]: n for n in g.node if n.op_type == "DequantizeLinear"}
    new_nodes, split = [], 0
    for node in g.node:
        s = shape.get(node.input[0]) if node.op_type == "MatMul" else None
        if not (s and node.input[1] in dq and len(s) == 3 and s[1] > rows and s[1] % rows == 0):
            new_nodes.append(node)
            continue
        chunks = s[1] // rows
        base = node.name or node.output[0]
        sizes = numpy_helper.from_array(np.full(chunks, rows, np.int64), f"{base}/row_chunks")
        g.initializer.append(sizes)
        parts = [f"{base}/rows{i}" for i in range(chunks)]
        new_nodes.append(helper.make_node("Split", [node.input[0], sizes.name], parts, axis=1, name=f"{base}/split_rows"))
        d = dq[node.input[1]]
        attrs = {a.name: helper.get_attribute_value(a) for a in d.attribute}
        outs = []
        for i, part in enumerate(parts):
            w = f"{base}/w{i}"
            new_nodes.append(helper.make_node("DequantizeLinear", list(d.input), [w], name=f"{base}/dq{i}", **attrs))
            outs.append(f"{base}/out{i}")
            new_nodes.append(helper.make_node("MatMul", [part, w], [outs[-1]], name=f"{base}/mm{i}"))
        new_nodes.append(helper.make_node("Concat", outs, list(node.output), axis=1, name=f"{base}/concat_rows"))
        split += 1
    if split:
        used = {i for n in new_nodes for i in n.input} | {o.name for o in g.output}
        new_nodes = [n for n in new_nodes if n.op_type != "DequantizeLinear" or n.output[0] in used]
        del g.node[:]
        g.node.extend(new_nodes)
    return split


def _lift_rank2_int4_matmuls(model: onnx.ModelProto) -> int:
    """Run every rank-2 ``[M,K] x DequantizeLinear(INT4)`` MatMul as ``[1,M,K]``.

    torq-compile raises only batched matmuls to its int4 kernel (torq_hl.mixed_quant_matmul);
    a rank-2 one falls back to a path that cannot read int4 weights. Unsqueeze/Squeeze are
    free reshapes on the NPU."""
    g = model.graph
    shapes = onnx.shape_inference.infer_shapes(model)
    rank = {v.name: len(v.type.tensor_type.shape.dim)
            for v in list(shapes.graph.value_info) + list(shapes.graph.input) + list(shapes.graph.output)
            if v.type.tensor_type.HasField("shape")}
    dq = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
    axes = "q4/lift_axes"
    new_nodes, lifted = [], 0
    for node in g.node:
        if node.op_type == "MatMul" and node.input[1] in dq and rank.get(node.input[0]) == 2:
            a, y = node.input[0], node.output[0]
            new_nodes.append(helper.make_node("Unsqueeze", [a, axes], [f"{y}/lift_a"], name=f"{node.name}/lift_a"))
            node.input[0] = f"{y}/lift_a"
            node.output[0] = f"{y}/lift_y"
            new_nodes.append(node)
            new_nodes.append(helper.make_node("Squeeze", [f"{y}/lift_y", axes], [y], name=f"{node.name}/lift_y"))
            lifted += 1
        else:
            new_nodes.append(node)
    if lifted:
        g.initializer.append(numpy_helper.from_array(np.array([0], np.int64), axes))
        del g.node[:]
        g.node.extend(new_nodes)
    return lifted


def _pack_nibbles(u: np.ndarray) -> np.ndarray:
    """uint8 [..., 2n] with values < 16 -> uint8 [..., n], low nibble first."""
    if u.shape[-1] % 2:
        u = np.concatenate([u, np.zeros((*u.shape[:-1], 1), u.dtype)], axis=-1)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).astype(np.uint8)


def to_matmulnbits(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """For ONNX Runtime: turn each ``DequantizeLinear(INT4, axis=0, block_size=32) -> MatMul``
    back into ``com.microsoft::MatMulNBits`` (bits=4, block_size=32, no accuracy_level), the
    form of the onnx-community q4 graphs. ORT's own DequantizeLinear+MatMul fusion would run
    int8 activations instead; MatMulNBits without accuracy_level computes in fp32 and matches
    the q4 source graphs. Returns (model, MatMuls converted)."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    dq = {n.output[0]: n for n in g.node if n.op_type == "DequantizeLinear"
          and inits.get(n.input[0]) is not None and inits[n.input[0]].data_type == onnx.TensorProto.INT4}
    packed: dict[str, list[str]] = {}
    new_inits, new_nodes, converted = [], [], 0
    for node in g.node:
        if node.op_type == "DequantizeLinear" and node.output[0] in dq:
            continue
        if node.op_type == "MatMul" and node.input[1] in dq:
            d = dq[node.input[1]]
            attrs = {a.name: helper.get_attribute_value(a) for a in d.attribute}
            if attrs.get("axis") != 0 or attrs.get("block_size") != BLOCK:
                raise NotImplementedError(f"{d.name}: unsupported DequantizeLinear attrs {attrs}")
            q = numpy_helper.to_array(inits[d.input[0]]).astype(np.int8)        # [K, N]
            k, n = q.shape
            nb = k // BLOCK
            if d.output[0] not in packed:
                base = d.output[0].removesuffix("/dequantized")
                scale = numpy_helper.to_array(inits[d.input[1]]).astype(np.float32)  # [nb, N]
                zp = numpy_helper.to_array(inits[d.input[2]]).astype(np.int8)        # [nb, N]
                b = _pack_nibbles((q.T + 8).astype(np.uint8).reshape(n, nb, BLOCK))   # [N, nb, 16]
                z = _pack_nibbles((zp.T + 8).astype(np.uint8))                        # [N, ceil(nb/2)]
                names = [f"{base}/nbits_B", f"{base}/nbits_scales", f"{base}/nbits_zp"]
                new_inits += [numpy_helper.from_array(b, names[0]),
                              numpy_helper.from_array(np.ascontiguousarray(scale.T).reshape(-1), names[1]),
                              numpy_helper.from_array(z.reshape(-1), names[2])]
                packed[d.output[0]] = names
            new_nodes.append(helper.make_node(
                "MatMulNBits", [node.input[0], *packed[d.output[0]]], list(node.output), name=node.name,
                domain="com.microsoft", K=k, N=n, bits=4, block_size=BLOCK))
            converted += 1
        else:
            new_nodes.append(node)
    used = {i for nd in new_nodes for i in nd.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return model, converted
