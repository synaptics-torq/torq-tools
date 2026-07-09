#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.
"""Build a static, Torq-compilable bf16 LFM2-VL vision encoder.

Transforms the dynamic HF vision_encoder.onnx into a single-resolution static
graph the SL2610 can compile:
  1. decompose com.microsoft.MultiHeadAttention -> MatMul/Softmax (the importer
     has no MHA lowering)
  2. fix inputs to one full tile (pixel_values[1,N,768], mask=all-ones const,
     spatial_shapes=[[gh,gw]] const) so dynamic dims + the pos-embed Resize and
     the projector's Compress/ScatterND padding logic constant-fold / become
     identity, which we then drop
  3. decompose LayerNormalization into primitives, MATERIALIZING the
     mean/std/scale/bias broadcasts to full [1,N,768] tensors (Expand). The chip
     tiles the 1024-token norm across the token axis; a stride-0 broadcast
     operand would give a tiled binding a non-constant byte offset, so we
     materialize to keep every binding offset constant.

Outputs <out_prefix>.onnx (fp32) and <out_prefix>_bf16.onnx, and validates the
fp32 result against the original encoder.

    python scripts/build_vision_encoder_static.py [--patches 1024 --grid 32 32]
"""
import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort

from .export import LiquidModelExporter as L
from ...tools.convert_dtype.onnx import convert_model

SRC = "models/liquid-2p5-450M-VL/export/onnx/fp32/static/vision_encoder.onnx"
HEADS, HD = 12, 64


def decompose_mha(graph):
    n = 0
    for node in list(graph.nodes):
        if node.op != "MultiHeadAttention":
            continue
        q, k, v = node.inputs[0], node.inputs[1], node.inputs[2]
        mask = node.inputs[5] if len(node.inputs) > 5 and getattr(node.inputs[5], "name", "") else None
        scale = float(node.attrs.get("scale", 1.0 / np.sqrt(HD)))
        out = node.outputs[0]
        b = node.name

        def rsh(x, sh, nm):
            o = gs.Variable(nm, dtype=np.float32)
            graph.layer(op="Reshape", name=nm, inputs=[x, gs.Constant(nm + "_s", np.array(sh, np.int64))], outputs=[o])
            return o

        def tr(x, p, nm):
            o = gs.Variable(nm, dtype=np.float32)
            graph.layer(op="Transpose", name=nm, inputs=[x], outputs=[o], attrs={"perm": p})
            return o

        qh = tr(rsh(q, [1, -1, HEADS, HD], b + "/qr"), [0, 2, 1, 3], b + "/qt")
        kh = tr(rsh(k, [1, -1, HEADS, HD], b + "/kr"), [0, 2, 1, 3], b + "/kt")
        vh = tr(rsh(v, [1, -1, HEADS, HD], b + "/vr"), [0, 2, 1, 3], b + "/vt")
        kT = tr(kh, [0, 1, 3, 2], b + "/kT")
        sc = gs.Variable(b + "/s", dtype=np.float32)
        graph.layer(op="MatMul", name=b + "/qk", inputs=[qh, kT], outputs=[sc])
        scs = gs.Variable(b + "/ss", dtype=np.float32)
        graph.layer(op="Mul", name=b + "/sc", inputs=[sc, gs.Constant(b + "/scc", np.array(scale, np.float32))], outputs=[scs])
        if mask is not None:
            m2 = gs.Variable(b + "/sm_", dtype=np.float32)
            graph.layer(op="Add", name=b + "/ma", inputs=[scs, mask], outputs=[m2])
            scs = m2
        sm = gs.Variable(b + "/smx", dtype=np.float32)
        graph.layer(op="Softmax", name=b + "/smo", inputs=[scs], outputs=[sm], attrs={"axis": -1})
        av = gs.Variable(b + "/av", dtype=np.float32)
        graph.layer(op="MatMul", name=b + "/avo", inputs=[sm, vh], outputs=[av])
        merged = rsh(tr(av, [0, 2, 1, 3], b + "/avt"), [1, -1, HEADS * HD], b + "/mg")
        for c in graph.nodes:
            c.inputs = [merged if i is out else i for i in c.inputs]
        graph.outputs = [merged if o is out else o for o in graph.outputs]
        node.inputs.clear()
        node.outputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def decompose_ln(graph, shape_of, materialize=False):
    """LayerNorm -> ReduceMean/Sub/Mul/Add/Sqrt/Div primitives. With
    materialize=True the mean/std/scale/bias are Expand'd to full shape (avoids
    stride-0 broadcast operands but the Expand itself yields a dynamic-offset
    memref the chip rejects). Default keeps plain broadcasts."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "LayerNormalization":
            continue
        X, sc = node.inputs[0], node.inputs[1]
        bi = node.inputs[2] if len(node.inputs) > 2 else None
        ax = int(node.attrs.get("axis", -1))
        eps = float(node.attrs.get("epsilon", 1e-5))
        out = node.outputs[0]
        b = node.name
        full = shape_of(X.name) or [1, 1024, 768]
        axc = gs.Constant(b + "/ax", np.array([ax], np.int64))
        shp = gs.Constant(b + "/shp", np.array(full, np.int64))

        def mk(op, ins, nm, **a):
            o = gs.Variable(nm, dtype=np.float32)
            graph.layer(op=op, name=nm, inputs=ins, outputs=[o], attrs=a)
            return o

        def bc(t, nm):  # broadcast operand: materialize to full shape, or pass through
            return mk("Expand", [t, shp], nm) if materialize else t

        mean = mk("ReduceMean", [X, axc], b + "/m", keepdims=1)
        xc = mk("Sub", [X, bc(mean, b + "/me")], b + "/xc")
        sq = mk("Mul", [xc, xc], b + "/sq")
        var = mk("ReduceMean", [sq, axc], b + "/v", keepdims=1)
        vare = mk("Add", [var, gs.Constant(b + "/e", np.array(eps, np.float32))], b + "/ve")
        std = mk("Sqrt", [vare], b + "/st")
        norm = mk("Div", [xc, bc(std, b + "/se")], b + "/nm")
        scd = mk("Mul", [norm, bc(sc, b + "/sce")], b + "/scl")
        y = mk("Add", [scd, bc(bi, b + "/bie")], b + "/y") if bi is not None else scd
        for c in graph.nodes:
            c.inputs = [y if i is out else i for i in c.inputs]
        graph.outputs = [y if o is out else o for o in graph.outputs]
        node.inputs.clear()
        node.outputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def gemm_to_matmul(graph):
    """Replace Gemm(A,B,C) with MatMul(A,B)+Add(C) (alpha=beta=1, no transpose)."""
    n = 0
    for node in list(graph.nodes):
        if node.op != "Gemm":
            continue
        if float(node.attrs.get("alpha", 1.0)) != 1.0 or float(node.attrs.get("beta", 1.0)) != 1.0:
            continue
        if int(node.attrs.get("transA", 0)) != 0 or int(node.attrs.get("transB", 0)) != 0:
            continue
        A, B = node.inputs[0], node.inputs[1]
        C = node.inputs[2] if len(node.inputs) > 2 else None
        out = node.outputs[0]
        b = node.name
        mm = gs.Variable(b + "/mm", dtype=np.float32)
        graph.layer(op="MatMul", name=b + "/matmul", inputs=[A, B], outputs=[mm])
        y = mm
        if C is not None:
            y = gs.Variable(b + "/bias", dtype=np.float32)
            graph.layer(op="Add", name=b + "/add", inputs=[mm, C], outputs=[y])
        for c in graph.nodes:
            c.inputs = [y if i is out else i for i in c.inputs]
        graph.outputs = [y if o is out else o for o in graph.outputs]
        node.inputs.clear()
        node.outputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def split_large_matmuls(graph, shape_of, max_out=512):
    """Split MatMul/Gemm with a constant rhs weight whose output dim > max_out
    into output-dim chunks (+Concat), so each weight slice + I/O fits LRAM."""
    n = 0
    for node in list(graph.nodes):
        if node.op not in ("MatMul", "Gemm"):
            continue
        A, B = node.inputs[0], node.inputs[1]
        if not isinstance(B, gs.Constant) or B.values.ndim != 2:
            continue
        K, N = B.values.shape
        if N <= max_out:
            continue
        bias = node.inputs[2] if (node.op == "Gemm" and len(node.inputs) > 2) else None
        bias_v = bias.values if isinstance(bias, gs.Constant) else None
        out = node.outputs[0]
        b = node.name
        chunks = []
        for ci, s in enumerate(range(0, N, max_out)):
            e = min(s + max_out, N)
            wc = gs.Constant(f"{b}/w{ci}", B.values[:, s:e].copy())
            o = gs.Variable(f"{b}/mm{ci}", dtype=np.float32)
            graph.layer(op="MatMul", name=f"{b}/mm{ci}", inputs=[A, wc], outputs=[o])
            if bias_v is not None:
                bc = gs.Constant(f"{b}/b{ci}", bias_v[s:e].copy())
                o2 = gs.Variable(f"{b}/ba{ci}", dtype=np.float32)
                graph.layer(op="Add", name=f"{b}/ba{ci}", inputs=[o, bc], outputs=[o2])
                o = o2
            chunks.append(o)
        cc = gs.Variable(b + "/cat", dtype=np.float32)
        graph.layer(op="Concat", name=b + "/cat", inputs=chunks, outputs=[cc], attrs={"axis": -1})
        for c in graph.nodes:
            c.inputs = [cc if i is out else i for i in c.inputs]
        graph.outputs = [cc if o is out else o for o in graph.outputs]
        node.inputs.clear()
        node.outputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def split_ln_token_dim(graph, shape_of, poison=256, chunk=128):
    """Work around a torq-compile tiler crash on LayerNorm at exactly 256 tokens.

    torq-compile aborts ('linalg.generic op unhandled tiled implementation ...
    result is not accessed using a permuted projection') on a LayerNorm whose
    token axis is exactly ``poison`` (256) — the same op compiles fine at 64,
    128, 512, 1024. (Only the LN reduction hits it; attention/matmul at 256 are
    fine.) Slice that axis into ``chunk``-sized (<=128) pieces, LayerNorm each,
    then Concat back. LN is per-token so this is numerically exact. No-op unless
    a LN's token axis is exactly ``poison`` (so the 1024-token path is untouched).
    """
    n = 0
    for node in list(graph.nodes):
        if node.op != "LayerNormalization":
            continue
        X = node.inputs[0]
        full = shape_of(X.name)
        if not full:
            continue
        ax = int(node.attrs.get("axis", -1)) % len(full)
        tax = next((i for i, d in enumerate(full)
                    if d == poison and i != ax and i != 0), None)
        if tax is None:
            continue
        out = node.outputs[0]
        b = node.name
        parts = []
        for ci, s in enumerate(range(0, poison, chunk)):
            e = min(s + chunk, poison)
            sl = gs.Variable(f"{b}/sl{ci}", dtype=np.float32)
            graph.layer(
                op="Slice", name=f"{b}/slc{ci}",
                inputs=[X,
                        gs.Constant(f"{b}/s{ci}", np.array([s], np.int64)),
                        gs.Constant(f"{b}/e{ci}", np.array([e], np.int64)),
                        gs.Constant(f"{b}/a{ci}", np.array([tax], np.int64))],
                outputs=[sl])
            ln = gs.Variable(f"{b}/ln{ci}", dtype=np.float32)
            graph.layer(op="LayerNormalization", name=f"{b}/lnc{ci}",
                        inputs=[sl] + list(node.inputs[1:]), outputs=[ln],
                        attrs=dict(node.attrs))
            parts.append(ln)
        graph.layer(op="Concat", name=f"{b}/cat", inputs=parts, outputs=[out],
                    attrs={"axis": tax})
        node.outputs.clear()
        node.inputs.clear()
        n += 1
    graph.cleanup().toposort()
    return n


def const_of(t):
    return t.values if isinstance(t, gs.Constant) else None


def remove_identity_masking(graph):
    rew = 0
    for node in list(graph.nodes):
        if node.op == "Compress":
            c = const_of(node.inputs[1])
            if c is not None and bool(c.all()):
                d, o = node.inputs[0], node.outputs[0]
                for x in graph.nodes:
                    x.inputs = [d if i is o else i for i in x.inputs]
                graph.outputs = [d if z is o else z for z in graph.outputs]
                node.inputs.clear(); node.outputs.clear(); rew += 1
        elif node.op == "ScatterND":
            cv, upd, o = const_of(node.inputs[0]), node.inputs[2], node.outputs[0]
            if cv is not None:
                r = gs.Variable(node.name + "/rc", dtype=np.float32)
                graph.layer(op="Reshape", name=node.name + "/rs",
                            inputs=[upd, gs.Constant(node.name + "/sh", np.array(list(cv.shape), np.int64))], outputs=[r])
                for x in graph.nodes:
                    x.inputs = [r if i is o else i for i in x.inputs]
                graph.outputs = [r if z is o else z for z in graph.outputs]
                node.inputs.clear(); node.outputs.clear(); rew += 1
    return rew


# Image resolution -> (num_patches, grid) for the LFM2-VL SigLIP tower
# (patch_size 16: pixel_values is [1, N, 3*16*16=768]). A 256x256 image is
# 16x16=256 patches -> 64 image tokens (the board's image-decoder path, i.e.
# vision_encoder_256.vmfb); 128x128 is 8x8=64 patches -> 16 tokens.
VISION_RES = {128: (64, (8, 8)), 256: (256, (16, 16))}


def build_static_vision_encoder(
    out_prefix,
    *,
    patches=1024,
    grid=(32, 32),
    src=SRC,
    ln_expand=True,
    gemm_to_matmul_pass=False,
    split_matmul=0,
    validate=True,
):
    """Build a static, single-resolution, Torq-compilable bf16 vision encoder
    from the dynamic ``src`` ONNX. Writes ``<out_prefix>.onnx`` (fp32) and
    ``<out_prefix>_bf16.onnx``; returns the bf16 path.

    ``patches``/``grid`` select the resolution — see :data:`VISION_RES`
    (256-res -> 256/(16,16) -> 64 tokens; 128-res -> 64/(8,8) -> 16 tokens)."""
    S, (GH, GW) = patches, tuple(grid)

    g = gs.import_onnx(onnx.load(src))
    print("MHA decomposed:", decompose_mha(g))

    pvar = [i for i in g.inputs if i.name == "pixel_values"][0]
    pvar.shape = [1, S, 768]
    mc = gs.Constant("pam_c", np.ones((1, S), np.int64))
    ssc = gs.Constant("ss_c", np.array([[GH, GW]], np.int64))
    keep = []
    for inp in g.inputs:
        if inp.name == "pixel_attention_mask":
            for n in g.nodes:
                n.inputs = [mc if x is inp else x for x in n.inputs]
        elif inp.name == "spatial_shapes":
            for n in g.nodes:
                n.inputs = [ssc if x is inp else x for x in n.inputs]
        else:
            keep.append(inp)
    g.inputs = keep
    g.cleanup().toposort()
    m = gs.export_onnx(g); m.ir_version = 10

    for _ in range(40):
        m, nf = L._fold_shape_ops(m)
        m, ns = L._resolve_negative_slices(m)
        del m.graph.value_info[:]
        try:
            m = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
        except Exception:
            pass
        if nf == 0 and ns == 0:
            break
    onnx.save(m, out_prefix + "_a.onnx")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = out_prefix + "_b.onnx"
    ort.InferenceSession(out_prefix + "_a.onnx", so, providers=["CPUExecutionProvider"])

    m = onnx.shape_inference.infer_shapes(onnx.load(out_prefix + "_b.onnx"))
    vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}

    def shape_of(name):
        v = vi.get(name)
        return [d.dim_value for d in v.type.tensor_type.shape.dim] if v else None

    g = gs.import_onnx(m)
    print("masking nodes removed:", remove_identity_masking(g))
    g.cleanup().toposort()
    # Dodge the token=256 LayerNorm tiler crash (no-op unless a LN is 256-token).
    # Refresh shape_of afterwards so decompose_ln sees the post-split (128-token)
    # LN shapes for its broadcast materialization.
    ns = split_ln_token_dim(g, shape_of)
    if ns:
        print(f"LN token-split ({256}->2x128):", ns)
        mm = onnx.shape_inference.infer_shapes(gs.export_onnx(g))
        vi = {v.name: v for v in list(mm.graph.value_info)
              + list(mm.graph.input) + list(mm.graph.output)}

        def shape_of(name):
            v = vi.get(name)
            return [d.dim_value for d in v.type.tensor_type.shape.dim] if v else None

        g = gs.import_onnx(mm)
    print("LN decomposed:", decompose_ln(g, shape_of, materialize=ln_expand))
    if gemm_to_matmul_pass:
        print("Gemm->MatMul:", gemm_to_matmul(g))
    if split_matmul > 0:
        print(f"split matmuls (>{split_matmul}):", split_large_matmuls(g, shape_of, split_matmul))
    mm = gs.export_onnx(g); mm.ir_version = 10
    onnx.checker.check_model(mm, full_check=False)
    onnx.save(mm, out_prefix + ".onnx")
    from collections import Counter
    print("ops:", dict(sorted(Counter(n.op_type for n in mm.graph.node).items(), key=lambda kv: -kv[1])))

    if validate:
        pv = np.random.RandomState(0).randn(1, S, 768).astype(np.float32) * 0.1
        ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(
            None, {"pixel_values": pv, "pixel_attention_mask": np.ones((1, S), np.int64),
                   "spatial_shapes": np.array([[GH, GW]], np.int64)})[0]
        out = ort.InferenceSession(out_prefix + ".onnx", providers=["CPUExecutionProvider"]).run(None, {"pixel_values": pv})[0]
        mm_ = min(len(ref), len(out))
        print(f"validate vs original: out {out.shape}  max abs diff {np.abs(ref[:mm_] - out[:mm_]).max():.5f}")

    convert_model(out_prefix + ".onnx", out_prefix + "_bf16.onnx", "bf16", convert_io=True)
    print("bf16 saved:", out_prefix + "_bf16.onnx")
    return out_prefix + "_bf16.onnx"


def main():
    ap = argparse.ArgumentParser(description="Build a static Torq-compilable LFM2-VL vision encoder.")
    ap.add_argument("--patches", type=int, default=1024)
    ap.add_argument("--grid", type=int, nargs=2, default=[32, 32], metavar=("GH", "GW"))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="/tmp/vis_lnmat")
    ap.add_argument("--ln-expand", action="store_true", default=True,
                    help="materialize LN broadcasts via Expand (default: on)")
    ap.add_argument("--split-matmul", type=int, default=0,
                    help="split MatMul/Gemm output dims > this into chunks (0=off)")
    ap.add_argument("--gemm-to-matmul", action="store_true", help="replace Gemm with MatMul+Add")
    args = ap.parse_args()
    build_static_vision_encoder(
        args.out, patches=args.patches, grid=args.grid, src=args.src,
        ln_expand=args.ln_expand, gemm_to_matmul_pass=args.gemm_to_matmul,
        split_matmul=args.split_matmul,
    )


if __name__ == "__main__":
    main()
