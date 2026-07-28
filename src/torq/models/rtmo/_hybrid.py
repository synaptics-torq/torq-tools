# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Hybrid RTMO quantization: int8 conv layers + int16 (or bf16) transformer neck.

RTMO's neck is a HybridEncoder: the whole model is convolutional **except** one
AIFI transformer block (multi-head self-attention + 2 LayerNorms + FFN) that runs
on the stride-32 (``1x256x10x10`` -> ``1x100x256``) feature. Full int8 holds up
on the conv detection branches (cosine ~0.97) but the attention neck + keypoint
head drop to ~0.90; int16 activations recover the neck to ~0.99 but cost 2x on
every conv. This module measures/produces the middle ground — **int8 for all the
convs, higher precision only for the transformer** — by splitting the model at
the transformer boundaries and quantizing each part with the matching scheme:

    image ─▶ [backbone  int8] ─▶ P5 ─▶ [transformer int16x8/bf16] ─▶ P5' ─┐
                    │  ├──────────────── P4 (skip) ─────────────────────┐ │
                    │  └──────────────── P3 (skip) ────────────────┐    │ │
                    └───────────────────────────────────────────▶ [head int8] ─▶ 8 outputs

The three parts are chained (dequantize at each seam, so int8<->int16 requant
happens exactly as it would on-device) and compared to the fp32 ONNX. Each part
is a real, deployment-grade TFLite produced by the same ``onnx2tf`` + TFLite PTQ
path as the whole-model int8/int16x8 schemes in :mod:`quantize`, so the accuracy
number and the artifacts come from one code path.

The split boundaries are discovered automatically (anchored on the single neck
``Softmax``), so this is not hard-wired to one export of the graph.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# Reuse the proven quantize.py machinery whether imported as a package module or
# run as a standalone script (the package __init__ pulls LLM deps absent from the
# dedicated rtmo-quant venv; see README).
try:  # pragma: no cover - import shim
    from . import quantize as _q
except ImportError:  # pragma: no cover
    import importlib.util
    import os

    _spec = importlib.util.spec_from_file_location(
        "rtmo_quantize", os.path.join(os.path.dirname(__file__), "quantize.py")
    )
    _q = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_q)

logger = logging.getLogger("rtmo-hybrid")

DEFAULT_INPUT_SIZE = _q.DEFAULT_INPUT_SIZE


# --------------------------------------------------------------------------- #
# Boundary discovery + graph split
# --------------------------------------------------------------------------- #
def _shape_map(model):
    from onnx import shape_inference

    m = shape_inference.infer_shapes(model)
    vi = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        vi[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    return m, vi


def _find_transformer_cuts(model, vi):
    """Return ``(cut_in, cut_out)`` tensor names bounding the AIFI transformer.

    The transformer flattens a ``[1,C,H,W]`` feature to tokens and unflattens it
    back. ``cut_in`` is the 4-D tensor feeding the flatten ``Reshape`` that is an
    ancestor of the (single) ``Softmax``; ``cut_out`` is the 4-D tensor produced
    by the unflatten ``Reshape`` that descends from it.
    """
    g = model.graph
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}

    softmax = [n for n in g.node if n.op_type == "Softmax"]
    if len(softmax) != 1:
        raise RuntimeError(f"expected exactly one Softmax (AIFI attention), found {len(softmax)}")
    sm = softmax[0]

    def ancestors(start_tensors):
        seen, stack = set(), list(start_tensors)
        nodes = set()
        while stack:
            t = stack.pop()
            pi = prod.get(t)
            if pi is None or pi in nodes:
                continue
            nodes.add(pi)
            stack.extend(g.node[pi].input)
        return nodes

    def descendants(start_nodes):
        cons = {}
        for i, n in enumerate(g.node):
            for inp in n.input:
                cons.setdefault(inp, []).append(i)
        seen, stack = set(start_nodes), list(start_nodes)
        while stack:
            ni = stack.pop()
            for o in g.node[ni].output:
                for c in cons.get(o, []):
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
        return seen

    anc = ancestors(list(sm.input))
    rank4_reshape_in = [
        (i, g.node[i]) for i in anc
        if g.node[i].op_type == "Reshape" and len(vi.get(g.node[i].input[0], [])) == 4
    ]
    if not rank4_reshape_in:
        raise RuntimeError("could not locate the transformer flatten Reshape")
    # the flatten closest to the boundary = the one whose 4-D input has the most channels
    flatten = max(rank4_reshape_in, key=lambda x: vi[x[1].input[0]][1])[1]
    cut_in = flatten.input[0]

    desc = descendants({prod[o] for o in sm.output})
    rank4_reshape_out = [
        g.node[i] for i in desc
        if g.node[i].op_type == "Reshape" and len(vi.get(g.node[i].output[0], [])) == 4
    ]
    if not rank4_reshape_out:
        raise RuntimeError("could not locate the transformer unflatten Reshape")
    unflatten = min(rank4_reshape_out, key=lambda n: prod[n.output[0]])
    cut_out = unflatten.output[0]
    return cut_in, cut_out


def _partition(model, vi, cut_in, cut_out):
    """Assign every node to BB / TF / HEAD and return the cross-region skip
    tensors the head needs from the backbone (FPN skip connections)."""
    g = model.graph
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}
    init = {i.name for i in g.initializer}

    def ancestors_of(tensor):
        seen, stack = set(), [tensor]
        while stack:
            t = stack.pop()
            pi = prod.get(t)
            if pi is None or pi in seen:
                continue
            seen.add(pi)
            stack.extend(g.node[pi].input)
        return seen

    bb_nodes = ancestors_of(cut_in)          # everything needed to produce cut_in
    tf_nodes = ancestors_of(cut_out) - bb_nodes  # cut_in..cut_out (excl. backbone)

    def region(idx):
        if idx in bb_nodes:
            return "BB"
        if idx in tf_nodes:
            return "TF"
        return "HEAD"

    skips = []  # backbone tensors consumed by the head (besides cut_out)
    for i, n in enumerate(g.node):
        if region(i) != "HEAD":
            continue
        for inp in n.input:
            if inp in init or inp == "":
                continue
            pi = prod.get(inp)
            if pi is not None and region(pi) == "BB" and inp not in skips:
                skips.append(inp)
    # deterministic order: largest spatial (P3) ... smallest (P5-level), stable
    skips.sort(key=lambda t: -(vi[t][2] if len(vi[t]) == 4 else 0))
    return skips


def split_rtmo(prepared_onnx, out_dir):
    """Split a prepared RTMO ONNX into backbone / transformer / head sub-models.

    Returns a dict with the three ONNX paths and the boundary tensor metadata.
    """
    import onnx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(prepared_onnx))
    model, vi = _shape_map(model)
    inferred = out_dir / "rtmo_prepared_inferred.onnx"
    onnx.save(model, str(inferred))  # extract_model needs value_info on disk

    cut_in, cut_out = _find_transformer_cuts(model, vi)
    skips = _partition(model, vi, cut_in, cut_out)
    graph_in = model.graph.input[0].name
    graph_outs = [o.name for o in model.graph.output]

    logger.info("Split boundaries: cut_in=%s %s  cut_out=%s %s  skips=%s",
                cut_in, vi[cut_in], cut_out, vi[cut_out],
                [(s, vi[s]) for s in skips])

    paths = {
        "backbone": out_dir / "rtmo_part_backbone.onnx",
        "transformer": out_dir / "rtmo_part_transformer.onnx",
        "head": out_dir / "rtmo_part_head.onnx",
    }
    onnx.utils.extract_model(str(inferred), str(paths["backbone"]),
                             [graph_in], [cut_in] + skips)
    onnx.utils.extract_model(str(inferred), str(paths["transformer"]),
                             [cut_in], [cut_out])
    onnx.utils.extract_model(str(inferred), str(paths["head"]),
                             [cut_out] + skips, graph_outs)

    return {
        "paths": paths,
        "cut_in": cut_in, "cut_out": cut_out, "skips": skips,
        "graph_in": graph_in, "graph_outs": graph_outs,
        "shapes": {t: vi[t] for t in [cut_in, cut_out, *skips]},
    }


# --------------------------------------------------------------------------- #
# fp32 intermediate activations (calibration + verification taps)
# --------------------------------------------------------------------------- #
def collect_intermediates(onnx_path, inputs_nchw, tensor_names):
    """Run the full fp32 ONNX with ``tensor_names`` exposed as extra outputs.

    Returns a list (one per input) of ``{name: NCHW array}``.
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    have = {o.name for o in model.graph.output}
    from onnx import helper
    for t in tensor_names:
        if t not in have:
            model.graph.output.append(helper.ValueInfoProto(name=t))
    tmp = str(Path(onnx_path).with_suffix(".taps.onnx"))
    onnx.save(model, tmp)

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    want = list(tensor_names)
    return [dict(zip(want, sess.run(want, {in_name: x}))) for x in inputs_nchw]


# --------------------------------------------------------------------------- #
# Per-part convert + PTQ (multi-input aware)
# --------------------------------------------------------------------------- #
def _nhwc(a):
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def _saved_model_input_specs(saved_model_dir):
    """Return ``{signature_input_name: NHWC shape tuple}`` for the SavedModel.

    Read from the ``serving_default`` SignatureDef — those keys are exactly what
    the TFLite calibrator's signature runner uses to look up a dict sample, so a
    dict keyed by them (matched to arrays by shape) is order- and
    name-sanitisation-independent (``structured_input_signature`` kwargs keys and
    the tflite input order are both unreliable across loads).
    """
    from tensorflow.python.saved_model import loader_impl

    sm = loader_impl.parse_saved_model(str(saved_model_dir))
    sig = sm.meta_graphs[0].signature_def["serving_default"]
    return {key: tuple(int(d.size) for d in info.tensor_shape.dim)
            for key, info in sig.inputs.items()}


def convert_and_quantize_part(part_onnx, tf_dir, rep_samples_nhwc, scheme, out_path,
                              int8_io=True):
    """onnx2tf -> TFLite PTQ for one part.

    ``rep_samples_nhwc`` is a list of samples, each a list of NHWC float arrays
    (matched to the model's inputs by shape). ``scheme`` is ``int8`` /
    ``int16x8`` / ``bf16`` (bf16 = float fallback, weights kept high precision).
    """
    import tensorflow as tf

    tf_paths = _q.onnx_to_tf(part_onnx, tf_dir)
    sm = tf_paths["saved_model"]

    if scheme == "bf16":
        # float32 TFLite (no PTQ) — upper bound for the block; weights stay fp.
        Path(out_path).write_bytes(Path(tf_paths["float_tflite"]).read_bytes())
        logger.info("Wrote bf16(float) part TFLite: %s", out_path)
        return Path(out_path)

    # Feed a dict keyed by the SignatureDef input names (order-independent),
    # routing each calibration array to its slot by matching NHWC shape.
    specs = _saved_model_input_specs(sm)  # {signature_name: NHWC shape}

    def rep_gen():
        for sample in rep_samples_nhwc:
            by_shape = {tuple(int(d) for d in a.shape): a.astype(np.float32) for a in sample}
            yield {name: by_shape[shape] for name, shape in specs.items()}

    converter = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    if scheme == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if int8_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    elif scheme == "int16x8":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    Path(out_path).write_bytes(converter.convert())
    logger.info("Wrote %s part TFLite: %s", scheme, out_path)
    return Path(out_path)


# --------------------------------------------------------------------------- #
# Chained inference (layout-aware: canonical NCHW between parts)
# --------------------------------------------------------------------------- #
# onnx2tf transposes most 4-D tensors to NHWC, but at a split seam a boundary
# tensor can come out in either layout (e.g. the transformer's unflatten Reshape
# emits NCHW while the head expects NHWC for the same tensor). We keep every
# intermediate in canonical NCHW and reconcile against each part's actual tflite
# I/O shape by matching NCHW-vs-NHWC of the known logical shape.
def _nhwc_of(nchw):
    n = tuple(int(x) for x in nchw)
    return (n[0], n[2], n[3], n[1]) if len(n) == 4 else n


def _feed_layout(nchw_arr, tflite_shape):
    """Return ``nchw_arr`` transposed to whatever layout the tflite input wants."""
    s = tuple(int(x) for x in tflite_shape)
    n = tuple(int(x) for x in nchw_arr.shape)
    if s == n:
        return nchw_arr
    if len(n) == 4 and s == _nhwc_of(n):
        return np.transpose(nchw_arr, (0, 2, 3, 1))
    raise ValueError(f"cannot map nchw {n} onto tflite input {s}")


def _to_nchw(arr, nchw_shape):
    """Canonicalise a tflite output (NCHW or NHWC layout) to NCHW ``nchw_shape``."""
    s = tuple(int(x) for x in arr.shape)
    n = tuple(int(x) for x in nchw_shape)
    if s == n:
        return arr
    if len(n) == 4 and s == _nhwc_of(n):
        return np.transpose(arr, (0, 3, 1, 2))
    raise ValueError(f"cannot map tflite output {s} onto nchw {n}")


def _run_part(tflite_path, inputs_nchw, out_nchw_shapes):
    """Run one part. ``inputs_nchw`` are canonical-NCHW arrays (matched to the
    part's inputs by shape); returns ``{nchw_shape: nchw_array}`` (dequantized)."""
    interp = _q._tflite_interpreter(tflite_path)
    for d in interp.get_input_details():
        s = tuple(int(x) for x in d["shape"])
        arr = next(a for a in inputs_nchw
                   if tuple(int(x) for x in a.shape) == s or _nhwc_of(a.shape) == s)
        arr = _feed_layout(arr, s)
        if d["dtype"] in (np.int8, np.uint8, np.int16):
            scale, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            arr = np.clip(np.round(arr / scale + zp), info.min, info.max).astype(d["dtype"])
        interp.set_tensor(d["index"], arr.astype(d["dtype"]))
    interp.invoke()
    out = {}
    for od in interp.get_output_details():
        v = interp.get_tensor(od["index"])
        if od["dtype"] in (np.int8, np.uint8, np.int16):
            scale, zp = od["quantization"]
            v = (v.astype(np.float32) - zp) * scale
        s = tuple(int(x) for x in v.shape)
        nchw = next(n for n in out_nchw_shapes
                    if tuple(int(x) for x in n) == s or _nhwc_of(n) == s)
        out[tuple(int(x) for x in nchw)] = _to_nchw(v, nchw)
    return out


def run_hybrid(parts, image_nchw, cuts, out_nchw_shapes):
    """Chain backbone -> transformer -> head for one image; return
    ``{nchw_shape: nchw_array}`` for the eight model outputs."""
    cut_in, cut_out, skips = cuts["cut_in"], cuts["cut_out"], cuts["skips"]
    bb = _run_part(parts["backbone"], [image_nchw], [cut_in] + skips)
    p5 = bb[tuple(cut_in)]
    skip_arrs = [bb[tuple(s)] for s in skips]

    tf_out = _run_part(parts["transformer"], [p5], [cut_out])
    p5t = tf_out[tuple(cut_out)]

    return _run_part(parts["head"], [p5t] + skip_arrs, out_nchw_shapes)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def quantize_hybrid(
    onnx_path,
    out_dir,
    images_dir,
    input_size=DEFAULT_INPUT_SIZE,
    n_calib=100,
    n_verify=16,
    mean=0.0,
    std=1.0,
    transformer_scheme="int16x8",
    already_prepared=False,
):
    """Build + evaluate the hybrid (int8 convs / high-precision transformer).

    Returns a dict with the part TFLite paths and the per-output cosine accuracy
    (hybrid-vs-fp32), plus the split metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Preparing ONNX (simplify + quick-GELU)")
    if already_prepared:
        prepared = Path(onnx_path)
    else:
        prepared = _q.prepare_onnx(onnx_path, out_dir / "rtmo_prepared.onnx", input_size)

    logger.info("[2/6] Splitting at transformer boundaries")
    split = split_rtmo(prepared, out_dir)
    cut_in, cut_out, skips = split["cut_in"], split["cut_out"], split["skips"]

    logger.info("[3/6] Building calibration + verification sets")
    calib_nhwc = _q.build_calibration_set(str(images_dir), n_calib, input_size, mean, std)
    calib_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in calib_nhwc]
    verify_nhwc = _q.build_calibration_set(str(images_dir), n_verify, input_size, mean, std, seed=99)
    verify_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in verify_nhwc]

    logger.info("[4/6] Collecting fp32 intermediate activations for per-part calibration")
    taps = [cut_in, cut_out] + skips
    calib_mid = collect_intermediates(prepared, calib_nchw, taps)

    # NHWC views of the fp32 intermediates, keyed by tap name
    def mids_nhwc(records, name):
        return [_nhwc(r[name]) for r in records]

    logger.info("[5/6] Converting + quantizing the three parts")
    parts = {}
    # backbone: image -> int8
    parts["backbone"] = convert_and_quantize_part(
        split["paths"]["backbone"], out_dir / "tf_backbone",
        [[x] for x in calib_nhwc], "int8", out_dir / "rtmo_hybrid_backbone_int8.tflite")
    # transformer: cut_in -> {int16x8|bf16}
    parts["transformer"] = convert_and_quantize_part(
        split["paths"]["transformer"], out_dir / "tf_transformer",
        [[m] for m in mids_nhwc(calib_mid, cut_in)], transformer_scheme,
        out_dir / f"rtmo_hybrid_transformer_{transformer_scheme}.tflite", int8_io=False)
    # head: {cut_out, *skips} -> int8
    head_samples = [
        [_nhwc(r[cut_out])] + [_nhwc(r[s]) for s in skips] for r in calib_mid
    ]
    parts["head"] = convert_and_quantize_part(
        split["paths"]["head"], out_dir / "tf_head",
        head_samples, "int8", out_dir / "rtmo_hybrid_head_int8.tflite")

    logger.info("[6/6] Chaining parts + comparing to fp32 ONNX")
    onnx_res = _q.run_onnx(prepared, verify_nchw)
    out_nchw_shapes = [tuple(int(x) for x in v.shape) for v in onnx_res[0].values()]
    cuts = {
        "cut_in": tuple(split["shapes"][cut_in]),
        "cut_out": tuple(split["shapes"][cut_out]),
        "skips": [tuple(split["shapes"][s]) for s in skips],
    }
    hybrid_res = [run_hybrid(parts, x, cuts, out_nchw_shapes) for x in verify_nchw]
    acc = _q.compare_onnx_tflite(onnx_res, hybrid_res)
    for name, (cos, err) in sorted(acc.items()):
        logger.info("  hybrid %-16s cos=%.5f  maxerr=%.4g", name, cos, err)
    cosines = [c for c, _ in acc.values()]
    logger.info("  hybrid MEAN cos=%.5f  MIN cos=%.5f", float(np.mean(cosines)), float(np.min(cosines)))

    return {
        "prepared_onnx": prepared,
        "split": split,
        "parts": parts,
        "accuracy": acc,
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
    }


# Torq compiler flags for the NSS-only hybrid parts (no CSS/host ops).
_HYBRID_TORQ_FLAGS = [
    "--torq-hw=SL2610",
    "--torq-disable-css",
    "--torq-disable-host",
    "--torq-convert-dtypes",
    "--torq-convert-io-dtype",
]


def compile_hybrid(parts, out_dir, *, extra_flags=None, local_compile=False,
                   use_binary=False, compiler_path=None):
    """Compile the three hybrid TFLite parts to NSS-only vmfbs.

    Uses the Torq compiler **Python API** via ``torq.utils.compile`` — the same
    path as the other models (moonshine/gemma), so no ``torq-compile`` /
    ``tosa-converter`` binaries are required. int8 parts compile unsliced; the
    bf16 transformer sliced (slicing helps bf16, hurts int8).

    ``parts`` is the ``{"backbone","transformer","head"}`` TFLite-path dict from
    :func:`quantize_hybrid`. Returns ``{name: vmfb Path}``. Falls back to the
    ``torq-compile`` binary only if ``use_binary`` is set or the Python API is
    unavailable.
    """
    import tempfile

    from ...utils.compile import compile_mlir_for_vm, export_tflite_to_mlir

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = [
        ("backbone", "rtmo_hyb_backbone_int8.vmfb", True),
        ("transformer", "rtmo_hyb_transformer_bf16.vmfb", False),
        ("head", "rtmo_hyb_head_int8.vmfb", True),
    ]
    vmfbs = {}
    for key, vmfb_name, no_slice in spec:
        tfl = Path(parts[key])
        flags = list(_HYBRID_TORQ_FLAGS)
        if no_slice:
            flags.append("--torq-disable-slicing")
        flags += list(extra_flags or [])
        with tempfile.TemporaryDirectory() as td:
            mlir = Path(td) / f"{tfl.stem}.mlir"
            export_tflite_to_mlir(tfl, mlir)          # tflite -> tosa MLIR (Python API)
            vmfb = out_dir / vmfb_name
            compile_mlir_for_vm(mlir, vmfb, target="torq", compiler_args=flags,
                                local_compile=local_compile, use_binary=use_binary,
                                compiler_path=compiler_path)   # MLIR -> vmfb (Python API)
        logger.info("Compiled %s -> %s", key, vmfb)
        vmfbs[key] = vmfb
    return vmfbs


def add_rtmo_hybrid_args(parser):
    parser.add_argument("-i", "--onnx", default="models/rtmo/export/model_nopost_fp32.onnx",
                        help="Source fp32 ONNX (post-processing stripped) (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", default="models/rtmo/export/hybrid",
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--images-dir", default="models/rtmo/calib",
                        help="Directory of representative images (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--n-calib", type=int, default=100)
    parser.add_argument("--n-verify", type=int, default=16)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--transformer-scheme", choices=["int16x8", "bf16"], default="bf16",
                        help="Precision for the transformer part (default: %(default)s)")
    parser.add_argument("--already-prepared", action="store_true",
                        help="Source ONNX is already simplified + quick-GELU'd")
    parser.add_argument("--compile", action="store_true",
                        help="Also compile the three TFLite parts to NSS-only vmfbs "
                             "(Torq compiler Python API; --use-binary to force the binary)")
    from ...utils.compile import add_torq_args
    add_torq_args(parser)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Hybrid-quantize RTMO (int8 convs + int16/bf16 transformer)")
    add_rtmo_hybrid_args(parser)
    args = parser.parse_args()
    res = quantize_hybrid(
        args.onnx, args.out_dir, args.images_dir, args.input_size,
        args.n_calib, args.n_verify, args.mean, args.std,
        args.transformer_scheme, args.already_prepared,
    )
    print("backbone   :", res["parts"]["backbone"])
    print("transformer:", res["parts"]["transformer"])
    print("head       :", res["parts"]["head"])
    print(f"hybrid mean cosine: {res['mean_cosine']:.5f}  min: {res['min_cosine']:.5f}")

    if args.compile:
        vmfbs = compile_hybrid(
            res["parts"], args.out_dir,
            extra_flags=args.compile_flags,
            local_compile=args.local_compile,
            use_binary=args.use_binary,
            compiler_path=args.compiler_path,
        )
        for key, vmfb in vmfbs.items():
            print(f"{key + ' vmfb':<15}: {vmfb}")


if __name__ == "__main__":
    main()
