# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Hybrid RTMO quantization: int8 conv layers + int16/bf16 transformer neck.

RTMO is convolutional except one AIFI transformer block on the stride-32
feature. Full int8 holds on the conv branches (cosine ~0.97) but the attention
neck + keypoint head drop to ~0.90; int16 recovers them but costs 2x on every
conv. This module builds the middle ground — int8 convs, higher precision only
for the transformer — by splitting at the transformer boundaries (discovered
automatically, anchored on the single neck Softmax):

    image -> [backbone int8] -> P5 -> [transformer int16x8/bf16] -> P5' -\\
                  |------------- P3, P4 (skips) -------------> [head int8] -> 8 outputs

Each part is a deployment-grade TFLite from the same onnx2tf + PTQ path as
:mod:`quantize`; the chain is verified against the fp32 ONNX both per-head
(cosine) and after post-processing (decoded detections).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# Reuse quantize.py whether imported as a package module or run standalone (the
# package __init__ pulls LLM deps absent from the dedicated rtmo-quant venv).
try:  # pragma: no cover - import shim
    from . import quantize as _q
except ImportError:  # pragma: no cover
    import importlib.util
    import os

    _spec = importlib.util.spec_from_file_location("rtmo_quantize", os.path.join(os.path.dirname(__file__), "quantize.py"))
    _q = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_q)

logger = logging.getLogger("rtmo-hybrid")

DEFAULT_INPUT_SIZE = _q.DEFAULT_INPUT_SIZE


# ---- Boundary discovery + graph split ----
def _shape_map(model):
    from onnx import shape_inference

    m = shape_inference.infer_shapes(model)
    vi = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        vi[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    return m, vi


def _find_transformer_cuts(model, vi):
    """Return ``(cut_in, cut_out)``: the 4-D tensors feeding the AIFI flatten
    Reshape (ancestor of the single Softmax) and produced by its unflatten."""
    g = model.graph
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}

    softmax = [n for n in g.node if n.op_type == "Softmax"]
    if len(softmax) != 1:
        raise RuntimeError(f"expected exactly one Softmax (AIFI attention), found {len(softmax)}")
    sm = softmax[0]

    def ancestors(start_tensors):
        nodes, stack = set(), list(start_tensors)
        while stack:
            pi = prod.get(stack.pop())
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
            for o in g.node[stack.pop()].output:
                for c in cons.get(o, []):
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
        return seen

    anc = ancestors(list(sm.input))
    rank4_in = [(i, g.node[i]) for i in anc if g.node[i].op_type == "Reshape" and len(vi.get(g.node[i].input[0], [])) == 4]
    if not rank4_in:
        raise RuntimeError("could not locate the transformer flatten Reshape")
    flatten = max(rank4_in, key=lambda x: vi[x[1].input[0]][1])[1]  # boundary = most channels

    desc = descendants({prod[o] for o in sm.output})
    rank4_out = [g.node[i] for i in desc if g.node[i].op_type == "Reshape" and len(vi.get(g.node[i].output[0], [])) == 4]
    if not rank4_out:
        raise RuntimeError("could not locate the transformer unflatten Reshape")
    unflatten = min(rank4_out, key=lambda n: prod[n.output[0]])
    return flatten.input[0], unflatten.output[0]


def _partition(model, vi, cut_in, cut_out):
    """Assign nodes to BB / TF / HEAD; return the backbone->head FPN skip tensors."""
    g = model.graph
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}
    init = {i.name for i in g.initializer}

    def ancestors_of(tensor):
        seen, stack = set(), [tensor]
        while stack:
            pi = prod.get(stack.pop())
            if pi is None or pi in seen:
                continue
            seen.add(pi)
            stack.extend(g.node[pi].input)
        return seen

    bb_nodes = ancestors_of(cut_in)
    tf_nodes = ancestors_of(cut_out) - bb_nodes

    skips = []
    for i, n in enumerate(g.node):
        if i in bb_nodes or i in tf_nodes:
            continue
        for inp in n.input:
            if inp in init or inp == "":
                continue
            pi = prod.get(inp)
            if pi is not None and pi in bb_nodes and inp not in skips:
                skips.append(inp)
    skips.sort(key=lambda t: -(vi[t][2] if len(vi[t]) == 4 else 0))  # P3 (largest) first
    return skips


def split_rtmo(prepared_onnx, out_dir):
    """Split a prepared RTMO ONNX into backbone / transformer / head sub-models."""
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
    logger.info("Split boundaries: cut_in=%s %s  cut_out=%s %s  skips=%s", cut_in, vi[cut_in], cut_out, vi[cut_out], [(s, vi[s]) for s in skips])

    paths = {"backbone": out_dir / "rtmo_part_backbone.onnx", "transformer": out_dir / "rtmo_part_transformer.onnx", "head": out_dir / "rtmo_part_head.onnx"}
    onnx.utils.extract_model(str(inferred), str(paths["backbone"]), [graph_in], [cut_in] + skips)
    onnx.utils.extract_model(str(inferred), str(paths["transformer"]), [cut_in], [cut_out])
    onnx.utils.extract_model(str(inferred), str(paths["head"]), [cut_out] + skips, graph_outs)
    return {"paths": paths, "cut_in": cut_in, "cut_out": cut_out, "skips": skips, "graph_in": graph_in, "graph_outs": graph_outs, "shapes": {t: vi[t] for t in [cut_in, cut_out, *skips]}}


# ---- fp32 intermediate activations (calibration + verification taps) ----
def collect_intermediates(onnx_path, inputs_nchw, tensor_names):
    """Run the fp32 ONNX with ``tensor_names`` as extra outputs -> per-input {name: array}."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    model = onnx.load(str(onnx_path))
    have = {o.name for o in model.graph.output}
    for t in tensor_names:
        if t not in have:
            model.graph.output.append(helper.ValueInfoProto(name=t))
    tmp = str(Path(onnx_path).with_suffix(".taps.onnx"))
    onnx.save(model, tmp)

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    want = list(tensor_names)
    return [dict(zip(want, sess.run(want, {in_name: x}))) for x in inputs_nchw]


# ---- Per-part convert + PTQ (multi-input aware) ----
def _nhwc(a):
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def _saved_model_input_specs(saved_model_dir):
    """{serving_default input name: NHWC shape} — the keys the TFLite calibrator
    looks up, so dict samples are order/name-sanitisation independent."""
    from tensorflow.python.saved_model import loader_impl

    sig = loader_impl.parse_saved_model(str(saved_model_dir)).meta_graphs[0].signature_def["serving_default"]
    return {key: tuple(int(d.size) for d in info.tensor_shape.dim) for key, info in sig.inputs.items()}


def convert_and_quantize_part(part_onnx, tf_dir, rep_samples_nhwc, scheme, out_path, int8_io=True):
    """onnx2tf -> TFLite PTQ for one part. ``rep_samples_nhwc``: list of samples,
    each a list of NHWC arrays matched to inputs by shape. ``scheme``: int8 /
    int16x8 / bf16 (bf16 = float TFLite, no PTQ)."""
    import tensorflow as tf

    tf_paths = _q.onnx_to_tf(part_onnx, tf_dir)
    sm = tf_paths["saved_model"]

    if scheme == "bf16":
        Path(out_path).write_bytes(Path(tf_paths["float_tflite"]).read_bytes())
        logger.info("Wrote bf16(float) part TFLite: %s", out_path)
        return Path(out_path)

    specs = _saved_model_input_specs(sm)

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
            converter.inference_input_type = converter.inference_output_type = tf.int8
    elif scheme == "int16x8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    Path(out_path).write_bytes(converter.convert())
    logger.info("Wrote %s part TFLite: %s", scheme, out_path)
    return Path(out_path)


# ---- Chained inference. onnx2tf transposes most 4-D tensors to NHWC, but a
# seam tensor can come out in either layout; keep intermediates in canonical
# NCHW and reconcile against each part's actual tflite I/O shape. ----
def _nhwc_of(nchw):
    n = tuple(int(x) for x in nchw)
    return (n[0], n[2], n[3], n[1]) if len(n) == 4 else n


def _feed_layout(nchw_arr, tflite_shape):
    """Transpose a canonical-NCHW array to whatever layout the tflite input wants."""
    s, n = tuple(int(x) for x in tflite_shape), tuple(int(x) for x in nchw_arr.shape)
    if s == n:
        return nchw_arr
    if len(n) == 4 and s == _nhwc_of(n):
        return np.transpose(nchw_arr, (0, 2, 3, 1))
    raise ValueError(f"cannot map nchw {n} onto tflite input {s}")


def _to_nchw(arr, nchw_shape):
    """Canonicalise a tflite output (either layout) to NCHW ``nchw_shape``."""
    s, n = tuple(int(x) for x in arr.shape), tuple(int(x) for x in nchw_shape)
    if s == n:
        return arr
    if len(n) == 4 and s == _nhwc_of(n):
        return np.transpose(arr, (0, 3, 1, 2))
    raise ValueError(f"cannot map tflite output {s} onto nchw {n}")


def _run_part(tflite_path, inputs_nchw, out_nchw_shapes):
    """Run one part (inputs matched by shape); return {nchw_shape: dequantized nchw array}."""
    interp = _q._tflite_interpreter(tflite_path)
    for d in interp.get_input_details():
        s = tuple(int(x) for x in d["shape"])
        arr = _feed_layout(next(a for a in inputs_nchw if tuple(int(x) for x in a.shape) == s or _nhwc_of(a.shape) == s), s)
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
        nchw = next(n for n in out_nchw_shapes if tuple(int(x) for x in n) == s or _nhwc_of(n) == s)
        out[tuple(int(x) for x in nchw)] = _to_nchw(v, nchw)
    return out


def run_hybrid(parts, image_nchw, cuts, out_nchw_shapes):
    """Chain backbone -> transformer -> head for one image -> {nchw_shape: array}."""
    cut_in, cut_out, skips = cuts["cut_in"], cuts["cut_out"], cuts["skips"]
    bb = _run_part(parts["backbone"], [image_nchw], [cut_in] + skips)
    p5t = _run_part(parts["transformer"], [bb[tuple(cut_in)]], [cut_out])[tuple(cut_out)]
    return _run_part(parts["head"], [p5t] + [bb[tuple(s)] for s in skips], out_nchw_shapes)


# ---- Orchestration ----
def quantize_hybrid(onnx_path, out_dir, images_dir, input_size=DEFAULT_INPUT_SIZE, n_calib=100, n_verify=16, mean=0.0, std=1.0, transformer_scheme="int16x8", already_prepared=False):
    """Build + evaluate the hybrid; return part paths, accuracy, split metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Preparing ONNX (simplify + quick-GELU)")
    prepared = Path(onnx_path) if already_prepared else _q.prepare_onnx(onnx_path, out_dir / "rtmo_prepared.onnx", input_size)

    logger.info("[2/6] Splitting at transformer boundaries")
    split = split_rtmo(prepared, out_dir)
    cut_in, cut_out, skips = split["cut_in"], split["cut_out"], split["skips"]

    logger.info("[3/6] Building calibration + verification sets")
    calib_nhwc = _q.build_calibration_set(str(images_dir), n_calib, input_size, mean, std)
    calib_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in calib_nhwc]
    verify_nhwc = _q.build_calibration_set(str(images_dir), n_verify, input_size, mean, std, seed=99)
    verify_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in verify_nhwc]

    logger.info("[4/6] Collecting fp32 intermediate activations for per-part calibration")
    calib_mid = collect_intermediates(prepared, calib_nchw, [cut_in, cut_out] + skips)

    logger.info("[5/6] Converting + quantizing the three parts")
    parts = {}
    parts["backbone"] = convert_and_quantize_part(split["paths"]["backbone"], out_dir / "tf_backbone", [[x] for x in calib_nhwc], "int8", out_dir / "rtmo_hybrid_backbone_int8.tflite")
    parts["transformer"] = convert_and_quantize_part(split["paths"]["transformer"], out_dir / "tf_transformer", [[_nhwc(r[cut_in])] for r in calib_mid], transformer_scheme, out_dir / f"rtmo_hybrid_transformer_{transformer_scheme}.tflite", int8_io=False)
    head_samples = [[_nhwc(r[cut_out])] + [_nhwc(r[s]) for s in skips] for r in calib_mid]
    parts["head"] = convert_and_quantize_part(split["paths"]["head"], out_dir / "tf_head", head_samples, "int8", out_dir / "rtmo_hybrid_head_int8.tflite")

    logger.info("[6/6] Chaining parts + comparing to fp32 ONNX")
    onnx_res = _q.run_onnx(prepared, verify_nchw)
    out_nchw_shapes = [tuple(int(x) for x in v.shape) for v in onnx_res[0].values()]
    cuts = {"cut_in": tuple(split["shapes"][cut_in]), "cut_out": tuple(split["shapes"][cut_out]), "skips": [tuple(split["shapes"][s]) for s in skips]}
    hybrid_res = [run_hybrid(parts, x, cuts, out_nchw_shapes) for x in verify_nchw]
    acc = _q.compare_onnx_tflite(onnx_res, hybrid_res)
    for name, (cos, err) in sorted(acc.items()):
        logger.info("  hybrid %-16s cos=%.5f  maxerr=%.4g", name, cos, err)
    cosines = [c for c, _ in acc.values()]
    logger.info("  hybrid MEAN cos=%.5f  MIN cos=%.5f", float(np.mean(cosines)), float(np.min(cosines)))

    # Also compare after post-processing (decoded detections, not just raw heads).
    pp = compare_postprocess(onnx_res, hybrid_res)
    logger.info("  postproc: dets onnx=%d hybrid=%d matched=%d (%d/%d imgs same count) | IoU=%.4f  kptdelta=%.2fpx  scoreMAE=%.4f",
                pp["n_onnx"], pp["n_hybrid"], pp["matched"], pp["count_agree"], pp["n_images"], pp["mean_iou"], pp["mean_kpt_px"], pp["mean_score_mae"])

    return {"prepared_onnx": prepared, "split": split, "parts": parts, "accuracy": acc, "mean_cosine": float(np.mean(cosines)), "min_cosine": float(np.min(cosines)), "postprocess": pp}


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_postprocess(onnx_res, hybrid_res, score_thr=0.30, iou_thr=0.5):
    """Compare fp32-ONNX vs hybrid *after* post-processing: decode both head sets,
    threshold at ``score_thr``, greedily match boxes by IoU; report count
    agreement, mean IoU, keypoint pixel error, and score MAE."""
    from ._postprocess import model_postprocess

    ious, kpt_px, score_ae = [], [], []
    tot_o = tot_h = matched = count_agree = 0
    for ro, rh in zip(onnx_res, hybrid_res):
        # Re-key hybrid outputs to head names by (unique) NCHW shape.
        shape_to_name = {np.asarray(v).shape: k for k, v in ro.items()}
        rh = {shape_to_name[np.asarray(v).shape]: v for v in rh.values()}
        do, ko = model_postprocess(ro)
        dh, kh = model_postprocess(rh)
        do, ko, dh, kh = do[0], ko[0], dh[0], kh[0]
        mo, mh = do[:, 4] > score_thr, dh[:, 4] > score_thr
        do, ko, dh, kh = do[mo], ko[mo], dh[mh], kh[mh]
        tot_o += len(do)
        tot_h += len(dh)
        count_agree += int(len(do) == len(dh))
        used = set()
        for j in range(len(do)):
            best, bj = 0.0, -1
            for k in range(len(dh)):
                if k not in used and (iou := _box_iou(do[j, :4], dh[k, :4])) > best:
                    best, bj = iou, k
            if bj >= 0 and best >= iou_thr:
                used.add(bj)
                matched += 1
                ious.append(best)
                score_ae.append(abs(float(do[j, 4]) - float(dh[bj, 4])))
                vis = ko[j, :, 2] > score_thr
                if vis.any():
                    kpt_px.append(float(np.linalg.norm(ko[j, vis, :2] - kh[bj, vis, :2], axis=1).mean()))
    _mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"n_images": len(onnx_res), "n_onnx": tot_o, "n_hybrid": tot_h, "matched": matched, "count_agree": count_agree, "mean_iou": _mean(ious), "mean_kpt_px": _mean(kpt_px), "mean_score_mae": _mean(score_ae)}


# Torq compiler flags for the NSS-only hybrid parts (no CSS/host ops).
_HYBRID_TORQ_FLAGS = ["--torq-hw=SL2610", "--torq-disable-css", "--torq-disable-host", "--torq-convert-dtypes", "--torq-convert-io-dtype"]


def _compile_parts(parts, out_dir, *, extra_flags=None, local_compile=False, use_binary=False, compiler_path=None):
    """In-process compile of the three TFLite parts to NSS-only vmfbs. Must run
    TensorFlow-free (see :func:`compile_hybrid`) — call that, not this."""
    import tempfile

    from ...utils.compile import compile_mlir_for_vm, export_tflite_to_mlir

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # int8 parts compile unsliced; the bf16 transformer sliced (slicing helps bf16, hurts int8).
    spec = [("backbone", "rtmo_hyb_backbone_int8.vmfb", True), ("transformer", "rtmo_hyb_transformer_bf16.vmfb", False), ("head", "rtmo_hyb_head_int8.vmfb", True)]
    vmfbs = {}
    for key, vmfb_name, no_slice in spec:
        tfl = Path(parts[key])
        flags = _HYBRID_TORQ_FLAGS + (["--torq-disable-slicing"] if no_slice else []) + list(extra_flags or [])
        with tempfile.TemporaryDirectory() as td:
            mlir = Path(td) / f"{tfl.stem}.mlir"
            export_tflite_to_mlir(tfl, mlir)
            vmfb = out_dir / vmfb_name
            compile_mlir_for_vm(mlir, vmfb, target="torq", compiler_args=flags, local_compile=local_compile, use_binary=use_binary, compiler_path=compiler_path)
        logger.info("Compiled %s -> %s", key, vmfb)
        vmfbs[key] = vmfb
    return vmfbs


def compile_hybrid(parts, out_dir, *, extra_flags=None, local_compile=False, use_binary=False, compiler_path=None):
    """Compile the three hybrid TFLite parts to NSS-only vmfbs via the Torq
    compiler Python API (no binaries; ``use_binary`` forces the fallback).

    If TensorFlow is already imported (e.g. right after :func:`quantize_hybrid`),
    the compile runs in a fresh subprocess (:mod:`._compile_worker`): TF and the
    ``torq.compiler`` wheel each statically link an LLVM, and both in one process
    abort with "Option 'remarks-section' registered more than once".
    """
    import sys

    if "tensorflow" not in sys.modules:
        return _compile_parts(parts, out_dir, extra_flags=extra_flags, local_compile=local_compile, use_binary=use_binary, compiler_path=compiler_path)

    import json
    import subprocess
    import tempfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TensorFlow is loaded; compiling in a fresh subprocess (LLVM isolation)")
    with tempfile.TemporaryDirectory() as td:
        job, res = Path(td) / "job.json", Path(td) / "vmfbs.json"
        job.write_text(json.dumps({"parts": {k: str(v) for k, v in parts.items()}, "out_dir": str(out_dir), "extra_flags": list(extra_flags or []), "local_compile": local_compile, "use_binary": use_binary, "compiler_path": str(compiler_path) if compiler_path else None, "result": str(res)}))
        subprocess.run([sys.executable, "-m", "torq.models.rtmo._compile_worker", str(job)], check=True)
        return {k: Path(v) for k, v in json.loads(res.read_text()).items()}


def add_rtmo_hybrid_args(parser):
    parser.add_argument("-i", "--onnx", default="models/rtmo/export/model_nopost_fp32.onnx", help="Source fp32 ONNX (post-processing stripped) (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", default="models/rtmo/export/hybrid", help="Output directory (default: %(default)s)")
    parser.add_argument("--images-dir", default="models/rtmo/calib", help="Directory of representative images (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--n-calib", type=int, default=100)
    parser.add_argument("--n-verify", type=int, default=16)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--transformer-scheme", choices=["int16x8", "bf16"], default="bf16", help="Precision for the transformer part (default: %(default)s)")
    parser.add_argument("--already-prepared", action="store_true", help="Source ONNX is already simplified + quick-GELU'd")
    parser.add_argument("--compile", action="store_true", help="Also compile the parts to NSS-only vmfbs (Python API; --use-binary forces the binary)")
    from ...utils.compile import add_torq_args
    add_torq_args(parser)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Hybrid-quantize RTMO (int8 convs + int16/bf16 transformer)")
    add_rtmo_hybrid_args(parser)
    args = parser.parse_args()
    res = quantize_hybrid(args.onnx, args.out_dir, args.images_dir, args.input_size, args.n_calib, args.n_verify, args.mean, args.std, args.transformer_scheme, args.already_prepared)
    print("backbone   :", res["parts"]["backbone"])
    print("transformer:", res["parts"]["transformer"])
    print("head       :", res["parts"]["head"])
    print(f"hybrid mean cosine: {res['mean_cosine']:.5f}  min: {res['min_cosine']:.5f}")

    if args.compile:
        vmfbs = compile_hybrid(res["parts"], args.out_dir, extra_flags=args.compile_flags, local_compile=args.local_compile, use_binary=args.use_binary, compiler_path=args.compiler_path)
        for key, vmfb in vmfbs.items():
            print(f"{key + ' vmfb':<15}: {vmfb}")


if __name__ == "__main__":
    main()
