# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Quantize the (post-processing-stripped) YOLO26 model to int8 TFLite (weights AND
activations both int8 -- TFLite's full-integer PTQ scheme).

Pipeline: (1) onnxsim folds any residual shape guards; (2) onnx2tf with the
``tf_converter`` backend (the default flatbuffer_direct backend's strict quantizer
rejects the neck attention block's activation x activation MatMul -- same issue as
RTMO's AIFI), rewriting NCHW->NHWC; (3) verify float TFLite vs ONNX (cosine);
(4) PTQ on a representative image set (int8, or int16x8 for higher accuracy on the
attention block).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("yolo26-quantize")

DEFAULT_INPUT_SIZE = 320


# ---- ONNX preparation ----
def prepare_onnx(onnx_path, out_path, input_size: int = DEFAULT_INPUT_SIZE) -> Path:
    """Simplify the ONNX so it converts and quantizes cleanly."""
    import onnx
    import onnxsim

    model = onnx.load(str(onnx_path))
    model, ok = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name: [1, 3, input_size, input_size]})
    if not ok:
        logger.warning("onnxsim reported it could not fully simplify the model")
    onnx.checker.check_model(model)
    out_path = Path(out_path)
    onnx.save(model, str(out_path))
    return out_path


# ---- Preprocessing / representative dataset ----
def preprocess(path, input_size=DEFAULT_INPUT_SIZE, *, crop=None, flip=False) -> np.ndarray:
    """Return an NHWC [1,S,S,3] float32 tensor (RGB, [0,1] -- matches Ultralytics' own preprocessing)."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    if crop is not None:
        h, w = img.shape[:2]
        y0, x0, y1, x1 = crop
        img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if flip:
        img = img[:, ::-1]
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
    return img[None].astype(np.float32)


def build_calibration_set(images_dir, n_samples=200, input_size=DEFAULT_INPUT_SIZE, seed=0) -> list[np.ndarray]:
    """~n_samples NHWC calibration tensors from natural images (crop/flip aug after one clean pass)."""
    paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(images_dir, "**", ext), recursive=True) if ":Zone.Identifier" not in p)
    if not paths:
        raise FileNotFoundError(f"no images found under {images_dir}")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    i = 0
    while len(samples) < n_samples and paths:
        path = paths[i % len(paths)]
        i += 1
        if len(samples) < len(paths):
            crop, flip = None, False
        else:
            y0, x0 = rng.uniform(0, 0.3, size=2)
            y1, x1 = rng.uniform(0.7, 1.0, size=2)
            crop, flip = (float(y0), float(x0), float(y1), float(x1)), bool(rng.integers(2))
        try:
            samples.append(preprocess(path, input_size, crop=crop, flip=flip))
        except ValueError:
            paths.remove(path)
    logger.info("Built %d calibration samples from %s", len(samples), images_dir)
    return samples


# ---- Conversion / backends ----
def onnx_to_tf(onnx_path, out_dir) -> dict[str, Path]:
    """ONNX -> TF SavedModel + float32 TFLite via onnx2tf (tf_converter backend)."""
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(out_dir), "-tb", "tf_converter", "-n"]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "TF_USE_LEGACY_KERAS": "1"})
    return {"saved_model": out_dir, "float_tflite": out_dir / f"{Path(onnx_path).stem}_float32.tflite"}


def run_onnx(onnx_path, inputs_nchw: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    return [{n: o for n, o in zip(out_names, sess.run(out_names, {in_name: x}))} for x in inputs_nchw]


def _tflite_interpreter(tflite_path):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    return interp


def run_tflite(tflite_path, inputs_nhwc: list[np.ndarray]) -> list[dict[tuple, np.ndarray]]:
    """Run a TFLite model; outputs dequantized, NCHW, keyed by shape for ONNX matching."""
    interp = _tflite_interpreter(tflite_path)
    inp = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    results = []
    for x in inputs_nhwc:
        xin = x
        if inp["dtype"] in (np.int8, np.uint8):
            scale, zp = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            xin = np.clip(np.round(x / scale + zp), info.min, info.max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], xin)
        interp.invoke()
        out = {}
        for od in out_details:
            v = interp.get_tensor(od["index"])
            if od["dtype"] in (np.int8, np.uint8, np.int16):
                scale, zp = od["quantization"]
                v = (v.astype(np.float32) - zp) * scale
            if v.ndim == 4:
                v = np.transpose(v, (0, 3, 1, 2))
            out[v.shape] = v
        results.append(out)
    return results


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def compare_onnx_tflite(onnx_results, tflite_results) -> dict[str, tuple[float, float]]:
    """Match ONNX to TFLite outputs by (NCHW) shape -> {name: (mean cosine, max abs err)}."""
    per_name: dict[str, list[tuple[float, float]]] = {}
    for oref, otf in zip(onnx_results, tflite_results):
        for name, ov in oref.items():
            tv = otf.get(ov.shape)
            pair = (float("nan"), float("nan")) if tv is None else (_cosine(ov, tv), float(np.abs(ov.astype(np.float64) - tv.astype(np.float64)).max()))
            per_name.setdefault(name, []).append(pair)
    return {name: (float(np.nanmean([c for c, _ in v])), float(np.nanmax([e for _, e in v]))) for name, v in per_name.items()}


# ---- PTQ ----
def quantize_int8(saved_model_dir, calibration_set, out_path, scheme="int8") -> Path:
    """PTQ from a SavedModel: 'int8' (full int8, weights AND activations) or
    'int16x8' (int8 weights / int16 activations)."""
    import tensorflow as tf

    def rep_gen():
        for x in calibration_set:
            yield [x.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    if scheme == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = converter.inference_output_type = tf.int8
    elif scheme == "int16x8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    model_bytes = converter.convert()
    out_path = Path(out_path)
    out_path.write_bytes(model_bytes)
    logger.info("Wrote %s TFLite: %s", scheme, out_path)
    return out_path


# ---- Orchestration / CLI ----
def quantize_yolo26(onnx_path, out_dir, images_dir, input_size=DEFAULT_INPUT_SIZE, n_calib=200, n_verify=16, scheme="int8") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Preparing ONNX (simplify)")
    prepared = prepare_onnx(onnx_path, out_dir / "yolo26_prepared.onnx", input_size)

    logger.info("[2/4] ONNX -> TensorFlow")
    tf_paths = onnx_to_tf(prepared, out_dir / "tf")

    logger.info("[3/4] Verifying float conversion, then %s PTQ", scheme)
    calib = build_calibration_set(str(images_dir), n_calib, input_size)
    verify_nhwc = build_calibration_set(str(images_dir), n_verify, input_size, seed=99)
    verify_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in verify_nhwc]
    onnx_res = run_onnx(prepared, verify_nchw)
    float_acc = compare_onnx_tflite(onnx_res, run_tflite(tf_paths["float_tflite"], verify_nhwc))
    for name, (cos, err) in sorted(float_acc.items()):
        logger.info("  float  %-16s cos=%.5f", name, cos)

    int8_path = quantize_int8(tf_paths["saved_model"], calib, out_dir / f"yolo26_{scheme}.tflite", scheme)

    logger.info("[4/4] Verifying quantized accuracy vs fp32 ONNX")
    q_acc = compare_onnx_tflite(onnx_res, run_tflite(int8_path, verify_nhwc))
    for name, (cos, err) in sorted(q_acc.items()):
        logger.info("  %-6s %-16s cos=%.5f", scheme, name, cos)

    return {"prepared_onnx": prepared, "float_tflite": tf_paths["float_tflite"], "quant_tflite": int8_path, "float_accuracy": float_acc, "quant_accuracy": q_acc}


def add_yolo26_quantize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--onnx", default="models/yolo26/export/model_nopost_fp32.onnx", help="Source fp32 ONNX (post-processing stripped) (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", default="models/yolo26/export/int8", help="Output directory (default: %(default)s)")
    parser.add_argument("--images-dir", default="models/yolo26/calib", help="Directory of representative images (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--n-calib", type=int, default=200, help="Calibration sample count")
    parser.add_argument("--n-verify", type=int, default=16, help="Accuracy-check sample count")
    parser.add_argument("--scheme", choices=["int8", "int16x8"], default="int8", help="Quantization scheme (default: %(default)s)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Quantize YOLO26 to int8/int16x8 TFLite")
    add_yolo26_quantize_args(parser)
    args = parser.parse_args()
    quantize_yolo26(args.onnx, args.out_dir, args.images_dir, args.input_size, args.n_calib, args.n_verify, args.scheme)


if __name__ == "__main__":
    main()
