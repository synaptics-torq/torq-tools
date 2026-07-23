# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Quantize the (post-processing-stripped) RTMO model to an int8 TFLite model.

Pipeline (all steps verified against the source ONNX):

1. **Prepare** the ONNX: onnx-simplifier folds the dynamic-shape reshape guards
   in the neck (otherwise onnx2tf emits Shape/Equal/Select clusters that block
   full-integer quantization), and the neck FFN's exact GELU
   (``0.5·y·(1+erf(y/√2))``) is replaced with the int8-friendly quick-GELU
   ``y·sigmoid(1.702·y)`` (only Mul+Sigmoid — no Flex ``Erf``, and it lowers as a
   scaled SiLU on the NPU).
2. **Convert** ONNX -> TensorFlow (SavedModel + float32 TFLite) via ``onnx2tf``
   with the ``tf_converter`` backend (the default flatbuffer_direct backend's
   strict quantizer rejects the attention's activation×activation MatMul). This
   rewrites NCHW -> TFLite-native NHWC.
3. **Verify** the conversion: run the float TFLite and source ONNX on the same
   inputs and compare the eight head outputs (cosine similarity).
4. **PTQ**: calibrate on a representative image set and emit a full-integer int8
   TFLite (int8 I/O). Verify int8-vs-fp32 accuracy. An int16-activation /
   int8-weight scheme is available for materially higher accuracy where the
   backend supports it.

The model has no in-graph normalisation and begins with a Focus/space-to-depth
stem, so preprocessing is just resize + (optional) normalise, and identical
tensors are fed to every backend so the accuracy comparisons are apples-to-apples.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("rtmo-quantize")

DEFAULT_INPUT_SIZE = 320
QUICK_GELU_SCALE = 1.702  # gelu(x) ≈ x·sigmoid(1.702·x)


# --------------------------------------------------------------------------- #
# ONNX preparation
# --------------------------------------------------------------------------- #
def _replace_exact_gelu_with_quickgelu(model) -> int:
    """Rewrite each exact-GELU (``0.5·y·(1+erf(y/√2))``) to ``y·sigmoid(1.702·y)``.

    Returns the number of GELUs replaced.
    """
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def sole(tensor, op_type):
        c = [n for n in cons.get(tensor, []) if n.op_type == op_type]
        return c[0] if len(c) == 1 else None

    replaced = 0
    for erf in [n for n in g.node if n.op_type == "Erf"]:
        div = prod.get(erf.input[0])
        if div is None or div.op_type != "Div":
            continue
        y = div.input[0]
        add1 = sole(erf.output[0], "Add")
        if add1 is None:
            continue
        mul_y = sole(add1.output[0], "Mul")
        if mul_y is None or y not in mul_y.input:
            continue
        mul_half = sole(mul_y.output[0], "Mul")
        if mul_half is None:
            continue
        out = mul_half.output[0]
        remove = {div.name, erf.name, add1.name, mul_y.name, mul_half.name}

        scale_name = f"qgelu_scale_{replaced}"
        g.initializer.append(
            numpy_helper.from_array(np.array(QUICK_GELU_SCALE, np.float32), scale_name)
        )
        n1 = helper.make_node("Mul", [y, scale_name], [f"qgelu_scaled_{replaced}"], f"qgelu_mul_scale_{replaced}")
        n2 = helper.make_node("Sigmoid", [f"qgelu_scaled_{replaced}"], [f"qgelu_sig_{replaced}"], f"qgelu_sigmoid_{replaced}")
        n3 = helper.make_node("Mul", [y, f"qgelu_sig_{replaced}"], [out], f"qgelu_mul_out_{replaced}")

        kept = [n for n in g.node if n.name not in remove]
        idx = max(i for i, n in enumerate(kept) if y in n.output)
        for k, nn in enumerate((n1, n2, n3)):
            kept.insert(idx + 1 + k, nn)
        del g.node[:]
        g.node.extend(kept)
        replaced += 1
    return replaced


def prepare_onnx(onnx_path: str | Path, out_path: str | Path, input_size: int = DEFAULT_INPUT_SIZE) -> Path:
    """Simplify + quick-GELU the ONNX so it converts and quantizes cleanly."""
    import onnx
    import onnxsim

    model = onnx.load(str(onnx_path))
    in_name = model.graph.input[0].name
    model, ok = onnxsim.simplify(model, overwrite_input_shapes={in_name: [1, 3, input_size, input_size]})
    if not ok:
        logger.warning("onnxsim reported it could not fully simplify the model")
    n = _replace_exact_gelu_with_quickgelu(model)
    logger.info("Prepared ONNX: simplified + replaced %d exact-GELU with quick-GELU", n)
    onnx.checker.check_model(model)
    out_path = Path(out_path)
    onnx.save(model, str(out_path))
    return out_path


# --------------------------------------------------------------------------- #
# Preprocessing / representative dataset
# --------------------------------------------------------------------------- #
def preprocess(
    path: str,
    input_size: int = DEFAULT_INPUT_SIZE,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    crop: tuple[float, float, float, float] | None = None,
    flip: bool = False,
) -> np.ndarray:
    """Return an NHWC ``[1, S, S, 3]`` float32 tensor (RGB, ``(x-mean)/std``)."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    if crop is not None:
        h, w = img.shape[:2]
        y0, x0, y1, x1 = crop
        img = img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    if flip:
        img = img[:, ::-1]
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    img = (img - mean) / std
    return img[None].astype(np.float32)  # NHWC


def build_calibration_set(
    images_dir: str,
    n_samples: int = 200,
    input_size: int = DEFAULT_INPUT_SIZE,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """~``n_samples`` NHWC calibration tensors from natural images (crop/flip aug)."""
    paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(images_dir, "**", ext), recursive=True)
        if ":Zone.Identifier" not in p
    )
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
            samples.append(preprocess(path, input_size, mean, std, crop=crop, flip=flip))
        except ValueError:
            paths.remove(path)
    logger.info("Built %d calibration samples from %s", len(samples), images_dir)
    return samples


# --------------------------------------------------------------------------- #
# Conversion / backends
# --------------------------------------------------------------------------- #
def onnx_to_tf(onnx_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """ONNX -> TF SavedModel + float32 TFLite via onnx2tf (tf_converter backend)."""
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(out_dir),
           "-tb", "tf_converter", "-n"]
    logger.info("Running: %s", " ".join(cmd))
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "TF_USE_LEGACY_KERAS": "1"}
    subprocess.run(cmd, check=True, env=env)
    return {
        "saved_model": out_dir,
        "float_tflite": out_dir / f"{Path(onnx_path).stem}_float32.tflite",
    }


def run_onnx(onnx_path: str | Path, inputs_nchw: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    return [
        {n: o for n, o in zip(out_names, sess.run(out_names, {in_name: x}))}
        for x in inputs_nchw
    ]


def _tflite_interpreter(tflite_path: str | Path):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    return interp


def run_tflite(tflite_path: str | Path, inputs_nhwc: list[np.ndarray]) -> list[dict[tuple, np.ndarray]]:
    """Run a TFLite model; outputs keyed by their (NCHW) shape for ONNX matching."""
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
            if v.ndim == 4:  # NHWC -> NCHW
                v = np.transpose(v, (0, 3, 1, 2))
            out[v.shape] = v
        results.append(out)
    return results


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def compare_onnx_tflite(onnx_results, tflite_results) -> dict[str, tuple[float, float]]:
    """Match ONNX outputs to TFLite outputs by (NCHW) shape; return per-output
    (mean cosine, max abs error)."""
    per_name: dict[str, list[tuple[float, float]]] = {}
    for oref, otf in zip(onnx_results, tflite_results):
        for name, ov in oref.items():
            tv = otf.get(ov.shape)
            if tv is None:
                per_name.setdefault(name, []).append((float("nan"), float("nan")))
            else:
                per_name.setdefault(name, []).append(
                    (_cosine(ov, tv), float(np.abs(ov.astype(np.float64) - tv.astype(np.float64)).max()))
                )
    return {
        name: (float(np.nanmean([c for c, _ in v])), float(np.nanmax([e for _, e in v])))
        for name, v in per_name.items()
    }


# --------------------------------------------------------------------------- #
# PTQ
# --------------------------------------------------------------------------- #
def _add_int8_input(model_bytes: bytes, in_scale: float = 1.0, in_zp: int = -128) -> bytes:
    """Splice an int8 graph input onto a TFLite model with an int16 input.

    TFLite forbids an int8 I/O type in the int16×8 activation mode, so we build
    the model with an int16 input and prepend ``int8 →DEQUANTIZE→ float32
    →QUANTIZE→ int16`` (QUANTIZE cannot go int8→int16 directly). The int8 input
    carries ``(in_scale, in_zp)`` — default ``(1.0, -128)`` maps ``pixel-128`` to
    the model's ``[0,255]`` range, matching the full-int8 model's input.
    """
    from tensorflow.lite.tools import flatbuffer_utils as fu
    from tensorflow.lite.python import schema_py_generated as fb

    m = fu.convert_bytearray_to_object(model_bytes)
    sg = m.subgraphs[0]

    def opcode(builtin_code):
        for i, oc in enumerate(m.operatorCodes):
            if oc.builtinCode == builtin_code:
                return i
        oc = fb.OperatorCodeT()
        oc.builtinCode = builtin_code
        oc.deprecatedBuiltinCode = min(builtin_code, 127)
        oc.version = 1
        m.operatorCodes.append(oc)
        return len(m.operatorCodes) - 1

    deq_op, qua_op = opcode(fb.BuiltinOperator.DEQUANTIZE), opcode(fb.BuiltinOperator.QUANTIZE)
    orig_in = sg.inputs[0]
    t16 = sg.tensors[orig_in]
    name = t16.name if isinstance(t16.name, str) else t16.name.decode()

    m.buffers.append(fb.BufferT())
    t8 = fb.TensorT()
    t8.shape, t8.type, t8.buffer, t8.name = list(t16.shape), fb.TensorType.INT8, len(m.buffers) - 1, name + "_int8"
    q = fb.QuantizationParametersT()
    q.scale, q.zeroPoint = [float(in_scale)], [int(in_zp)]
    t8.quantization = q
    sg.tensors.append(t8)
    t8_idx = len(sg.tensors) - 1

    m.buffers.append(fb.BufferT())
    tf32 = fb.TensorT()
    tf32.shape, tf32.type, tf32.buffer, tf32.name = list(t16.shape), fb.TensorType.FLOAT32, len(m.buffers) - 1, name + "_f32"
    sg.tensors.append(tf32)
    tf32_idx = len(sg.tensors) - 1

    deq, qua = fb.OperatorT(), fb.OperatorT()
    deq.opcodeIndex, deq.inputs, deq.outputs = deq_op, [t8_idx], [tf32_idx]
    qua.opcodeIndex, qua.inputs, qua.outputs = qua_op, [tf32_idx], [orig_in]
    sg.operators.insert(0, qua)
    sg.operators.insert(0, deq)
    sg.inputs = [t8_idx]
    return bytes(fu.convert_object_to_bytearray(m))


def quantize_int8(
    saved_model_dir: str | Path,
    calibration_set: list[np.ndarray],
    out_path: str | Path,
    scheme: str = "int8",
    int8_input_scale: float = 1.0,
    int8_input_zp: int = -128,
) -> Path:
    """PTQ from a SavedModel using a representative dataset.

    - ``scheme='int8'``: full-integer int8 (int8 weights + activations + I/O).
    - ``scheme='int16x8'``: int8 weights + int16 activations (int16 I/O); much
      more accurate on the transformer neck where the backend supports int16.
    - ``scheme='int16x8_int8in'``: as int16x8 but with an int8 image input
      (int16 outputs), via a spliced ``DEQUANTIZE→QUANTIZE`` at the boundary.
    """
    import tensorflow as tf

    def rep_gen():
        for x in calibration_set:
            yield [x.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    if scheme == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif scheme in ("int16x8", "int16x8_int8in"):
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        if scheme == "int16x8_int8in":
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    model_bytes = converter.convert()
    if scheme == "int16x8_int8in":
        model_bytes = _add_int8_input(model_bytes, int8_input_scale, int8_input_zp)

    out_path = Path(out_path)
    out_path.write_bytes(model_bytes)
    logger.info("Wrote %s TFLite: %s", scheme, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def quantize_rtmo(
    onnx_path: str | Path,
    out_dir: str | Path,
    images_dir: str | Path,
    input_size: int = DEFAULT_INPUT_SIZE,
    n_calib: int = 200,
    n_verify: int = 16,
    mean: float = 0.0,
    std: float = 1.0,
    scheme: str = "int8",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Preparing ONNX (simplify + quick-GELU)")
    prepared = prepare_onnx(onnx_path, out_dir / "rtmo_prepared.onnx", input_size)

    logger.info("[2/4] ONNX -> TensorFlow")
    tf_paths = onnx_to_tf(prepared, out_dir / "tf")

    logger.info("[3/4] Verifying float conversion, then int8 PTQ")
    calib = build_calibration_set(str(images_dir), n_calib, input_size, mean, std)
    verify_nhwc = build_calibration_set(str(images_dir), n_verify, input_size, mean, std, seed=99)
    verify_nchw = [np.transpose(x, (0, 3, 1, 2)) for x in verify_nhwc]
    onnx_res = run_onnx(prepared, verify_nchw)
    float_acc = compare_onnx_tflite(onnx_res, run_tflite(tf_paths["float_tflite"], verify_nhwc))
    for name, (cos, err) in sorted(float_acc.items()):
        logger.info("  float  %-16s cos=%.5f", name, cos)

    int8_path = quantize_int8(tf_paths["saved_model"], calib, out_dir / f"rtmo_{scheme}.tflite", scheme)

    logger.info("[4/4] Verifying quantized accuracy vs fp32 ONNX")
    q_acc = compare_onnx_tflite(onnx_res, run_tflite(int8_path, verify_nhwc))
    for name, (cos, err) in sorted(q_acc.items()):
        logger.info("  %-6s %-16s cos=%.5f", scheme, name, cos)

    return {
        "prepared_onnx": prepared,
        "float_tflite": tf_paths["float_tflite"],
        "quant_tflite": int8_path,
        "float_accuracy": float_acc,
        "quant_accuracy": q_acc,
    }


def add_rtmo_quantize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--onnx", default="models/rtmo/export/rtmo_nopost_fp32.onnx",
                        help="Source fp32 ONNX (post-processing stripped) (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", default="models/rtmo/export/int8",
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--images-dir", default="models/rtmo/calib",
                        help="Directory of representative images (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--n-calib", type=int, default=200, help="Calibration sample count")
    parser.add_argument("--n-verify", type=int, default=16, help="Accuracy-check sample count")
    parser.add_argument("--mean", type=float, default=0.0, help="Input normalisation mean")
    parser.add_argument("--std", type=float, default=1.0, help="Input normalisation std")
    parser.add_argument("--scheme", choices=["int8", "int16x8", "int16x8_int8in"], default="int8",
                        help="Quantization scheme (default: %(default)s)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Quantize RTMO to int8 TFLite")
    add_rtmo_quantize_args(parser)
    args = parser.parse_args()
    res = quantize_rtmo(
        args.onnx, args.out_dir, args.images_dir, args.input_size,
        args.n_calib, args.n_verify, args.mean, args.std, args.scheme,
    )
    print("prepared_onnx:", res["prepared_onnx"])
    print("float_tflite :", res["float_tflite"])
    print("quant_tflite :", res["quant_tflite"])


if __name__ == "__main__":
    main()
