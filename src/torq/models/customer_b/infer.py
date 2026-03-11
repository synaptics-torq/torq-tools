# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Run inference on Customer B models using TFLite Runtime or IREE Runtime.

Usage
~~~~~
::

    # TFLite inference (int8)
    python -m torq.models.customer_b.infer -m output_customer_b/all_fc --component all_fc

    # IREE / VMFB inference
    python -m torq.models.customer_b.infer -m output_customer_b/all_fc --component all_fc
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from torq.utils.logging import configure_logging

from . import add_customer_b_infer_args


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tflite_io_details(tflite_path: str | os.PathLike):
    """Return (input_details, output_details) for a TFLite model."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter.get_input_details(), interpreter.get_output_details()


def _run_tflite(tflite_path: str | os.PathLike, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Run inference with TFLite Runtime."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp_detail, data in zip(input_details, inputs):
        interpreter.set_tensor(inp_detail["index"], data)

    interpreter.invoke()

    outputs = []
    for out_detail in output_details:
        outputs.append(interpreter.get_tensor(out_detail["index"]))
    return outputs


def _run_iree(vmfb_path: str | os.PathLike, inputs: list[np.ndarray], device: str = "local-task") -> list[np.ndarray]:
    """Run inference with IREE Runtime via VMFBInferenceRunner."""
    from torq.runtime import VMFBInferenceRunner

    runner = VMFBInferenceRunner(
        model_path=str(vmfb_path),
        device_uri=device,
        load_method="mmap",
    )
    return runner.infer(inputs)


def _generate_random_inputs(input_details: list[dict], seed: int = 42) -> list[np.ndarray]:
    """Generate random input data matching the model's expected shapes/dtypes."""
    rng = np.random.default_rng(seed)
    inputs = []
    for inp in input_details:
        shape = list(inp["shape"])
        dtype = inp["dtype"]
        if dtype == np.int8:
            data = rng.integers(-128, 127, size=shape, dtype=np.int8)
        elif dtype == np.uint8:
            data = rng.integers(0, 255, size=shape, dtype=np.uint8)
        elif dtype == np.int16:
            data = rng.integers(-32768, 32767, size=shape, dtype=np.int16)
        elif dtype == np.float32:
            data = rng.standard_normal(shape).astype(np.float32)
        else:
            data = rng.standard_normal(shape).astype(np.float32)
        inputs.append(data)
    return inputs


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def infer_customer_b(args: argparse.Namespace):
    """Run inference on a Customer B model."""
    configure_logging(getattr(args, "logging", "info"))
    model_dir = Path(args.model_dir)

    # Find model files
    vmfbs = sorted(model_dir.rglob("*.vmfb"))
    tflites = sorted(model_dir.rglob("*_int8.tflite"))

    if not vmfbs and not tflites:
        tflites = sorted(model_dir.rglob("*.tflite"))

    if not vmfbs and not tflites:
        _logger.error("No .vmfb or .tflite models found in %s", model_dir)
        return

    # Prefer TFLite for getting I/O metadata
    tflite_path = tflites[0] if tflites else None
    vmfb_path = vmfbs[0] if vmfbs else None

    # Get input shapes from TFLite model
    if tflite_path:
        input_details, output_details = _get_tflite_io_details(tflite_path)
    else:
        _logger.warning("No TFLite model found; cannot determine I/O shapes automatically")
        return

    # Generate or load inputs
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".npy":
            inputs = [np.load(str(input_path))]
        elif input_path.suffix == ".bin":
            dtype = input_details[0]["dtype"]
            shape = list(input_details[0]["shape"])
            data = np.frombuffer(input_path.read_bytes(), dtype=dtype).reshape(shape)
            inputs = [data]
        else:
            _logger.error("Unsupported input format: %s", input_path.suffix)
            return
    else:
        _logger.info("Generating random inputs …")
        inputs = _generate_random_inputs(input_details)

    _logger.info("Input shapes: %s", [i.shape for i in inputs])

    # Run TFLite inference
    if tflite_path:
        _logger.info("Running TFLite inference: %s", tflite_path)
        tflite_outputs = _run_tflite(tflite_path, inputs)
        for i, out in enumerate(tflite_outputs):
            _logger.info("  TFLite output[%d]: shape=%s  dtype=%s  range=[%s, %s]",
                         i, out.shape, out.dtype, out.min(), out.max())

    # Run IREE inference
    if vmfb_path:
        _logger.info("Running IREE inference: %s", vmfb_path)
        iree_outputs = _run_iree(vmfb_path, inputs)
        for i, out in enumerate(iree_outputs):
            _logger.info("  IREE output[%d]: shape=%s  dtype=%s  range=[%s, %s]",
                         i, out.shape, out.dtype, out.min(), out.max())

    # Compare if both available
    if tflite_path and vmfb_path:
        _logger.info("Comparing TFLite vs IREE outputs …")
        for i, (tf_out, iree_out) in enumerate(zip(tflite_outputs, iree_outputs)):
            if tf_out.shape != iree_out.shape:
                _logger.warning("  output[%d]: shape mismatch: TFLite=%s  IREE=%s",
                                i, tf_out.shape, iree_out.shape)
                continue
            diff = np.abs(tf_out.astype(np.float32) - iree_out.astype(np.float32))
            n_diff = np.count_nonzero(diff)
            total = diff.size
            _logger.info("  output[%d]: max_abs_diff=%.6f  num_diff=%d/%d [%.2f%%]",
                         i, diff.max(), n_diff, total, 100.0 * n_diff / total)


def main():
    parser = argparse.ArgumentParser(description="Run inference on Customer B models")
    add_customer_b_infer_args(parser)
    args = parser.parse_args()
    infer_customer_b(args)


if __name__ == "__main__":
    main()
