# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Quantize FULLY_CONNECTED ops in a fp32 TFLite model to int16×int8 mixed precision.

Each FC op is wrapped: fp32 → QUANTIZE(int16) → FC(int16, int8, int64_bias) → DEQUANTIZE(int16→fp32).
All other ops remain fp32.  The result can be converted to TOSA MLIR via
``iree-import-tflite`` + ``iree-opt`` without the i48 crash.

Key design decision: every FC gets an explicit INT64 bias (zero-filled if the
original had no bias).  This prevents TF's TOSA lowering from synthesising an
implicit i48 zero-bias tensor that crashes ``iree-opt``.
"""

import copy
import logging
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as tfl

_logger = logging.getLogger(__name__)

# TFLite builtin operator codes
_FULLY_CONNECTED_CODE = 9
_QUANTIZE_CODE = 114
_DEQUANTIZE_CODE = 6


def _find_or_add_opcode(model: tfl.ModelT, builtin_code: int) -> int:
    """Return the opcode index for *builtin_code*, adding one if needed."""
    for idx, oc in enumerate(model.operatorCodes):
        c = oc.deprecatedBuiltinCode if oc.deprecatedBuiltinCode != 127 else oc.builtinCode
        if c == builtin_code:
            return idx
    new_oc = tfl.OperatorCodeT()
    new_oc.deprecatedBuiltinCode = min(builtin_code, 127)
    new_oc.builtinCode = builtin_code
    new_oc.version = 1
    model.operatorCodes.append(new_oc)
    return len(model.operatorCodes) - 1


def _make_quantized_tensor(
    orig_tensor: tfl.TensorT,
    dtype: int,
    suffix: str,
    scale: float,
    zero_point: int,
    buffer_idx: int,
) -> tfl.TensorT:
    """Create a new tensor descriptor with quantization parameters."""
    t = tfl.TensorT()
    name = (orig_tensor.name or b"tensor").decode("utf-8", errors="replace")
    t.name = (name + suffix).encode("utf-8")
    t.shape = copy.deepcopy(orig_tensor.shape)
    if orig_tensor.shapeSignature is not None:
        t.shapeSignature = copy.deepcopy(orig_tensor.shapeSignature)
    t.type = dtype
    t.buffer = buffer_idx
    t.isVariable = False
    t.hasRank = True

    qp = tfl.QuantizationParametersT()
    qp.scale = np.array([scale], dtype=np.float32)
    qp.zeroPoint = np.array([zero_point], dtype=np.int64)
    qp.quantizedDimension = 0
    t.quantization = qp
    return t


def quantize_fc_ops_in_tflite(
    fp32_tflite_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Quantize all FC ops in a fp32 TFLite to int16×int8 mixed precision.

    Parameters
    ----------
    fp32_tflite_path : Path
        Input fp32 TFLite model.
    output_path : Path or None
        Where to write the mixed-precision TFLite.  Defaults to
        ``<stem>_fc_int16x8_mixed.tflite`` next to the input.

    Returns
    -------
    Path
        Path to the written mixed-precision TFLite file.
    """
    fp32_tflite_path = Path(fp32_tflite_path)
    if output_path is None:
        output_path = fp32_tflite_path.parent / f"{fp32_tflite_path.stem}_fc_int16x8_mixed.tflite"
    output_path = Path(output_path)

    _logger.info("Quantizing FC ops in %s → %s", fp32_tflite_path.name, output_path.name)

    with open(fp32_tflite_path, "rb") as f:
        buf = bytearray(f.read())

    model = tfl.ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]

    quantize_opcode_idx = _find_or_add_opcode(model, _QUANTIZE_CODE)
    dequantize_opcode_idx = _find_or_add_opcode(model, _DEQUANTIZE_CODE)

    # Add empty buffer for new activation tensors
    empty_buf = tfl.BufferT()
    empty_buf.data = None
    model.buffers.append(empty_buf)
    empty_buf_idx = len(model.buffers) - 1

    # Identify all FC operator indices
    fc_indices = []
    for oi, op in enumerate(sg.operators):
        oc = model.operatorCodes[op.opcodeIndex]
        code = oc.deprecatedBuiltinCode if oc.deprecatedBuiltinCode != 127 else oc.builtinCode
        if code == _FULLY_CONNECTED_CODE:
            fc_indices.append(oi)

    _logger.info("Found %d FULLY_CONNECTED ops to quantize", len(fc_indices))

    # Process each FC in reverse order (so insertion indices stay valid)
    ops_inserted = 0
    for fc_oi in reversed(fc_indices):
        fc_op = sg.operators[fc_oi]

        inp_tidx = int(fc_op.inputs[0])
        wt_tidx = int(fc_op.inputs[1])
        bias_tidx = int(fc_op.inputs[2]) if len(fc_op.inputs) > 2 and int(fc_op.inputs[2]) >= 0 else -1
        out_tidx = int(fc_op.outputs[0])

        inp_tensor = sg.tensors[inp_tidx]
        wt_tensor = sg.tensors[wt_tidx]
        out_tensor = sg.tensors[out_tidx]

        # --- Quantize weights: fp32 → int8 (symmetric) ---
        wt_buf_idx = wt_tensor.buffer
        wt_data_raw = model.buffers[wt_buf_idx].data
        if wt_data_raw is None:
            _logger.warning("FC[%d]: weight buffer is empty, skipping", fc_oi)
            continue

        wt_fp32 = np.frombuffer(bytes(wt_data_raw), dtype=np.float32)
        wt_abs_max = max(abs(float(wt_fp32.min())), abs(float(wt_fp32.max())))
        if wt_abs_max == 0:
            wt_abs_max = 1e-8
        wt_scale = wt_abs_max / 127.0
        wt_int8 = np.clip(np.round(wt_fp32 / wt_scale), -127, 127).astype(np.int8)

        wt_int8_buf = tfl.BufferT()
        wt_int8_buf.data = list(wt_int8.tobytes())
        model.buffers.append(wt_int8_buf)
        wt_int8_buf_idx = len(model.buffers) - 1

        wt_int8_tensor = _make_quantized_tensor(
            wt_tensor, tfl.TensorType.INT8, "_int8",
            scale=wt_scale, zero_point=0,
            buffer_idx=wt_int8_buf_idx,
        )
        sg.tensors.append(wt_int8_tensor)
        wt_int8_tidx = len(sg.tensors) - 1

        # --- Activation scales ---
        inp_scale = 1.0 / 32767.0  # covers [-1, 1] in int16
        inp_zp = 0
        out_scale = inp_scale
        out_zp = 0

        # --- Create int16 input tensor ---
        inp_int16_tensor = _make_quantized_tensor(
            inp_tensor, tfl.TensorType.INT16, "_int16",
            scale=inp_scale, zero_point=inp_zp,
            buffer_idx=empty_buf_idx,
        )
        sg.tensors.append(inp_int16_tensor)
        inp_int16_tidx = len(sg.tensors) - 1

        # --- Create int16 output tensor ---
        out_int16_tensor = _make_quantized_tensor(
            out_tensor, tfl.TensorType.INT16, "_int16",
            scale=out_scale, zero_point=out_zp,
            buffer_idx=empty_buf_idx,
        )
        sg.tensors.append(out_int16_tensor)
        out_int16_tidx = len(sg.tensors) - 1

        # --- INT64 bias (always explicit to avoid i48 in TOSA) ---
        bias_scale = inp_scale * wt_scale
        bias_dim = int(wt_tensor.shape[0])  # [output_units, input_units]

        if bias_tidx >= 0:
            bias_tensor = sg.tensors[bias_tidx]
            bias_data_raw = model.buffers[bias_tensor.buffer].data
            if bias_data_raw is not None:
                bias_fp32 = np.frombuffer(bytes(bias_data_raw), dtype=np.float32)
                bias_int64 = np.round(bias_fp32 / bias_scale).astype(np.int64)
            else:
                bias_int64 = np.zeros(bias_dim, dtype=np.int64)
        else:
            bias_int64 = np.zeros(bias_dim, dtype=np.int64)

        bias_int64_buf = tfl.BufferT()
        bias_int64_buf.data = list(bias_int64.tobytes())
        model.buffers.append(bias_int64_buf)
        bias_int64_buf_idx = len(model.buffers) - 1

        bias_ref_tensor = sg.tensors[bias_tidx] if bias_tidx >= 0 else wt_tensor
        bias_int64_tensor = _make_quantized_tensor(
            bias_ref_tensor, tfl.TensorType.INT64, "_int64",
            scale=bias_scale, zero_point=0,
            buffer_idx=bias_int64_buf_idx,
        )
        bias_int64_tensor.shape = np.array([bias_dim], dtype=np.int32)
        if bias_int64_tensor.shapeSignature is not None:
            bias_int64_tensor.shapeSignature = np.array([bias_dim], dtype=np.int32)
        sg.tensors.append(bias_int64_tensor)
        bias_int64_tidx = len(sg.tensors) - 1

        # --- QUANTIZE op: fp32 → int16 ---
        q_op = tfl.OperatorT()
        q_op.opcodeIndex = quantize_opcode_idx
        q_op.inputs = np.array([inp_tidx], dtype=np.int32)
        q_op.outputs = np.array([inp_int16_tidx], dtype=np.int32)
        q_op.builtinOptionsType = 0
        q_op.builtinOptions = None

        # --- Rewire FC ---
        fc_op.inputs = np.array([inp_int16_tidx, wt_int8_tidx, bias_int64_tidx], dtype=np.int32)
        fc_op.outputs = np.array([out_int16_tidx], dtype=np.int32)

        # --- DEQUANTIZE op: int16 → fp32 ---
        dq_op = tfl.OperatorT()
        dq_op.opcodeIndex = dequantize_opcode_idx
        dq_op.inputs = np.array([out_int16_tidx], dtype=np.int32)
        dq_op.outputs = np.array([out_tidx], dtype=np.int32)
        dq_op.builtinOptionsType = 0
        dq_op.builtinOptions = None

        # Insert QUANTIZE before FC, DEQUANTIZE after
        sg.operators.insert(fc_oi, q_op)
        sg.operators.insert(fc_oi + 2, dq_op)
        ops_inserted += 2

    _logger.info("Inserted %d QUANTIZE/DEQUANTIZE ops", ops_inserted)

    # Serialize
    builder = flatbuffers.Builder(len(buf) * 2)
    packed = model.Pack(builder)
    builder.Finish(packed, b"TFL3")
    output_path.write_bytes(bytes(builder.Output()))

    _logger.info("Wrote mixed-precision TFLite: %s (%.1f KB)",
                 output_path, output_path.stat().st_size / 1024)
    return output_path
