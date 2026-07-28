# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2.5 (Liquid) custom-op replacements.

Rewrites the ORT ``com.microsoft`` fused ops that appear in the LiquidAI
LFM2.5 ONNX export — ``SimplifiedLayerNormalization``,
``SkipSimplifiedLayerNormalization`` and ``GroupQueryAttention`` — into
standard ONNX ops the Torq compiler can lower.  Split out of the monolithic
``edits.py`` when it was refactored into the ``graph_edit.edits`` package.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto
from onnx_graphsurgeon.ir.tensor import LazyValues

from ..onnx import OnnxGraphEdit, rewire_consumers


def _is_bfloat16(dtype) -> bool:
    """`gs.Tensor.dtype` for a BFLOAT16 tensor is usually a proper
    `np.dtype(ml_dtypes.bfloat16)`, but onnx_graphsurgeon's `import_onnx`
    was observed falling back to the raw `onnx.TensorProto.BFLOAT16` enum
    int instead, for *Variables* specifically (not Constants -- their
    `.values` still resolve to real `ml_dtypes.bfloat16` arrays; this
    affects only the `.dtype` attribute used for type comparisons), when
    importing a decoder that had already been bf16-converted by a separate
    tool before this edit runs. Check both representations.
    """
    if isinstance(dtype, int):
        return dtype == int(TensorProto.BFLOAT16)
    return np.dtype(dtype) == np.dtype(ml_dtypes.bfloat16)


def _tensor_proto_dtype(dtype) -> int:
    """Resolve a `gs.Tensor.dtype` to its `onnx.TensorProto.*` enum int for
    a `Cast` node's `to` attribute -- passes through the raw-int fallback
    from `_is_bfloat16` unchanged (it's already the enum value), otherwise
    converts a real numpy/ml_dtypes dtype the normal way.
    """
    if isinstance(dtype, int):
        return dtype
    return int(onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype)))


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """[..., n_bytes] uint8, 2 packed 4-bit values/byte (low nibble = even
    index, high nibble = odd index) -> [..., 2*n_bytes] uint8 unpacked.
    Convention verified empirically against a real MatMulNBits node's
    onnxruntime output (see gemma4's op_repros/decompose_matmul_nbits.py).
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _pack_int4(values: np.ndarray) -> bytes:
    """[..., n] signed int8 values in [-8,7] -> ONNX-packed INT4 raw bytes
    (2 values/byte, low nibble first, stored as the low 4 bits of the
    value's two's-complement byte representation). Pads to even length
    with a zero nibble if needed. Exact convention used by
    `torq.tools.quantization.weight_quantization.quantize._pack_int4` (the
    established, torq-compile-tested int4 path) -- matched here on purpose,
    not reinvented, since that's the pathway `DecomposeMatMulNBits`'s output
    needs to match for the compiler to accept it (see its docstring).
    """
    flat = values.flatten().astype(np.int8)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int8(0))
    low = flat[0::2].astype(np.uint8) & 0x0F
    high = flat[1::2].astype(np.uint8) & 0x0F
    packed = low | (high << 4)
    return packed.tobytes()


def _make_signed_int4_constant(name: str, unsigned_nibbles: np.ndarray) -> gs.Constant:
    """Build a `gs.Constant` backed by a native ONNX `INT4` (packed, signed
    [-8,7]) tensor from `unsigned_nibbles` (values in MatMulNBits's native
    unsigned [0,15] range, e.g. from `_unpack_nibbles`).

    Shifts every value by -8 before packing (`[0,15] -> [-8,7]`) rather than
    packing as unsigned `UINT4`: the established, torq-compile-tested int4
    path (`weight_quantization/quantize.py`) uses signed INT4 throughout, and
    the shift is exact and lossless for `DequantizeLinear`'s `(x - zero_point)
    * scale` -- shifting *both* `x` and `zero_point` by the same constant
    leaves `x - zero_point` unchanged (`(x-8) - (zp-8) == x - zp`), so the
    dequantized result is bit-for-bit identical to the unsigned version.

    Constructed via a raw `onnx.TensorProto` + `LazyValues` wrapper, not
    `gs.Constant(name, values, export_dtype=...)`: verified empirically that
    onnx_graphsurgeon's export path only supports converting a `float32`
    *source* array via `export_dtype` (mirrors the existing bf16 patch's
    `_NUMPY_ARRAY_CONVERTERS` mechanism) and separately that assigning the
    array's own dtype to `ml_dtypes.int4`/`uint4` directly produces an
    unpacked (1 byte/value) `raw_data` that `onnx.helper.make_tensor` rejects
    as the wrong size for a packed INT4 tensor -- neither path works for an
    already-integer source. Building the packed bytes directly and wrapping
    them in `LazyValues` (never `.load()`ed, so `gs.export_onnx` passes the
    TensorProto through unchanged) sidesteps both gaps.
    """
    shape = tuple(unsigned_nibbles.shape)
    signed = (unsigned_nibbles.astype(np.int16) - 8).astype(np.int8)
    tensor_proto = TensorProto()
    tensor_proto.name = name
    tensor_proto.dims[:] = list(shape)
    tensor_proto.data_type = TensorProto.INT4
    tensor_proto.raw_data = _pack_int4(signed)
    return gs.Constant(name, LazyValues(tensor_proto))


@dataclass
class ReplaceSimplifiedLayerNorm(OnnxGraphEdit):
    """
    Replace SimplifiedLayerNormalization (ORT fused RMS norm) with standard ONNX ops.

    Produces: Pow(x,2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(1,sqrt) -> Mul(x,rcp) -> Mul(weight)
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "SimplifiedLayerNormalization"

    def transform(self, node: gs.Node):
        inp = node.inputs[0]
        weight = node.inputs[1]
        epsilon = node.attrs.get("epsilon", 1e-6)
        out_var = node.outputs[0]
        prefix = node.name

        pow_out = self.graph.layer(
            name=f"{prefix}/Pow", op="Pow",
            inputs=[inp, gs.Constant(f"{prefix}/pow_exp", np.array(2.0, dtype=np.float32))],
            outputs=[gs.Variable(f"{prefix}/Pow_output_0")],
        )[0]
        mean_out = self.graph.layer(
            name=f"{prefix}/ReduceMean", op="ReduceMean",
            inputs=[pow_out, gs.Constant(f"{prefix}/reduce_axes", np.array([-1], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/ReduceMean_output_0")],
        )[0]
        add_out = self.graph.layer(
            name=f"{prefix}/Add", op="Add",
            inputs=[mean_out, gs.Constant(f"{prefix}/epsilon", np.array(epsilon, dtype=np.float32))],
            outputs=[gs.Variable(f"{prefix}/Add_output_0")],
        )[0]
        sqrt_out = self.graph.layer(
            name=f"{prefix}/Sqrt", op="Sqrt",
            inputs=[add_out],
            outputs=[gs.Variable(f"{prefix}/Sqrt_output_0")],
        )[0]
        div_out = self.graph.layer(
            name=f"{prefix}/Div", op="Div",
            inputs=[gs.Constant(f"{prefix}/one", np.array(1.0, dtype=np.float32)), sqrt_out],
            outputs=[gs.Variable(f"{prefix}/Div_output_0")],
        )[0]
        mul_out = self.graph.layer(
            name=f"{prefix}/Mul", op="Mul",
            inputs=[inp, div_out],
            outputs=[gs.Variable(f"{prefix}/Mul_output_0")],
        )[0]
        mul_w_out = self.graph.layer(
            name=f"{prefix}/Mul_1", op="Mul",
            inputs=[mul_out, weight],
            outputs=[gs.Variable(f"{prefix}/Mul_1_output_0")],
        )[0]

        rewire_consumers(out_var.outputs.copy(), out_var, mul_w_out)

        # Also update graph outputs if needed
        for i, go in enumerate(self.graph.outputs):
            if go is out_var:
                self.graph.outputs[i] = mul_w_out

        node.outputs.clear()
        self._logger.debug("Replaced SimplifiedLayerNormalization '%s'", node.name)


@dataclass
class ReplaceSkipSimplifiedLayerNorm(OnnxGraphEdit):
    """
    Replace SkipSimplifiedLayerNormalization (ORT fused skip-connection + RMS norm)
    with standard ONNX ops.

    SkipSimplifiedLayerNormalization(input, skip, weight, epsilon):
        sum = input + skip
        output[0] = RMSNorm(sum, weight, epsilon)
        output[3] = sum

    Produces: Add(input, skip) -> [RMS norm chain] -> output; skip sum forwarded.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "SkipSimplifiedLayerNormalization"

    def transform(self, node: gs.Node):
        inp = node.inputs[0]
        skip = node.inputs[1]
        weight = node.inputs[2]
        epsilon = node.attrs.get("epsilon", 1e-6)
        prefix = node.name

        # output[0] = RMSNorm result, output[3] = skip sum
        out_norm = node.outputs[0]
        out_skip_sum = node.outputs[3] if len(node.outputs) > 3 and node.outputs[3].name else None

        # Step 1: skip sum = input + skip
        skip_sum = self.graph.layer(
            name=f"{prefix}/SkipAdd", op="Add",
            inputs=[inp, skip],
            outputs=[gs.Variable(f"{prefix}/skip_sum")],
        )[0]

        # Step 2: RMSNorm on skip_sum
        pow_out = self.graph.layer(
            name=f"{prefix}/Pow", op="Pow",
            inputs=[skip_sum, gs.Constant(f"{prefix}/pow_exp", np.array(2.0, dtype=np.float32))],
            outputs=[gs.Variable(f"{prefix}/Pow_output_0")],
        )[0]
        mean_out = self.graph.layer(
            name=f"{prefix}/ReduceMean", op="ReduceMean",
            inputs=[pow_out, gs.Constant(f"{prefix}/reduce_axes", np.array([-1], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/ReduceMean_output_0")],
        )[0]
        add_out = self.graph.layer(
            name=f"{prefix}/Add", op="Add",
            inputs=[mean_out, gs.Constant(f"{prefix}/epsilon", np.array(epsilon, dtype=np.float32))],
            outputs=[gs.Variable(f"{prefix}/Add_output_0")],
        )[0]
        sqrt_out = self.graph.layer(
            name=f"{prefix}/Sqrt", op="Sqrt",
            inputs=[add_out],
            outputs=[gs.Variable(f"{prefix}/Sqrt_output_0")],
        )[0]
        div_out = self.graph.layer(
            name=f"{prefix}/Div", op="Div",
            inputs=[gs.Constant(f"{prefix}/one", np.array(1.0, dtype=np.float32)), sqrt_out],
            outputs=[gs.Variable(f"{prefix}/Div_output_0")],
        )[0]
        mul_out = self.graph.layer(
            name=f"{prefix}/Mul", op="Mul",
            inputs=[skip_sum, div_out],
            outputs=[gs.Variable(f"{prefix}/Mul_output_0")],
        )[0]
        mul_w_out = self.graph.layer(
            name=f"{prefix}/Mul_1", op="Mul",
            inputs=[mul_out, weight],
            outputs=[gs.Variable(f"{prefix}/Mul_1_output_0")],
        )[0]

        # Rewire norm output consumers
        rewire_consumers(out_norm.outputs.copy(), out_norm, mul_w_out)
        for i, go in enumerate(self.graph.outputs):
            if go is out_norm:
                self.graph.outputs[i] = mul_w_out

        # Rewire skip sum output consumers
        if out_skip_sum is not None:
            rewire_consumers(out_skip_sum.outputs.copy(), out_skip_sum, skip_sum)
            for i, go in enumerate(self.graph.outputs):
                if go is out_skip_sum:
                    self.graph.outputs[i] = skip_sum

        node.outputs.clear()
        self._logger.debug("Replaced SkipSimplifiedLayerNormalization '%s'", node.name)


@dataclass
class ReplaceGroupQueryAttention(OnnxGraphEdit):
    """
    Replace GroupQueryAttention (ORT fused op) with standard ONNX ops.

    Decomposes into: RoPE -> KV concat -> Q*K^T*scale -> mask -> Softmax -> *V
    Matches the fp32 model's expanded attention structure.
    """

    num_heads: int
    kv_num_heads: int
    head_dim: int

    def match(self, node: gs.Node) -> bool:
        return node.op == "GroupQueryAttention"

    def _apply_rope(self, x, cos_cache, sin_cache, seqlen_k, prefix):
        """Apply rotary position embeddings: x*cos + rotate_half(x)*sin"""
        # Reshape to (B, num_heads, seq, head_dim)
        # cos/sin caches are (max_seq, head_dim/2) or (max_seq, head_dim)
        # Gather cos/sin for positions 0..seqlen_k (only current position matters for single-token)

        # x * cos
        mul_cos = self.graph.layer(
            name=f"{prefix}/rope_mul_cos", op="Mul",
            inputs=[x, cos_cache],
            outputs=[gs.Variable(f"{prefix}/rope_mul_cos_out")],
        )[0]

        # rotate_half(x): split x into two halves, negate second, concat
        half_dim = self.head_dim // 2
        x_first = self.graph.layer(
            name=f"{prefix}/rope_slice_first", op="Slice",
            inputs=[
                x,
                gs.Constant(f"{prefix}/rope_start_0", np.array([0], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_end_half", np.array([half_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_axis_neg1", np.array([-1], dtype=np.int64)),
            ],
            outputs=[gs.Variable(f"{prefix}/rope_first_half")],
        )[0]
        x_second = self.graph.layer(
            name=f"{prefix}/rope_slice_second", op="Slice",
            inputs=[
                x,
                gs.Constant(f"{prefix}/rope_start_half", np.array([half_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_end_full", np.array([self.head_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_axis_neg1b", np.array([-1], dtype=np.int64)),
            ],
            outputs=[gs.Variable(f"{prefix}/rope_second_half")],
        )[0]
        neg_second = self.graph.layer(
            name=f"{prefix}/rope_neg", op="Neg",
            inputs=[x_second],
            outputs=[gs.Variable(f"{prefix}/rope_neg_out")],
        )[0]
        rotated = self.graph.layer(
            name=f"{prefix}/rope_concat", op="Concat",
            inputs=[neg_second, x_first],
            outputs=[gs.Variable(f"{prefix}/rope_rotated")],
            attrs={"axis": -1},
        )[0]
        mul_sin = self.graph.layer(
            name=f"{prefix}/rope_mul_sin", op="Mul",
            inputs=[rotated, sin_cache],
            outputs=[gs.Variable(f"{prefix}/rope_mul_sin_out")],
        )[0]
        result = self.graph.layer(
            name=f"{prefix}/rope_add", op="Add",
            inputs=[mul_cos, mul_sin],
            outputs=[gs.Variable(f"{prefix}/rope_out")],
        )[0]
        return result

    def transform(self, node: gs.Node):
        # GQA inputs:
        # [0] Q (B, seq, num_heads*head_dim)
        # [1] K (B, seq, kv_num_heads*head_dim)
        # [2] V (B, seq, kv_num_heads*head_dim)
        # [3] past_key (B, kv_num_heads, past_seq, head_dim)
        # [4] past_value (B, kv_num_heads, past_seq, head_dim)
        # [5] seqlen_k (B, 1) int32
        # [6] total_seq_len scalar int32
        # [7] cos_cache (max_seq, head_dim)
        # [8] sin_cache (max_seq, head_dim)
        # [9] (unused)
        # [10] attn_bias (B, 1, seq, total_seq) float32

        # GQA outputs (may have been partially cleaned up):
        # Typically [0]=attn_output, [1]=present_key, [2]=present_value
        # But cleanup can remove unused outputs, so we identify by name

        q_inp = node.inputs[0]
        k_inp = node.inputs[1]
        v_inp = node.inputs[2]
        past_key = node.inputs[3]
        past_value = node.inputs[4]
        seqlen_k = node.inputs[5] if (
            len(node.inputs) > 5
            and isinstance(node.inputs[5], (gs.Variable, gs.Constant))
            and node.inputs[5].name
        ) else None
        cos_cache = node.inputs[7]
        sin_cache = node.inputs[8]

        # attn_bias may be absent (empty input) in int4 quantized models
        attn_bias = node.inputs[10] if (
            len(node.inputs) > 10
            and isinstance(node.inputs[10], (gs.Variable, gs.Constant))
            and node.inputs[10].name
        ) else None

        # Identify outputs by name pattern rather than fixed index
        out_attn = None
        out_present_k = None
        out_present_v = None
        for out in node.outputs:
            if "present" in out.name and "key" in out.name:
                out_present_k = out
            elif "present" in out.name and "value" in out.name:
                out_present_v = out
            else:
                out_attn = out

        scale = node.attrs.get("scale", 1.0 / (self.head_dim ** 0.5))
        prefix = node.name

        # Some LFM2.5 exports apply rotary via separate q_rotary/k_rotary
        # RotaryEmbedding ops *before* the GQA op (do_rotary=0, empty cos/sin
        # cache inputs) — so the Q/K arriving here are already rotated. Only
        # build+apply RoPE in this decomposition when the GQA op does rotary
        # itself (do_rotary=1, real cos/sin caches); otherwise applying it would
        # double-rotate and building cos_full from an empty cache leaves the
        # cos/sin tensors with dynamic (None) shapes.
        do_rotary = int(node.attrs.get("do_rotary", 1))

        # Disconnect the old GQA node immediately so that the original output
        # variables can be reused without creating duplicate-name warnings.
        node.inputs.clear()
        node.outputs.clear()

        # Squeeze seqlen_k from (B, 1) to (B,) if needed — model_q4 uses (B,1)
        if seqlen_k is not None and getattr(seqlen_k, 'shape', None) and len(seqlen_k.shape) == 2:
            seqlen_k = self.graph.layer(
                name=f"{prefix}/seqlen_k_squeeze", op="Squeeze",
                inputs=[seqlen_k, gs.Constant(f"{prefix}/seqlen_k_squeeze_axes", np.array([1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/seqlen_k_squeezed", dtype=np.int32)],
            )[0]

        nh = self.num_heads
        kvh = self.kv_num_heads
        hd = self.head_dim

        # Reshape Q: (B, S, nh*hd) -> (B, S, nh, hd) -> transpose to (B, nh, S, hd)
        q_reshaped = self.graph.layer(
            name=f"{prefix}/q_reshape", op="Reshape",
            inputs=[q_inp, gs.Constant(f"{prefix}/q_shape", np.array([0, -1, nh, hd], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/q_reshaped")],
        )[0]
        q_transposed = self.graph.layer(
            name=f"{prefix}/q_transpose", op="Transpose",
            inputs=[q_reshaped],
            outputs=[gs.Variable(f"{prefix}/q_transposed")],
            attrs={"perm": [0, 2, 1, 3]},
        )[0]

        # Reshape K: (B, S, kvh*hd) -> (B, S, kvh, hd) -> transpose to (B, kvh, S, hd)
        k_reshaped = self.graph.layer(
            name=f"{prefix}/k_reshape", op="Reshape",
            inputs=[k_inp, gs.Constant(f"{prefix}/k_shape", np.array([0, -1, kvh, hd], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/k_reshaped")],
        )[0]
        k_transposed = self.graph.layer(
            name=f"{prefix}/k_transpose", op="Transpose",
            inputs=[k_reshaped],
            outputs=[gs.Variable(f"{prefix}/k_transposed")],
            attrs={"perm": [0, 2, 1, 3]},
        )[0]

        # Reshape V: same as K
        v_reshaped = self.graph.layer(
            name=f"{prefix}/v_reshape", op="Reshape",
            inputs=[v_inp, gs.Constant(f"{prefix}/v_shape", np.array([0, -1, kvh, hd], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/v_reshaped")],
        )[0]
        v_transposed = self.graph.layer(
            name=f"{prefix}/v_transpose", op="Transpose",
            inputs=[v_reshaped],
            outputs=[gs.Variable(f"{prefix}/v_transposed")],
            attrs={"perm": [0, 2, 1, 3]},
        )[0]

        # Apply RoPE to Q and K using runtime sin/cos computation.
        # Instead of storing massive cos/sin lookup tables (64MB),
        # extract inv_freq from the cache and compute cos/sin at runtime.
        # This matches the fp32 reference model's rotary_emb subgraph pattern.

        # Extract inv_freq from cos/sin caches: inv_freq = atan2(sin[1,:], cos[1,:])
        if isinstance(cos_cache, gs.Constant) and isinstance(sin_cache, gs.Constant):
            cos_vals = np.asarray(cos_cache.values, dtype=np.float32)
            sin_vals = np.asarray(sin_cache.values, dtype=np.float32)
            inv_freq = np.arctan2(sin_vals[1, :], cos_vals[1, :]).astype(np.float32)
            # inv_freq shape: (head_dim//2,) → reshape to (1, head_dim//2, 1) for MatMul
            inv_freq_const = gs.Constant(
                f"{prefix}/inv_freq",
                inv_freq.reshape(1, hd // 2, 1)
            )
        else:
            inv_freq_const = None

        cos_unsq = sin_unsq = None
        if do_rotary and inv_freq_const is not None and seqlen_k is not None:
            # Runtime RoPE computation: cos(position * inv_freq), sin(position * inv_freq)
            # Cast position (int32 scalar) to float32 and reshape to (1, 1, 1) for MatMul
            pos_float = self.graph.layer(
                name=f"{prefix}/pos_cast", op="Cast",
                inputs=[seqlen_k],
                outputs=[gs.Variable(f"{prefix}/pos_float")],
                attrs={"to": int(onnx.TensorProto.FLOAT)},
            )[0]
            pos_3d = self.graph.layer(
                name=f"{prefix}/pos_reshape", op="Reshape",
                inputs=[pos_float, gs.Constant(f"{prefix}/pos_shape", np.array([1, 1, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/pos_3d")],
            )[0]
            # MatMul: inv_freq(1, hd//2, 1) @ position(1, 1, 1) → (1, hd//2, 1)
            angles = self.graph.layer(
                name=f"{prefix}/rope_angles", op="MatMul",
                inputs=[inv_freq_const, pos_3d],
                outputs=[gs.Variable(f"{prefix}/rope_angles_out")],
            )[0]
            # Transpose: (1, hd//2, 1) → (1, 1, hd//2)
            angles_t = self.graph.layer(
                name=f"{prefix}/rope_angles_t", op="Transpose",
                inputs=[angles],
                outputs=[gs.Variable(f"{prefix}/rope_angles_transposed")],
                attrs={"perm": [0, 2, 1]},
            )[0]
            # Duplicate: Concat(angles, angles) → (1, 1, hd)
            angles_full = self.graph.layer(
                name=f"{prefix}/rope_angles_dup", op="Concat",
                inputs=[angles_t, angles_t],
                outputs=[gs.Variable(f"{prefix}/rope_angles_full")],
                attrs={"axis": -1},
            )[0]
            # Cos / Sin
            cos_val = self.graph.layer(
                name=f"{prefix}/rope_cos", op="Cos",
                inputs=[angles_full],
                outputs=[gs.Variable(f"{prefix}/rope_cos_out")],
            )[0]
            sin_val = self.graph.layer(
                name=f"{prefix}/rope_sin", op="Sin",
                inputs=[angles_full],
                outputs=[gs.Variable(f"{prefix}/rope_sin_out")],
            )[0]
            # Unsqueeze to (1, 1, 1, hd) for broadcast with (B, nh, S=1, hd)
            cos_unsq = self.graph.layer(
                name=f"{prefix}/cos_unsqueeze", op="Unsqueeze",
                inputs=[cos_val, gs.Constant(f"{prefix}/unsq_axes_01", np.array([0], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/cos_unsqueezed")],
            )[0]
            sin_unsq = self.graph.layer(
                name=f"{prefix}/sin_unsqueeze", op="Unsqueeze",
                inputs=[sin_val, gs.Constant(f"{prefix}/unsq_axes_01b", np.array([0], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/sin_unsqueezed")],
            )[0]
        elif do_rotary and seqlen_k is not None:
            # Fallback: use original cos/sin cache with Gather
            cos_full = self.graph.layer(
                name=f"{prefix}/cos_dup", op="Concat",
                inputs=[cos_cache, cos_cache],
                outputs=[gs.Variable(f"{prefix}/cos_full")],
                attrs={"axis": -1},
            )[0]
            sin_full = self.graph.layer(
                name=f"{prefix}/sin_dup", op="Concat",
                inputs=[sin_cache, sin_cache],
                outputs=[gs.Variable(f"{prefix}/sin_full")],
                attrs={"axis": -1},
            )[0]
            cos_pos = self.graph.layer(
                name=f"{prefix}/cos_gather", op="Gather",
                inputs=[cos_full, seqlen_k],
                outputs=[gs.Variable(f"{prefix}/cos_at_pos")],
                attrs={"axis": 0},
            )[0]
            sin_pos = self.graph.layer(
                name=f"{prefix}/sin_gather", op="Gather",
                inputs=[sin_full, seqlen_k],
                outputs=[gs.Variable(f"{prefix}/sin_at_pos")],
                attrs={"axis": 0},
            )[0]
            cos_unsq = self.graph.layer(
                name=f"{prefix}/cos_unsqueeze", op="Unsqueeze",
                inputs=[cos_pos, gs.Constant(f"{prefix}/unsq_axes_01", np.array([0, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/cos_unsqueezed")],
            )[0]
            sin_unsq = self.graph.layer(
                name=f"{prefix}/sin_unsqueeze", op="Unsqueeze",
                inputs=[sin_pos, gs.Constant(f"{prefix}/unsq_axes_01b", np.array([0, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/sin_unsqueezed")],
            )[0]
        elif do_rotary:
            # No seqlen_k: use full cos/sin cache directly (e.g. multi-token prefill)
            cos_full = self.graph.layer(
                name=f"{prefix}/cos_dup", op="Concat",
                inputs=[cos_cache, cos_cache],
                outputs=[gs.Variable(f"{prefix}/cos_full")],
                attrs={"axis": -1},
            )[0]
            sin_full = self.graph.layer(
                name=f"{prefix}/sin_dup", op="Concat",
                inputs=[sin_cache, sin_cache],
                outputs=[gs.Variable(f"{prefix}/sin_full")],
                attrs={"axis": -1},
            )[0]
            cos_unsq = self.graph.layer(
                name=f"{prefix}/cos_unsqueeze", op="Unsqueeze",
                inputs=[cos_full, gs.Constant(f"{prefix}/unsq_axes_01", np.array([0, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/cos_unsqueezed")],
            )[0]
            sin_unsq = self.graph.layer(
                name=f"{prefix}/sin_unsqueeze", op="Unsqueeze",
                inputs=[sin_full, gs.Constant(f"{prefix}/unsq_axes_01b", np.array([0, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/sin_unsqueezed")],
            )[0]

        if do_rotary:
            q_rope = self._apply_rope(q_transposed, cos_unsq, sin_unsq, None, f"{prefix}/q")
            k_rope = self._apply_rope(k_transposed, cos_unsq, sin_unsq, None, f"{prefix}/k")
        else:
            # Q/K already rotated by the external q_rotary/k_rotary ops.
            q_rope = q_transposed
            k_rope = k_transposed

        # KV cache concat: concat past with new along sequence dim (axis=2)
        # Reuse original output variables directly to avoid duplicate-name warnings
        if out_present_k is not None:
            # Disconnect from old producer, reuse as Concat output
            out_present_k.inputs.clear()
            present_k_var = out_present_k
        else:
            present_k_var = gs.Variable(f"{prefix}/present_key", dtype=np.float32)
        if out_present_v is not None:
            out_present_v.inputs.clear()
            present_v_var = out_present_v
        else:
            present_v_var = gs.Variable(f"{prefix}/present_value", dtype=np.float32)

        present_k = self.graph.layer(
            name=f"{prefix}/k_concat", op="Concat",
            inputs=[past_key, k_rope],
            outputs=[present_k_var],
            attrs={"axis": -2},
        )[0]
        present_v = self.graph.layer(
            name=f"{prefix}/v_concat", op="Concat",
            inputs=[past_value, v_transposed],
            outputs=[present_v_var],
            attrs={"axis": -2},
        )[0]

        # GQA broadcast: if num_heads > kv_num_heads, expand K,V heads
        if nh != kvh:
            repeat_factor = nh // kvh
            # Unsqueeze K to (B, kvh, 1, S, hd) then Expand to (B, kvh, repeat, S, hd)
            k_for_attn = self.graph.layer(
                name=f"{prefix}/k_unsq_expand", op="Unsqueeze",
                inputs=[present_k, gs.Constant(f"{prefix}/k_unsq_ax", np.array([2], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/k_unsqueezed")],
            )[0]
            k_expanded = self.graph.layer(
                name=f"{prefix}/k_expand", op="Expand",
                inputs=[k_for_attn, gs.Constant(f"{prefix}/k_expand_shape", np.array([1, 1, repeat_factor, 1, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/k_expanded")],
            )[0]
            k_for_attn = self.graph.layer(
                name=f"{prefix}/k_reshape_expanded", op="Reshape",
                inputs=[k_expanded, gs.Constant(f"{prefix}/k_exp_shape", np.array([0, nh, -1, hd], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/k_attn_ready")],
            )[0]

            v_for_attn = self.graph.layer(
                name=f"{prefix}/v_unsq_expand", op="Unsqueeze",
                inputs=[present_v, gs.Constant(f"{prefix}/v_unsq_ax", np.array([2], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/v_unsqueezed")],
            )[0]
            v_expanded = self.graph.layer(
                name=f"{prefix}/v_expand", op="Expand",
                inputs=[v_for_attn, gs.Constant(f"{prefix}/v_expand_shape", np.array([1, 1, repeat_factor, 1, 1], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/v_expanded")],
            )[0]
            v_for_attn = self.graph.layer(
                name=f"{prefix}/v_reshape_expanded", op="Reshape",
                inputs=[v_expanded, gs.Constant(f"{prefix}/v_exp_shape", np.array([0, nh, -1, hd], dtype=np.int64))],
                outputs=[gs.Variable(f"{prefix}/v_attn_ready")],
            )[0]
        else:
            k_for_attn = present_k
            v_for_attn = present_v

        # Q * K^T: transpose K last two dims, then matmul
        k_t = self.graph.layer(
            name=f"{prefix}/k_transpose_attn", op="Transpose",
            inputs=[k_for_attn],
            outputs=[gs.Variable(f"{prefix}/k_transposed_attn")],
            attrs={"perm": [0, 1, 3, 2]},
        )[0]

        # Scale Q
        q_scaled = self.graph.layer(
            name=f"{prefix}/q_scale", op="Mul",
            inputs=[q_rope, gs.Constant(f"{prefix}/scale", np.array(scale, dtype=np.float32))],
            outputs=[gs.Variable(f"{prefix}/q_scaled")],
        )[0]

        # Attention scores: Q_scaled @ K^T
        attn_scores = self.graph.layer(
            name=f"{prefix}/attn_matmul", op="MatMul",
            inputs=[q_scaled, k_t],
            outputs=[gs.Variable(f"{prefix}/attn_scores", dtype=np.float32)],
        )[0]

        # Add attention bias (causal mask) if present; otherwise Softmax sees raw scores
        # and the causal mask is applied later by mask_future_attn_scores
        if attn_bias is not None:
            softmax_input = self.graph.layer(
                name=f"{prefix}/attn_add_bias", op="Add",
                inputs=[attn_scores, attn_bias],
                outputs=[gs.Variable(f"{prefix}/attn_masked")],
            )[0]
        else:
            softmax_input = attn_scores

        # Softmax — name must end with "self_attn/Softmax" so
        # MaskFutureAttentionScores can find it later.
        softmax_name = prefix.rsplit("/", 1)[0] + "/self_attn/Softmax"
        attn_weights = self.graph.layer(
            name=softmax_name, op="Softmax",
            inputs=[softmax_input],
            outputs=[gs.Variable(f"{prefix}/attn_weights")],
            attrs={"axis": -1},
        )[0]

        # Attention output: weights @ V
        attn_output = self.graph.layer(
            name=f"{prefix}/attn_v_matmul", op="MatMul",
            inputs=[attn_weights, v_for_attn],
            outputs=[gs.Variable(f"{prefix}/attn_output")],
        )[0]

        # Transpose back: (B, nh, S, hd) -> (B, S, nh, hd) -> reshape to (B, S, nh*hd)
        attn_transposed = self.graph.layer(
            name=f"{prefix}/attn_transpose", op="Transpose",
            inputs=[attn_output],
            outputs=[gs.Variable(f"{prefix}/attn_transposed")],
            attrs={"perm": [0, 2, 1, 3]},
        )[0]
        attn_reshaped = self.graph.layer(
            name=f"{prefix}/attn_reshape", op="Reshape",
            inputs=[attn_transposed, gs.Constant(f"{prefix}/attn_out_shape", np.array([0, -1, nh * hd], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/attn_reshaped")],
        )[0]

        # Rewire outputs
        if out_attn is not None:
            rewire_consumers(out_attn.outputs.copy(), out_attn, attn_reshaped)
        # present_k/v variables are reused directly as Concat outputs,
        # so no rewiring or graph output update is needed for them.

        self._logger.debug("Replaced GroupQueryAttention '%s'", node.name)


@dataclass
class ReplaceRotaryEmbedding(OnnxGraphEdit):
    """
    Replace a standalone (non-fused) ``com.microsoft.RotaryEmbedding`` node
    with standard ONNX ops, gathering ``cos``/``sin`` host-side instead of
    on-device.

    Some exports apply RoPE to Q/K via a separate ``RotaryEmbedding`` node
    *before* feeding them into an attention op (as opposed to the attention
    op doing rotary internally, e.g. ``GroupQueryAttention`` with
    ``do_rotary=1`` -- see ``ReplaceGroupQueryAttention``). This edit handles
    that standalone case: ``x*cos + rotate_half(x)*sin``, non-interleaved
    (GPT-NeoX style), matching ``ReplaceGroupQueryAttention._apply_rope``'s
    convention.

    Unlike an earlier version of this edit, ``cos_at_pos``/``sin_at_pos``
    are **not** computed in-graph via ``Gather(cos_cache, position_ids)`` +
    a self-``Concat`` duplication to full ``head_dim``. Confirmed
    (torq-tools-dev issue investigation, gemma4 static export) that pattern
    reliably fails to compile: `cos_cache`/`sin_cache` are precomputed for
    the source model's *full* max-context-length (Gemma-4:
    ``max_position_embeddings=131072``), and once that ``Gather`` is fused
    with any real downstream elementwise consumer (the RoPE ``Mul``), the
    torq compiler's tile-and-fuse pass can't reduce the huge constant's
    footprint (`"no more domains to tile"` -- the *output* is already
    minimal, but a fully data-dependent single-element extract has no
    affine access pattern into the huge *input* to tile), leaving the whole
    untiled table to blow the on-chip LRAM budget. This is unrelated to and
    survives removing the self-``Concat`` alone -- confirmed via isolated
    minimal repros extracted directly from the real model.

    Since ``cos_at_pos``/``sin_at_pos`` are a pure, deterministic function
    of ``position_ids`` with no learned weights, there is no need for the
    NPU to ever see the full table at all -- exactly the same reasoning
    already applied to the embedding table (see
    ``ExtractGatherBlockQuantizedLUT``: ``embed_tokens`` needs zero NPU
    compilation because its lookup runs host-side instead). This edit does
    the same for RoPE: each distinct ``cos_cache``/``sin_cache`` pair
    referenced by any matched node is saved once, already duplicated to
    full ``head_dim`` (row ``i`` of the saved array is exactly what
    ``cos_at_pos``/``sin_at_pos`` would have been for
    ``position_ids == i``), to ``save_to`` -- and a new, tiny
    ``[1,1,head_dim]`` graph input (named by transforming the cache's own
    name, e.g. ``cos_cache_local`` -> ``cos_full_local``) replaces the
    ``Gather``+``Concat`` in the graph. Only the real per-token math
    (``Mul``/rotate-half/``Add``) remains on the NPU. Host-side code must
    look up the row for the current ``position_ids`` and feed it as that
    input at inference time (see ``_RopeCacheLUT``/`` _lookup_rope_caches``
    in ``models/gemma4/export_int4.py`` for the reference implementation).
    Multiple ``RotaryEmbedding`` nodes sharing the same source cache (e.g.
    every sliding-window layer's Q and K rotary) all resolve to the *same*
    graph input -- one lookup per cache variant per forward step, not one
    per layer.

    Unlike ``ReplaceGroupQueryAttention``, ``head_dim``/``num_heads`` are
    derived per-node from that node's own ``cos_cache`` shape and the input
    tensor's (by then static) hidden size, since a single graph can
    legitimately mix multiple RoPE configurations (e.g. Gemma-4's
    local/sliding layers at head_dim=256 alongside global/full-attention
    layers at head_dim=512, both using this same op).

    Args:
        save_to: destination ``.npz`` path for the (deduplicated,
            already-duplicated-to-``head_dim``) cache tables. All distinct
            cache variants referenced by matched nodes in this graph are
            written to the *same* file, one array per variant.

    Raises:
        ValueError: If ``interleaved=1`` or a non-default
            ``rotary_embedding_dim`` is requested (unsupported -- no known
            export uses them yet) -- or if ``cos_cache``/``sin_cache`` aren't
            constants, or the input's hidden dim isn't statically known
            (this edit must run after `fix_io_dims`).
    """

    save_to: os.PathLike | str
    _cache_arrays: dict = field(default_factory=dict, init=False, repr=False)
    _cache_inputs: dict = field(default_factory=dict, init=False, repr=False)

    def match(self, node: gs.Node) -> bool:
        return node.op == "RotaryEmbedding"

    def _host_gathered_cache_input(self, cache: gs.Constant, head_dim: int) -> gs.Variable:
        """Return the (possibly newly-created) graph input holding
        ``cache``'s rows already duplicated to ``head_dim``, deduplicated
        across every node in this graph that shares the same source cache.
        """
        inp_name = cache.name.replace("_cache_", "_full_")
        existing = self._cache_inputs.get(inp_name)
        if existing is not None:
            return existing

        values = np.asarray(cache.values)
        duplicated = np.concatenate([values, values], axis=-1)
        self._cache_arrays[inp_name] = duplicated

        # A directory of individual `.npy` files, NOT a single `.npz`: only
        # one row per variant is read per inference step, and numpy cannot
        # memory-map a `.npz` (zip container -- `mmap_mode` silently has no
        # effect). These tables are sized for the source model's full max
        # context, so loading them resident is a large, pure waste (~768MB
        # measured for gemma4). See `_RopeCacheLUT` for the reader.
        save_to = Path(self.save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        np.save(save_to / f"{inp_name}.npy", duplicated)

        new_inp = gs.Variable(name=inp_name, dtype=duplicated.dtype, shape=[1, 1, head_dim])
        self.graph.inputs.append(new_inp)
        self._cache_inputs[inp_name] = new_inp
        return new_inp

    def transform(self, node: gs.Node):
        self._check_node_op(node, "RotaryEmbedding")

        if int(node.attrs.get("interleaved", 0)) != 0:
            raise ValueError(
                f"'{node.name}': interleaved RotaryEmbedding is not supported by this edit"
            )

        x, _position_ids, cos_cache, sin_cache = node.inputs[:4]
        if not (isinstance(cos_cache, gs.Constant) and isinstance(sin_cache, gs.Constant)):
            raise ValueError(
                f"'{node.name}': cos_cache/sin_cache must be constants for this edit"
            )
        half_dim = int(cos_cache.values.shape[-1])
        head_dim = 2 * half_dim
        rotary_dim = int(node.attrs.get("rotary_embedding_dim", 0) or head_dim)
        if rotary_dim != head_dim:
            raise ValueError(
                f"'{node.name}': partial rotation (rotary_embedding_dim={rotary_dim} != "
                f"head_dim={head_dim}) is not supported by this edit"
            )

        x_shape = getattr(x, "shape", None)
        if not x_shape or not isinstance(x_shape[-1], int):
            raise ValueError(
                f"'{node.name}': input's hidden dim must be statically known -- "
                "run `fix_io_dims` before this edit"
            )
        hidden = int(x_shape[-1])
        num_heads = int(node.attrs.get("num_heads", 0)) or hidden // head_dim

        out_var = node.outputs[0]
        prefix = node.name
        node.inputs.clear()
        node.outputs.clear()

        # (B, S, H) -> (B, S, nh, hd)
        x_heads = self.graph.layer(
            name=f"{prefix}/split_heads", op="Reshape",
            inputs=[x, gs.Constant(f"{prefix}/split_heads_shape", np.array([0, 0, num_heads, head_dim], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/x_heads")],
        )[0]

        # cos/sin at the current position, already duplicated to full
        # head_dim: gathered host-side (see class docstring for why), fed
        # in as a small graph input instead of computed in-graph. Then
        # unsqueeze a heads axis to broadcast against (B, S, nh, hd).
        cos_full = self._host_gathered_cache_input(cos_cache, head_dim)
        sin_full = self._host_gathered_cache_input(sin_cache, head_dim)
        cos_unsq = self.graph.layer(
            name=f"{prefix}/cos_unsqueeze", op="Unsqueeze",
            inputs=[cos_full, gs.Constant(f"{prefix}/cos_unsq_axes", np.array([2], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/cos_unsqueezed")],
        )[0]
        sin_unsq = self.graph.layer(
            name=f"{prefix}/sin_unsqueeze", op="Unsqueeze",
            inputs=[sin_full, gs.Constant(f"{prefix}/sin_unsq_axes", np.array([2], dtype=np.int64))],
            outputs=[gs.Variable(f"{prefix}/sin_unsqueezed")],
        )[0]

        # x*cos + rotate_half(x)*sin, rotate_half = concat(-x[..., hd/2:], x[..., :hd/2])
        mul_cos = self.graph.layer(
            name=f"{prefix}/rope_mul_cos", op="Mul",
            inputs=[x_heads, cos_unsq],
            outputs=[gs.Variable(f"{prefix}/rope_mul_cos_out")],
        )[0]
        x_first = self.graph.layer(
            name=f"{prefix}/rope_slice_first", op="Slice",
            inputs=[
                x_heads,
                gs.Constant(f"{prefix}/rope_start_0", np.array([0], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_end_half", np.array([half_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_axis_neg1", np.array([-1], dtype=np.int64)),
            ],
            outputs=[gs.Variable(f"{prefix}/rope_first_half")],
        )[0]
        x_second = self.graph.layer(
            name=f"{prefix}/rope_slice_second", op="Slice",
            inputs=[
                x_heads,
                gs.Constant(f"{prefix}/rope_start_half", np.array([half_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_end_full", np.array([head_dim], dtype=np.int64)),
                gs.Constant(f"{prefix}/rope_axis_neg1b", np.array([-1], dtype=np.int64)),
            ],
            outputs=[gs.Variable(f"{prefix}/rope_second_half")],
        )[0]
        neg_second = self.graph.layer(
            name=f"{prefix}/rope_neg", op="Neg",
            inputs=[x_second],
            outputs=[gs.Variable(f"{prefix}/rope_neg_out")],
        )[0]
        rotated = self.graph.layer(
            name=f"{prefix}/rope_concat", op="Concat",
            inputs=[neg_second, x_first],
            outputs=[gs.Variable(f"{prefix}/rope_rotated")],
            attrs={"axis": -1},
        )[0]
        mul_sin = self.graph.layer(
            name=f"{prefix}/rope_mul_sin", op="Mul",
            inputs=[rotated, sin_unsq],
            outputs=[gs.Variable(f"{prefix}/rope_mul_sin_out")],
        )[0]
        rope_out = self.graph.layer(
            name=f"{prefix}/rope_add", op="Add",
            inputs=[mul_cos, mul_sin],
            outputs=[gs.Variable(f"{prefix}/rope_out")],
        )[0]

        # (B, S, nh, hd) -> (B, S, H), restoring the original op's output contract.
        merged = self.graph.layer(
            name=f"{prefix}/merge_heads", op="Reshape",
            inputs=[rope_out, gs.Constant(f"{prefix}/merge_heads_shape", np.array([0, 0, -1], dtype=np.int64))],
            outputs=[out_var],
        )[0]

        self._logger.debug("Replaced standalone RotaryEmbedding '%s'", node.name)


@dataclass
class DecomposeMatMulNBits(OnnxGraphEdit):
    """
    Decompose a `com.microsoft.MatMulNBits` node (`bits=4`, blockwise
    scale/zero_point) into a standard-ONNX chain the compiler already
    supports: `DequantizeLinear` (packed INT4 weight, bf16 scale, packed
    INT4 zero_point, opset-21 blocked quantization) -> `MatMul`.

    Validated against the real target compiler (see
    `models/gemma4-e2b-int4/export/onnx/op_repros/decompose_matmul_nbits.py`,
    the standalone repro this was ported from, and
    `STATIC_EXPORT_PLAN.md`): confirmed the compiler supports bf16 `MatMul`
    directly, and blocked `DequantizeLinear` with `axis=0`. Three design
    choices carried over from that validation:

    - `DequantizeLinear` runs with `axis=0` against a weight already
      transposed to `[K,N]` (not the `[N,K]` layout `MatMulNBits` stores),
      so its output is already in the layout `MatMul` needs -- no runtime
      `Transpose` node, and no data movement over the *full* dequantized
      matrix just to fix up layout. The transpose happens once, in NumPy,
      on the still-quantized (small) packed weight/scale/zero_point at
      *this* (export) time -- correct by construction, since blocked
      dequant is elementwise per-block (`y_T = y.T`).
    - If the node's activation input is already bf16 (true once this runs
      on a bf16-converted graph -- the intended integration point), no
      `Cast` is inserted at all: `DequantizeLinear -> MatMul`, straight bf16.
      Otherwise a `Cast` bridges dtypes on the *activation and output*
      (tiny: `[...,K]`/`[...,N]`) rather than the weight (`[K,N]`, orders of
      magnitude larger) -- same node count, far less data touched per
      forward pass.
    - The weight/zero_point are packed as native `TensorProto.INT4` (see
      `_make_signed_int4_constant`), not plain int8 as an earlier version of
      this class used. Motivated by a real, reproduced failure: with plain
      int8, torq-compile's own automatic tensor-slicing pass (needed
      regardless, since every one of this model's weights vastly exceeds the
      target's 512KB on-chip LRAM once dequantized) hit a
      `memref.collapse_shape` stride mismatch specific to the int8 operand.
      This matches the already-established, torq-compile-tested int4
      pathway instead (`torq.tools.quantization.weight_quantization.
      quantize`, which packs the same way) -- not yet re-verified end-to-end
      against the real compiler after switching, so treat as a strong,
      well-motivated hypothesis until that check has actually run.

    Raises:
        NotImplementedError: if `bits != 4`.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "MatMulNBits"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMulNBits")

        K = int(node.attrs["K"])
        N = int(node.attrs["N"])
        bits = int(node.attrs["bits"])
        block_size = int(node.attrs["block_size"])
        if bits != 4:
            raise NotImplementedError(f"'{node.name}': only bits=4 is handled, got bits={bits}")
        n_blocks = K // block_size

        a, b_quant, scales, zero_points = node.inputs[:4]
        out_var = node.outputs[0]
        prefix = node.name or out_var.name

        for t, label in ((b_quant, "weight"), (scales, "scales"), (zero_points, "zero_points")):
            if not isinstance(t, gs.Constant):
                raise ValueError(f"'{node.name}': '{label}' input must be a constant to decompose")

        w_packed = np.asarray(b_quant.values)                        # [N, n_blocks, block_size*bits//8]
        zp_packed = np.asarray(zero_points.values)                   # [N, n_blocks*bits//8]
        scale_vals = np.asarray(scales.values).astype(np.float32)    # [N, n_blocks]

        # Unsigned nibble range [0,15] -- MatMulNBits's native convention,
        # NOT yet shifted to the signed [-8,7] range `_make_signed_int4_constant`
        # packs as INT4 (see that function's docstring for why the shift is
        # exact/lossless).
        w_int8 = _unpack_nibbles(w_packed).reshape(N, n_blocks * block_size)[:, :K].astype(np.int8)
        zp_int8 = _unpack_nibbles(zp_packed)[:, :n_blocks].astype(np.int8)

        # Transpose the still-quantized weight/zero_point/scale to
        # [K,N]/[n_blocks,N] here, in NumPy, once -- see class docstring.
        w_int8_t = np.ascontiguousarray(w_int8.T)                              # [K, N]
        zp_int8_t = np.ascontiguousarray(zp_int8.T)                            # [n_blocks, N]
        scale_bf16_t = np.ascontiguousarray(scale_vals.T).astype(ml_dtypes.bfloat16)  # [n_blocks, N]

        # Packed native INT4 (not plain int8): confirmed the established,
        # torq-compile-tested int4 pathway (`weight_quantization/quantize.py`)
        # uses packed `TensorProto.INT4`, not int8, for both weight and
        # zero_point -- and that using plain int8 here (as an earlier version
        # of this class did) produced a `memref.collapse_shape` stride
        # mismatch specifically in the compiler's slicing pass. See class
        # docstring.
        w_const = _make_signed_int4_constant(f"{prefix}/W_int4_T", w_int8_t)
        zp_const = _make_signed_int4_constant(f"{prefix}/zp_int4_T", zp_int8_t)
        scale_const = gs.Constant(f"{prefix}/scale_bf16_T", scale_bf16_t)

        node.inputs.clear()
        node.outputs.clear()

        w_dq_bf16 = self.graph.layer(
            name=f"{prefix}/DequantizeLinear", op="DequantizeLinear",
            inputs=[w_const, scale_const, zp_const],
            outputs=[gs.Variable(f"{prefix}/W_dq_bf16", dtype=np.dtype(ml_dtypes.bfloat16), shape=[K, N])],
            attrs={"axis": 0, "block_size": block_size},
        )[0]

        bf16 = np.dtype(ml_dtypes.bfloat16)
        act_is_bf16 = _is_bfloat16(a.dtype)
        if act_is_bf16:
            matmul_a, matmul_out = a, out_var
        else:
            matmul_a = self.graph.layer(
                name=f"{prefix}/CastActBf16", op="Cast",
                inputs=[a],
                outputs=[gs.Variable(f"{prefix}/A_bf16", dtype=bf16, shape=a.shape)],
                attrs={"to": int(TensorProto.BFLOAT16)},
            )[0]
            matmul_out = gs.Variable(f"{prefix}/Y_bf16", dtype=bf16, shape=out_var.shape)
            self.graph.layer(
                name=f"{prefix}/CastOut", op="Cast",
                inputs=[matmul_out],
                outputs=[out_var],
                attrs={"to": _tensor_proto_dtype(out_var.dtype)},
            )

        self.graph.layer(
            name=f"{prefix}/MatMul", op="MatMul",
            inputs=[matmul_a, w_dq_bf16],
            outputs=[matmul_out],
        )

        self._logger.debug(
            "Decomposed MatMulNBits '%s' (K=%d, N=%d, block_size=%d, cast=%s)",
            node.name, K, N, block_size, not act_is_bf16,
        )
