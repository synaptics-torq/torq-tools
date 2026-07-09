# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2.5 (Liquid) custom-op replacements.

Rewrites the ORT ``com.microsoft`` fused ops that appear in the LiquidAI
LFM2.5 ONNX export — ``SimplifiedLayerNormalization``,
``SkipSimplifiedLayerNormalization`` and ``GroupQueryAttention`` — into
standard ONNX ops the Torq compiler can lower.  Split out of the monolithic
``edits.py`` when it was refactored into the ``graph_edit.edits`` package.
"""

import re
from dataclasses import dataclass

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers


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

        if inv_freq_const is not None and seqlen_k is not None:
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
        elif seqlen_k is not None:
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
        else:
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

        q_rope = self._apply_rope(q_transposed, cos_unsq, sin_unsq, None, f"{prefix}/q")
        k_rope = self._apply_rope(k_transposed, cos_unsq, sin_unsq, None, f"{prefix}/k")

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
