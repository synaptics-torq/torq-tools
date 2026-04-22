# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import hashlib
import os
import re

import onnx
import onnx_graphsurgeon as gs
import numpy as np

from .onnx import (
    OnnxGraphEdit,
    rewire_consumers
)

from ..utils.onnx import (
    normalize_layer_name
)

__all__ = [
    "ReplaceDynamicKVCache",
    "MaskFutureAttentionScores",
    "AddCurrLenInput",
    "ConvertToStaticIndex",
    "DequantizeProjectionsMatMul",
    "RemoveIsNaN",
    "RemoveRedundantCasts",
    "FoldScalarMatMul",
    "ConstantBroadcastPolicy",
    "BroadcastOpInputs",
    "ExtractConstantLUT",
    "EliminateTranspose",
    "CollapseReshapeChain",
    "CollapseGQABroadcast",
    "ReplaceSimplifiedLayerNorm",
    "ReplaceMatMulNBits",
    "ReplaceGroupQueryAttention",
    "CombineKVCacheMixin",
    "CommonGraphEditsMixin",
]


@dataclass
class ReplaceDynamicKVCache(OnnxGraphEdit):
    """
    Replace dynamic key-value cache updates with a static in-place blend.

    `cache[i] = new_value if i == cur_len else cache[i]`

    Args:
        cur_len (gs.Variable): Graph input to represent current sequence length
        max_tokens (int): Maximum sequence length

    Raises:
        ValueError: If Concat node doesn't have expected attributes

    Notes:
        - Builds a mask that is true for the current position
        - Blends the new cache value into the existing cache using the mask
        - Disconnects old Concat node from the graph
        - Optimizers may CSE-deduplicate identical masks into one shared tensor
    """

    cur_len: gs.Variable
    max_tokens: int

    def __post_init__(self):
        self.output_names = {o.name for o in self.graph.outputs}
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        return node.op == "Concat" and node.outputs[0].name in self.output_names

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Concat")
        cache_output = node.outputs[0].name
        if node.attrs["axis"] != -2:
            raise ValueError(
                f"Static KV Cache: '{node.name}' expected Concat axis to be -2, got {node.attrs['axis']}"
            )
        if len(node.inputs) != 2:
            raise ValueError(
                f"Static KV Cache: '{node.name}' expected Concat node to have 2 inputs, got {len(node.inputs)}"
            )

        past_cache_vals, new_cache_val = node.inputs
        output = node.outputs[0]

        # create mask for current position
        mask_shape = [1, 1, self.max_tokens, 1]
        if not (time_ids := self.graph.tensors().get("time_ids")):
            time_ids = gs.Constant(
                "time_ids", np.arange(self.max_tokens, dtype=np.int64).reshape(*mask_shape)
            )
        mask = self.graph.layer(
            name=output.name + "_update_mask",
            op="Equal",
            inputs=[time_ids, self.cur_len],
            outputs=[
                gs.Variable(
                    f"{output.name}_mask_eq", dtype=onnx.TensorProto.BOOL, shape=mask_shape
                )
            ],
        )[0]

        # blend new value into cache using the mask
        self.graph.layer(
            name=output.name + "_blend_kv",
            op="Where",
            inputs=[mask, new_cache_val, past_cache_vals],
            outputs=[output],
        )

        # disconnect Concat node
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug("Added static KV cache for output '%s'", cache_output)


@dataclass
class MaskFutureAttentionScores(OnnxGraphEdit):
    """
    Add causal masking to attention scores to prevent attending to future tokens.

    Enforces left-to-right causality by assigning a large negative value to positions > `cur_len`, thereby blocking future positions.

    Args:
        cur_len (gs.Variable): Graph input to represent current sequence length
        max_tokens (int): Maximum number of tokens in sequence
        export_dtype (onnx.TensorProto.DataType): ONNX export data type for tensors

    Raises:
        ValueError: If Softmax producer is not the expected op

    Notes:
        - Creates a mask that is only true for positions <= cur_len
        - Rewires the attention score producer to use this mask
        - Optimizers may CSE-deduplicate identical masks into one shared tensor
    """

    cur_len: gs.Variable
    max_tokens: int
    export_dtype: onnx.TensorProto.DataType

    def __post_init__(self):
        if self.export_dtype not in onnx.TensorProto.DataType.values():
            raise RuntimeError(f"A valid export dtype is required for this edit, received {type(self.export_dtype)}")
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        if node.op == "Softmax" and node.name.endswith("self_attn/Softmax"):
            return isinstance(node.i(), gs.Node)
        return False

    def transform(self, node: gs.Node):
        if not self.export_dtype:
            raise RuntimeError("ONNX export dtype is requried for this graph edit, provide via `export_dtype`")

        self._check_node_op(node, "Softmax")

        # create bool mask where positions > cur_len are effectively blocked
        # by being set to a large negative value
        mask_shape = [1, 1, 1, self.max_tokens]
        if not (time_axis := self.graph.tensors().get("time_axis")):
            time_axis = gs.Constant(
                "time_axis", np.arange(self.max_tokens, dtype=np.int64).reshape(*mask_shape)
            )
        if not (attn_mask_keep := self.graph.tensors().get("attn_mask_keep")):
            attn_mask_keep = gs.Constant(
                "attn_mask_keep", np.asarray(0.0, dtype=np.float32),
                export_dtype=self.export_dtype
            )
        if not (attn_mask_block := self.graph.tensors().get("attn_mask_block")):
            max_float = -65504 if self.export_dtype == onnx.TensorProto.FLOAT16 else -1e9
            attn_mask_block = gs.Constant(
                "attn_mask_block", np.asarray(max_float, dtype=np.float32),
                export_dtype=self.export_dtype
            )
        mask_lte = self.graph.layer(
            name=node.name + "_lte_cur_len",
            op="LessOrEqual",
            inputs=[time_axis, self.cur_len],
            outputs=[
                gs.Variable(
                    node.name + "_less", dtype=onnx.TensorProto.BOOL, shape=mask_shape
                )
            ],
        )[0]
        mask = self.graph.layer(
            name=node.name + "_mask_attn",
            op="Where",
            inputs=[mask_lte, attn_mask_keep, attn_mask_block],
            outputs=[
                gs.Variable(node.name + "_where", dtype=node.inputs[0].dtype or np.float32, shape=mask_shape)
            ],
        )[0]

        # rewire producer node to use mask
        producer_node: gs.Node = node.i()
        if producer_node.op != "Add":
            producer_output: gs.Variable = node.inputs[0]
            consumers: list[gs.Node] = producer_output.outputs.copy()
            add_output: gs.Variable = self.graph.layer(
                name=node.name + "_bias_add",
                op="Add",
                inputs=[node.inputs[0], mask],
                outputs=[
                    gs.Variable(node.name + "_biased", dtype=producer_output.dtype or np.float32, shape=producer_output.shape)
                ],
            )[0]
            rewire_consumers(consumers, producer_output, add_output)
        else:
            producer_node.inputs[1] = mask

        self._logger.debug("Added causal attention mask to scores at node '%s'", producer_node.name)


@dataclass
class AddCurrLenInput(OnnxGraphEdit):
    """
    Replace dynamic sequence length computation with runtime model input.

    Removes the Shape->Gather runtime-calculated sequence length and replaces it with the model input `cur_len`.

    Args:
        cur_len (gs.Variable): Graph input to represent current sequence length

    Raises:
        ValueError: If Shape consumer is not a `Gather` op

    Notes:
        - Replaces `Shape(past_key_values) -> Gather(i=2)` with `cur_len`
        - Disconnects original Shape and Gather nodes
    """

    cur_len: gs.Variable

    def match(self, node: gs.Node) -> bool:
        if node.op == "Shape" and "past_key_values" in node.inputs[0].name:
            return isinstance(node.o(), gs.Node) and node.o().op == "Gather"
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Shape")
        gather_node: gs.Node = node.o()
        if not isinstance(gather_node, gs.Node) or gather_node.op != "Gather":
            raise ValueError(f"Expected Gather node after Shape, got {gather_node}")

        gather_out: gs.Variable = gather_node.outputs[0]
        consumers: list[gs.Node] = list(gather_out.outputs)
        rewire_consumers(consumers, gather_out, self.cur_len)

        # disconnect Shape + Gather branch
        node.inputs.clear()
        gather_node.outputs.clear()

        self._logger.debug("Replaced dynamic seq len getter at node '%s'", node.name)


@dataclass
class ConvertToStaticIndex(OnnxGraphEdit):
    """
    Convert dynamic Range-based indexing to static indexing if `index = Range(start, start + 1, 1)`.

    Replaces redundant index computation `Range(start, start + 1, 1)` by wiring consumers to directly accept `start`.

    Raises:
        ValueError: If Range limit is not produced by an `Add` op
        ValueError: If Range start and limit don't share a common producer

    Notes:
        - Directly connects Range start to consumers of Range node
        - Disconnects Range node from the graph
    """

    def match(self, node: gs.Node) -> bool:
        return (
            node.op == "Range"
            and node.i(1).op == "Add"
            and any(inp is node.inputs[0] for inp in node.i(1).inputs)
        )

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Range")
        start = node.inputs[0]
        limit_prod = node.i(1)
        if limit_prod.op != "Add":
            raise ValueError(
                f"Expected Add node for limit, got {limit_prod.op} for dynamic range replacement"
            )
        if not any(inp is start for inp in limit_prod.inputs):
            raise ValueError(
                f"Range node and limit node must have common producer for dynamic range replacement"
            )
        range_out: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(range_out.outputs)
        for consumer in consumers:
            for i, inp in enumerate(consumer.inputs):
                if inp is range_out:
                    consumer.inputs[i] = start

        # disconnect Range node
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug("Replaced dynamic range index for node '%s'", node.name)


@dataclass
class DequantizeProjectionsMatMul(OnnxGraphEdit):
    """
    Manually dequantize projection scores MatMul producer to prevent MLIR warnings.

    Args:
        hidden_size (int): Model hidden KV dims size
        vocab_size (int): Model vocabulary size
        export_dtype (onnx.TensorProto.DataType): ONNX export data type for tensors

    Raises:
        ValueError: If MatMul producer is not a `DequantizeLinear` op
        ValueError: If weights are not correctly formatted
        ValueError: If dequantization params are not correctly formatted
    """

    hidden_size: int
    vocab_size: int
    export_dtype: onnx.TensorProto.DataType

    def __post_init__(self):
        if self.export_dtype not in onnx.TensorProto.DataType.values():
            raise RuntimeError(f"A valid export dtype is required for this edit, received {type(self.export_dtype)}")
        return super().__post_init__()

    def match(self, node: gs.Node):
        if node.op == "MatMul" and node.outputs[0].name == "logits":
            return isinstance(node.i(1), gs.Node) and node.i(1).op == "DequantizeLinear"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        dequant_node: gs.Node = node.i(1)
        try:
            transpose_node: gs.Node = dequant_node.i()
        except IndexError:
            self._logger.debug("Dequantize node does not have Transpose input, looking in inputs for const weight")
            quant_weights: gs.Constant = dequant_node.inputs[0]
        else:
            quant_weights: gs.Constant = transpose_node.inputs[0]
        if not isinstance(quant_weights, gs.Constant):
            self._logger.warning("Dequantization weights not found, skipping")
            return

        self._check_node_op(dequant_node, "DequantizeLinear")

        W_q: np.ndarray = quant_weights.values
        if W_q.shape == (self.vocab_size, self.hidden_size):
            W_q = W_q.T
        if W_q.shape != (self.hidden_size, self.vocab_size):
            raise ValueError(f"Expected weight shape of {(self.vocab_size, self.hidden_size)} or {(self.hidden_size, self.vocab_size)}, got {W_q.shape}")
        if W_q.dtype != np.uint8:
            raise ValueError(f"Expected uint8 weights, got {W_q.dtype}")

        if len(dequant_node.inputs) < 3:
            raise ValueError(f"Expected 3 inputs (x, scale, zp) for DequantizeLinear node, got {len(dequant_node.inputs)}")
        scale_inp, zp_inp = dequant_node.inputs[1], dequant_node.inputs[2]
        if not isinstance(scale_inp, gs.Constant):
            raise ValueError(f"Expected constant scale, got {type(scale_inp)}")
        if not isinstance(zp_inp, gs.Constant):
            raise ValueError(f"Expected constant zp, got {type(scale_inp)}")
        scale = scale_inp.values.item()
        zp = zp_inp.values.item()
        node.inputs[1] = gs.Constant(
            node.inputs[1].name + "_float_folded",
            (W_q.astype(np.int32) - np.int32(zp)).astype(np.float32) * np.float32(scale),
            export_dtype=self.export_dtype
        )

        dequant_node.outputs.clear()

        self._logger.debug("Dequantized projection scores producer")


@dataclass
class RemoveIsNaN(OnnxGraphEdit):
    """
    Remove unsupported IsNaN operations.

    Raises:
        ValueError: If IsNaN is not consumed by a `Where` op
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "IsNaN"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "IsNaN")
        producer: gs.Tensor = node.inputs[0]
        where_node: gs.Node = node.o()
        if where_node.op != "Where":
            raise ValueError(
                f"Expected Where node consumer, got {where_node.op} for IsNaN replacement"
            )
        where_out: gs.Variable = where_node.outputs[0]
        consumers: list[gs.Node] = list(where_out.outputs)
        rewire_consumers(consumers, where_out, producer)

        # disconnect IsNaN -> Where chain
        node.inputs.clear()
        where_node.inputs.clear()
        where_node.outputs.clear()

        self._logger.debug("Removed unsupported IsNaN op '%s'", node.name)


@dataclass
class RemoveRedundantCasts(OnnxGraphEdit):
    """
    Remove redundant Cast ops where input dtype == output dtype
    """

    @staticmethod
    def _to_onnx_dtype(dtype: np.dtype | int | None) -> int | None:
        if dtype is None:
            return None
        if isinstance(dtype, int):
            return dtype
        try:
            return onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        except Exception:
            return None

    def match(self, node: gs.Node) -> bool:
        if node.op != "Cast" or not node.inputs or not node.outputs:
            return False
        inp_dtype = self._to_onnx_dtype(getattr(node.inputs[0], "dtype", None))
        if inp_dtype is None:
            return False
        cast_to = node.attrs.get("to", None)
        if isinstance(cast_to, int) and inp_dtype == cast_to:
            return True
        out_dtype = self._to_onnx_dtype(getattr(node.outputs[0], "dtype", None))
        return out_dtype is not None and inp_dtype == out_dtype

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Cast")
        inp = node.inputs[0]
        out = node.outputs[0]
        consumers: list[gs.Node] = list(out.outputs)
        rewire_consumers(consumers, out, inp)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is out:
                self.graph.outputs[i] = inp
        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug("Removed redundant Cast node '%s'", node.name)


@dataclass
class FoldScalarMatMul(OnnxGraphEdit):
    """
    Fold `MatMul A @ B`, where B is a batched scalar, into Mul.

    Raises:
        ValueError: If MatMul operand shapes are incompatible
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul":
            return False

        a, b = node.inputs
        a_shape = getattr(a, "shape", None)
        b_shape = getattr(b, "shape", None)
        if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
            return a_shape[-1] == 1 and b_shape[-2] == 1 and b_shape[-1] == 1
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        a, b = node.inputs
        a_shape = getattr(a, "shape", None)
        b_shape = getattr(b, "shape", None)
        y = node.outputs[0]

        if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("Invalid MatMul operand shapes for scalar scale matmul replacement")
        if not (a_shape[-1] == 1 and b_shape[-2] == 1 and b_shape[-1] == 1):
            raise ValueError(f"Expected scalar-compatible MatMul shapes, got A={a_shape}, B={b_shape}")
        
        self.graph.layer(
            name=node.name + "_mul_fold",
            op="Mul",
            inputs=[a, b],
            outputs=[y]
        )
        node.outputs.clear()

        self._logger.debug("Folded scalar MatMul node '%s' into Mul", node.name)


@dataclass
class EliminateTranspose(OnnxGraphEdit):
    """
    Eliminate Transpose ops that don't rearrange data in memory.

    Handles two cases:
    1. No-op: permuted shape equals input shape (e.g., transposing two dims of equal size).
       The Transpose is bypassed entirely.
    2. Data-preserving: permutation only swaps dims where at most one per cycle has size > 1
       (e.g., ``[1,1,H,D]`` with ``perm=[0,2,1,3]`` → ``[1,H,1,D]`` when seq_len dim is 1).
       The Transpose is replaced with an equivalent Reshape.
    """

    @staticmethod
    def _is_data_preserving_perm(perm: list[int], shape: list[int]) -> bool:
        visited = [False] * len(perm)
        for i in range(len(perm)):
            if visited[i] or perm[i] == i:
                visited[i] = True
                continue
            cycle_dims: list[int] = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle_dims.append(shape[j])
                j = perm[j]
            if sum(1 for d in cycle_dims if d > 1) > 1:
                return False
        return True

    def match(self, node: gs.Node) -> bool:
        if node.op != "Transpose" or not node.inputs or not node.outputs:
            return False
        perm = node.attrs.get("perm", None)
        if perm is None:
            return False
        inp_shape = getattr(node.inputs[0], "shape", None)
        if inp_shape is None or not all(isinstance(d, (int, np.integer)) for d in inp_shape):
            return False

        perm = [int(p) for p in perm]
        shape = [int(d) for d in inp_shape]

        return self._is_data_preserving_perm(perm, shape)

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Transpose")
        inp = node.inputs[0]
        out = node.outputs[0]
        inp_shape = [int(d) for d in inp.shape]
        out_shape = [int(d) for d in out.shape]
        consumers: list[gs.Node] = list(out.outputs)

        if inp_shape == out_shape:
            rewire_consumers(consumers, out, inp)
            for i, graph_out in enumerate(self.graph.outputs):
                if graph_out is out:
                    self.graph.outputs[i] = inp
        else:
            shape_const = gs.Constant(
                name=node.name + "_fold_shape",
                values=np.array(out_shape, dtype=np.int64)
            )
            reshape_out: gs.Variable = self.graph.layer(
                name=node.name + "_fold_reshape",
                op="Reshape",
                inputs=[inp, shape_const],
                outputs=[gs.Variable(
                    name=out.name + "_reshaped",
                    dtype=out.dtype,
                    shape=out_shape
                )]
            )[0]
            rewire_consumers(consumers, out, reshape_out)
            for i, graph_out in enumerate(self.graph.outputs):
                if graph_out is out:
                    self.graph.outputs[i] = reshape_out

        node.inputs.clear()
        node.outputs.clear()
        if inp_shape == out_shape:
            self._logger.debug(
                "Eliminated Transpose '%s': %s -> %s", node.name, inp_shape, out_shape
            )
        else:
            self._logger.debug(
                "Folded Transpose '%s' into Reshape '%s'", node.name, node.name + "_fold_reshape"
            )


@dataclass
class CollapseReshapeChain(OnnxGraphEdit):
    """
    Collapse consecutive Reshape ops into a single Reshape.

    Matches a Reshape node whose only consumer is another Reshape,
    and replaces the chain with a single Reshape from the first input
    to the last output shape.
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "Reshape" or not node.inputs or not node.outputs:
            return False
        out = node.outputs[0]
        if len(out.outputs) != 1:
            return False
        consumer: gs.Node = out.outputs[0]
        return consumer.op == "Reshape"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Reshape")
        data_inp = node.inputs[0]

        # walk forward through all consecutive Reshapes
        current = node
        collapsed: list[str] = [node.name]
        while True:
            out = current.outputs[0]
            if len(out.outputs) != 1 or out.outputs[0].op != "Reshape":
                break
            next_node: gs.Node = out.outputs[0]
            current.inputs.clear()
            current.outputs.clear()
            collapsed.append(next_node.name)
            current = next_node

        # wire original data input into the final Reshape
        current.inputs[0] = data_inp
        self._logger.debug(
            "Collapsed %d Reshapes into '%s'", len(collapsed), current.name
        )


@dataclass
class CollapseGQABroadcast(OnnxGraphEdit):
    """
    Collapse Unsqueeze → Expand → Reshape GQA head broadcast into a single Expand.

    In GQA attention, KV tensors ``[B, H_kv, S, D]`` are broadcast to match Q heads
    via ``Unsqueeze(axis) → Expand → Reshape → [B, H_q, S, D]``.  When ``H_kv == 1``
    the chain is equivalent to a single ``Expand`` because the head dim can be
    broadcast directly.
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "Unsqueeze" or not node.inputs or not node.outputs:
            return False
        unsqueeze_out = node.outputs[0]
        if len(unsqueeze_out.outputs) != 1:
            return False
        expand_node: gs.Node = unsqueeze_out.outputs[0]
        if expand_node.op != "Expand" or not expand_node.outputs:
            return False
        expand_out = expand_node.outputs[0]
        if len(expand_out.outputs) != 1:
            return False
        reshape_node: gs.Node = expand_out.outputs[0]
        if reshape_node.op != "Reshape" or not reshape_node.outputs:
            return False

        inp_shape = getattr(node.inputs[0], "shape", None)
        final_shape = getattr(reshape_node.outputs[0], "shape", None)
        if inp_shape is None or final_shape is None:
            return False
        if not all(isinstance(d, (int, np.integer)) for d in inp_shape):
            return False
        if not all(isinstance(d, (int, np.integer)) for d in final_shape):
            return False

        inp_shape = [int(d) for d in inp_shape]
        final_shape = [int(d) for d in final_shape]

        if len(inp_shape) != len(final_shape):
            return False

        axes = None
        if len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
            axes = node.inputs[1].values.flatten().tolist()
        elif "axes" in node.attrs:
            axes = list(node.attrs["axes"])
        if axes is None or len(axes) != 1:
            return False
        axis = int(axes[0])
        ndim_after = len(inp_shape) + 1
        if axis < 0:
            axis = ndim_after + axis

        # Collapsible when the dim adjacent to the insertion point is 1,
        # meaning a direct Expand on the input can replace the full chain.
        return 0 < axis <= len(inp_shape) and inp_shape[axis - 1] == 1

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Unsqueeze")
        expand_node: gs.Node = node.outputs[0].outputs[0]
        reshape_node: gs.Node = expand_node.outputs[0].outputs[0]

        inp = node.inputs[0]
        final_out = reshape_node.outputs[0]
        final_shape = [int(d) for d in final_out.shape]
        consumers: list[gs.Node] = list(final_out.outputs)

        expand_shape_const = gs.Constant(
            name=node.name + "_gqa_expand_shape",
            values=np.array(final_shape, dtype=np.int64)
        )
        new_expand_out: gs.Variable = self.graph.layer(
            name=node.name + "_gqa_expand",
            op="Expand",
            inputs=[inp, expand_shape_const],
            outputs=[gs.Variable(
                name=final_out.name + "_expanded",
                dtype=final_out.dtype,
                shape=final_shape
            )]
        )[0]
        rewire_consumers(consumers, final_out, new_expand_out)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is final_out:
                self.graph.outputs[i] = new_expand_out

        node.inputs.clear()
        node.outputs.clear()
        expand_node.inputs.clear()
        expand_node.outputs.clear()
        reshape_node.inputs.clear()
        reshape_node.outputs.clear()
        self._logger.debug(
            "Collapsed GQA broadcast at '%s' into single Expand -> %s",
            node.name, final_shape
        )


@dataclass
class ReplaceConstantDivWithMul(OnnxGraphEdit):
    """
    Replaces x/C with x * C' where C' = 1/C is a newly computed constant.

    Args:
        export_dtype (onnx.TensorProto.DataType): ONNX export data type for tensors

    Raises:
        TypeError: If divisor is not a constant tensor
    """
    export_dtype: onnx.TensorProto.DataType

    def match(self, node: gs.Node) -> bool:
        if node.op == "Div" and len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
            return True
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Div")
        if not len(node.inputs) > 1 or not isinstance(node.inputs[1], gs.Constant):
            raise TypeError("Expected second operand of Div to be a `gs.Constant`")

        # x/C -> x * C' where C' = 1/C
        divisor: gs.Constant = node.inputs[1]
        if not (reciprocal := self.graph.tensors().get(divisor.name + "_reciprocal")):
            reciprocal = gs.Constant(
                name=divisor.name + "_reciprocal",
                values=np.array(np.float32(1.0) / divisor.values.astype(np.float32)),
                export_dtype=self.export_dtype,
            )
        node.op = "Mul"
        node.inputs[1] = reciprocal

        self._logger.debug("Replaced Div @ '%s' by constant '%s' with Mul by reciprocal", node.name, divisor.name)


class ConstantBroadcastPolicy(Enum):
    """
    Strategy for handling broadcastable constants during graph edits.

    - `DEFER_RUNTIME`: Insert `Expand` nodes so constants broadcast at runtime (lower memory, slower inference).
    - `MATERIALIZE`: Pre-broadcast constants and store the expanded tensor (faster inference, higher memory).
    - `SKIP`: Leave constants untouched and let downstream tools handle broadcasting.
    """
    DEFER_RUNTIME = auto()
    MATERIALIZE = auto()
    SKIP = auto()

@dataclass
class BroadcastOpInputs(OnnxGraphEdit):
    """
    Add explicit `Expand` nodes for broadcasting op inputs to output shape.

    Args:
        ops (list[str]): Ops to apply explicit input broadcasting, will apply to all ops if list is empty.
        out_idx (int): Index of output to use as broadcast target shape (default: 0).
        inp_idx (list[int]): Only broadcast inputs at these indices (default: None, broadcast all inputs).
        constants_policy (ConstantBroadcastPolicy): How to treat constant inputs (default: skip).
    """

    ops: list[str]
    out_idx: int = 0
    inp_idx: list[int] | None = None
    constants_policy: ConstantBroadcastPolicy = ConstantBroadcastPolicy.SKIP

    def __post_init__(self):
        self.inp_idx = self.inp_idx or []
        return super().__post_init__()

    @staticmethod
    def _has_valid_shape(tensor: gs.Constant | gs.Variable) -> bool:
        try:
            shape = getattr(tensor, "shape", None)
            return shape is not None and all(isinstance(d, (int, np.integer)) for d in shape)
        except TypeError:
            raise ValueError(f"{tensor.name}, {tensor.shape}")

    @staticmethod
    def _unique_tensor_id(tensor: gs.Constant | gs.Variable, hash_length: int = 8) -> str:
        inputs = [getattr(n, "name", str(n)) for n in tensor.inputs]
        outputs = [getattr(n, "name", str(n)) for n in tensor.outputs]
        id_str = tensor.name + ":" + "|".join(inputs) + ">>" + "|".join(outputs)
        return hashlib.sha256(id_str.encode()).hexdigest()[:hash_length]

    def _add_broadcast_to_tensor(self, tensor: gs.Constant | gs.Variable, bcast_shape: list[int]):
        # create copy of initial consumers to prevent cycle later
        consumers: list[gs.Node] = tensor.outputs.copy()
        bcast_shape_const: gs.Constant = gs.Constant(
            name=tensor.name + "_bcast_shape",
            values=np.array(bcast_shape).astype(np.int64)
        )
        bcast_out: gs.Variable = self.graph.layer(
            name=tensor.name + "_bcast",
            op="Expand",
            inputs=[tensor, bcast_shape_const],
            outputs=[gs.Variable(name=tensor.name + "_expanded", dtype=tensor.dtype, shape=bcast_shape)]
        )[0]
        rewire_consumers(consumers, tensor, bcast_out)

    def match(self, node: gs.Node) -> bool:
        if self.ops and node.op not in self.ops:
            return False
        if not node.inputs or not node.outputs:
            return False

        if not (0 <= self.out_idx < len(node.outputs)):
            self._logger.warning(
                "Received invalid output index; valid: %s, received: %s",
                list(range(len(node.outputs))), self.out_idx
            )
            return False
        if not self._has_valid_shape(node.outputs[self.out_idx]):
            return False

        target_inp_idxs = self.inp_idx or list(range(len(node.inputs)))
        if any(i < 0 or i >= len(node.inputs) for i in target_inp_idxs):
            self._logger.warning(
                "Received invalid input indices; valid: %s, received: %s",
                list(range(len(node.inputs))), self.inp_idx
            )
            return False
        return all(self._has_valid_shape(node.inputs[i]) for i in target_inp_idxs)

    def transform(self, node: gs.Node):
        target_out: gs.Variable = node.outputs[self.out_idx]
        assert isinstance(target_out, gs.Variable), "Node output must be `gs.Variable`"
        if not self._has_valid_shape(target_out):
            raise ValueError(
                "Missing valid integer shape info for output '%s' (node: %s, '%s')",
                target_out.name, node.op, node.name
            )
        bcast_shape: list[int] = list(target_out.shape)
        target_inp_idxs = self.inp_idx or list(range(len(node.inputs)))
        bcast_done: set[str] = set()

        for i in target_inp_idxs:
            inp = node.inputs[i]
            if self._unique_tensor_id(inp) in bcast_done:
                continue

            if not self._has_valid_shape(inp):
                self._logger.warning(
                    "Broadcasting input '%s' with no valid integer shape info (node: %s, '%s')",
                    inp.name, node.op, node.name
                )
            
            if list(inp.shape) == bcast_shape:
                continue
            
            if isinstance(inp, gs.Variable):
                self._add_broadcast_to_tensor(inp, bcast_shape)
            elif isinstance(inp, gs.Constant):
                if getattr(inp, "dtype", None) is None:
                    self._logger.warning(
                        "Skipping broadcast of initializer '%s' due to missing dtype info",
                        inp.name
                    )
                    continue
                if self.constants_policy == ConstantBroadcastPolicy.SKIP:
                    continue
                if self.constants_policy == ConstantBroadcastPolicy.DEFER_RUNTIME:
                    self._add_broadcast_to_tensor(inp, bcast_shape)
                elif self.constants_policy == ConstantBroadcastPolicy.MATERIALIZE:
                    export_dtype = inp.export_dtype
                    if inp.dtype == onnx.TensorProto.BFLOAT16:
                        dtype = np.float32
                        export_dtype = onnx.TensorProto.BFLOAT16
                    else:
                        dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.dtype) \
                            if isinstance(inp.dtype, int) else inp.dtype
                    bcast_values = np.broadcast_to(inp.values, bcast_shape).astype(dtype)
                    bcast_const = gs.Constant(
                        name=inp.name + "_bcast",
                        values=bcast_values,
                        export_dtype=export_dtype
                    )
                    bcast_const.outputs = inp.outputs
                    inp.outputs.clear()
                else:
                    raise ValueError(f"Invalid constant broadcast policy '{self.constants_policy}'")
            else:
                raise ValueError(f"Invalid input tensor type '{type(inp)}'")
            
            bcast_done.add(self._unique_tensor_id(inp))
            self._logger.debug(
                "Broadcasted input '%s' of %s node '%s' to %s",
                inp.name, node.op, node.name, bcast_shape
            )


@dataclass
class ExtractConstantLUT(OnnxGraphEdit):

    lut_shape: tuple[int, ...]
    save_to: os.PathLike | str
    inp_name: str | None = None

    def match(self, node: gs.Node) -> bool:
        if node.op != "Gather" or len(node.inputs) < 2:
            return False
        if node.attrs.get("axis", None) != 0:
            return False
        lut = node.inputs[0]
        if not isinstance(lut, gs.Constant):
            return False
        lut_shape = lut.values.shape
        if lut_shape == self.lut_shape:
            return True
        return False

    def transform(self, node: gs.Node):
        if not (node.op == "Gather" and len(node.inputs) >= 2 and isinstance((lut := node.inputs[0]), gs.Constant)):
            raise ValueError(f"Gather node '{node.name}' does not have a constant data input")
        if (axis := node.attrs.get("axis", None)) != 0:
            raise ValueError(f"Only support axis = 0 for LUT, found axis = {axis} for Gather node '{node.name}'")
        
        lut_data = lut.values
        if not isinstance(lut_data, np.ndarray):
            self._logger.warning("Constant data is not NumPy array, attempting to load lazy values")
            try:
                lut_data = lut_data.load()
            except AttributeError as e:
                raise ValueError(f"Constant data for {node.name} is not loadable") from e
            if not isinstance(lut_data, np.ndarray):
                raise ValueError(f"Invalid Constant data type: {type(lut_data)}")
        
        self.save_to = Path(self.save_to)
        self.save_to.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.save_to, lut_data)

        if not self.inp_name:
            self.inp_name = f"extracted_lut_{normalize_layer_name(node.name)}_input"
        lut_out: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(lut_out.outputs)        
        lut_entry_inp = gs.Variable(
            name=self.inp_name,
            dtype=lut_out.dtype,
            shape=lut_out.shape
        )
        rewire_consumers(consumers, lut_out, lut_entry_inp)
        self.graph.inputs.append(lut_entry_inp)
        node.outputs.clear()
        self._logger.debug(
            "Extracted LUT from '%s', consumers redirected to graph input '%s'",
            node.name, self.inp_name
        )


@dataclass
class ExtractQuantizedEmbedding(OnnxGraphEdit):
    """
    Extract token embeddings from GatherBlockQuantized (int4 quantized embedding).

    Dequantizes the int4 packed weights to fp32, saves as .npy, and replaces
    the GatherBlockQuantized node output with a graph input (like ExtractConstantLUT).
    """

    hidden_size: int
    vocab_size: int
    save_to: os.PathLike | str
    inp_name: str = "token_embedding"

    def match(self, node: gs.Node) -> bool:
        if node.op != "GatherBlockQuantized":
            return False
        # Must gather from input_ids (graph input)
        if len(node.inputs) < 2:
            return False
        idx_inp = node.inputs[1]
        return any(idx_inp is gi for gi in self.graph.inputs)

    def transform(self, node: gs.Node):
        weight_q = node.inputs[0]   # uint8 packed int4: (vocab, K_packed)
        scales = node.inputs[2]     # float32: (vocab, n_blocks)
        zp = node.inputs[3] if len(node.inputs) > 3 else None

        bits = node.attrs.get("bits", 4)
        block_size = node.attrs.get("block_size", 32)

        w_data = np.asarray(weight_q.values, dtype=np.uint8)
        s_data = np.asarray(scales.values, dtype=np.float32)

        # Unpack uint8 → two int4 values per byte (low nibble first)
        low = (w_data & 0x0F).astype(np.float32)
        high = ((w_data >> 4) & 0x0F).astype(np.float32)
        unpacked = np.stack([low, high], axis=-1).reshape(w_data.shape[0], -1)
        # Trim to hidden_size (padding may exist)
        unpacked = unpacked[:, :self.hidden_size]

        # Unpack zero points
        if zp is not None and zp.values is not None and zp.values.size > 0:
            zp_data = np.asarray(zp.values, dtype=np.uint8)
            zp_low = (zp_data & 0x0F).astype(np.float32)
            zp_high = ((zp_data >> 4) & 0x0F).astype(np.float32)
            zp_unpacked = np.stack([zp_low, zp_high], axis=-1).reshape(zp_data.shape[0], -1)
        else:
            n_blocks_per_row = s_data.shape[1]
            zp_unpacked = np.full((w_data.shape[0], n_blocks_per_row), 8.0, dtype=np.float32)

        # Dequantize: for each block of block_size elements along hidden dim
        n_blocks = s_data.shape[1]
        result = np.empty((self.vocab_size, self.hidden_size), dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, self.hidden_size)
            scale_col = s_data[:self.vocab_size, b:b+1]          # (V, 1)
            zp_col = zp_unpacked[:self.vocab_size, b:b+1]        # (V, 1)
            result[:, start:end] = (unpacked[:self.vocab_size, start:end] - zp_col) * scale_col

        # Save
        save_path = Path(self.save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result)

        # Replace node output with graph input
        out_var: gs.Variable = node.outputs[0]
        consumers = list(out_var.outputs)
        out_dtype = out_var.dtype or np.float32
        # Use explicit shape: embedding output is (B, S, hidden_size)
        out_shape = out_var.shape if out_var.shape else [1, 1, self.hidden_size]
        new_inp = gs.Variable(
            name=self.inp_name,
            dtype=out_dtype,
            shape=out_shape,
        )
        rewire_consumers(consumers, out_var, new_inp)
        self.graph.inputs.append(new_inp)
        node.outputs.clear()
        self._logger.debug(
            "Extracted quantized embedding from '%s' (%d x %d), saved to '%s'",
            node.name, self.vocab_size, self.hidden_size, self.save_to
        )


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


def _dequantize_matmulnbits_weights(
    W_q: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None,
    K: int,
    N: int,
    bits: int,
    block_size: int,
) -> np.ndarray:
    """
    Dequantize MatMulNBits int4 packed weights to fp32.

    MatMulNBits weight layout:
        W_q: (N, n_blocks, blob_size) uint8 — each byte packs two 4-bit values
        scales: (N, n_blocks) float32
        zero_points: (N, n_blocks // 2) uint8 — each byte packs two 4-bit zp values (or None)

    Returns:
        fp32 weight of shape (K, N)
    """
    n_blocks = (K + block_size - 1) // block_size

    # Unpack uint8 -> two int4 values per byte (little-endian nibble order)
    low = (W_q & 0x0F).astype(np.int8)
    high = ((W_q >> 4) & 0x0F).astype(np.int8)
    # Interleave: for each byte, low nibble comes first
    unpacked = np.stack([low, high], axis=-1).reshape(N, n_blocks, block_size)

    # Ensure scales are 2D (N, n_blocks) — may arrive as flat (N * n_blocks,)
    scales = np.asarray(scales, dtype=np.float32).reshape(N, n_blocks)

    # Unpack zero points
    if zero_points is not None and zero_points.size > 0:
        zp_low = (zero_points & 0x0F).astype(np.int8)
        zp_high = ((zero_points >> 4) & 0x0F).astype(np.int8)
        zp_unpacked = np.stack([zp_low, zp_high], axis=-1).reshape(N, n_blocks)
    else:
        zp_unpacked = np.zeros((N, n_blocks), dtype=np.int8)

    # Dequantize: float_val = (int_val - zero_point) * scale
    W_float = (unpacked.astype(np.float32) - zp_unpacked[:, :, np.newaxis].astype(np.float32)) * scales[:, :, np.newaxis]

    # Reshape to (N, K) and transpose to (K, N) to match standard MatMul convention
    W_float = W_float.reshape(N, -1)[:, :K]
    return W_float.T.astype(np.float32)


@dataclass
class ReplaceMatMulNBits(OnnxGraphEdit):
    """
    Replace MatMulNBits (ORT 4-bit quantized matmul) with DequantizeLinear + Reshape + MatMul.

    Dequantizes weights at graph-edit time to fp32 and creates a standard MatMul node.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "MatMulNBits"

    def transform(self, node: gs.Node):
        activation = node.inputs[0]
        W_q_const: gs.Constant = node.inputs[1]
        scales_const: gs.Constant = node.inputs[2]
        zp_const: gs.Constant = node.inputs[3] if len(node.inputs) > 3 else None

        K = node.attrs["K"]
        N = node.attrs["N"]
        bits = node.attrs.get("bits", 4)
        block_size = node.attrs.get("block_size", 32)
        out_var = node.outputs[0]
        prefix = node.name.replace("_Quant", "")

        W_float = _dequantize_matmulnbits_weights(
            W_q_const.values,
            scales_const.values,
            zp_const.values if zp_const is not None else None,
            K, N, bits, block_size,
        )

        weight_const = gs.Constant(
            f"{prefix}/weight_dequantized",
            W_float,
        )

        matmul_out = self.graph.layer(
            name=f"{prefix}/MatMul", op="MatMul",
            inputs=[activation, weight_const],
            outputs=[gs.Variable(f"{prefix}/MatMul_output_0", dtype=out_var.dtype, shape=out_var.shape)],
        )[0]

        rewire_consumers(out_var.outputs.copy(), out_var, matmul_out)

        # Also update graph outputs if this node's output was a graph output
        for i, go in enumerate(self.graph.outputs):
            if go is out_var:
                self.graph.outputs[i] = matmul_out

        node.outputs.clear()
        self._logger.debug(
            "Replaced MatMulNBits '%s' (K=%d, N=%d) with dequantized MatMul",
            node.name, K, N
        )


@dataclass
class ReplaceMatMulNBitsLinear(OnnxGraphEdit):
    """
    Replace MatMulNBits with DequantizeLinear + Transpose + MatMul.

    Keeps weights in packed UINT4 form and inserts a standard
    DequantizeLinear node (opset 21, block_size) to dequantize at runtime.

    Flow per node:
        DequantizeLinear(x_uint4(N,K), scale(N,n_blocks), zp_uint4(N,n_blocks),
                         axis=1, block_size) → Transpose(1,0) → MatMul

    After gs.export_onnx(), call ``pack_dq_weights_uint4(model)`` to convert the
    uint8 placeholder tensors to proper ONNX UINT4 packed format, halving storage.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "MatMulNBits"

    def transform(self, node: gs.Node):
        activation = node.inputs[0]
        W_q_const: gs.Constant = node.inputs[1]
        scales_const: gs.Constant = node.inputs[2]
        zp_const: gs.Constant = node.inputs[3] if len(node.inputs) > 3 else None

        K = node.attrs["K"]
        N = node.attrs["N"]
        bits = node.attrs.get("bits", 4)
        block_size = node.attrs.get("block_size", 32)
        out_var = node.outputs[0]
        prefix = node.name.replace("_Quant", "").replace("_Q4", "")

        n_blocks = (K + block_size - 1) // block_size

        # Unpack int4 packed weights: (N, n_blocks, blob_size) uint8 → (N, K) uint8
        # Each uint8 in W_q holds 2 int4 values (low nibble first).
        # After export, pack_dq_weights_uint4() converts these back to UINT4.
        W_q = W_q_const.values
        low = (W_q & 0x0F).astype(np.uint8)
        high = ((W_q >> 4) & 0x0F).astype(np.uint8)
        x_data = np.stack([low, high], axis=-1).reshape(N, K)

        # Scales: (N*n_blocks,) → (N, n_blocks) for block_size DQ
        scales_2d = np.asarray(scales_const.values, dtype=np.float32).reshape(N, n_blocks)

        # Zero points: MatMulNBits uses unsigned int4 with default zero_point=8
        if zp_const is not None and zp_const.values is not None and zp_const.values.size > 0:
            zp_raw = zp_const.values
            zp_low = (zp_raw & 0x0F).astype(np.uint8)
            zp_high = ((zp_raw >> 4) & 0x0F).astype(np.uint8)
            zp_2d = np.stack([zp_low, zp_high], axis=-1).reshape(N, n_blocks)
        else:
            zp_2d = np.full((N, n_blocks), 8, dtype=np.uint8)

        # Pre-transpose weights so DequantizeLinear outputs (K, N) directly,
        # eliminating the need for a separate Transpose node.
        # Layout: x(K, N), scale(n_blocks, N), zp(n_blocks, N), axis=0 (blocks along K)
        x_data_t = np.ascontiguousarray(x_data.T)       # (N, K) → (K, N)
        scales_t = np.ascontiguousarray(scales_2d.T)     # (N, n_blocks) → (n_blocks, N)
        zp_t = np.ascontiguousarray(zp_2d.T)             # (N, n_blocks) → (n_blocks, N)

        # DequantizeLinear: x(K, N) uint8→UINT4, scale(n_blocks, N), zp(n_blocks, N)
        # Naming convention: /dq_weights and /dq_zero_points suffix triggers UINT4 packing
        dq_out = self.graph.layer(
            name=f"{prefix}/DequantizeLinear", op="DequantizeLinear",
            inputs=[
                gs.Constant(f"{prefix}/dq_weights", x_data_t),
                gs.Constant(f"{prefix}/dq_scales", scales_t),
                gs.Constant(f"{prefix}/dq_zero_points", zp_t),
            ],
            outputs=[gs.Variable(f"{prefix}/dq_output", dtype=np.float32)],
            attrs={"axis": 0, "block_size": block_size},
        )[0]

        # MatMul: activation(B, S, K) @ weight(K, N) → (B, S, N)
        matmul_out = self.graph.layer(
            name=f"{prefix}/MatMul", op="MatMul",
            inputs=[activation, dq_out],
            outputs=[gs.Variable(f"{prefix}/MatMul_output_0", dtype=out_var.dtype, shape=out_var.shape)],
        )[0]

        rewire_consumers(out_var.outputs.copy(), out_var, matmul_out)

        for i, go in enumerate(self.graph.outputs):
            if go is out_var:
                self.graph.outputs[i] = matmul_out

        node.outputs.clear()
        self._logger.debug(
            "Replaced MatMulNBits '%s' (K=%d, N=%d, block_size=%d) with DequantizeLinear+MatMul",
            node.name, K, N, block_size
        )


def pack_dq_weights_uint4(model):
    """Convert DequantizeLinear weight/zp initializers from uint8 to ONNX UINT4.

    After ``gs.export_onnx()``, DQ weight and zero-point tensors are stored as
    uint8 (one value per byte).  This function repacks them into the ONNX UINT4
    type (two values per byte), halving storage.

    Tensors are identified by the naming convention established in
    ``ReplaceMatMulNBitsLinear``: names ending with ``/dq_weights`` or
    ``/dq_zero_points``.
    """
    from onnx import TensorProto, numpy_helper

    for init in model.graph.initializer:
        if not (init.name.endswith("/dq_weights") or init.name.endswith("/dq_zero_points")):
            continue
        if init.data_type != TensorProto.UINT8:
            continue
        data = numpy_helper.to_array(init).flatten()
        # Pad to even length if needed
        if len(data) % 2 != 0:
            data = np.append(data, np.uint8(0))
        packed = (data[0::2] & 0x0F) | ((data[1::2] & 0x0F) << 4)
        init.data_type = TensorProto.UINT4
        init.raw_data = packed.tobytes()
        # Clear float/int data fields
        del init.float_data[:]
        del init.int32_data[:]
        del init.int64_data[:]

    return model


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


class CombineKVCacheMixin:
    """
    Mixin for combining separate key/value cache I/O tensors along the heads axis.

    Pairs key+value tensors per layer and merges them into single tensors
    with doubled head dimension: [..., H, L, D] -> [..., 2*H, L, D].

    Must be used with OnnxGraphEditor (defines self._graph, self._logger).
    """

    def combine_kv_io_tensors(
        self,
        kv_tensor_shape: list[int],
        *,
        input_prefix: str = "past_key_values",
        output_prefix: str = "present",
        kv_layer_re: str | re.Pattern = r"\.(\d+)\.(key|value)$",
        combined_name_fmt: str = "{prefix}.{layer}"
    ):
        if isinstance(kv_layer_re, str):
            kv_layer_re = re.compile(kv_layer_re)
        # concatenate along H axis: [..., H, L, D] <-> [..., 2*H, L, D]
        _H_DIM_AXIS = len(kv_tensor_shape) - 3

        def _get_kv_pairs(
            io_coll: list[gs.Variable], prefix: str
        ) -> list[tuple[int, gs.Variable, gs.Variable]]:
            io_dict: dict[int, dict[str, gs.Variable]] = defaultdict(dict)
            for io in io_coll:
                if not isinstance(io, gs.Variable):
                    raise TypeError(f"Expected gs.Variable, got {type(io)}")
                if io.shape is None or list(io.shape) != kv_tensor_shape:
                    continue
                if not io.name.startswith(prefix):
                    continue
                m = kv_layer_re.search(io.name)
                if m is None:
                    continue
                layer, role = int(m.group(1)), m.group(2)
                if role in io_dict[layer]:
                    raise ValueError(
                        f"Duplicate {role} tensor for layer {layer}: "
                        f"'{io_dict[layer][role].name}' and '{io.name}'"
                    )
                io_dict[layer][role] = io
            kv_pairs: list[tuple[int, gs.Variable, gs.Variable]] = []
            for layer in sorted(io_dict):
                entry = io_dict[layer]
                if "key" not in entry or "value" not in entry:
                    raise ValueError(
                        f"Layer {layer} is missing "
                        f"{'key' if 'key' not in entry else 'value'} tensor"
                    )
                kv_pairs.append((layer, entry["key"], entry["value"]))
            return kv_pairs

        def _remove_io(tensor: gs.Variable, io_coll: list[gs.Variable]) -> int:
            for idx, io_tensor in enumerate(io_coll):
                if tensor is io_tensor:
                    io_coll.pop(idx)
                    return idx
            return -1

        def _concatenate_kv_input(
            layer: int,
            key_input: gs.Variable,
            value_input: gs.Variable,
            prefix: str,
        ):
            assert key_input.dtype == value_input.dtype
            assert list(key_input.shape) == kv_tensor_shape
            assert list(value_input.shape) == kv_tensor_shape
            key_consumers: list[gs.Node] = key_input.outputs
            value_consumers: list[gs.Node] = value_input.outputs

            base = combined_name_fmt.format(prefix=prefix, layer=layer)
            n_kv_heads = kv_tensor_shape[_H_DIM_AXIS]
            combined_shape = kv_tensor_shape.copy()
            combined_shape[_H_DIM_AXIS] *= 2
            combined_input = gs.Variable(
                name=f"{base}.key_value",
                dtype=key_input.dtype,
                shape=combined_shape
            )
            if not (kv_concat_axis := self._graph.tensors().get("kv_concat_axis")):
                kv_concat_axis = gs.Constant(
                    "kv_concat_axis",
                    np.array([_H_DIM_AXIS], dtype=np.int64),
                )
            if not (kv_inp_key_starts := self._graph.tensors().get(
                "kv_inp_key_starts"
            )):
                kv_inp_key_starts = gs.Constant(
                    "kv_inp_key_starts", np.array([0], dtype=np.int64)
                )
            if not (kv_inp_key_ends := self._graph.tensors().get(
                "kv_inp_key_ends"
            )):
                kv_inp_key_ends = gs.Constant(
                    "kv_inp_key_ends",
                    np.array([n_kv_heads], dtype=np.int64),
                )
            if not (kv_inp_value_starts := self._graph.tensors().get(
                "kv_inp_value_starts"
            )):
                kv_inp_value_starts = gs.Constant(
                    "kv_inp_value_starts",
                    np.array([n_kv_heads], dtype=np.int64),
                )
            if not (kv_inp_value_ends := self._graph.tensors().get(
                "kv_inp_value_ends"
            )):
                kv_inp_value_ends = gs.Constant(
                    "kv_inp_value_ends",
                    np.array([2 * n_kv_heads], dtype=np.int64),
                )
            key_slice: gs.Variable = self._graph.layer(
                name=f"{base}.key_slice",
                op="Slice",
                inputs=[
                    combined_input,
                    kv_inp_key_starts,
                    kv_inp_key_ends,
                    kv_concat_axis,
                ],
                outputs=[
                    gs.Variable(
                        name=f"{base}.key_from_combined",
                        dtype=key_input.dtype,
                        shape=key_input.shape,
                    )
                ],
            )[0]
            value_slice: gs.Variable = self._graph.layer(
                name=f"{base}.value_slice",
                op="Slice",
                inputs=[
                    combined_input,
                    kv_inp_value_starts,
                    kv_inp_value_ends,
                    kv_concat_axis,
                ],
                outputs=[
                    gs.Variable(
                        name=f"{base}.value_from_combined",
                        dtype=value_input.dtype,
                        shape=value_input.shape,
                    )
                ],
            )[0]

            rewire_consumers(key_consumers, key_input, key_slice)
            rewire_consumers(value_consumers, value_input, value_slice)
            key_input.outputs.clear()
            value_input.outputs.clear()
            insert_pos = _remove_io(key_input, self._graph.inputs)
            _remove_io(value_input, self._graph.inputs)
            if insert_pos >= 0:
                self._graph.inputs.insert(insert_pos, combined_input)
            else:
                self._graph.inputs.append(combined_input)

            self._logger.debug(
                "Combined KV input layer %d: '%s' + '%s' -> '%s'",
                layer,
                key_input.name,
                value_input.name,
                combined_input.name,
            )

        def _concatenate_kv_output(
            layer: int,
            key_output: gs.Variable,
            value_output: gs.Variable,
            prefix: str,
        ):
            assert key_output.dtype == value_output.dtype
            assert list(key_output.shape) == kv_tensor_shape
            assert list(value_output.shape) == kv_tensor_shape

            base = combined_name_fmt.format(prefix=prefix, layer=layer)
            combined_shape = kv_tensor_shape.copy()
            combined_shape[_H_DIM_AXIS] *= 2
            combined_tensor: gs.Variable = self._graph.layer(
                name=f"{base}_kv_concat",
                op="Concat",
                inputs=[key_output, value_output],
                outputs=[
                    gs.Variable(
                        name=f"{base}.key_value",
                        dtype=key_output.dtype,
                        shape=combined_shape,
                    )
                ],
                attrs={"axis": _H_DIM_AXIS},
            )[0]
            insert_pos = _remove_io(key_output, self._graph.outputs)
            _remove_io(value_output, self._graph.outputs)
            if insert_pos >= 0:
                self._graph.outputs.insert(insert_pos, combined_tensor)
            else:
                self._graph.outputs.append(combined_tensor)

            self._logger.debug(
                "Combined KV output layer %d: '%s' + '%s' -> '%s'",
                layer,
                key_output.name,
                value_output.name,
                combined_tensor.name,
            )

        input_pairs = _get_kv_pairs(self._graph.inputs, input_prefix)
        output_pairs = _get_kv_pairs(self._graph.outputs, output_prefix)
        self._logger.debug(
            "Combining KV tensors: %d input pairs, %d output pairs (axis=%d)",
            len(input_pairs),
            len(output_pairs),
            _H_DIM_AXIS,
        )
        for kv_info in input_pairs:
            _concatenate_kv_input(*kv_info, input_prefix)
        for kv_info in output_pairs:
            _concatenate_kv_output(*kv_info, output_prefix)
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()
        return self


class CommonGraphEditsMixin:
    """
    Mixin providing convenience methods for common graph edits.
    
    Must be used with OnnxGraphEditor (defines self._graph, self._graph_name, 
    self._export_dtype, self.apply_edit).
    """

    def replace_dynamic_kv_cache(self, cur_len, max_tokens):
        self.apply_edit(ReplaceDynamicKVCache(self._graph, self._graph_name, cur_len, max_tokens))
        return self

    def mask_future_attn_scores(self, cur_len, max_tokens):
        self.apply_edit(MaskFutureAttentionScores(self._graph, self._graph_name, cur_len, max_tokens, self._export_dtype))
        return self

    def add_curr_len_input(self, cur_len):
        self.apply_edit(AddCurrLenInput(self._graph, self._graph_name, cur_len))
        return self

    def convert_to_static_index(self):
        self.apply_edit(ConvertToStaticIndex(self._graph, self._graph_name))
        return self

    def dequantize_projections_matmul(self, hidden_size, vocab_size):
        self.apply_edit(DequantizeProjectionsMatMul(self._graph, self._graph_name, hidden_size, vocab_size, self._export_dtype))
        return self

    def remove_isNaN(self):
        self.apply_edit(RemoveIsNaN(self._graph, self._graph_name))
        return self

    def remove_redundant_casts(
        self
    ):
        self.apply_edit(RemoveRedundantCasts(self._graph, self._graph_name))
        return self

    def fold_scalar_matmul(self):
        self.apply_edit(FoldScalarMatMul(self._graph, self._graph_name))
        return self
    
    def replace_constant_div_with_mul(self):
        self.apply_edit(ReplaceConstantDivWithMul(self._graph, self._graph_name, self._export_dtype))

    def broadcast_op_inputs(self, ops, output_idx=0, inputs_idx=None, constants_policy=ConstantBroadcastPolicy.SKIP):
        self.apply_edit(BroadcastOpInputs(self._graph, self._graph_name, ops, output_idx, inputs_idx, constants_policy))
        return self

    def extract_token_embeddings(self, hidden_size, vocab_size, save_to, inp_name="token_embedding"):
        self.apply_edit(ExtractConstantLUT(self._graph, self._graph_name, (vocab_size, hidden_size), save_to, inp_name))
        return self

    def extract_quantized_embeddings(self, hidden_size, vocab_size, save_to, inp_name="token_embedding"):
        self.apply_edit(ExtractQuantizedEmbedding(self._graph, self._graph_name, hidden_size, vocab_size, save_to, inp_name))
        return self

    def eliminate_transposes(self):
        self.apply_edit(EliminateTranspose(self._graph, self._graph_name))
        return self

    def collapse_reshape_chains(self):
        self.apply_edit(CollapseReshapeChain(self._graph, self._graph_name))
        return self

    def collapse_gqa_broadcast(self):
        self.apply_edit(CollapseGQABroadcast(self._graph, self._graph_name))
        return self

    def replace_simplified_layer_norm(self):
        self.apply_edit(ReplaceSimplifiedLayerNorm(self._graph, self._graph_name))
        return self

    def replace_skip_simplified_layer_norm(self):
        self.apply_edit(ReplaceSkipSimplifiedLayerNorm(self._graph, self._graph_name))
        return self

    def replace_matmul_nbits(self):
        self.apply_edit(ReplaceMatMulNBits(self._graph, self._graph_name))
        return self

    def replace_matmul_nbits_linear(self):
        self.apply_edit(ReplaceMatMulNBitsLinear(self._graph, self._graph_name))
        return self

    def replace_group_query_attention(self, num_heads, kv_num_heads, head_dim):
        self.apply_edit(ReplaceGroupQueryAttention(self._graph, self._graph_name, num_heads, kv_num_heads, head_dim))
        return self
