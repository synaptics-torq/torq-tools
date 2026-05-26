# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Optional
import hashlib
import os
import re

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from onnx import helper, numpy_helper

from .onnx import (
    OnnxGraphEdit,
    rewire_consumers
)

from ..utils.onnx import normalize_layer_name

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
    "TrimLMHeadVocab",
    "SplitLMHead",
    "EliminateRank0Gather",
    "EliminateSingletonGatherUnsqueeze",
    "RewriteNegativePads",
    "AbsorbPadding",
    "WidenStridedDepthwiseConv",
    "DecomposeBidirectionalRnn",
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
                gs.Variable(node.name + "_where", dtype=node.inputs[0].dtype, shape=mask_shape)
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
                    gs.Variable(node.name + "_biased", dtype=producer_output.dtype, shape=producer_output.shape)
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
class EliminateExpand(OnnxGraphEdit):
    """
    Eliminate Expand ops at inputs for specific ops and let the compiler handle implicit broadcasting.
    """

    ops: list[str]

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _iter_producers(node):
        for tensor in node.inputs or []:
            if tensor is None:
                continue
            for producer in getattr(tensor, "inputs", []) or []:
                if producer is not None:
                    yield producer

    @staticmethod
    def _clear_downstream_shapes(node: gs.Node, graph_output_ids: set[int]):
        """BFS forward from node, clearing .shape on all intermediate variables."""
        queue = [node]
        visited = set()
        while queue:
            n = queue.pop(0)
            if id(n) in visited:
                continue
            visited.add(id(n))
            for out in n.outputs:
                if not isinstance(out, gs.Variable):
                    continue
                if id(out) not in graph_output_ids:
                    out.shape = None
                for consumer in out.outputs:
                    if id(consumer) not in visited:
                        queue.append(consumer)

    def match(self, node: gs.Node) -> bool:
        if node.op not in self.ops:
            return False
        return any(
            producer.op == "Expand"
            for producer in self._iter_producers(node)
        )

    def transform(self, node: gs.Node):
        if node.op not in self.ops:
            raise ValueError(
                f"Expected node op in {self.ops} for Expand elimination, got '{node.op}'"
            )
        graph_output_ids = {id(o) for o in self.graph.outputs}
        for prod in self._iter_producers(node):
            if prod.op != "Expand":
                continue
            inp_tensor = prod.inputs[0]
            expand_out = prod.outputs[0]
            rewire_consumers([node], expand_out, inp_tensor)
            # Clear stale shape metadata on all downstream variables so ONNX
            # shape inference can recompute them from the pre-expand input.
            self._clear_downstream_shapes(node, graph_output_ids)
            if not expand_out.outputs and not any(id(out) == id(expand_out) for out in self.graph.outputs):
                prod.inputs.clear()
                prod.outputs.clear()
            self._logger.debug(
                "Eliminated Expand '%s' feeding %s node '%s'",
                prod.name, node.op, node.name
            )


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
        """Cycle heuristic: True when at most one dim > 1 per perm cycle.

        This is a *necessary* condition for ``Transpose(x, perm)`` to be
        equivalent to a Reshape, but not sufficient (e.g. shape
        ``[1, 32, 16]`` with ``perm=[2, 1, 0]``). Used as a fast pre-filter.
        """
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

    @staticmethod
    def _transpose_equals_reshape(shape: list[int], perm: list[int]) -> bool:
        """True when ``Reshape(x, permuted_shape)`` matches ``Transpose(x, perm)``.

        Hybrid check: first apply a cheap cycle heuristic that rules out
        permutations that cannot be data-preserving, then confirm with an
        exact element-comparison on a synthetic index tensor when feasible.
        For tensors larger than ~1M elements (common in LLMs) materializing
        the probe is too expensive, so the heuristic is trusted.
        """
        if len(shape) != len(perm):
            return False
        out_shape = [shape[p] for p in perm]
        if shape == out_shape:
            return True
        if not EliminateTranspose._is_data_preserving_perm(perm, shape):
            return False
        numel = 1
        for d in shape:
            numel *= int(d)
        if numel > 1_000_000:
            return True
        x = np.arange(numel, dtype=np.int64).reshape(shape)
        return bool(
            np.array_equal(np.transpose(x, perm), x.reshape(out_shape))
        )

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

        return self._transpose_equals_reshape(shape, perm)

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
        if node.attrs.get("axis", 0) != 0:
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
        if (axis := node.attrs.get("axis", 0)) != 0:
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
class TrimLMHeadVocab(OnnxGraphEdit):
    """
    Trim LM head weight matrix to a subset of tokens.

    The weight matrix is sliced from [hidden, vocab] to [hidden, kept_count].
    If include_argmax is True, an ArgMax is appended to output the compact index.
    Otherwise the output is the trimmed logits tensor [1, 1, kept_count].

    The caller is responsible for mapping compact_idx -> original token ID via
    kept_token_ids[compact_idx].

    Args:
        kept_token_ids (np.ndarray): 1-D array of original token IDs to keep, in the
            order they should appear in the trimmed weight (sorted recommended).
        output_name (str): Name of the MatMul output to match (default: "logits").
        save_lut (Path | str | None): If provided, save kept_token_ids to this .npy path.
        include_argmax (bool): If True, append ArgMax to the graph (default: False).
    """

    kept_token_ids: np.ndarray
    output_name: str = "logits"
    save_lut: Path | str | None = None
    include_argmax: bool = False

    def __post_init__(self):
        self.kept_token_ids = np.asarray(self.kept_token_ids, dtype=np.int64)
        if self.kept_token_ids.ndim != 1 or len(self.kept_token_ids) == 0:
            raise ValueError("kept_token_ids must be a non-empty 1-D array")
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul" or not node.outputs:
            return False
        return node.outputs[0].name == self.output_name

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        weight_inp = node.inputs[1]
        if not isinstance(weight_inp, gs.Constant):
            raise ValueError(
                f"Expected constant weight for LM head MatMul, got {type(weight_inp).__name__}"
            )

        W = weight_inp.values
        if W.ndim != 2:
            raise ValueError(f"Expected 2-D weight matrix, got shape {W.shape}")

        hidden_size, vocab_size = W.shape
        if np.any(self.kept_token_ids >= vocab_size) or np.any(self.kept_token_ids < 0):
            raise ValueError(
                f"kept_token_ids contains values outside [0, {vocab_size})"
            )
        kept_count = len(self.kept_token_ids)

        W_trimmed = W[:, self.kept_token_ids]
        trimmed_weight = gs.Constant(
            name=weight_inp.name + "_trimmed",
            values=W_trimmed,
            export_dtype=getattr(weight_inp, "export_dtype", None),
        )

        node.inputs[1] = trimmed_weight

        logits_out = node.outputs[0]
        old_shape = list(logits_out.shape) if logits_out.shape else None
        if old_shape and len(old_shape) >= 1:
            new_logits_shape = old_shape[:-1] + [kept_count]
        else:
            new_logits_shape = [1, 1, kept_count]

        trimmed_logits = gs.Variable(
            name="logits",
            dtype=logits_out.dtype,
            shape=new_logits_shape,
        )
        consumers = list(logits_out.outputs)
        node.outputs[0] = trimmed_logits

        if self.include_argmax:
            final_out = self.graph.layer(
                name="lm_head_argmax",
                op="ArgMax",
                attrs={"axis": -1, "keepdims": 0},
                inputs=[trimmed_logits],
                outputs=[gs.Variable(
                    name="compact_token_idx",
                    dtype=np.int64,
                    shape=new_logits_shape[:-1],
                )],
            )[0]
        else:
            final_out = trimmed_logits

        rewire_consumers(consumers, logits_out, final_out)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is logits_out:
                self.graph.outputs[i] = final_out

        if self.save_lut is not None:
            lut_path = Path(self.save_lut)
            lut_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(lut_path, self.kept_token_ids)
            self._logger.debug("Saved token ID LUT to '%s'", lut_path)

        self._logger.debug(
            "Trimmed LM head vocab: %d -> %d tokens (argmax=%s)",
            vocab_size, kept_count, self.include_argmax,
        )


@dataclass
class SplitLMHead(OnnxGraphEdit):
    """
    Extract the final LM head into a standalone graph.

    The main graph output is replaced with the non-constant MatMul input, named
    ``hidden_states_name``. The extracted LM head graph accepts that tensor as
    its only input, preserves the original LM head output, and is saved to
    ``save_to``.
    """

    save_to: Path | str
    output_name: str = "logits"
    hidden_states_name: str = "last_hidden_states"

    def _find_lm_head_matmul(self) -> gs.Node:
        for node in self.graph.nodes:
            if node.op != "MatMul" or not node.outputs:
                continue
            output = node.outputs[0]
            if output.name != self.output_name:
                continue
            if any(graph_output is output for graph_output in self.graph.outputs):
                return node
        raise ValueError(f"Could not find final LM head MatMul feeding graph output '{self.output_name}'")

    @staticmethod
    def _select_hidden_states(node: gs.Node) -> gs.Variable:
        if len(node.inputs) != 2:
            raise ValueError(f"LM head MatMul '{node.name}' must have 2 inputs, found {len(node.inputs)}")
        if all(isinstance(inp, gs.Constant) for inp in node.inputs):
            raise ValueError(f"LM head MatMul '{node.name}' is invalid because both inputs are constant")
        hidden_states = next(
            (inp for inp in node.inputs if not isinstance(inp, gs.Constant)),
            None
        )
        if not isinstance(hidden_states, gs.Variable):
            raise ValueError(
                f"Expected LM head hidden states to be a graph variable, got {type(hidden_states).__name__}"
            )
        return hidden_states

    def _extract_lm_head_graph(self) -> gs.Graph:
        lm_head = self._find_lm_head_matmul()
        lm_head_logits = lm_head.outputs[0]
        hidden_states = self._select_hidden_states(lm_head)
        lm_head_input = gs.Variable(
            name=self.hidden_states_name,
            dtype=hidden_states.dtype,
            shape=hidden_states.shape,
        )
        for idx, inp in enumerate(lm_head.inputs):
            if inp is hidden_states:
                lm_head.inputs[idx] = lm_head_input
                break
        lm_head_graph = gs.Graph(
            name="main",
            nodes=[lm_head],
            inputs=[lm_head_input],
            outputs=[lm_head_logits],
        )
        return lm_head_graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul" or not node.outputs:
            return False
        output = node.outputs[0]
        if output.name != self.output_name:
            return False
        return any(graph_output is output for graph_output in self.graph.outputs)

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        hidden_states = self._select_hidden_states(node)
        lm_head_graph = self._extract_lm_head_graph()
        lm_head_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(lm_head_graph),
            True, True, True
        )
        save_to = Path(self.save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(lm_head_model, save_to)

        logits = node.outputs[0]
        hidden_states.name = self.hidden_states_name
        for idx, graph_output in enumerate(self.graph.outputs):
            if graph_output is logits:
                self.graph.outputs[idx] = hidden_states

        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug(
            "Split LM head MatMul '%s'; graph output '%s' now exposes '%s'",
            node.name,
            logits.name,
            hidden_states.name,
        )
        self._logger.debug("Saved split LM head to '%s'", save_to)


def _static_int_shape(var) -> list[int] | None:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None
    out: list[int] = []
    for dim in shape:
        try:
            val = int(dim)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        out.append(val)
    return out


def _const_array(t) -> np.ndarray | None:
    if isinstance(t, gs.Constant):
        return np.asarray(t.values)
    producers = getattr(t, "inputs", None) or []
    if len(producers) != 1 or producers[0].op != "Constant":
        return None
    value = producers[0].attrs.get("value")
    if value is None or not hasattr(value, "values"):
        return None
    return np.asarray(value.values)


def _const_int_list(t) -> list[int] | None:
    arr = _const_array(t)
    if arr is None:
        return None
    return [int(v) for v in arr.reshape(-1)]


def _consumers_in_graph(graph: gs.Graph, name: str) -> list[gs.Node]:
    return [
        n for n in graph.nodes
        if any(getattr(inp, "name", None) == name for inp in n.inputs)
    ]


@dataclass
class EliminateRank0Gather(OnnxGraphEdit):
    """
    Rewrite ``Gather(rank-0) -> Unsqueeze(axes=[0])`` chains to a rank-1 Gather.

    IREE's ``FoldCollapseShape{,FullSlice}IntoInterfaceTensorStore`` codegen
    patterns trip a ranks-don't-match verifier when a tensor of rank > 0 is
    collapsed to rank 0 inside a dispatch; making the Gather produce ``[1]``
    directly avoids the rank-0 intermediate. Only fires when every consumer of
    the rank-0 Gather is ``Unsqueeze(axes=[0])``.
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _output_rank(var) -> int | None:
        shape = getattr(var, "shape", None)
        return None if shape is None else len(shape)

    @staticmethod
    def _unsqueeze_axes_is_zero_only(node: gs.Node) -> bool:
        if node.op != "Unsqueeze":
            return False
        if len(node.inputs) >= 2:
            return _const_int_list(node.inputs[1]) == [0]
        axes_attr = node.attrs.get("axes")
        if axes_attr is None:
            return False
        return list(axes_attr) == [0]

    def match(self, node: gs.Node) -> bool:
        if node.op != "Gather" or len(node.inputs) < 2 or len(node.outputs) != 1:
            return False
        data, indices = node.inputs[0], node.inputs[1]
        if (
            self._output_rank(node.outputs[0]) != 0
            or self._output_rank(data) != 1
            or self._output_rank(indices) != 0
        ):
            return False
        consumers = _consumers_in_graph(self.graph, node.outputs[0].name)
        if not consumers:
            return False
        return all(self._unsqueeze_axes_is_zero_only(c) for c in consumers)

    def _make_rank1_indices(self, idx, name_prefix: str):
        if isinstance(idx, gs.Constant):
            arr = np.asarray(idx.values, dtype=np.int64).reshape(1)
            return gs.Constant(name=f"{name_prefix}_idx1d", values=arr)
        shape_const = gs.Constant(
            name=f"{name_prefix}_idx_shape",
            values=np.array([1], dtype=np.int64),
        )
        out = gs.Variable(
            name=f"{name_prefix}_idx1d",
            dtype=getattr(idx, "dtype", np.int64),
            shape=(1,),
        )
        self.graph.nodes.append(
            gs.Node(
                op="Reshape",
                name=f"{name_prefix}_idx_reshape",
                inputs=[idx, shape_const],
                outputs=[out],
            )
        )
        return out

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Gather")
        out = node.outputs[0]
        consumers = _consumers_in_graph(self.graph, out.name)

        base = node.name or "rank0_gather"
        new_idx = self._make_rank1_indices(node.inputs[1], base)
        new_out = gs.Variable(name=f"{out.name}_rank1", dtype=out.dtype, shape=(1,))
        node.inputs[1] = new_idx
        node.outputs = [new_out]

        for unsq in consumers:
            unsq_out = unsq.outputs[0]
            for n in self.graph.nodes:
                n.inputs = [
                    new_out if getattr(i, "name", None) == unsq_out.name else i
                    for i in n.inputs
                ]
            for i, go in enumerate(self.graph.outputs):
                if getattr(go, "name", None) == unsq_out.name:
                    self.graph.outputs[i] = new_out
            unsq.inputs = []
            unsq.outputs = []

        self._logger.debug(
            "rewrote rank-0 Gather %r (consumed by %d Unsqueeze(s)) to rank-1 path",
            node.name, len(consumers),
        )


@dataclass
class EliminateSingletonGatherUnsqueeze(OnnxGraphEdit):
    """
    Fold ``Gather(axis=k, indices=0) -> elementwise_unary -> Unsqueeze(axes=[k])``
    when ``data.shape[k] == 1`` and the final shape equals the original data
    shape.

    Once shapes are static, the squeeze/unsqueeze around an elementwise unary
    is a no-op and can be removed by feeding the unary directly.
    """

    _ELEMENTWISE_UNARY_OPS = frozenset(
        {
            "Abs", "Ceil", "Cos", "Erf", "Exp", "Floor", "Log", "Neg",
            "Relu", "Sigmoid", "Sin", "Sqrt", "Tanh",
        }
    )

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _normalize_axis(axis: int, rank: int) -> int | None:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        return axis

    @staticmethod
    def _is_scalar_zero(t) -> bool:
        arr = _const_array(t)
        if arr is None or arr.shape != ():
            return False
        return int(arr.reshape(())) == 0

    @staticmethod
    def _unsqueeze_axes(node: gs.Node, out_rank: int) -> list[int] | None:
        if node.op != "Unsqueeze":
            return None
        if len(node.inputs) >= 2:
            axes = _const_int_list(node.inputs[1])
        else:
            raw = node.attrs.get("axes")
            axes = [int(v) for v in raw] if raw is not None else None
        if axes is None:
            return None
        normalized: list[int] = []
        for axis in axes:
            if axis < 0:
                axis += out_rank
            if axis < 0 or axis >= out_rank:
                return None
            normalized.append(axis)
        return normalized

    def _resolve_chain(self, gather: gs.Node):
        """Return (data_shape, unary, unsqueeze) or None if pattern doesn't match."""
        if (
            gather.op != "Gather"
            or len(gather.inputs) < 2
            or len(gather.outputs) != 1
        ):
            return None
        data, indices = gather.inputs[0], gather.inputs[1]
        data_shape = _static_int_shape(data)
        gather_shape = _static_int_shape(gather.outputs[0])
        if data_shape is None or gather_shape is None:
            return None

        axis = self._normalize_axis(
            int(gather.attrs.get("axis", 0)), len(data_shape)
        )
        if axis is None or data_shape[axis] != 1 or not self._is_scalar_zero(indices):
            return None
        if gather_shape != data_shape[:axis] + data_shape[axis + 1:]:
            return None

        gather_consumers = _consumers_in_graph(self.graph, gather.outputs[0].name)
        if len(gather_consumers) != 1:
            return None
        unary = gather_consumers[0]
        if (
            unary.op not in self._ELEMENTWISE_UNARY_OPS
            or len(unary.inputs) != 1
            or len(unary.outputs) != 1
        ):
            return None

        unary_consumers = _consumers_in_graph(self.graph, unary.outputs[0].name)
        if len(unary_consumers) != 1:
            return None
        unsqueeze = unary_consumers[0]
        out_shape = (
            _static_int_shape(unsqueeze.outputs[0])
            if len(unsqueeze.outputs) == 1
            else None
        )
        if out_shape != data_shape:
            return None
        if self._unsqueeze_axes(unsqueeze, len(out_shape)) is None:
            return None
        return data_shape, unary, unsqueeze

    def match(self, node: gs.Node) -> bool:
        return self._resolve_chain(node) is not None

    def transform(self, node: gs.Node):
        chain = self._resolve_chain(node)
        if chain is None:
            return
        data_shape, unary, unsqueeze = chain

        unary.inputs[0] = node.inputs[0]
        unary.outputs[0] = unsqueeze.outputs[0]
        node.inputs.clear()
        node.outputs.clear()
        unsqueeze.inputs.clear()
        unsqueeze.outputs.clear()

        self._logger.debug(
            "removed singleton Gather->%s->Unsqueeze shim at %r shape=%s",
            unary.op, node.name, data_shape,
        )


@dataclass
class RewriteNegativePads(OnnxGraphEdit):
    """
    Rewrite ``Pad`` ops with negative (crop) paddings into ``Pad(positive)+Slice``.

    IREE/Torch-MLIR handle ``Slice`` more reliably than ``Pad`` with negative
    pads. Limited to ``mode == constant`` with a constant ``pads`` initializer
    and a fully-static ranked data input.
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _pad_mode(node: gs.Node) -> str:
        raw = node.attrs.get("mode", "constant")
        if isinstance(raw, bytes):
            return raw.decode("ascii")
        return str(raw)

    def match(self, node: gs.Node) -> bool:
        if node.op != "Pad" or len(node.inputs) < 2 or self._pad_mode(node) != "constant":
            return False
        pads_inp = node.inputs[1]
        if not isinstance(pads_inp, gs.Constant):
            return False
        pads_flat = np.asarray(pads_inp.values).astype(np.int64).reshape(-1)
        if pads_flat.size % 2 != 0 or np.all(pads_flat >= 0):
            return False
        rank = pads_flat.size // 2
        shape_in = _static_int_shape(node.inputs[0])
        return shape_in is not None and len(shape_in) == rank

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Pad")
        pads_flat = (
            np.asarray(node.inputs[1].values).astype(np.int64).reshape(-1)
        )
        rank = pads_flat.size // 2
        shape_in = _static_int_shape(node.inputs[0])

        pos_begin = np.maximum(pads_flat[:rank], 0)
        neg_begin = np.minimum(pads_flat[:rank], 0)
        pos_end = np.maximum(pads_flat[rank:], 0)
        neg_end = np.minimum(pads_flat[rank:], 0)

        inter_shape = [
            int(shape_in[i] + pos_begin[i] + pos_end[i]) for i in range(rank)
        ]
        slice_starts = (-neg_begin).astype(np.int64)
        slice_ends = (np.asarray(inter_shape, dtype=np.int64) + neg_end).astype(np.int64)

        if not all(
            0 <= slice_starts[i] <= slice_ends[i] <= inter_shape[i]
            for i in range(rank)
        ):
            self._logger.debug(
                "skip Pad %s: invalid slice range starts=%s ends=%s inter=%s",
                node.name, slice_starts.tolist(), slice_ends.tolist(), inter_shape,
            )
            return

        out = node.outputs[0]
        consumers = list(out.outputs)
        graph_output_idx = [
            i for i, g in enumerate(self.graph.outputs) if g is out
        ]
        out_dtype = out.dtype
        needs_pos_pad = bool(np.any(pos_begin > 0) or np.any(pos_end > 0))

        base = node.name or out.name

        if needs_pos_pad:
            pads_pos_const = gs.Constant(
                name=f"{base}_pos_pads",
                values=np.concatenate([pos_begin, pos_end]).astype(np.int64),
            )
            pad_inputs = [node.inputs[0], pads_pos_const]
            if len(node.inputs) >= 3:
                pad_inputs.append(node.inputs[2])
            mid = gs.Variable(
                name=f"{base}_pos_pad_tmp",
                dtype=out_dtype,
                shape=inter_shape,
            )
            self.graph.nodes.append(
                gs.Node(
                    op="Pad",
                    name=f"{base}_pos_only",
                    inputs=pad_inputs,
                    outputs=[mid],
                    attrs={"mode": "constant"},
                )
            )
            slice_data = mid
        else:
            slice_data = node.inputs[0]

        starts_const = gs.Constant(
            f"{base}_slice_starts", slice_starts
        )
        ends_const = gs.Constant(
            f"{base}_slice_ends", slice_ends
        )
        sliced = gs.Variable(
            name=f"{base}_cropped", dtype=out_dtype, shape=list(out.shape) if out.shape else None,
        )
        self.graph.nodes.append(
            gs.Node(
                op="Slice",
                name=f"{base}_crop_slice",
                inputs=[slice_data, starts_const, ends_const],
                outputs=[sliced],
            )
        )
        rewire_consumers(consumers, out, sliced)
        for i in graph_output_idx:
            self.graph.outputs[i] = sliced
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "rewrote Pad %s -> %sSlice (inter_shape=%s)",
            node.name, "Pad+" if needs_pos_pad else "(no pos Pad) ", inter_shape,
        )


@dataclass
class AbsorbPadding(OnnxGraphEdit):
    """
    Fuse non-negative ``Pad -> Conv`` chains into a single ``Conv`` with merged pads.

    Pads with negative paddings are out of scope: run :class:`RewriteNegativePads`
    first to convert them to ``Pad+Slice``. Convs that set ``auto_pad`` are
    skipped (per ONNX spec ``auto_pad`` overrides explicit ``pads``).
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        if node.op != "Pad" or not node.inputs or not node.outputs:
            return False
        users = list(node.outputs[0].outputs)
        if len(users) != 1 or users[0].op != "Conv":
            return False
        pads_inp = node.inputs[1] if len(node.inputs) >= 2 else None
        if not isinstance(pads_inp, gs.Constant):
            return False
        pads_flat = np.asarray(pads_inp.values).astype(np.int64).reshape(-1)
        if np.any(pads_flat < 0):
            return False
        conv = users[0]
        if "pads" not in conv.attrs or "kernel_shape" not in conv.attrs:
            return False
        if "auto_pad" in conv.attrs:
            return False
        spatial_rank = len(conv.attrs["kernel_shape"])
        return len(conv.attrs["pads"]) == 2 * spatial_rank

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Pad")
        conv = node.outputs[0].outputs[0]
        pads_flat = (
            np.asarray(node.inputs[1].values).astype(np.int64).reshape(-1)
        )
        rank = len(pads_flat) // 2
        spatial_rank = len(conv.attrs["kernel_shape"])
        spatial_axes = list(range(rank - spatial_rank, rank))
        conv_pads = list(conv.attrs["pads"])

        pb_half = pads_flat[:rank]
        pa_half = pads_flat[rank:]
        new_conv_pads: list[int] = []
        for j in range(spatial_rank):
            new_conv_pads.append(int(conv_pads[j] + pb_half[spatial_axes[j]]))
        for j in range(spatial_rank):
            new_conv_pads.append(
                int(conv_pads[j + spatial_rank] + pa_half[spatial_axes[j]])
            )

        conv.attrs["pads"] = new_conv_pads
        conv.inputs[0] = node.inputs[0]
        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug(
            "fused Pad %s -> Conv %s (new pads=%s)",
            node.name, conv.name, new_conv_pads,
        )


@dataclass
class WidenStridedDepthwiseConv(OnnxGraphEdit):
    """
    Widen narrow strided-depthwise Convs to dodge the Torq DEDR codegen path.

    For depthwise Convs with ``stride[-1] > 1`` and a small last spatial output
    dim, Torq picks a DEDR ``G(L)[sgGroups>1]`` SIMD scatter-gather descriptor
    that the current NSS CModel hangs on. Padding the trailing axis to widen
    the output past the threshold and slicing back forces the compiler onto
    the dense ``V(H)`` path. Defaults match Torq HW: ``bus_width_bytes=72``
    (``iram_seg_width``), ``sg_groups_max=4``.
    """

    bus_width_bytes: int = 72
    sg_groups_max: int = 4

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _element_size_bytes(dtype) -> int | None:
        if isinstance(dtype, np.dtype):
            return dtype.itemsize
        if isinstance(dtype, int):
            sizes = {
                onnx.TensorProto.FLOAT: 4,
                onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1,
                onnx.TensorProto.UINT16: 2, onnx.TensorProto.INT16: 2,
                onnx.TensorProto.INT32: 4, onnx.TensorProto.UINT32: 4,
                onnx.TensorProto.INT64: 8, onnx.TensorProto.UINT64: 8,
                onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2,
                onnx.TensorProto.DOUBLE: 8, onnx.TensorProto.BOOL: 1,
            }
            return sizes.get(int(dtype))
        return None

    def _threshold_for_dtype(self, dtype) -> int | None:
        elem = self._element_size_bytes(dtype)
        if not elem or self.sg_groups_max <= 0:
            return None
        return (self.bus_width_bytes // elem) // self.sg_groups_max

    def _threshold_if_match(self, node: gs.Node) -> int | None:
        if node.op != "Conv" or len(node.inputs) < 2 or not node.outputs:
            return None
        x, w = node.inputs[0], node.inputs[1]
        out = node.outputs[0]
        if not (
            _static_int_shape(x) is not None
            and _static_int_shape(w) is not None
            and _static_int_shape(out) is not None
        ):
            return None
        if len(x.shape) < 3 or len(w.shape) != len(x.shape):
            return None
        in_channels = int(x.shape[1])
        group = int(node.attrs.get("group", 1))
        if group != in_channels or int(w.shape[1]) != 1:
            return None
        rank = len(x.shape) - 2
        strides = list(node.attrs.get("strides", [1] * rank))
        if len(strides) != rank or int(strides[-1]) <= 1:
            return None
        threshold = self._threshold_for_dtype(out.dtype)
        if threshold is None or threshold <= 0:
            return None
        if int(out.shape[-1]) > threshold:
            return None
        return threshold

    def match(self, node: gs.Node) -> bool:
        return self._threshold_if_match(node) is not None

    def transform(self, node: gs.Node):
        threshold = self._threshold_if_match(node)
        if threshold is None:
            return
        x = node.inputs[0]
        out = node.outputs[0]
        rank = len(x.shape) - 2
        cur_out_w = int(out.shape[-1])
        target_out_w = threshold + 1
        extra = target_out_w - cur_out_w

        strides = list(node.attrs.get("strides", [1] * rank))
        s_w = int(strides[-1])
        extra_pad = extra * s_w

        pads = list(node.attrs.get("pads", [0] * (2 * rank)))
        if len(pads) != 2 * rank:
            raise ValueError(
                f"Conv {node.name!r} has unexpected 'pads' length {len(pads)} "
                f"(expected {2 * rank})"
            )
        pads[2 * rank - 1] = int(pads[2 * rank - 1]) + extra_pad
        node.attrs["pads"] = pads
        node.attrs["auto_pad"] = "NOTSET"

        new_out_shape = list(out.shape)
        new_out_shape[-1] = target_out_w
        widened = gs.Variable(
            name=f"{out.name}_widened", dtype=out.dtype, shape=new_out_shape,
        )
        consumers = list(out.outputs)
        graph_output_indices = [
            i for i, g in enumerate(self.graph.outputs) if g is out
        ]
        node.outputs[0] = widened

        last_axis = len(new_out_shape) - 1
        starts = gs.Constant(
            f"{node.name}_widen_slice_starts", np.array([0], dtype=np.int64)
        )
        ends = gs.Constant(
            f"{node.name}_widen_slice_ends", np.array([cur_out_w], dtype=np.int64)
        )
        axes = gs.Constant(
            f"{node.name}_widen_slice_axes", np.array([last_axis], dtype=np.int64)
        )
        steps = gs.Constant(
            f"{node.name}_widen_slice_steps", np.array([1], dtype=np.int64)
        )
        sliced = gs.Variable(name=out.name, dtype=out.dtype, shape=list(out.shape))
        self.graph.nodes.append(
            gs.Node(
                op="Slice",
                name=f"{node.name}_widen_slice",
                inputs=[widened, starts, ends, axes, steps],
                outputs=[sliced],
            )
        )
        rewire_consumers(consumers, out, sliced)
        for i in graph_output_indices:
            self.graph.outputs[i] = sliced
        self._logger.debug(
            "widened depthwise Conv %s: out_w %d -> %d (pad+%d on axis %d, slice [:%d])",
            node.name, cur_out_w, target_out_w, extra_pad, last_axis, cur_out_w,
        )


@dataclass
class DecomposeBidirectionalRnn(OnnxGraphEdit):
    """
    Normalize ONNX ``GRU`` / ``LSTM`` / ``RNN`` ops for the torq backend.

    Two behaviors, controlled together so a single edit can fully prepare
    recurrent ops for the torq import + lowering pipeline:

    1. **Bidirectional decomposition** (always on). torch-mlir's ONNX
       importer only supports ``direction == "forward"``. Bidirectional
       ops are split into two forward ops -- one on ``X``, one on the
       time-reversed ``X`` -- with the reversed branch's sequence output
       flipped back before ``Concat``. Only ``layout == 0`` is supported;
       ``layout == 1`` raises ``NotImplementedError`` rather than silently
       miscompiling.

    2. **Long-sequence chunking** (opt-in via ``max_chunk_len``).
       *Originally-forward* ops whose statically-known sequence length
       exceeds ``K = max_chunk_len`` are rewritten into ``ceil(T / K)``
       shorter forward ops chained along the sequence axis:
       ``Split(axis=0)`` slices ``X``, each chunk threads its ``Y_h``
       (and ``Y_c`` for LSTM) into the next chunk's ``initial_h`` (and
       ``initial_c``), and per-step ``Y`` outputs are stitched back with
       ``Concat(axis=0)``. Weights (``W``/``R``/``B``, and ``P`` for
       LSTM) are shared by reference -- no duplication.

       Forward branches synthesized by the bidirectional decomposition
       above are intentionally **not** chunked: splitting the
       time-reversed branch along the sequence axis broke per-step Y
       stitching on real Synaptics bidirectional GRUs (e.g.
       ``voice_filter``), so bidirectional sources keep the full
       contiguous sequence in each direction.

       Why: torq's executable lowering fully unrolls the recurrent loop
       (the backend cannot consume dynamic dispatch parameters) and
       produces one giant dispatch per RNN op. That scales poorly through
       ``TileAndFusePass``'s memory-fit simulation; splitting into
       ``K``-step chunks yields several modest dispatches that compile in
       seconds while preserving semantics exactly. ``K = 4`` matches the
       longest configuration that currently compiles within CI timeouts
       for the NNNR3 fGRU stack on torq.

       Chunking constraints (silently skipped when not met): ``layout ==
       0``, empty ``sequence_lens``, statically-known sequence length
       strictly greater than ``K``.

    Args:
        max_chunk_len (int | None): When set, chunk forward ops whose
            sequence length exceeds this value. ``None`` (default)
            disables chunking, preserving the original
            decompose-only behavior.
    """

    max_chunk_len: int | None = None

    _RNN_OUTPUT_SLOT_SUFFIX: ClassVar[dict[str, tuple[str, ...]]] = {
        "GRU": ("Y", "Y_h"),
        "RNN": ("Y", "Y_h"),
        "LSTM": ("Y", "Y_h", "Y_c"),
    }
    _RNN_OPS: tuple[str, ...] = ("GRU", "LSTM", "RNN")
    _STATE_INPUT_POSITIONS: ClassVar[dict[str, tuple[int, ...]]] = {
        "GRU": (5,), "RNN": (5,), "LSTM": (5, 6),
    }
    _STATE_OUTPUT_POSITIONS: ClassVar[dict[str, tuple[int, ...]]] = {
        "GRU": (1,), "RNN": (1,), "LSTM": (1, 2),
    }
    # Full ONNX input arity, including optional LSTM peepholes at index 7.
    _FULL_INPUT_COUNT: ClassVar[dict[str, int]] = {"GRU": 6, "RNN": 6, "LSTM": 8}

    def __post_init__(self):
        if self.max_chunk_len is not None and self.max_chunk_len <= 0:
            raise ValueError(
                f"max_chunk_len must be >= 1 when set, got {self.max_chunk_len}"
            )
        self.requires_shape_inference = True
        # ``apply_edit`` iterates the live node list, so any forward branch
        # we synthesize during bidirectional decomposition gets re-visited
        # by the outer loop. Track those branch identities so the
        # forward-chunking path explicitly skips them; bidirectional
        # sources must run each direction's full contiguous sequence.
        self._decompose_emitted: set[int] = set()
        return super().__post_init__()

    @staticmethod
    def _attr_str(attrs: dict, key: str, default: str) -> str:
        val = attrs.get(key, default)
        if isinstance(val, bytes):
            val = val.decode()
        return val

    @staticmethod
    def _is_empty_input(v) -> bool:
        return v is None or (isinstance(v, gs.Variable) and v.name == "")

    @classmethod
    def _unused_output_shape(
        cls,
        op: str,
        out_pos: int,
        X,
        hidden_size: int | None,
        layout: int = 0,
    ) -> tuple[int, ...] | None:
        if not getattr(X, "shape", None) or hidden_size is None:
            return None
        xshape = list(X.shape)
        if len(xshape) < 3:
            return None
        seq_len = xshape[0] if layout == 0 else xshape[1]
        batch = xshape[1] if layout == 0 else xshape[0]
        if out_pos == 0:
            return (
                (seq_len, 1, batch, hidden_size)
                if layout == 0
                else (batch, seq_len, 1, hidden_size)
            )
        if op == "LSTM" and out_pos == 2:
            return (1, batch, hidden_size) if layout == 0 else (batch, 1, hidden_size)
        return (1, batch, hidden_size) if layout == 0 else (batch, 1, hidden_size)

    @classmethod
    def _typed_rnn_output(
        cls,
        node_name: str,
        op: str,
        out_pos: int,
        *,
        dtype,
        shape: tuple[int, ...] | None,
    ) -> gs.Variable:
        suffix = cls._RNN_OUTPUT_SLOT_SUFFIX[op][out_pos]
        return gs.Variable(name=f"{node_name}:{suffix}", dtype=dtype, shape=shape)

    @classmethod
    def _unused_slot(
        cls,
        node_name: str,
        op: str,
        out_pos: int,
        X,
        hidden_size: int | None,
        layout: int,
        dtype,
    ) -> gs.Variable:
        return cls._typed_rnn_output(
            node_name,
            op,
            out_pos,
            dtype=dtype,
            shape=cls._unused_output_shape(op, out_pos, X, hidden_size, layout),
        )

    @classmethod
    def _bidir_branch_output(
        cls,
        base: str,
        branch: str,
        op: str,
        out_pos: int,
        out,
        X,
        hidden_size: int | None,
        layout: int,
    ) -> gs.Variable:
        if cls._is_empty_input(out):
            return cls._unused_slot(
                f"{base}_{branch}", op, out_pos, X, hidden_size, layout, X.dtype
            )
        return gs.Variable(
            name=f"{base}_out{out_pos}_{branch}",
            dtype=out.dtype,
            shape=cls._per_direction_shape_for_output(out_pos, out, layout),
        )

    @classmethod
    def restore_output_arity(cls, graph: gs.Graph) -> None:
        """Re-append typed omitted RNN outputs dropped by GraphSurgeon cleanup."""
        for node in graph.nodes:
            op = node.op
            if op not in cls._RNN_OUTPUT_SLOT_SUFFIX:
                continue
            X = node.inputs[0] if node.inputs else None
            hidden = node.attrs.get("hidden_size")
            hidden_size = int(hidden) if hidden is not None else None
            layout = int(node.attrs.get("layout", 0))
            dtype = getattr(X, "dtype", None) or np.float32
            while len(node.outputs) < cls._full_output_count(op):
                node.outputs.append(
                    cls._unused_slot(
                        node.name or op.lower(),
                        op,
                        len(node.outputs),
                        X,
                        hidden_size,
                        layout,
                        dtype,
                    )
                )

    @staticmethod
    def _per_direction_size(op: str) -> int:
        return {"RNN": 5, "GRU": 6, "LSTM": 7}[op]

    @staticmethod
    def _full_output_count(op: str) -> int:
        return {"RNN": 2, "GRU": 2, "LSTM": 3}[op]

    @staticmethod
    def _split_constant_axis0(
        c: gs.Constant, name_prefix: str
    ) -> tuple[gs.Constant, gs.Constant]:
        arr = np.asarray(c.values)
        if arr.shape[0] != 2:
            raise ValueError(
                f"expected leading dim 2 for {c.name!r}, got shape {tuple(arr.shape)}"
            )
        fwd = gs.Constant(name=f"{name_prefix}_fwd", values=np.ascontiguousarray(arr[0:1]))
        rev = gs.Constant(name=f"{name_prefix}_rev", values=np.ascontiguousarray(arr[1:2]))
        return fwd, rev

    @staticmethod
    def _constant_of_shape_fill(var) -> np.ndarray | None:
        producers = getattr(var, "inputs", None) or []
        if len(producers) != 1 or producers[0].op != "ConstantOfShape":
            return None
        value = producers[0].attrs.get("value")
        if value is None:
            dtype = getattr(var, "dtype", None) or np.float32
            return np.array(0, dtype=dtype)
        if hasattr(value, "values"):
            arr = np.asarray(value.values)
        else:
            arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(
                f"ConstantOfShape fill for {getattr(var, 'name', '<unnamed>')!r} "
                f"must be scalar, got shape {arr.shape}"
            )
        return arr.reshape(())

    @classmethod
    def _materialize_constant_of_shape(
        cls, var, shape: tuple[int, ...] | None, name: str
    ) -> gs.Constant | None:
        if shape is None or any(int(d) <= 0 for d in shape):
            return None
        fill = cls._constant_of_shape_fill(var)
        if fill is None:
            return None
        values = np.full(shape, fill.item(), dtype=fill.dtype)
        return gs.Constant(name=name, values=np.ascontiguousarray(values))

    def _split_runtime_axis0(
        self,
        var: gs.Variable,
        name_prefix: str,
        half_shape: tuple | None,
    ) -> tuple[gs.Variable, gs.Variable]:
        fwd = gs.Variable(name=f"{name_prefix}_fwd", dtype=var.dtype, shape=half_shape)
        rev = gs.Variable(name=f"{name_prefix}_rev", dtype=var.dtype, shape=half_shape)
        sizes = gs.Constant(
            name=f"{name_prefix}_split_sizes",
            values=np.array([1, 1], dtype=np.int64),
        )
        self.graph.nodes.append(
            gs.Node(
                op="Split",
                name=f"{name_prefix}_split",
                inputs=[var, sizes],
                outputs=[fwd, rev],
                attrs={"axis": 0},
            )
        )
        return fwd, rev

    def _split_input(
        self, var, name_prefix: str, materialized_shape: tuple[int, ...] | None = None
    ) -> tuple[object, object]:
        if isinstance(var, gs.Constant):
            return self._split_constant_axis0(var, name_prefix)
        materialized = self._materialize_constant_of_shape(
            var, materialized_shape, f"{name_prefix}_constant"
        )
        if materialized is not None:
            return self._split_constant_axis0(materialized, name_prefix)
        half_shape = None
        if getattr(var, "shape", None):
            full = list(var.shape)
            if len(full) >= 1:
                half_shape = tuple([1, *full[1:]])
        return self._split_runtime_axis0(var, name_prefix, half_shape)

    @staticmethod
    def _split_per_direction_attr(
        attrs: dict, key: str
    ) -> tuple[list | None, list | None]:
        if key not in attrs:
            return None, None
        raw = list(attrs[key])
        if len(raw) % 2 != 0:
            raise ValueError(
                f"per-direction attribute {key!r} has odd length {len(raw)}"
            )
        half = len(raw) // 2
        return raw[:half], raw[half:]

    @staticmethod
    def _per_direction_shape_for_output(out_pos: int, original_out, layout: int):
        if not getattr(original_out, "shape", None):
            return None
        full = list(original_out.shape)
        if out_pos == 0:
            ax = 1 if layout == 0 else 2
        else:
            ax = 0 if layout == 0 else 1
        if ax >= len(full):
            return None
        return tuple(1 if d == ax else full[d] for d in range(len(full)))

    @staticmethod
    def _initial_state_shape(
        input_pos: int, X, hidden_size: int | None
    ) -> tuple[int, ...] | None:
        if input_pos not in (5, 6) or hidden_size is None:
            return None
        if not getattr(X, "shape", None) or len(X.shape) < 2:
            return None
        try:
            batch = int(X.shape[1])
        except (TypeError, ValueError):
            return None
        if batch <= 0:
            return None
        return (2, batch, hidden_size)

    def _gather_flip(
        self, src, axis: int, length: int, name_prefix: str
    ) -> gs.Variable:
        idx = gs.Constant(
            name=f"{name_prefix}_flip_idx",
            values=np.arange(length - 1, -1, -1, dtype=np.int64),
        )
        out_shape = tuple(src.shape) if getattr(src, "shape", None) else None
        out = gs.Variable(
            name=f"{name_prefix}_flipped", dtype=src.dtype, shape=out_shape,
        )
        self.graph.nodes.append(
            gs.Node(
                op="Gather",
                name=f"{name_prefix}_flip",
                inputs=[src, idx],
                outputs=[out],
                attrs={"axis": axis},
            )
        )
        return out

    def _needs_chunking(self, node: gs.Node) -> bool:
        """Whether ``node`` is a forward RNN that the chunking pass should split."""
        if self.max_chunk_len is None:
            return False
        if id(node) in self._decompose_emitted:
            return False
        attrs = dict(node.attrs)
        if self._attr_str(attrs, "direction", "forward") != "forward":
            return False
        if int(attrs.get("layout", 0)) != 0:
            return False
        if not node.inputs or self._is_empty_input(node.inputs[0]):
            return False
        X = node.inputs[0]
        xshape = getattr(X, "shape", None)
        if not xshape or len(xshape) < 1:
            return False
        try:
            seq_len = int(xshape[0])
        except (TypeError, ValueError):
            return False
        if seq_len <= self.max_chunk_len:
            return False
        # sequence_lens (input 4), if present, must be empty; ragged batches
        # can't be naively chunked along the sequence axis.
        if len(node.inputs) >= 5 and not self._is_empty_input(node.inputs[4]):
            return False
        return True

    def match(self, node: gs.Node) -> bool:
        if node.op not in self._RNN_OPS:
            return False
        direction = self._attr_str(dict(node.attrs), "direction", "forward")
        if direction == "bidirectional":
            return True
        return self._needs_chunking(node)

    def transform(self, node: gs.Node):
        attrs = dict(node.attrs)
        op = node.op
        direction = self._attr_str(attrs, "direction", "forward")

        if direction != "bidirectional":
            # Forward op flagged by _needs_chunking; rewrite it in place.
            self._chunk_in_place(node)
            return

        layout = int(attrs.get("layout", 0))
        if layout != 0:
            raise NotImplementedError(
                f"{op}: bidirectional decomposition only supports layout=0, got {layout}"
            )

        n_per_dir = self._per_direction_size(op)
        raw_inputs = list(node.inputs)
        while len(raw_inputs) < n_per_dir:
            raw_inputs.append(gs.Variable.empty())

        X = raw_inputs[0]
        seq_axis = 0
        yh_axis = 0
        y_dir_axis = 1

        if not getattr(X, "shape", None) or len(X.shape) <= seq_axis:
            raise ValueError(
                f"{op}: cannot determine sequence length for X={X.name!r}; "
                "shape inference is required before decomposition"
            )
        seq_len = int(X.shape[seq_axis])
        if seq_len <= 0:
            raise ValueError(
                f"{op}: non-static sequence length {seq_len} on X={X.name!r}"
            )

        base = node.name or f"{op.lower()}_bidir"
        hidden_size_attr = attrs.get("hidden_size")
        hidden_size = int(hidden_size_attr) if hidden_size_attr is not None else None

        pd_indices = [i for i in range(1, n_per_dir) if i != 4]
        fwd_inputs: list[object] = list(raw_inputs)
        rev_inputs: list[object] = list(raw_inputs)
        for i in pd_indices:
            var = raw_inputs[i]
            if self._is_empty_input(var):
                continue
            fwd_var, rev_var = self._split_input(
                var,
                f"{base}_in{i}",
                self._initial_state_shape(i, X, hidden_size),
            )
            fwd_inputs[i] = fwd_var
            rev_inputs[i] = rev_var

        rev_inputs[0] = self._gather_flip(X, seq_axis, seq_len, f"{base}_x")

        fwd_attrs: dict = {}
        rev_attrs: dict = {}
        for key in ("activations", "activation_alpha", "activation_beta"):
            fwd_part, rev_part = self._split_per_direction_attr(attrs, key)
            if fwd_part is not None:
                fwd_attrs[key] = fwd_part
                rev_attrs[key] = rev_part
        for key, val in attrs.items():
            if key in ("direction", "activations", "activation_alpha", "activation_beta"):
                continue
            fwd_attrs[key] = val
            rev_attrs[key] = val
        fwd_attrs["direction"] = "forward"
        rev_attrs["direction"] = "forward"

        raw_outputs = list(node.outputs)
        if not raw_outputs:
            return

        full_out_count = self._full_output_count(op)
        original_out_count = len(raw_outputs)
        while len(raw_outputs) < full_out_count:
            raw_outputs.append(gs.Variable.empty())

        new_nodes: list[gs.Node] = []
        fwd_outs: list[gs.Variable] = []
        rev_outs: list[gs.Variable] = []

        for out_pos, out in enumerate(raw_outputs):
            fwd_outs.append(
                self._bidir_branch_output(
                    base, "fwd", op, out_pos, out, X, hidden_size, layout
                )
            )
            rev_outs.append(
                self._bidir_branch_output(
                    base, "rev", op, out_pos, out, X, hidden_size, layout
                )
            )

        fwd_node = gs.Node(
            op=op, name=f"{base}_fwd", inputs=fwd_inputs, outputs=fwd_outs, attrs=fwd_attrs,
        )
        rev_node = gs.Node(
            op=op, name=f"{base}_rev", inputs=rev_inputs, outputs=rev_outs, attrs=rev_attrs,
        )
        new_nodes.append(fwd_node)
        new_nodes.append(rev_node)
        self._decompose_emitted.add(id(fwd_node))
        self._decompose_emitted.add(id(rev_node))

        for out_pos in range(original_out_count):
            out = raw_outputs[out_pos]
            if self._is_empty_input(out):
                continue
            ax = y_dir_axis if out_pos == 0 else yh_axis
            rev_branch = rev_outs[out_pos]
            if out_pos == 0:
                rev_branch = self._gather_flip(
                    rev_branch, seq_axis, seq_len, f"{base}_out{out_pos}_rev"
                )
            out.inputs.clear()
            new_nodes.append(
                gs.Node(
                    op="Concat",
                    name=f"{base}_out{out_pos}_concat",
                    inputs=[fwd_outs[out_pos], rev_branch],
                    outputs=[out],
                    attrs={"axis": ax},
                )
            )

        node.inputs.clear()
        node.outputs.clear()
        self.graph.nodes.extend(new_nodes)
        self._logger.debug(
            "decomposed bidirectional %s %r at layout=%d", op, base, layout
        )
        # Note: the forward branches synthesized here are intentionally not
        # passed through ``_chunk_in_place``. Splitting the time-reversed
        # branch along the sequence axis broke per-step Y stitching on
        # real Synaptics bidirectional GRUs (e.g. voice_filter); preserve
        # the original full-sequence semantics for bidirectional sources.

    def _chunk_in_place(self, node: gs.Node) -> None:
        """
        Rewrite a forward ``GRU`` / ``LSTM`` / ``RNN`` node into a chain of
        shorter equivalent ops along the sequence axis, preserving the
        identity of its output tensors so consumers stay wired.

        Caller must have already confirmed via :meth:`_needs_chunking` that
        ``node`` is a chunkable forward op.
        """
        op = node.op
        attrs = dict(node.attrs)
        X = node.inputs[0]
        seq_len = int(X.shape[0])
        K = self.max_chunk_len
        n_chunks = (seq_len + K - 1) // K
        chunk_sizes = [K] * (n_chunks - 1) + [seq_len - K * (n_chunks - 1)]

        base = node.name or f"{op.lower()}_chunk"

        hidden_size_attr = attrs.get("hidden_size")
        hidden_size = int(hidden_size_attr) if hidden_size_attr is not None else None
        batch: int | None = None
        if len(X.shape) >= 2:
            try:
                batch = int(X.shape[1])
            except (TypeError, ValueError):
                batch = None
        state_shape = (
            (1, batch, hidden_size)
            if (hidden_size is not None and batch is not None)
            else None
        )

        n_inputs = self._FULL_INPUT_COUNT[op]
        n_outputs = self._full_output_count(op)
        raw_inputs = list(node.inputs)
        while len(raw_inputs) < n_inputs:
            raw_inputs.append(gs.Variable.empty())
        raw_outputs = list(node.outputs)
        while len(raw_outputs) < n_outputs:
            raw_outputs.append(gs.Variable.empty())

        split_sizes = gs.Constant(
            name=f"{base}_split_sizes",
            values=np.asarray(chunk_sizes, dtype=np.int64),
        )
        x_chunks: list[gs.Variable] = []
        for i, size in enumerate(chunk_sizes):
            chunk_shape = (
                (size, *tuple(X.shape[1:])) if X.shape and len(X.shape) >= 1 else None
            )
            x_chunks.append(
                gs.Variable(name=f"{base}_x{i}", dtype=X.dtype, shape=chunk_shape)
            )
        new_nodes: list[gs.Node] = [
            gs.Node(
                op="Split",
                name=f"{base}_x_split",
                inputs=[X, split_sizes],
                outputs=x_chunks,
                attrs={"axis": 0},
            )
        ]

        state_in_positions = self._STATE_INPUT_POSITIONS[op]
        state_out_positions = self._STATE_OUTPUT_POSITIONS[op]
        emit_y = not self._is_empty_input(raw_outputs[0])

        # Running state vars threaded between chunks. Initial values are
        # whatever the original op had (may be empty -- ONNX defaults to zero).
        state_vars: list[object] = [raw_inputs[p] for p in state_in_positions]
        y_chunks: list[gs.Variable] = []
        y_dtype = raw_outputs[0].dtype if emit_y else X.dtype

        for i, size in enumerate(chunk_sizes):
            is_last = (i == n_chunks - 1)
            chunk_inputs: list[object] = list(raw_inputs)
            chunk_inputs[0] = x_chunks[i]
            for slot, in_pos in enumerate(state_in_positions):
                chunk_inputs[in_pos] = state_vars[slot]

            chunk_outputs: list[object] = [gs.Variable.empty()] * n_outputs

            if emit_y:
                y_shape = (size, 1, batch, hidden_size) if state_shape else None
                y_chunk = gs.Variable(
                    name=f"{base}_y{i}", dtype=y_dtype, shape=y_shape,
                )
                chunk_outputs[0] = y_chunk
                y_chunks.append(y_chunk)

            next_state_vars: list[object] = []
            for slot, out_pos in enumerate(state_out_positions):
                orig_state_out = raw_outputs[out_pos]
                if is_last:
                    if self._is_empty_input(orig_state_out):
                        state_var = self._unused_slot(
                            f"{base}_chunk{i}",
                            op,
                            out_pos,
                            X,
                            hidden_size,
                            int(attrs.get("layout", 0)),
                            X.dtype,
                        )
                    else:
                        # Reuse the original tensor so any graph-output /
                        # consumer wiring is preserved.
                        orig_state_out.inputs.clear()
                        state_var = orig_state_out
                else:
                    dtype_for_state = (
                        orig_state_out.dtype
                        if not self._is_empty_input(orig_state_out)
                        else X.dtype
                    )
                    state_var = gs.Variable(
                        name=f"{base}_state{slot}_chunk{i}",
                        dtype=dtype_for_state,
                        shape=state_shape,
                    )
                chunk_outputs[out_pos] = state_var
                next_state_vars.append(state_var)

            while chunk_outputs and self._is_empty_input(chunk_outputs[-1]):
                chunk_outputs.pop()

            new_nodes.append(
                gs.Node(
                    op=op,
                    name=f"{base}_chunk{i}",
                    inputs=chunk_inputs,
                    outputs=chunk_outputs,
                    attrs=dict(attrs),
                )
            )
            state_vars = next_state_vars

        if emit_y:
            orig_y = raw_outputs[0]
            orig_y.inputs.clear()
            new_nodes.append(
                gs.Node(
                    op="Concat",
                    name=f"{base}_y_concat",
                    inputs=y_chunks,
                    outputs=[orig_y],
                    attrs={"axis": 0},
                )
            )

        node.inputs.clear()
        node.outputs.clear()
        self.graph.nodes.extend(new_nodes)
        self._logger.debug(
            "chunked %s %r: seq_len=%d -> %d chunk(s) of size %s",
            op, base, seq_len, n_chunks, chunk_sizes,
        )


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
                if list(io.shape) != kv_tensor_shape:
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

    def eliminate_expands(self, ops: list[str]):
        self.apply_edit(EliminateExpand(self._graph, self._graph_name, ops))
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

    def trim_lm_head_vocab(self, kept_token_ids, save_lut=None, output_name="logits", include_argmax=False):
        self.apply_edit(TrimLMHeadVocab(self._graph, self._graph_name, kept_token_ids, output_name, save_lut, include_argmax))
        return self

    def split_lm_head(self, save_to, output_name="logits", hidden_states_name="last_hidden_states"):
        self.apply_edit(SplitLMHead(self._graph, self._graph_name, save_to, output_name, hidden_states_name))
        return self

    def eliminate_rank0_gather(self):
        self.apply_edit(EliminateRank0Gather(self._graph, self._graph_name))
        return self

    def eliminate_singleton_gather_unsqueeze(self):
        self.apply_edit(EliminateSingletonGatherUnsqueeze(self._graph, self._graph_name))
        return self

    def rewrite_negative_pads(self):
        self.apply_edit(RewriteNegativePads(self._graph, self._graph_name))
        return self

    def absorb_padding(self):
        self.apply_edit(AbsorbPadding(self._graph, self._graph_name))
        return self

    def widen_strided_depthwise_conv(self, bus_width_bytes: int = 72, sg_groups_max: int = 4):
        self.apply_edit(
            WidenStridedDepthwiseConv(
                self._graph, self._graph_name, bus_width_bytes, sg_groups_max
            )
        )
        return self

    def decompose_bidirectional_rnn(self, max_chunk_len: int | None = None):
        self.apply_edit(
            DecomposeBidirectionalRnn(self._graph, self._graph_name, max_chunk_len)
        )
        return self
