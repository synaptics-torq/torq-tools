# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import hashlib
import os

import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import (
    OnnxGraphEdit,
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor
)

from ...utils.onnx import (
    normalize_layer_name
)

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
        ValueError: If Softmax producer is not an `Add` op

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
            return isinstance(node.i(), gs.Node) and node.i().op == "Add"
        return False

    def transform(self, node: gs.Node):
        if not self.export_dtype:
            raise RuntimeError("ONNX export dtype is requried for this graph edit, provide via `export_dtype`")

        self._check_node_op(node, "Softmax")
        if (add_node := node.i()).op != "Add":
            raise ValueError(
                f"Causal Attention Mask: '{node.name}' expected producer to be Add node, got {add_node.op} ({add_node.name})"
            )

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
            name=add_node.name + "_lte_cur_len",
            op="LessOrEqual",
            inputs=[time_axis, self.cur_len],
            outputs=[
                gs.Variable(
                    add_node.name + "_less", dtype=onnx.TensorProto.BOOL, shape=mask_shape
                )
            ],
        )[0]
        mask = self.graph.layer(
            name=add_node.name + "_mask_attn",
            op="Where",
            inputs=[mask_lte, attn_mask_keep, attn_mask_block],
            outputs=[
                gs.Variable(add_node.name + "_where", dtype=add_node.inputs[0].dtype, shape=mask_shape)
            ],
        )[0]

        # rewire Add node to use mask
        add_node.inputs[1] = mask

        self._logger.debug("Added causal attention mask to scores at node '%s'", add_node.name)


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
        self.rewire_consumers(consumers, gather_out, self.cur_len)

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
        hidden_size (int): Moonshine hidden KV dims size
        vocab_size (int): Moonshine vocabulary size
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
        self.rewire_consumers(consumers, where_out, producer)

        # disconnect IsNaN -> Where chain
        node.inputs.clear()
        where_node.inputs.clear()
        where_node.outputs.clear()

        self._logger.debug("Removed unsupported IsNaN op '%s'", node.name)


@dataclass
class MoveOutputFromConcat(OnnxGraphEdit):
    """
    Move outputs from Concat nodes to their consumer Pad nodes for compatibility.

    This is requried to prevent errors with Acuity compilation.

    Args:
        pad_len (int): Length of padding to apply
    """

    pad_len: int

    def __post_init__(self):
        self.output_names = {o.name for o in self.graph.outputs}
        return super().__post_init__()

    def match(self, node: gs.Node):
        return node.op == "Concat" and node.outputs[0].name in self.output_names

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Concat")
        output_name = node.outputs[0].name
        consumers: list[gs.Node] = list(node.outputs[0].outputs)
        for consumer in consumers:
            if consumer.op == "Pad":

                concat_output: gs.Variable = node.outputs[0]
                pad_output: gs.Variable = consumer.outputs[0]

                tensors = self.graph.tensors()
                if not (output_slice_starts := tensors.get("output_slice_starts")):
                    output_slice_starts = gs.Constant(
                        "output_slice_starts", np.array([0], dtype=np.int64)
                    )
                if not (output_slice_ends := tensors.get("output_slice_ends")):
                    output_slice_ends = gs.Constant(
                        "output_slice_ends", np.array([-self.pad_len], dtype=np.int64)
                    )
                if not (output_slice_axes := tensors.get("output_slice_axes")):
                    output_slice_axes = gs.Constant(
                        "output_slice_axes", np.array([3], dtype=np.int64)
                    )
                if not (output_slice_steps := tensors.get("output_slice_steps")):
                    output_slice_steps = gs.Constant(
                        "output_slice_steps", np.array([1], dtype=np.int64)
                    )
                slice_output: gs.Variable = self.graph.layer(
                    name=pad_output.name + "_slice",
                    op="Slice",
                    inputs=[
                        pad_output,
                        output_slice_starts,
                        output_slice_ends,
                        output_slice_axes,
                        output_slice_steps,
                    ],
                    outputs=[
                        gs.Variable(
                            concat_output.name, dtype=concat_output.dtype, shape=concat_output.shape
                        )
                    ],
                )[0]

                for i, output in enumerate(self.graph.outputs):
                    if output is concat_output:
                        self.graph.outputs[i] = slice_output

                orig = concat_output.name
                concat_output.name = orig + "_prepad"
                slice_output.name = orig

                self._logger.debug("Moved output '%s' to Pad node '%s'", output_name, consumer.name)

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
class ReplaceInt64FloatCast(OnnxGraphEdit):
    """
    Replace int64 -> float casts with a look-up table: `output<fp32|fp16|bf16> = LUT[input<int64>]`
    """

    max_int: int

    @staticmethod
    def _is_integer_type(dtype: np.dtype | int) -> bool:
        if isinstance(dtype, np.dtype):
            return np.issubdtype(dtype, np.integer)
        elif isinstance(dtype, int):
            return dtype in {
                onnx.TensorProto.INT8, onnx.TensorProto.INT16, onnx.TensorProto.INT32, onnx.TensorProto.INT64,
                onnx.TensorProto.UINT8, onnx.TensorProto.UINT16, onnx.TensorProto.UINT32, onnx.TensorProto.UINT64,
            }
        else:
            return False

    def match(self, node: gs.Node) -> bool:
        if node.op == "Cast":
            if self._is_integer_type(node.inputs[0].dtype) and node.attrs["to"] in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.BFLOAT16,
            ):
                int_inp: gs.Variable | gs.Constant = node.inputs[0]
                cast_dtype = node.attrs["to"]
                if cast_dtype not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as target is not float",
                        node.name
                    )
                    return False
                if not int_inp.shape or not all(isinstance(d, (int, np.integer)) for d in int_inp.shape):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as input is not static",
                        node.name
                    )
                    return False
                if not all(i == 1 for i in list(int_inp.shape)[:-1]):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as input has non-batch dims",
                        node.name
                    )
                    return False
                return True 
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Cast")
        cast_dtype_str = onnx.helper.tensor_dtype_to_string(node.attrs["to"])
        self._logger.warning("Replacing int -> %s cast with lookup table, disable if env supports this casting", cast_dtype_str)

        int_inp: gs.Variable | gs.Constant = node.inputs[0]
        inp_dtype = int_inp.dtype
        if not self._is_integer_type(inp_dtype):
            raise ValueError(
                f"Cast input must be integer, found {int_inp.dtype} for int -> float cast replacement"
            )
        cast_dtype = node.attrs["to"]
        if cast_dtype not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
            raise ValueError(
                f"Cast output must be float, found {onnx.helper.tensor_dtype_to_string(cast_dtype)} for int -> float cast replacement"
            )

        if not int_inp.shape or not all(isinstance(d, (int, np.integer)) for d in int_inp.shape):
            raise ValueError(
                f"Cast input must have static shape, found {int_inp.shape} for int -> float cast replacement"
            )
        if not all(i == 1 for i in list(int_inp.shape)[:-1]):
            raise ValueError(
                f"Cast input must be batched, found non-batch dims in shape {int_inp.shape} for int -> float cast replacement"
            )
        tensors = self.graph.tensors()
        if not (shape_scalar := tensors.get(int_inp.name + "_shape_scalar")):
            shape_scalar = gs.Constant(int_inp.name + "_shape_scalar", np.array([], dtype=np.int64))
        if not (shape_batched := tensors.get(int_inp.name + "_shape_batched")):
            shape_batched = gs.Constant(int_inp.name + "_shape_batched", np.asarray(int_inp.shape, dtype=np.int64))
        float_lut_name = int_inp.name + f"_lut_{self.max_int}_{onnx.helper.tensor_dtype_to_string(cast_dtype)}"
        if not (lookup_table := tensors.get(float_lut_name)):
            lookup_table = gs.Constant(float_lut_name, np.arange(self.max_int, dtype=np.float32), export_dtype=cast_dtype)
        float_out: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(float_out.outputs)
        assert float_out.shape == int_inp.shape, f"Cast node '{node.name}': input shape {int_inp.shape} != output shape {float_out.shape}"

        int_inp_scalar: gs.Variable = self.graph.layer(
            name=int_inp.name + "_to_scalar",
            op="Reshape",
            inputs=[int_inp, shape_scalar],
            outputs=[gs.Variable(name=int_inp.name + "_scalar", dtype=int_inp.dtype, shape=[])]
        )[0]
        lookup_output: gs.Variable = self.graph.layer(
            name=int_inp.name + "_lookup",
            op="Gather",
            inputs=[lookup_table, int_inp_scalar],
            outputs=[gs.Variable(name=int_inp.name + "_float_value", dtype=cast_dtype, shape=[])]
        )[0]
        lookup_output_batched: gs.Variable = self.graph.layer(
            name=int_inp.name + "_lookup_batch",
            op="Reshape",
            inputs=[lookup_output, shape_batched],
            outputs=[gs.Variable(name=lookup_output.name + "_batched", dtype=cast_dtype, shape=int_inp.shape)]
        )[0]
        self.rewire_consumers(consumers, float_out, lookup_output_batched)
        node.outputs.clear()

        self._logger.debug("Replaced int -> %s Cast node '%s' with look-up table", cast_dtype_str, node.name)

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
        self.rewire_consumers(consumers, tensor, bcast_out)

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
        self.rewire_consumers(consumers, lut_out, lut_entry_inp)
        self.graph.inputs.append(lut_entry_inp)
        node.outputs.clear()
        self._logger.debug(
            "Extracted LUT from '%s', consumers redirected to graph input '%s'",
            node.name, self.inp_name
        )


class MoonshineOnnxGraphEditor(OnnxGraphEditor):

    def __init__(
        self,
        graph: gs.Graph,
        component: str,
        export_dtype: onnx.TensorProto.DataType | None = None
    ):
        super().__init__(
            graph,
            component,
            export_dtype=export_dtype
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        component: str,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ) -> "MoonshineOnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            component,
            export_dtype
        )

    def fix_encoder_io(
        self,
        num_samples: int,
        enc_seq_len: int,
        dims: list[FixedDimMapping] | None = None,
        *,
        batch_dim: str = "batch_size",
        num_samples_dim: str = "num_samples",
        enc_seq_len_dims: list[str] = ["encoder_sequence_length", "floor(floor(floor(num_samples/64 - 127/64)/3)/2) - 1"],
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(num_samples_dim, DimMatchType.EXACT, num_samples),
        ]
        to_fix.extend([
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, enc_seq_len)
            for seq_len_dim in enc_seq_len_dims
        ])
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def fix_decoder_io(
        self,
        enc_seq_len: int,
        dec_seq_len: int,
        with_past: bool,
        dims: list[FixedDimMapping] | None = None,
        *,
        batch_dim: str = "batch_size",
        enc_seq_len_dim: str = "encoder_sequence_length",
        dec_seq_len_dim: str = "decoder_sequence_length",
        past_dec_seq_len_dim: str = "past_decoder_sequence_length"
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(enc_seq_len_dim, DimMatchType.CONTAINS, enc_seq_len),
            FixedDimMapping(dec_seq_len_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(past_dec_seq_len_dim, DimMatchType.CONTAINS, dec_seq_len if with_past else 1)
        ]
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def replace_dynamic_kv_cache(
        self,
        cur_len: gs.Variable,
        max_tokens: int
    ):
        self.apply_edit(ReplaceDynamicKVCache(self._graph, self._graph_name, cur_len, max_tokens))
        return self

    def mask_future_attn_scores(
        self,
        cur_len: gs.Variable,
        max_tokens: int
    ):
        self.apply_edit(MaskFutureAttentionScores(self._graph, self._graph_name, cur_len, max_tokens, self._export_dtype))
        return self

    def add_curr_len_input(
        self,
        cur_len: gs.Variable
    ):
        self.apply_edit(AddCurrLenInput(self._graph, self._graph_name, cur_len))
        return self

    def convert_to_static_index(
        self
    ):
        self.apply_edit(ConvertToStaticIndex(self._graph, self._graph_name))
        return self

    def dequantize_projections_matmul(
        self,
        hidden_size: int,
        vocab_size: int
    ):
        self.apply_edit(DequantizeProjectionsMatMul(self._graph, self._graph_name, hidden_size, vocab_size, self._export_dtype))
        return self

    def remove_isNaN(
        self
    ):
        self.apply_edit(RemoveIsNaN(self._graph, self._graph_name))
        return self

    def move_output_from_concat(
        self,
        pad_len: int
    ):
        self.apply_edit(MoveOutputFromConcat(self._graph, self._graph_name, pad_len))
        return self

    def fold_scalar_matmul(
        self
    ):
        self.apply_edit(FoldScalarMatMul(self._graph, self._graph_name))
        return self

    def replace_int64_float_cast(
        self,
        max_int: int
    ):
        self.apply_edit(ReplaceInt64FloatCast(self._graph, self._graph_name, max_int))
        return self
    
    def broadcast_op_inputs(
        self,
        ops: list[str],
        output_idx: int = 0,
        inputs_idx: list[int] = None,
        constants_policy: ConstantBroadcastPolicy = ConstantBroadcastPolicy.SKIP,
    ):
        self.apply_edit(BroadcastOpInputs(self._graph, self._graph_name, ops, output_idx, inputs_idx, constants_policy))
        return self

    def extract_token_embeddings(
        self,
        hidden_size: int,
        vocab_size: int,
        save_to: os.PathLike | str,
        inp_name: str = "token_embedding"
    ):
        self.apply_edit(ExtractConstantLUT(self._graph, self._graph_name, (vocab_size, hidden_size), save_to, inp_name))
        return self
