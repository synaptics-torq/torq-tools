# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Graph edits specific to Moonshine Streaming.

Two kinds of edits live here, and neither belongs in the shared
``torq.graph_edit.edits`` package:

* Streaming-specialised variants of shared edits — ``ReplaceDynamicKVCache``,
  ``MaskFutureAttentionScores`` and ``AddCurrLenInput``. Their ``match``/``transform``
  logic is tuned for the dynamo *stacked-cache* decoder export and therefore
  intentionally diverges from the shared versions (e.g. matching the Gather-from-
  graph-input KV pattern and shape-based self-attention detection).
* Genuinely new decompositions — ``DecomposeLayerNormalization``(``MulReciprocal``),
  ``DecomposeGelu`` and ``DecomposeBooleanAnd``.

Edits that are byte-for-byte identical to the shared package are NOT duplicated
here; they are reused from ``torq.graph_edit.edits`` via ``CommonGraphEditsMixin``.
"""

from dataclasses import dataclass

import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import OnnxGraphEdit, rewire_consumers

__all__ = [
    "ReplaceDynamicKVCache",
    "MaskFutureAttentionScores",
    "AddCurrLenInput",
    "DecomposeLayerNormalization",
    "DecomposeLayerNormalizationMulReciprocal",
    "DecomposeGelu",
    "DecomposeBooleanAnd",
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
        if node.op != "Concat" or node.attrs.get("axis") != -2:
            return False
        # Primary: Concat output is directly a graph output (torchscript / non-dynamo export)
        if node.outputs[0].name in self.output_names:
            return True
        # Secondary: per-layer KV concat in dynamo stacked-cache export.
        # One input is a Gather whose data source is a graph input (the past KV buffer).
        for inp in node.inputs:
            if inp.inputs:
                producer = inp.inputs[0]
                if producer.op == "Gather" and producer.inputs:
                    gather_src = producer.inputs[0]
                    if isinstance(gather_src, gs.Variable) and not gather_src.inputs:
                        return True
        return False

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

        # Update output shape to match past_cache_vals (pre-allocated buffer).
        # The Where blend keeps the buffer size fixed; the old Concat shape
        # (past+1) no longer applies.
        if getattr(past_cache_vals, 'shape', None) is not None:
            output.shape = list(past_cache_vals.shape)

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
        if node.op == "Softmax":
            is_self_attn = node.name.endswith("self_attn/Softmax")
            if not is_self_attn and node.inputs:
                inp_shape = getattr(node.inputs[0], "shape", None)
                if inp_shape and len(inp_shape) >= 1:
                    last_dim = inp_shape[-1]
                    if isinstance(last_dim, str):
                        is_self_attn = ("past_seq" in last_dim)
                    else:
                        # Accept max_tokens (after KV replacement) or max_tokens+1 (traced shape)
                        is_self_attn = last_dim in (self.max_tokens, self.max_tokens + 1)
            if is_self_attn:
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
        if node.op == "Shape" and any(x in node.inputs[0].name for x in ("past_key_values", "past_self")):
            return isinstance(node.o(), gs.Node) and node.o().op in ("Gather", "Squeeze")
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Shape")
        gather_node: gs.Node = node.o()
        if not isinstance(gather_node, gs.Node) or gather_node.op not in ("Gather", "Squeeze"):
            raise ValueError(f"Expected Gather or Squeeze node after Shape, got {gather_node}")

        gather_out: gs.Variable = gather_node.outputs[0]
        consumers: list[gs.Node] = list(gather_out.outputs)
        rewire_consumers(consumers, gather_out, self.cur_len)

        # disconnect Shape + Gather/Squeeze branch
        node.inputs.clear()
        gather_node.outputs.clear()

        self._logger.debug("Replaced dynamic seq len getter at node '%s'", node.name)


@dataclass
class DecomposeLayerNormalizationMulReciprocal(OnnxGraphEdit):
    """
    Decompose ONNX LayerNormalization into basic arithmetic operations using Mul and Reciprocal.
    Useful for hardware targets that do not natively legalize LayerNormalization.
    """
    dim_map: dict[str, int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.dim_map is None:
            self.dim_map = {"batch": 1}

    def match(self, node: gs.Node) -> bool:
        return node.op == "LayerNormalization"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "LayerNormalization")
        
        X = node.inputs[0]
        scale = node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        
        axis = node.attrs.get("axis", -1)
        epsilon = node.attrs.get("epsilon", 1e-05)
        
        # Normalize axis and calculate reduction axes (all axes from `axis` to `rank - 1`)
        rank = len(X.shape) if X.shape is not None else 0
        if rank > 0:
            axis = axis % rank
            reduce_axes = list(range(axis, rank))
        else:
            reduce_axes = [axis]
            
        Y = node.outputs[0]
        
        # Resolve target shape to static integers using dim_map
        target_shape = []
        if X.shape is not None:
            for d in X.shape:
                if isinstance(d, str):
                    target_shape.append(self.dim_map.get(d, 1))
                else:
                    target_shape.append(d)
        else:
            target_shape = [1, 150, 320]
            
        target_shape_const = gs.Constant(
            name=node.name + "_target_shape",
            values=np.array(target_shape, dtype=np.int64)
        )

        def expand_tensor(tensor, name_suffix):
            # If shape is already matching target_shape, skip expand
            if tensor.shape is not None and list(tensor.shape) == list(target_shape):
                return tensor
            return self.graph.layer(
                name=node.name + "_" + name_suffix + "_expand",
                op="Expand",
                inputs=[tensor, target_shape_const],
                outputs=[gs.Variable(name=node.name + "_" + name_suffix + "_expanded", dtype=tensor.dtype, shape=target_shape)]
            )[0]

        # Create axes constant for ReduceMean
        axes_const = gs.Constant(
            name=node.name + "_axes",
            values=np.array(reduce_axes, dtype=np.int64)
        )
        
        # 1. Compute mean: mean = ReduceMean(X, axes)
        mean = self.graph.layer(
            name=node.name + "_mean",
            op="ReduceMean",
            inputs=[X, axes_const],
            outputs=[gs.Variable(name=node.name + "_mean_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]
        
        # 2. Subtract mean: X_diff = Sub(X, mean_expanded)
        mean_expanded = expand_tensor(mean, "mean")
        x_diff = self.graph.layer(
            name=node.name + "_sub_mean",
            op="Sub",
            inputs=[X, mean_expanded],
            outputs=[gs.Variable(name=node.name + "_diff", dtype=X.dtype)]
        )[0]
        
        # 3. Compute variance: var = ReduceMean((X - mean)^2, axes)
        x_diff_sq = self.graph.layer(
            name=node.name + "_diff_sq",
            op="Mul",
            inputs=[x_diff, x_diff],
            outputs=[gs.Variable(name=node.name + "_diff_sq_val", dtype=X.dtype)]
        )[0]
        
        var = self.graph.layer(
            name=node.name + "_var",
            op="ReduceMean",
            inputs=[x_diff_sq, axes_const],
            outputs=[gs.Variable(name=node.name + "_var_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]
        
        # 4. Standard deviation: stddev = Sqrt(var + eps)
        # Ensure we use a valid NumPy dtype matching X.dtype for the epsilon constant
        if isinstance(X.dtype, int):
            try:
                import onnx.helper
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(X.dtype)
            except Exception:
                np_dtype = np.float32
        else:
            np_dtype = X.dtype if X.dtype is not None else np.float32

        eps_const = gs.Constant(
            name=node.name + "_eps",
            values=np.array(epsilon, dtype=np_dtype),
        )
        
        var_eps = self.graph.layer(
            name=node.name + "_var_eps",
            op="Add",
            inputs=[var, eps_const],
            outputs=[gs.Variable(name=node.name + "_var_eps_val", dtype=X.dtype)]
        )[0]
        
        stddev = self.graph.layer(
            name=node.name + "_stddev",
            op="Sqrt",
            inputs=[var_eps],
            outputs=[gs.Variable(name=node.name + "_stddev_val", dtype=X.dtype)]
        )[0]
        
        # 5. Normalize: X_norm = Mul(X_diff, stddev_inv_expanded) where stddev_inv = Reciprocal(stddev)
        stddev_inv = self.graph.layer(
            name=node.name + "_stddev_inv",
            op="Reciprocal",
            inputs=[stddev],
            outputs=[gs.Variable(name=node.name + "_stddev_inv_val", dtype=X.dtype)]
        )[0]
        
        stddev_inv_expanded = expand_tensor(stddev_inv, "stddev_inv")
        
        x_norm = self.graph.layer(
            name=node.name + "_mul_stddev_inv",
            op="Mul",
            inputs=[x_diff, stddev_inv_expanded],
            outputs=[gs.Variable(name=node.name + "_norm", dtype=X.dtype)]
        )[0]
        
        # 6. Apply scale and optional bias
        scale_expanded = expand_tensor(scale, "scale")
        if bias is not None:
            bias_expanded = expand_tensor(bias, "bias")
            x_scaled = self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[gs.Variable(name=node.name + "_scaled", dtype=X.dtype)]
            )[0]
            self.graph.layer(
                name=node.name + "_add_bias",
                op="Add",
                inputs=[x_scaled, bias_expanded],
                outputs=[Y]
            )
        else:
            self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[Y]
            )
            
        # Rewire other optional outputs if they are used by any consumer
        if len(node.outputs) > 1 and len(node.outputs[1].outputs) > 0:
            rewire_consumers(node.outputs[1].outputs, node.outputs[1], mean)
        if len(node.outputs) > 2 and len(node.outputs[2].outputs) > 0:
            inv_std_dev = self.graph.layer(
                name=node.name + "_inv_stddev",
                op="Reciprocal",
                inputs=[stddev],
                outputs=[gs.Variable(name=node.name + "_inv_stddev_val", dtype=X.dtype)]
            )[0]
            rewire_consumers(node.outputs[2].outputs, node.outputs[2], inv_std_dev)
            
        # Disconnect node
        node.inputs.clear()
        node.outputs.clear()
        
        self._logger.debug("Decomposed LayerNormalization (Mul/Reciprocal) node '%s'", node.name)


@dataclass
class DecomposeLayerNormalization(OnnxGraphEdit):
    """
    Decompose ONNX LayerNormalization into basic arithmetic operations using Pow and Div.
    Closely matches standard PyTorch export/decomposition.
    """
    dim_map: dict[str, int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.dim_map is None:
            self.dim_map = {"batch": 1}

    def match(self, node: gs.Node) -> bool:
        return node.op == "LayerNormalization"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "LayerNormalization")
        
        X = node.inputs[0]
        scale = node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        
        axis = node.attrs.get("axis", -1)
        epsilon = node.attrs.get("epsilon", 1e-05)
        
        # Normalize axis and calculate reduction axes
        rank = len(X.shape) if X.shape is not None else 0
        if rank > 0:
            axis = axis % rank
            reduce_axes = list(range(axis, rank))
        else:
            reduce_axes = [axis]
            
        Y = node.outputs[0]
        
        # Resolve target shape to static integers using dim_map
        target_shape = []
        if X.shape is not None:
            for d in X.shape:
                if isinstance(d, str):
                    target_shape.append(self.dim_map.get(d, 1))
                else:
                    target_shape.append(d)
        else:
            target_shape = [1, 150, 320]
            
        target_shape_const = gs.Constant(
            name=node.name + "_target_shape",
            values=np.array(target_shape, dtype=np.int64)
        )

        def expand_tensor(tensor, name_suffix):
            if tensor.shape is not None and list(tensor.shape) == list(target_shape):
                return tensor
            return self.graph.layer(
                name=node.name + "_" + name_suffix + "_expand",
                op="Expand",
                inputs=[tensor, target_shape_const],
                outputs=[gs.Variable(name=node.name + "_" + name_suffix + "_expanded", dtype=tensor.dtype, shape=target_shape)]
            )[0]

        # Create axes constant for ReduceMean
        axes_const = gs.Constant(
            name=node.name + "_axes",
            values=np.array(reduce_axes, dtype=np.int64)
        )
        
        # 1. Compute mean: mean = ReduceMean(X, axes)
        mean = self.graph.layer(
            name=node.name + "_mean",
            op="ReduceMean",
            inputs=[X, axes_const],
            outputs=[gs.Variable(name=node.name + "_mean_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]
        
        # 2. Subtract mean: X_diff = Sub(X, mean_expanded)
        mean_expanded = expand_tensor(mean, "mean")
        x_diff = self.graph.layer(
            name=node.name + "_sub_mean",
            op="Sub",
            inputs=[X, mean_expanded],
            outputs=[gs.Variable(name=node.name + "_diff", dtype=X.dtype)]
        )[0]
        
        # 3. Compute variance: var = ReduceMean((X - mean)^2, axes) using Pow
        # Ensure we use a valid NumPy dtype matching X.dtype
        if isinstance(X.dtype, int):
            try:
                import onnx.helper
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(X.dtype)
            except Exception:
                np_dtype = np.float32
        else:
            np_dtype = X.dtype if X.dtype is not None else np.float32

        pow_exp = gs.Constant(
            name=node.name + "_pow_exp",
            values=np.array(2.0, dtype=np_dtype)
        )
        x_diff_sq = self.graph.layer(
            name=node.name + "_diff_sq",
            op="Pow",
            inputs=[x_diff, pow_exp],
            outputs=[gs.Variable(name=node.name + "_diff_sq_val", dtype=X.dtype)]
        )[0]
        
        var = self.graph.layer(
            name=node.name + "_var",
            op="ReduceMean",
            inputs=[x_diff_sq, axes_const],
            outputs=[gs.Variable(name=node.name + "_var_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]
        
        # 4. Standard deviation: stddev = Sqrt(var + eps)
        eps_const = gs.Constant(
            name=node.name + "_eps",
            values=np.array(epsilon, dtype=np_dtype),
        )
        
        var_eps = self.graph.layer(
            name=node.name + "_var_eps",
            op="Add",
            inputs=[var, eps_const],
            outputs=[gs.Variable(name=node.name + "_var_eps_val", dtype=X.dtype)]
        )[0]
        
        stddev = self.graph.layer(
            name=node.name + "_stddev",
            op="Sqrt",
            inputs=[var_eps],
            outputs=[gs.Variable(name=node.name + "_stddev_val", dtype=X.dtype)]
        )[0]
        
        # 5. Normalize: X_norm = Div(X_diff, stddev_expanded)
        stddev_expanded = expand_tensor(stddev, "stddev")
        
        x_norm = self.graph.layer(
            name=node.name + "_div_stddev",
            op="Div",
            inputs=[x_diff, stddev_expanded],
            outputs=[gs.Variable(name=node.name + "_norm", dtype=X.dtype)]
        )[0]
        
        # 6. Apply scale and optional bias
        scale_expanded = expand_tensor(scale, "scale")
        if bias is not None:
            bias_expanded = expand_tensor(bias, "bias")
            x_scaled = self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[gs.Variable(name=node.name + "_scaled", dtype=X.dtype)]
            )[0]
            self.graph.layer(
                name=node.name + "_add_bias",
                op="Add",
                inputs=[x_scaled, bias_expanded],
                outputs=[Y]
            )
        else:
            self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[Y]
            )
            
        # Rewire other optional outputs if they are used by any consumer
        if len(node.outputs) > 1 and len(node.outputs[1].outputs) > 0:
            rewire_consumers(node.outputs[1].outputs, node.outputs[1], mean)
        if len(node.outputs) > 2 and len(node.outputs[2].outputs) > 0:
            inv_std_dev = self.graph.layer(
                name=node.name + "_inv_stddev",
                op="Reciprocal",
                inputs=[stddev],
                outputs=[gs.Variable(name=node.name + "_inv_stddev_val", dtype=X.dtype)]
            )[0]
            rewire_consumers(node.outputs[2].outputs, node.outputs[2], inv_std_dev)
            
        # Disconnect node
        node.inputs.clear()
        node.outputs.clear()
        
        self._logger.debug("Decomposed LayerNormalization node '%s'", node.name)


@dataclass
class DecomposeGelu(OnnxGraphEdit):
    """
    Decompose ONNX Gelu into basic arithmetic operations (Mul, Add, Erf).
    Formula: Gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "Gelu"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Gelu")
        X = node.inputs[0]
        Y = node.outputs[0]

        # Resolve correct numpy dtype matching X.dtype
        if isinstance(X.dtype, int):
            try:
                import onnx.helper
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(X.dtype)
            except Exception:
                np_dtype = np.float32
        else:
            np_dtype = X.dtype if X.dtype is not None else np.float32

        # Create constant values
        const_half = gs.Constant(
            name=node.name + "_gelu_half",
            values=np.array(0.5, dtype=np_dtype)
        )
        const_one = gs.Constant(
            name=node.name + "_gelu_one",
            values=np.array(1.0, dtype=np_dtype)
        )
        const_inv_sqrt2 = gs.Constant(
            name=node.name + "_gelu_inv_sqrt2",
            values=np.array(1.0 / np.sqrt(2.0), dtype=np_dtype)
        )

        # 1. Mul: x_scaled = Mul(X, 1 / sqrt(2))
        x_scaled = self.graph.layer(
            name=node.name + "_scale",
            op="Mul",
            inputs=[X, const_inv_sqrt2],
            outputs=[gs.Variable(name=node.name + "_scaled_val", dtype=X.dtype)]
        )[0]

        # 2. Erf: erf_val = Erf(x_scaled)
        erf_val = self.graph.layer(
            name=node.name + "_erf",
            op="Erf",
            inputs=[x_scaled],
            outputs=[gs.Variable(name=node.name + "_erf_val", dtype=X.dtype)]
        )[0]

        # 3. Add: erf_plus_1 = Add(erf_val, 1)
        erf_plus_1 = self.graph.layer(
            name=node.name + "_add_one",
            op="Add",
            inputs=[erf_val, const_one],
            outputs=[gs.Variable(name=node.name + "_plus_one_val", dtype=X.dtype)]
        )[0]

        # 4. Mul: x_half = Mul(X, 0.5)
        x_half = self.graph.layer(
            name=node.name + "_half",
            op="Mul",
            inputs=[X, const_half],
            outputs=[gs.Variable(name=node.name + "_half_val", dtype=X.dtype)]
        )[0]

        # 5. Mul: Y = Mul(x_half, erf_plus_1)
        self.graph.layer(
            name=node.name + "_mul_final",
            op="Mul",
            inputs=[x_half, erf_plus_1],
            outputs=[Y]
        )

        # Disconnect node
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug("Decomposed Gelu node '%s'", node.name)


@dataclass
class DecomposeBooleanAnd(OnnxGraphEdit):
    """
    Decompose ONNX And on boolean (i1) tensors into Cast(int8) + Mul + Cast(bool).

    Hardware DMA engines (e.g. torq_api `_parse_mem_bdg_ldims`) assert that every
    dimension count n[i] > 0.  Bit-packed i1 tensors cause this assertion to fire
    because the byte-count for a small boolean dimension rounds to 0.  Casting
    inputs through int8 gives the hardware byte-aligned, element-per-byte data.

    `Mul(int8, int8)` is semantically equivalent to `And(bool, bool)` for 0/1
    values, so the output is cast back to bool to preserve downstream type
    expectations.
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "And":
            return False
        # Match whenever at least one input is boolean.
        return any(
            isinstance(inp, (gs.Variable, gs.Constant)) and inp.dtype == np.bool_
            for inp in node.inputs
        )

    def transform(self, node: gs.Node):
        self._check_node_op(node, "And")

        # Cast each input i1 -> int8
        inputs_int8 = []
        for i, inp in enumerate(node.inputs):
            cast_out = self.graph.layer(
                name=f"{node.name}_inp{i}_to_int8",
                op="Cast",
                inputs=[inp],
                outputs=[gs.Variable(
                    name=f"{node.name}_inp{i}_int8",
                    dtype=np.int8,
                    shape=inp.shape,
                )],
                attrs={"to": onnx.TensorProto.INT8},
            )[0]
            inputs_int8.append(cast_out)

        # Mul(int8, int8) is And for 0/1 values; supports implicit broadcasting.
        orig_output = node.outputs[0]
        mul_out = self.graph.layer(
            name=f"{node.name}_mul_int8",
            op="Mul",
            inputs=inputs_int8,
            outputs=[gs.Variable(
                name=f"{node.name}_mul_int8_out",
                dtype=np.int8,
                shape=orig_output.shape,
            )],
        )[0]

        # Cast result back to bool; reuse orig_output variable so consumers update
        # automatically.
        self.graph.layer(
            name=f"{node.name}_to_bool",
            op="Cast",
            inputs=[mul_out],
            outputs=[orig_output],
            attrs={"to": onnx.TensorProto.BOOL},
        )

        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug("Decomposed boolean And node '%s' into Cast+Mul+Cast", node.name)
