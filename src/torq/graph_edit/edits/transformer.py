# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers


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
                gs.Variable(node.name + "_where",
                            dtype=node.inputs[0].dtype or self.export_dtype,
                            shape=mask_shape)
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
class RetargetCrossAttnKeyLayout(OnnxGraphEdit):
    """
    Remove the redundant cross-attention key-cache transpose round-trip
    shared between the encoder (cache producer) and decoder (cache consumer).

    The cross-attn key cache is produced (in the encoder, or in
    ``gen_encoder_cache`` when the cache is not folded) as
    ``Reshape -> Transpose(perm=[0,2,1,3])`` yielding ``[B, H, L, D]``, then
    re-transposed in the decoder via ``Transpose(perm=[0,1,3,2])`` to
    ``[B, H, D, L]`` for the Q.K^T score MatMul. Those two transposes compose
    to a single ``[B, L, H, D] -> [B, H, D, L]`` permutation, so the cache can
    simply be carried in ``[B, H, D, L]`` end to end.

    The edit infers its role from the graph it is applied to:

    * Producer side (the key tensor is a graph **output**): rewrite the
      feeding Transpose's perm ``[0,2,1,3] -> [0,2,3,1]`` so it emits
      ``[B, H, D, L]`` directly, and swap the last two dims of the output.
    * Consumer side (the key tensor is a graph **input**): drop the
      ``perm=[0,1,3,2]`` Transpose and swap the last two dims of the input so
      the cache feeds the score MatMul directly.

    Net effect: one Transpose removed per layer from the autoregressive
    decoder at no added cost in the encoder. Values are deliberately left in
    ``[B, H, L, D]`` (they feed the score.V MatMul as-is), so keys and values
    no longer share a layout -- this is incompatible with combined key/value
    cache I/O and must be gated on individual KV I/O.
    """

    key_tensor_re: str | re.Pattern = r"\.encoder\.key$"
    producer_perm: tuple[int, ...] = (0, 2, 1, 3)
    consumer_perm: tuple[int, ...] = (0, 1, 3, 2)
    retargeted_perm: tuple[int, ...] = (0, 2, 3, 1)

    def __post_init__(self):
        if isinstance(self.key_tensor_re, str):
            self.key_tensor_re = re.compile(self.key_tensor_re)
        self._output_names = {o.name for o in self.graph.outputs}
        self._input_names = {i.name for i in self.graph.inputs}
        super().__post_init__()
        self.requires_shape_inference = True

    @staticmethod
    def _static_perm(node: gs.Node) -> tuple[int, ...] | None:
        perm = node.attrs.get("perm", None)
        if perm is None:
            return None
        return tuple(int(p) for p in perm)

    def match(self, node: gs.Node) -> bool:
        if node.op != "Transpose" or not node.inputs or not node.outputs:
            return False
        perm = self._static_perm(node)
        if perm is None:
            return False
        out, inp = node.outputs[0], node.inputs[0]
        # Producer side: this Transpose feeds a graph output named *.encoder.key
        if (
            perm == self.producer_perm
            and out.name in self._output_names
            and self.key_tensor_re.search(out.name)
        ):
            inp_shape = getattr(inp, "shape", None)
            if inp_shape is None or not all(isinstance(d, (int, np.integer)) for d in inp_shape):
                self._logger.warning(
                    "Skipping cross-attn key producer '%s': non-static input shape %s",
                    node.name, inp_shape
                )
                return False
            return True
        # Consumer side: this Transpose reads a graph input named *.encoder.key
        if (
            perm == self.consumer_perm
            and inp.name in self._input_names
            and self.key_tensor_re.search(inp.name)
        ):
            # Only safe to drop if the input feeds this Transpose alone; any
            # other consumer would still expect the original [B, H, L, D].
            if len(inp.outputs) != 1:
                self._logger.warning(
                    "Skipping cross-attn key consumer '%s': input '%s' has %d consumers",
                    node.name, inp.name, len(inp.outputs)
                )
                return False
            inp_shape = getattr(inp, "shape", None)
            if inp_shape is None or not all(isinstance(d, (int, np.integer)) for d in inp_shape):
                self._logger.warning(
                    "Skipping cross-attn key consumer '%s': non-static input shape %s",
                    node.name, inp_shape
                )
                return False
            return True
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Transpose")
        if self._static_perm(node) == self.producer_perm:
            self._retarget_producer(node)
        else:
            self._drop_consumer(node)

    def _retarget_producer(self, node: gs.Node):
        out = node.outputs[0]
        inp_shape = [int(d) for d in node.inputs[0].shape]
        old_out_shape = list(out.shape) if out.shape is not None else None
        node.attrs["perm"] = list(self.retargeted_perm)
        out.shape = [inp_shape[p] for p in self.retargeted_perm]
        self._logger.debug(
            "Retargeted cross-attn key producer '%s': perm %s -> %s, output %s -> %s",
            node.name, list(self.producer_perm), list(self.retargeted_perm),
            old_out_shape, out.shape
        )

    def _drop_consumer(self, node: gs.Node):
        inp = node.inputs[0]
        out = node.outputs[0]
        old_inp_shape = [int(d) for d in inp.shape]
        consumers: list[gs.Node] = list(out.outputs)
        # The graph input now carries the already-transposed [B, H, D, L] layout.
        inp.shape = [old_inp_shape[p] for p in self.consumer_perm]
        rewire_consumers(consumers, out, inp)
        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug(
            "Dropped cross-attn key consumer Transpose '%s'; input '%s' %s -> %s",
            node.name, inp.name, old_inp_shape, inp.shape
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
