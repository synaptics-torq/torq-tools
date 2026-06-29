# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import OnnxGraphEditor, FixedDimMapping, DimMatchType
from ...graph_edit.edits import CommonGraphEditsMixin, CombineKVCacheMixin
from .edits import (
    ReplaceDynamicKVCache,
    MaskFutureAttentionScores,
    AddCurrLenInput,
    DecomposeLayerNormalization,
    DecomposeGelu,
    DecomposeBooleanAnd,
)


class MoonshineStreamingOnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):

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
    ) -> "MoonshineStreamingOnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            component,
            export_dtype
        )

    def fix_io_dims(self, to_fix: list[FixedDimMapping] | None = None):
        """Fix dynamic I/O dims, additionally recording a ``name -> value`` map.

        The shared ``OnnxGraphEditor.fix_io_dims`` does not expose this map, but the
        streaming-local ``DecomposeLayerNormalization`` edits need it to resolve the
        normalized (symbolic) dimension to a concrete size.
        """
        to_fix = list(to_fix or [])
        self._dim_map = {m.match_name: m.value for m in to_fix}
        self._dim_map["batch"] = 1
        super().fix_io_dims(to_fix)

    def fix_encoder_io(
        self,
        chunk_len: int,
        feat_len: int,
        batch_dim: str = "batch",
        chunk_len_dim: str = "chunk_len",
        stable_dim: str = "t_stable",
    ):
        """Fix all dynamic dims on the fused encoder (frontend + encoder + adapter + cross_kv)."""
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(chunk_len_dim, DimMatchType.EXACT, chunk_len),
            FixedDimMapping(stable_dim, DimMatchType.EXACT, feat_len),
            # feat_len may also appear as a CONTAINS match in output names
            FixedDimMapping("features_dim_1", DimMatchType.CONTAINS, feat_len),
        ]
        self.fix_io_dims(to_fix)

    def make_decoder_static(self, max_tokens: int):
        """Convert the dynamic decoder to a static pre-allocated KV-cache decoder."""
        cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
        self._graph.inputs.append(cur_len_2d)
        squeeze_axes = gs.Constant("squeeze_axes_cur_len", np.array([0], dtype=np.int64))
        cur_len = self._graph.layer(
            name="current_len_to_1d",
            op="Squeeze",
            inputs=[cur_len_2d, squeeze_axes],
            outputs=[gs.Variable("current_len_squeezed", dtype=np.int64, shape=[1])],
        )[0]

        (
            self
            .replace_dynamic_kv_cache(cur_len, max_tokens)
            .mask_future_attn_scores(cur_len, max_tokens)
            .add_curr_len_input(cur_len)
        )

    def clear_intermediate_shapes(self):
        """Clear shape annotations on all non-input tensors so shape inference recomputes them."""
        graph_inputs = {t.name for t in self._graph.inputs}
        for node in self._graph.nodes:
            for tensor in node.outputs:
                if isinstance(tensor, gs.Variable) and tensor.name not in graph_inputs:
                    tensor.shape = None
        for t in self._graph.outputs:
            if isinstance(t, gs.Variable):
                t.shape = None
        return self

    # Streaming-specialised overrides
    # These shadow CommonGraphEditsMixin so the local divergent edit variants
    # (tuned for the dynamo stacked-cache decoder export) are used instead of the
    # shared ones. ``decompose_strided_conv1d`` is intentionally NOT overridden —
    # the shared edit is functionally identical and is inherited from the mixin.

    def replace_dynamic_kv_cache(self, cur_len, max_tokens):
        self.apply_edit(ReplaceDynamicKVCache(self._graph, self._graph_name, cur_len, max_tokens))
        return self

    def mask_future_attn_scores(self, cur_len, max_tokens):
        self.apply_edit(MaskFutureAttentionScores(self._graph, self._graph_name, cur_len, max_tokens, self._export_dtype))
        return self

    def add_curr_len_input(self, cur_len):
        self.apply_edit(AddCurrLenInput(self._graph, self._graph_name, cur_len))
        return self

    # New decompositions (model-local edits)

    def decompose_layer_normalization(self):
        dim_map = getattr(self, "_dim_map", None)
        self.apply_edit(DecomposeLayerNormalization(self._graph, self._graph_name, dim_map=dim_map))
        return self

    def decompose_gelu(self):
        self.apply_edit(DecomposeGelu(self._graph, self._graph_name))
        return self

    def decompose_boolean_and(self):
        self.apply_edit(DecomposeBooleanAnd(self._graph, self._graph_name))
        return self

    def remove_identity_gather_nd(self):
        gather_nd_nodes = [node for node in self._graph.nodes if node.op == "GatherND"]
        for node in gather_nd_nodes:
            if len(node.inputs) == 2:
                data_input = node.inputs[0]
                indices_input = node.inputs[1]
                if isinstance(indices_input, gs.Constant):
                    indices_arr = indices_input.values
                    if indices_arr.ndim == 3 and indices_arr.shape[0] == 1 and indices_arr.shape[2] == 2:
                        is_identity = True
                        for i in range(indices_arr.shape[1]):
                            if not np.array_equal(indices_arr[0, i], [0, i]):
                                is_identity = False
                                break
                        if is_identity:
                            output_var = node.outputs[0]
                            for consumer in list(output_var.outputs):
                                for idx, inp in enumerate(consumer.inputs):
                                    if inp is output_var:
                                        consumer.inputs[idx] = data_input
                            node.outputs.clear()
                            node.inputs.clear()
        self._graph.cleanup().toposort()
        return self

    def decompose_asinh(self):
        """Find Asinh nodes and decompose them into Log, Sqrt, Mul, and Add nodes."""
        asinh_nodes = [node for node in self._graph.nodes if node.op == "Asinh"]
        for node in asinh_nodes:
            if len(node.inputs) == 1:
                x = node.inputs[0]
                y = node.outputs[0]
                
                x_sq = gs.Variable(
                    name=node.name + "_x_sq",
                    dtype=x.dtype,
                    shape=x.shape
                )
                mul_node = gs.Node(
                    op="Mul",
                    name=node.name + "_x_sq_node",
                    inputs=[x, x],
                    outputs=[x_sq]
                )
                
                const_1 = gs.Constant(
                    name=node.name + "_const_1.0",
                    values=np.array(1.0, dtype=x.dtype)
                )
                
                x_sq_plus_1 = gs.Variable(
                    name=node.name + "_x_sq_plus_1",
                    dtype=x.dtype,
                    shape=x.shape
                )
                add_1_node = gs.Node(
                    op="Add",
                    name=node.name + "_x_sq_plus_1_node",
                    inputs=[x_sq, const_1],
                    outputs=[x_sq_plus_1]
                )
                
                sqrt_val = gs.Variable(
                    name=node.name + "_sqrt_val",
                    dtype=x.dtype,
                    shape=x.shape
                )
                sqrt_node = gs.Node(
                    op="Sqrt",
                    name=node.name + "_sqrt_node",
                    inputs=[x_sq_plus_1],
                    outputs=[sqrt_val]
                )
                
                sum_val = gs.Variable(
                    name=node.name + "_sum_val",
                    dtype=x.dtype,
                    shape=x.shape
                )
                add_sum_node = gs.Node(
                    op="Add",
                    name=node.name + "_add_sum_node",
                    inputs=[x, sqrt_val],
                    outputs=[sum_val]
                )
                
                log_node = gs.Node(
                    op="Log",
                    name=node.name + "_log_node",
                    inputs=[sum_val],
                    outputs=[y]
                )
                
                self._graph.nodes.extend([mul_node, add_1_node, sqrt_node, add_sum_node, log_node])
                
                node.inputs.clear()
                node.outputs.clear()
                
        self._graph.cleanup().toposort()
        return self
