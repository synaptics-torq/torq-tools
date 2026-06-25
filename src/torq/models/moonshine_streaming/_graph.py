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
    DecomposeLayerNormalizationMulReciprocal,
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

    def fix_fused_encoder_io(
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

    def fix_decoder_kv_io(
        self,
        dec_seq_len: int,
        enc_seq_len: int,
        batch_dim: str = "batch",
        dec_seq_dim: str = "dec_seq",
        enc_seq_dim: str = "enc_seq",
        past_seq_dim: str = "past_seq",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(dec_seq_dim, DimMatchType.EXACT, dec_seq_len),
            FixedDimMapping(enc_seq_dim, DimMatchType.EXACT, enc_seq_len),
            FixedDimMapping(past_seq_dim, DimMatchType.CONTAINS, dec_seq_len),
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

    def fix_preprocessor_io(
        self,
        num_samples: int,
        batch_dim: str = "batch",
        audio_len_dim: str = "audio_length",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(audio_len_dim, DimMatchType.EXACT, num_samples),
        ]
        self.fix_io_dims(to_fix)

    def fix_encoder_io(
        self,
        enc_seq_len: int,
        batch_dim: str = "batch",
        seq_len_dim: str = "seq_length",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, enc_seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_decoder_io(
        self,
        enc_seq_len: int,
        dec_seq_len: int,
        with_past: bool,
        batch_dim: str = "batch",
        enc_seq_dim: str = "enc_seq",
        past_seq_dim: str = "past_seq",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(enc_seq_dim, DimMatchType.EXACT, enc_seq_len),
        ]
        if with_past:
            # past_seq and past_seq + 1 are handled via DimMatchType.CONTAINS
            to_fix.append(FixedDimMapping(past_seq_dim, DimMatchType.CONTAINS, dec_seq_len))
        self.fix_io_dims(to_fix)

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

    # ── Streaming-specialised overrides ──────────────────────────────────────
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

    # ── New decompositions (model-local edits) ───────────────────────────────

    def decompose_layer_normalization(self):
        dim_map = getattr(self, "_dim_map", None)
        self.apply_edit(DecomposeLayerNormalization(self._graph, self._graph_name, dim_map=dim_map))
        return self

    def decompose_layer_normalization_mul_reciprocal(self):
        dim_map = getattr(self, "_dim_map", None)
        self.apply_edit(DecomposeLayerNormalizationMulReciprocal(self._graph, self._graph_name, dim_map=dim_map))
        return self

    def decompose_gelu(self):
        self.apply_edit(DecomposeGelu(self._graph, self._graph_name))
        return self

    def decompose_boolean_and(self):
        self.apply_edit(DecomposeBooleanAnd(self._graph, self._graph_name))
        return self

    def remove_identity_gather_nd(self):
        import numpy as np
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
    
    def decompose_reduce_sum(self):
        """Find ReduceSum nodes over the last dimension of a 2D tensor and replace with MatMul + Reshape.
        
        Performs computation in float32 by casting input to float32 first.
        To avoid hardware accelerator register constraints where the contraction dimension K of MatMul
        cannot exceed 65536 (1 << 16), it splits the input tensor along the sequence dimension into two
        halves (each size N/2 <= 40000), runs two smaller MatMuls, and sums the results.
        Finally, it reshapes and casts the result back to the original integer type.
        """
        import numpy as np
        reduce_sums = [node for node in self._graph.nodes if node.op == "ReduceSum"]
        for node in reduce_sums:
            if len(node.inputs) == 2:
                inp = node.inputs[0]
                axes_const = node.inputs[1]
                out = node.outputs[0]
                
                if isinstance(axes_const, gs.Constant) and inp.shape is not None:
                    axes = axes_const.values.flatten().tolist()
                    if len(inp.shape) == 2 and axes in ([1], [-1]):
                        N = inp.shape[1]
                        
                        def to_proto(np_dtype):
                            if np_dtype == np.int32: return onnx.TensorProto.INT32
                            if np_dtype == np.int64: return onnx.TensorProto.INT64
                            return int(onnx.helper.np_dtype_to_tensor_dtype(np_dtype))

                        # Cast input to FLOAT32
                        inp_float = gs.Variable(
                            name=node.name + "_inp_float",
                            dtype=np.float32,
                            shape=inp.shape
                        )
                        cast_inp_node = gs.Node(
                            op="Cast",
                            name=node.name + "_cast_inp_node",
                            inputs=[inp],
                            outputs=[inp_float],
                            attrs={"to": onnx.TensorProto.FLOAT}
                        )
                        
                        # Split along axis 1 into two halves
                        N1 = N // 2
                        N2 = N - N1
                        
                        part1 = gs.Variable(
                            name=node.name + "_part1",
                            dtype=np.float32,
                            shape=[1, N1]
                        )
                        part2 = gs.Variable(
                            name=node.name + "_part2",
                            dtype=np.float32,
                            shape=[1, N2]
                        )
                        
                        split_node = gs.Node(
                            op="Split",
                            name=node.name + "_split_node",
                            inputs=[inp_float],
                            outputs=[part1, part2],
                            attrs={"axis": 1, "num_outputs": 2}
                        )
                        
                        # Create ones float32 constants for each half
                        ones1 = gs.Constant(
                            name=node.name + "_ones1",
                            values=np.ones((N1, 1), dtype=np.float32)
                        )
                        ones2 = gs.Constant(
                            name=node.name + "_ones2",
                            values=np.ones((N2, 1), dtype=np.float32)
                        )
                        
                        # MatMuls on float32
                        matmul1_out = gs.Variable(
                            name=node.name + "_matmul1_out",
                            dtype=np.float32,
                            shape=[1, 1]
                        )
                        matmul1_node = gs.Node(
                            op="MatMul",
                            name=node.name + "_matmul1_node",
                            inputs=[part1, ones1],
                            outputs=[matmul1_out]
                        )
                        
                        matmul2_out = gs.Variable(
                            name=node.name + "_matmul2_out",
                            dtype=np.float32,
                            shape=[1, 1]
                        )
                        matmul2_node = gs.Node(
                            op="MatMul",
                            name=node.name + "_matmul2_node",
                            inputs=[part2, ones2],
                            outputs=[matmul2_out]
                        )
                        
                        # Add outputs together
                        add_out = gs.Variable(
                            name=node.name + "_add_out",
                            dtype=np.float32,
                            shape=[1, 1]
                        )
                        add_node = gs.Node(
                            op="Add",
                            name=node.name + "_add_node",
                            inputs=[matmul1_out, matmul2_out],
                            outputs=[add_out]
                        )
                        
                        # Reshape sum to [1]
                        reshape_shape = gs.Constant(
                            name=node.name + "_reshape_shape",
                            values=np.array([1], dtype=np.int64)
                        )
                        reshape_out = gs.Variable(
                            name=node.name + "_reshape_out",
                            dtype=np.float32,
                            shape=[1]
                        )
                        reshape_node = gs.Node(
                            op="Reshape",
                            name=node.name + "_reshape_node",
                            inputs=[add_out, reshape_shape],
                            outputs=[reshape_out]
                        )
                        
                        # Cast back to original integer type
                        to_type = to_proto(out.dtype)
                        cast_out_node = gs.Node(
                            op="Cast",
                            name=node.name + "_cast_out_node",
                            inputs=[reshape_out],
                            outputs=[out],
                            attrs={"to": to_type}
                        )
                        
                        self._graph.nodes.extend([
                            cast_inp_node, split_node,
                            matmul1_node, matmul2_node,
                            add_node, reshape_node, cast_out_node
                        ])
                        node.inputs.clear()
                        node.outputs.clear()
                        
        self._graph.cleanup().toposort()
        return self

    def decompose_asinh(self):
        """Find Asinh nodes and decompose them into Log, Sqrt, Mul, and Add nodes."""
        import numpy as np
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
