# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import OnnxGraphEdit, OnnxGraphEditor, FixedDimMapping, DimMatchType
from ...graph_edit.edits import CommonGraphEditsMixin, CombineKVCacheMixin
from ...graph_edit.edits.transformer import MaskFutureAttentionScores
from ...graph_edit.edits.arithmetic import DecomposeLayerNormalization


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
    # This shadows CommonGraphEditsMixin so the shared edit is instantiated with
    # its opt-in shape-based fallback matching: dynamo doesn't preserve
    # hierarchical node names, so the name-based primary match the shared
    # (non-streaming) class relies on never fires here — confirmed empirically
    # against real decoder exports; see duplicated.md. ``replace_dynamic_kv_cache``
    # and ``decompose_strided_conv1d`` are intentionally NOT overridden — the
    # shared edits are functionally identical here and are inherited from the
    # mixin.

    def mask_future_attn_scores(self, cur_len, max_tokens):
        self.apply_edit(
            MaskFutureAttentionScores(
                self._graph, self._graph_name, cur_len, max_tokens, self._export_dtype,
                match_shape_fallback=True,
            )
        )
        return self

    # New decompositions (model-local edits)

    def decompose_layer_normalization(self, enabled: bool = True):
        """Apply the shared LayerNormalization decomposition.

        ``enabled`` is a temporary workaround switch — flip to False once the
        target compiler natively supports LayerNormalization, without having
        to remove call sites.
        """
        self.apply_edit(
            DecomposeLayerNormalization(self._graph, self._graph_name, enabled=enabled)
        )
        return self

    def decompose_gelu(self):
        self.apply_edit(DecomposeGelu(self._graph, self._graph_name))
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
