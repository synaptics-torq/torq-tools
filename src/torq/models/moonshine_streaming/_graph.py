# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import (
    OnnxGraphEdit,
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
    rewire_consumers,
)
from ...graph_edit.edits import *

# Re-use the excellent graph surgeon edits from the standard Moonshine implementation
from ..moonshine._graph import (
    MoveOutputFromConcat,
    ReplaceInt64FloatCast,
    ReplacePadWithConcat,
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

    def move_output_from_concat(
        self,
        pad_len: int
    ):
        self.apply_edit(MoveOutputFromConcat(self._graph, self._graph_name, pad_len))
        return self

    def replace_int64_float_cast(
        self,
        max_int: int
    ):
        self.apply_edit(ReplaceInt64FloatCast(self._graph, self._graph_name, max_int))
        return self
    
    def replace_pad_with_concat(
        self,
    ):
        self.apply_edit(ReplacePadWithConcat(self._graph, self._graph_name))
        return self

    def decompose_strided_conv1d(self):
        self.apply_edit(DecomposeStridedConv1D(self._graph, self._graph_name))
        return self

    # def decompose_layer_normalization(self):
    #     self.apply_edit(DecomposeLayerNormalization(self._graph, self._graph_name))
    #     return self

    # def decompose_boolean_and(self):
    #     self.apply_edit(DecomposeBooleanAnd(self._graph, self._graph_name))
    #     return self

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


class DecomposeStridedConv1D(OnnxGraphEdit):
    """
    Decompose strided 1D convolutions into im2col unfold + MatMul.

    Handles two cases:

    1. Single-channel with Unsqueeze predecessor and kernel = 2*stride - 1:
       Uses an efficient reshape + slice trick to build sliding windows.

    2. General multi-channel Conv1D:
       Uses per-kernel-position strided slices to build the im2col matrix.

    Both produce: im2col patches @ weight.T [+ bias] -> Transpose to channel-first.
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "Conv":
            return False
        kernel_shape = node.attrs.get("kernel_shape", [])
        strides = node.attrs.get("strides", [1])
        group = node.attrs.get("group", 1)
        pads = node.attrs.get("pads", [0, 0])
        if len(kernel_shape) != 1 or len(strides) != 1:
            return False
        if strides[0] <= 1:
            return False
        if group != 1:
            return False
        if any(p != 0 for p in pads):
            return False
        weight = node.inputs[1]
        w_shape = weight.shape
        if w_shape is None or len(w_shape) != 3:
            return False
        inp = node.inputs[0]
        if inp.shape is None or not isinstance(inp.shape[-1], (int, np.integer)):
            return False
        return True

    def _is_single_channel_special_case(self, node: gs.Node) -> bool:
        k = node.attrs["kernel_shape"][0]
        s = node.attrs["strides"][0]
        weight = node.inputs[1]
        if weight.shape[1] != 1 or k != 2 * s - 1:
            return False
        conv_input = node.inputs[0]
        if conv_input.inputs and len(conv_input.inputs) == 1:
            producer = conv_input.inputs[0]
            if producer.op == "Unsqueeze":
                return True
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Conv")
        if self._is_single_channel_special_case(node):
            self._transform_single_channel(node)
        else:
            self._transform_general(node)

    def _transform_single_channel(self, node: gs.Node):
        """Efficient decomposition for single-channel Conv1D with k = 2*stride - 1."""
        k = node.attrs["kernel_shape"][0]
        s = node.attrs["strides"][0]
        weight = node.inputs[1]
        out_ch = weight.shape[0]
        overlap = k - s  # = s - 1

        conv_input: gs.Variable = node.inputs[0]
        conv_output: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(conv_output.outputs)

        unsqueeze_node: gs.Node = conv_input.inputs[0]
        raw_input = unsqueeze_node.inputs[0]

        raw_shape = raw_input.shape
        batch, n_samples = raw_shape

        n_chunks = n_samples // s
        n_windows = n_chunks - 1  # number of output positions
        expected_out = conv_output.shape
        if expected_out is not None and expected_out[-1] != n_windows:
            self._logger.warning(
                "Conv node '%s': expected %d output positions but output shape says %d",
                node.name, n_windows, expected_out[-1]
            )

        prefix = node.name or "decomposed_conv1d"

        # Reshape [batch, n_samples] -> [batch, n_chunks, stride]
        reshape_shape = gs.Constant(
            f"{prefix}_reshape_shape",
            np.array([batch, n_chunks, s], dtype=np.int64)
        )
        reshaped = self.graph.layer(
            name=f"{prefix}_reshape_input",
            op="Reshape",
            inputs=[raw_input, reshape_shape],
            outputs=[gs.Variable(
                f"{prefix}_reshaped", dtype=raw_input.dtype,
                shape=[batch, n_chunks, s]
            )],
            attrs={"allowzero": 0}
        )[0]

        # Slice left chunks [0:n_windows] along axis=1
        sl_starts_0 = gs.Constant(f"{prefix}_starts_0", np.array([0], dtype=np.int64))
        sl_ends_nw = gs.Constant(f"{prefix}_ends_{n_windows}", np.array([n_windows], dtype=np.int64))
        sl_axes_1 = gs.Constant(f"{prefix}_axes_1", np.array([1], dtype=np.int64))
        left = self.graph.layer(
            name=f"{prefix}_slice_left",
            op="Slice",
            inputs=[reshaped, sl_starts_0, sl_ends_nw, sl_axes_1],
            outputs=[gs.Variable(
                f"{prefix}_left", dtype=raw_input.dtype,
                shape=[batch, n_windows, s]
            )]
        )[0]

        # Slice right chunks [1:n_chunks] along axis=1
        sl_starts_1 = gs.Constant(f"{prefix}_starts_1", np.array([1], dtype=np.int64))
        sl_ends_nc = gs.Constant(f"{prefix}_ends_{n_chunks}", np.array([n_chunks], dtype=np.int64))
        right = self.graph.layer(
            name=f"{prefix}_slice_right",
            op="Slice",
            inputs=[reshaped, sl_starts_1, sl_ends_nc, sl_axes_1],
            outputs=[gs.Variable(
                f"{prefix}_right", dtype=raw_input.dtype,
                shape=[batch, n_windows, s]
            )]
        )[0]

        # Trim right to overlap along axis=2
        sl_ends_ov = gs.Constant(f"{prefix}_ends_{overlap}", np.array([overlap], dtype=np.int64))
        sl_axes_2 = gs.Constant(f"{prefix}_axes_2", np.array([2], dtype=np.int64))
        right_trimmed = self.graph.layer(
            name=f"{prefix}_slice_right_trim",
            op="Slice",
            inputs=[right, sl_starts_0, sl_ends_ov, sl_axes_2],
            outputs=[gs.Variable(
                f"{prefix}_right_trimmed", dtype=raw_input.dtype,
                shape=[batch, n_windows, overlap]
            )]
        )[0]

        # Concat [left, right_trimmed] along axis=2 -> [batch, n_windows, kernel]
        windows = self.graph.layer(
            name=f"{prefix}_im2col_concat",
            op="Concat",
            inputs=[left, right_trimmed],
            outputs=[gs.Variable(
                f"{prefix}_windows", dtype=raw_input.dtype,
                shape=[batch, n_windows, k]
            )],
            attrs={"axis": 2}
        )[0]

        # Reshape weight [out_ch, 1, kernel] -> [out_ch, kernel]
        w_reshape_shape = gs.Constant(
            f"{prefix}_w_reshape_shape",
            np.array([out_ch, k], dtype=np.int64)
        )
        w_2d = self.graph.layer(
            name=f"{prefix}_reshape_weight",
            op="Reshape",
            inputs=[weight, w_reshape_shape],
            outputs=[gs.Variable(
                f"{prefix}_w_2d", dtype=weight.dtype,
                shape=[out_ch, k]
            )],
            attrs={"allowzero": 0}
        )[0]

        # Transpose weight [out_ch, kernel] -> [kernel, out_ch]
        w_t = self.graph.layer(
            name=f"{prefix}_transpose_weight",
            op="Transpose",
            inputs=[w_2d],
            outputs=[gs.Variable(
                f"{prefix}_w_t", dtype=weight.dtype,
                shape=[k, out_ch]
            )],
            attrs={"perm": [1, 0]}
        )[0]

        # MatMul [batch, n_windows, kernel] x [kernel, out_ch] -> [batch, n_windows, out_ch]
        mm_out = self.graph.layer(
            name=f"{prefix}_matmul",
            op="MatMul",
            inputs=[windows, w_t],
            outputs=[gs.Variable(
                f"{prefix}_mm", dtype=raw_input.dtype,
                shape=[batch, n_windows, out_ch]
            )]
        )[0]

        # Transpose [batch, n_windows, out_ch] -> [batch, out_ch, n_windows]
        final = self.graph.layer(
            name=f"{prefix}_transpose_out",
            op="Transpose",
            inputs=[mm_out],
            outputs=[gs.Variable(
                f"{prefix}_out", dtype=conv_output.dtype,
                shape=[batch, out_ch, n_windows]
            )],
            attrs={"perm": [0, 2, 1]}
        )[0]

        rewire_consumers(consumers, conv_output, final)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is conv_output:
                self.graph.outputs[i] = final

        node.inputs.clear()
        node.outputs.clear()
        unsqueeze_node.inputs.clear()
        unsqueeze_node.outputs.clear()

        self._logger.debug(
            "Decomposed single-channel Conv1D '%s' (kernel=%d, stride=%d) into im2col + MatMul",
            node.name, k, s
        )

    def _transform_general(self, node: gs.Node):
        """General decomposition for multi-channel Conv1D using strided slices."""
        k = node.attrs["kernel_shape"][0]
        s = node.attrs["strides"][0]
        weight = node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        out_ch, c_in = weight.shape[0], weight.shape[1]

        conv_input = node.inputs[0]
        conv_output = node.outputs[0]
        consumers = list(conv_output.outputs)

        batch = conv_input.shape[0]
        l_in = int(conv_input.shape[2])
        l_out = (l_in - k) // s + 1
        prefix = node.name or "decomposed_conv1d"

        # im2col via strided slices: for each kernel position j,
        # extract input[:, :, j : j + l_out*s : s] -> [batch, c_in, l_out]
        sl_axes = gs.Constant(f"{prefix}_axes_2", np.array([2], dtype=np.int64))
        sl_steps = gs.Constant(f"{prefix}_steps_{s}", np.array([s], dtype=np.int64))
        unsq_axes = gs.Constant(f"{prefix}_unsq_axes_3", np.array([3], dtype=np.int64))

        slices = []
        for j in range(k):
            sl_start = gs.Constant(f"{prefix}_start_{j}", np.array([j], dtype=np.int64))
            sl_end = gs.Constant(f"{prefix}_end_{j}", np.array([j + l_out * s], dtype=np.int64))
            slice_j = self.graph.layer(
                name=f"{prefix}_slice_pos{j}",
                op="Slice",
                inputs=[conv_input, sl_start, sl_end, sl_axes, sl_steps],
                outputs=[gs.Variable(
                    f"{prefix}_pos{j}", dtype=conv_input.dtype,
                    shape=[batch, c_in, l_out]
                )]
            )[0]
            unsq_j = self.graph.layer(
                name=f"{prefix}_unsq_pos{j}",
                op="Unsqueeze",
                inputs=[slice_j, unsq_axes],
                outputs=[gs.Variable(
                    f"{prefix}_pos{j}_4d", dtype=conv_input.dtype,
                    shape=[batch, c_in, l_out, 1]
                )]
            )[0]
            slices.append(unsq_j)

        # Concat along axis=3: [batch, c_in, l_out, k]
        patches = self.graph.layer(
            name=f"{prefix}_im2col_concat",
            op="Concat",
            inputs=slices,
            outputs=[gs.Variable(
                f"{prefix}_patches", dtype=conv_input.dtype,
                shape=[batch, c_in, l_out, k]
            )],
            attrs={"axis": 3}
        )[0]

        # Transpose to [batch, l_out, c_in, k]
        patches_t = self.graph.layer(
            name=f"{prefix}_transpose_patches",
            op="Transpose",
            inputs=[patches],
            outputs=[gs.Variable(
                f"{prefix}_patches_t", dtype=conv_input.dtype,
                shape=[batch, l_out, c_in, k]
            )],
            attrs={"perm": [0, 2, 1, 3]}
        )[0]

        # Reshape to [batch, l_out, c_in * k]
        flat_shape = gs.Constant(
            f"{prefix}_flat_shape",
            np.array([batch, l_out, c_in * k], dtype=np.int64)
        )
        patches_flat = self.graph.layer(
            name=f"{prefix}_reshape_patches",
            op="Reshape",
            inputs=[patches_t, flat_shape],
            outputs=[gs.Variable(
                f"{prefix}_patches_flat", dtype=conv_input.dtype,
                shape=[batch, l_out, c_in * k]
            )],
            attrs={"allowzero": 0}
        )[0]

        # Reshape weight [out_ch, c_in, k] -> [out_ch, c_in * k]
        w_flat_shape = gs.Constant(
            f"{prefix}_w_flat_shape",
            np.array([out_ch, c_in * k], dtype=np.int64)
        )
        w_flat = self.graph.layer(
            name=f"{prefix}_reshape_weight",
            op="Reshape",
            inputs=[weight, w_flat_shape],
            outputs=[gs.Variable(
                f"{prefix}_w_flat", dtype=weight.dtype,
                shape=[out_ch, c_in * k]
            )],
            attrs={"allowzero": 0}
        )[0]

        # Transpose weight [out_ch, c_in*k] -> [c_in*k, out_ch]
        w_t = self.graph.layer(
            name=f"{prefix}_transpose_weight",
            op="Transpose",
            inputs=[w_flat],
            outputs=[gs.Variable(
                f"{prefix}_w_t", dtype=weight.dtype,
                shape=[c_in * k, out_ch]
            )],
            attrs={"perm": [1, 0]}
        )[0]

        # MatMul [batch, l_out, c_in*k] x [c_in*k, out_ch] -> [batch, l_out, out_ch]
        mm_out = self.graph.layer(
            name=f"{prefix}_matmul",
            op="MatMul",
            inputs=[patches_flat, w_t],
            outputs=[gs.Variable(
                f"{prefix}_mm", dtype=conv_input.dtype,
                shape=[batch, l_out, out_ch]
            )]
        )[0]

        # Add bias if present
        if bias is not None:
            mm_out = self.graph.layer(
                name=f"{prefix}_add_bias",
                op="Add",
                inputs=[mm_out, bias],
                outputs=[gs.Variable(
                    f"{prefix}_biased", dtype=conv_input.dtype,
                    shape=[batch, l_out, out_ch]
                )]
            )[0]

        # Transpose [batch, l_out, out_ch] -> [batch, out_ch, l_out]
        final = self.graph.layer(
            name=f"{prefix}_transpose_out",
            op="Transpose",
            inputs=[mm_out],
            outputs=[gs.Variable(
                f"{prefix}_out", dtype=conv_output.dtype,
                shape=[batch, out_ch, l_out]
            )],
            attrs={"perm": [0, 2, 1]}
        )[0]

        rewire_consumers(consumers, conv_output, final)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is conv_output:
                self.graph.outputs[i] = final

        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "Decomposed Conv1D '%s' (kernel=%d, stride=%d, in_ch=%d) into im2col + MatMul",
            node.name, k, s, c_in
        )


