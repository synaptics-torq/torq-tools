# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers
from ._helpers import _const_array, _static_int_shape


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
class FoldConvBatchNorm(OnnxGraphEdit):
    """
    Fold ``Conv -> Mul(per-channel const) -> Add(per-channel const)`` into the Conv.

    Eval-mode BatchNorm after a conv typically exports as a per-output-channel
    Mul (scale) and Add (shift). Algebraically the three ops are one conv:

        Add(Mul(Conv(x, W), s), b) = Conv(x, W')  with
        W'[oc] = W[oc] * s[oc],  bias'[oc] = c[oc] * s[oc] + b[oc]

    (``c`` is the conv's existing bias, or zero.) Only fires when the conv
    weight (and bias, if present) are constants, the Mul/Add constants
    broadcast exactly onto the conv output's channel axis (axis 1 of the
    ``N C spatial...`` layout), and the Conv/Mul outputs have no other
    consumers. Residual Adds of two activations are left alone. Works for any
    spatial rank and grouped convs. Fold math runs in fp32, then casts back to
    the weight's dtype (``export_dtype`` is preserved).
    """

    @staticmethod
    def _values_f32(constant) -> np.ndarray:
        return np.asarray(constant.values).astype(np.float32)

    def _sole_consumer(self, tensor) -> gs.Node | None:
        if any(out is tensor for out in self.graph.outputs):
            return None
        consumers = list(tensor.outputs)
        return consumers[0] if len(consumers) == 1 else None

    def _per_channel_const(
        self, node: gs.Node, data_tensor, cout: int, out_rank: int
    ) -> np.ndarray | None:
        """Return the fp32 per-channel vector [cout] of the node's constant
        operand, or None when it does not broadcast onto the channel axis."""
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            return None
        others = [t for t in node.inputs if t is not data_tensor]
        if len(others) != 1:
            return None
        values = _const_array(others[0])
        if values is None:
            return None
        values = values.astype(np.float32)
        if values.size == 1:
            return np.full(cout, values.reshape(-1)[0], dtype=np.float32)
        if values.ndim > out_rank:
            return None
        # Right-aligned ONNX broadcasting: the constant is per-channel only
        # when its (padded) shape places cout on axis 1 and 1 everywhere else.
        padded = (1,) * (out_rank - values.ndim) + tuple(values.shape)
        if padded[1] != cout:
            return None
        if any(dim != 1 for i, dim in enumerate(padded) if i != 1):
            return None
        return values.reshape(cout)

    def _resolve(self, node: gs.Node):
        if node.op != "Conv" or len(node.inputs) < 2 or len(node.outputs) != 1:
            return None
        weight = node.inputs[1]
        if not isinstance(weight, gs.Constant) or len(weight.shape) < 3:
            return None
        if len(node.inputs) > 2 and not isinstance(node.inputs[2], gs.Constant):
            return None
        cout = int(weight.shape[0])
        out_rank = len(weight.shape)  # conv output rank equals weight rank

        mul = self._sole_consumer(node.outputs[0])
        if mul is None or mul.op != "Mul":
            return None
        scale = self._per_channel_const(mul, node.outputs[0], cout, out_rank)
        if scale is None:
            return None
        add = self._sole_consumer(mul.outputs[0])
        if add is None or add.op != "Add":
            return None
        shift = self._per_channel_const(add, mul.outputs[0], cout, out_rank)
        if shift is None:
            return None
        return weight, mul, add, scale, shift, cout

    def match(self, node: gs.Node) -> bool:
        return self._resolve(node) is not None

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Conv")
        resolved = self._resolve(node)
        if resolved is None:
            return
        weight, mul, add, scale, shift, cout = resolved

        w_values = np.asarray(weight.values)
        folded_w = self._values_f32(weight) * scale.reshape(
            (cout,) + (1,) * (w_values.ndim - 1)
        )
        bias = (
            self._values_f32(node.inputs[2]).reshape(cout)
            if len(node.inputs) > 2
            else np.zeros(cout, dtype=np.float32)
        )
        folded_b = bias * scale + shift

        base = node.name or weight.name
        export_dtype = getattr(weight, "export_dtype", None)
        new_w = gs.Constant(
            f"{base}_bnfold_W",
            values=folded_w.astype(w_values.dtype),
            export_dtype=export_dtype,
        )
        new_b = gs.Constant(
            f"{base}_bnfold_b",
            values=folded_b.astype(w_values.dtype),
            export_dtype=export_dtype,
        )

        node.inputs[1] = new_w
        if len(node.inputs) > 2:
            node.inputs[2] = new_b
        else:
            node.inputs.append(new_b)
        # Take over the Add's output so downstream consumers (and graph
        # outputs) keep their tensor; the Mul/Add pair goes dead.
        node.outputs[0] = add.outputs[0]
        mul.inputs.clear()
        mul.outputs.clear()
        add.inputs.clear()
        add.outputs.clear()

        self._logger.debug(
            "folded Conv->Mul->Add at %r into the conv (cout=%d)", node.name, cout
        )

@dataclass
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
