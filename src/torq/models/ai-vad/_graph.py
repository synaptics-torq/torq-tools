# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
import os

import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import (
    OnnxGraphEdit,
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
    rewire_consumers
)
from ...graph_edit.edits import *


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
        rewire_consumers(consumers, float_out, lookup_output_batched)
        node.outputs.clear()

        self._logger.debug("Replaced int -> %s Cast node '%s' with look-up table", cast_dtype_str, node.name)

@dataclass
class ReplacePadWithConcat(OnnxGraphEdit):
    """
    Replace Pad ops with equivalent Concat ops using constant tensors.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "Pad"
    
    @staticmethod
    def _is_empty_variable(tensor: gs.Variable) -> bool:
        return (not tensor.name and not tensor.dtype and not tensor.shape)

    @staticmethod
    def _ensure_static_shape(tensor: gs.Tensor, node_name: str, label: str) -> list[int]:
        shape = getattr(tensor, "shape", None)
        if shape is None:
            raise ValueError(f"Pad node '{node_name}' {label} has no shape information")
        if not all(isinstance(d, (int, np.integer)) for d in shape):
            raise ValueError(
                f"Pad node '{node_name}' {label} has dynamic shape {shape}"
            )
        return [int(d) for d in shape]

    @staticmethod
    def _load_const_array(tensor: gs.Constant, node_name: str, label: str) -> np.ndarray:
        if not isinstance(tensor, gs.Constant):
            raise ValueError(f"Pad node '{node_name}' {label} must be constant, got {tensor}")
        values = tensor.values
        if not isinstance(values, np.ndarray):
            try:
                values = values.load()
            except AttributeError as e:
                raise ValueError(
                    f"Pad node '{node_name}' {label} is not a loadable constant"
                ) from e
        return np.asarray(values)

    @staticmethod
    def _normalize_dtype(dtype, node_name: str) -> tuple[np.dtype, int | None]:
        if dtype is None:
            raise ValueError(f"Pad node '{node_name}' is missing dtype information")
        if isinstance(dtype, int):
            if dtype == onnx.TensorProto.BFLOAT16:
                return np.dtype(np.float32), onnx.TensorProto.BFLOAT16
            try:
                return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dtype)), None
            except Exception as e:
                raise ValueError(
                    f"Pad node '{node_name}' has unsupported dtype {dtype}"
                ) from e
        try:
            return np.dtype(dtype), None
        except Exception as e:
            raise ValueError(
                f"Pad node '{node_name}' has unsupported dtype {dtype}"
            ) from e

    def _get_pad_value(self, node: gs.Node) -> object:
        if len(node.inputs) >= 3 and node.inputs[2] is not None:
            if self._is_empty_variable(node.inputs[2]):
                values = np.array(0).astype(np.int64)
            else:
                values = self._load_const_array(node.inputs[2], node.name, "constant_value")
        elif "value" in node.attrs:
            values = np.asarray(node.attrs["value"])
        elif "constant_value" in node.attrs:
            values = np.asarray(node.attrs["constant_value"])
        else:
            return 0
        if values.size != 1:
            raise ValueError(
                f"Pad node '{node.name}' constant_value must be scalar, got shape {values.shape}"
            )
        return values.reshape(()).item()

    def _get_axis_pads(self, node: gs.Node, rank: int) -> list[tuple[int, int]]:
        pads_values = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            pads_values = self._load_const_array(node.inputs[1], node.name, "pads")
        elif "pads" in node.attrs:
            pads_values = np.asarray(node.attrs["pads"])
        if pads_values is None:
            raise ValueError(f"Pad node '{node.name}' is missing pads")

        pads_values = np.asarray(pads_values)
        if not np.all(np.equal(pads_values, np.round(pads_values))):
            raise ValueError(f"Pad node '{node.name}' pads must be integers")
        pads_list = pads_values.astype(np.int64).flatten().tolist()

        axes_values = None
        if len(node.inputs) >= 4 and node.inputs[3] is not None:
            axes_values = self._load_const_array(node.inputs[3], node.name, "axes")
        elif "axes" in node.attrs:
            axes_values = np.asarray(node.attrs["axes"])

        if axes_values is None:
            if len(pads_list) != 2 * rank:
                raise ValueError(
                    f"Pad node '{node.name}' pads length {len(pads_list)} "
                    f"does not match rank {rank}"
                )
            return [(int(pads_list[i]), int(pads_list[i + rank])) for i in range(rank)]

        axes_values = np.asarray(axes_values)
        if not np.all(np.equal(axes_values, np.round(axes_values))):
            raise ValueError(f"Pad node '{node.name}' axes must be integers")
        axes = axes_values.astype(np.int64).flatten().tolist()
        if len(pads_list) != 2 * len(axes):
            raise ValueError(
                f"Pad node '{node.name}' pads length {len(pads_list)} does not match axes {axes}"
            )

        axis_pads: list[tuple[int, int]] = [(0, 0)] * rank
        for i, axis in enumerate(axes):
            axis = int(axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f"Pad node '{node.name}' has axis {axis} out of range for rank {rank}"
                )
            if axis_pads[axis] != (0, 0):
                raise ValueError(
                    f"Pad node '{node.name}' has duplicate axis {axis} in axes {axes}"
                )
            axis_pads[axis] = (int(pads_list[i]), int(pads_list[i + len(axes)]))
        return axis_pads

    @staticmethod
    def _build_pad_const(
        name: str,
        base_shape: list[int],
        axis: int,
        pad_len: int,
        pad_value: object,
        np_dtype: np.dtype,
        export_dtype: int | None
    ) -> gs.Constant:
        pad_shape = list(base_shape)
        pad_shape[axis] = pad_len
        if pad_value == 0:
            values = np.zeros(pad_shape, dtype=np_dtype)
        else:
            values = np.full(pad_shape, pad_value, dtype=np_dtype)
        return gs.Constant(name=name, values=values, export_dtype=export_dtype)

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Pad")
        if not node.inputs or not node.outputs:
            raise ValueError(f"Pad node '{node.name}' must have inputs and outputs")

        mode = node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode()
        if mode != "constant":
            raise ValueError(
                f"Pad node '{node.name}' has unsupported mode '{mode}'"
            )

        data = node.inputs[0]
        out = node.outputs[0]

        in_shape = self._ensure_static_shape(data, node.name, "input")
        out_shape = self._ensure_static_shape(out, node.name, "output")
        rank = len(in_shape)

        axis_pads = self._get_axis_pads(node, rank)
        if rank == 0 and any(b or a for b, a in axis_pads):
            raise ValueError(f"Pad node '{node.name}' cannot pad a scalar input")

        for before, after in axis_pads:
            if before < 0 or after < 0:
                raise ValueError(
                    f"Pad node '{node.name}' has negative pads {axis_pads}"
                )

        expected_shape = [
            in_shape[i] + axis_pads[i][0] + axis_pads[i][1] for i in range(rank)
        ]
        if out_shape != expected_shape:
            raise ValueError(
                f"Pad node '{node.name}' output shape {out_shape} does not match "
                f"expected {expected_shape}"
            )

        in_dtype = getattr(data, "dtype", None)
        out_dtype = getattr(out, "dtype", None)
        in_np_dtype, in_export_dtype = self._normalize_dtype(in_dtype, node.name)
        out_np_dtype, out_export_dtype = self._normalize_dtype(out_dtype, node.name)
        if in_np_dtype != out_np_dtype or in_export_dtype != out_export_dtype:
            raise ValueError(
                f"Pad node '{node.name}' dtype mismatch input {in_dtype} vs output {out_dtype}"
            )

        pad_value = self._get_pad_value(node)

        if not any(b or a for b, a in axis_pads):
            consumers: list[gs.Node] = list(out.outputs)
            rewire_consumers(consumers, out, data)
            for i, graph_out in enumerate(self.graph.outputs):
                if graph_out is out:
                    self.graph.outputs[i] = data
            node.inputs.clear()
            node.outputs.clear()
            self._logger.debug("Removed no-op Pad node '%s'", node.name)
            return

        pad_axes = [i for i, (b, a) in enumerate(axis_pads) if b or a]
        last_axis = pad_axes[-1]

        cur = data
        cur_shape = list(in_shape)
        for axis in range(rank):
            before, after = axis_pads[axis]
            if before == 0 and after == 0:
                continue

            concat_inputs: list[gs.Tensor] = []
            if before > 0:
                concat_inputs.append(
                    self._build_pad_const(
                        name=f"{node.name}_pad_pre_axis{axis}",
                        base_shape=cur_shape,
                        axis=axis,
                        pad_len=before,
                        pad_value=pad_value,
                        np_dtype=in_np_dtype,
                        export_dtype=in_export_dtype,
                    )
                )
            concat_inputs.append(cur)
            if after > 0:
                concat_inputs.append(
                    self._build_pad_const(
                        name=f"{node.name}_pad_post_axis{axis}",
                        base_shape=cur_shape,
                        axis=axis,
                        pad_len=after,
                        pad_value=pad_value,
                        np_dtype=in_np_dtype,
                        export_dtype=in_export_dtype,
                    )
                )

            new_shape = list(cur_shape)
            new_shape[axis] = new_shape[axis] + before + after
            if axis == last_axis:
                concat_out = out
                concat_out.shape = new_shape
            else:
                concat_out = gs.Variable(
                    name=f"{node.name}_pad_axis{axis}_out",
                    dtype=out_dtype,
                    shape=new_shape,
                )

            concat_node = gs.Node(
                op="Concat",
                name=f"{node.name}_pad_axis{axis}",
                inputs=concat_inputs,
                outputs=[concat_out],
                attrs={"axis": axis},
            )
            self.graph.nodes.append(concat_node)

            cur = concat_out
            cur_shape = new_shape

        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug("Replaced Pad node '%s' with Concat ops", node.name)


@dataclass
class WidenSmallStridedDepthwiseConv(OnnxGraphEdit):
    """
    Right-pad the trailing spatial input of a *narrow*, strided depthwise Conv
    so that its output dimension grows past the Torq compiler's DEDR
    scatter-gather (SIMD G-tag) selection threshold, then slice the conv output
    back to its original shape.

    Why
    ---
    For depthwise Conv with ``stride[-1] > 1`` and a small last spatial output
    dim, the Torq compiler picks a DEDR ``G(L)[sgGroups>1]`` SIMD scatter-gather
    descriptor. The current precompiled NSS CModel hangs while executing that
    descriptor (depthwise stride-2 + ``sgGroups=4`` codegen path), which makes
    pytest cases like ``test_onnx_model.py -k cleaned_bf16_layer_conv_13`` time
    out under the simulator. The DEDR codegen condition is roughly::

        sg fires iff   out_w * sgGroups <= bus_width_items

    where ``bus_width_items = bus_width_bytes / element_size`` and ``sgGroups``
    ranges over 1..``sg_groups_max``. Pushing ``out_w`` just past
    ``bus_width_items // sg_groups_max`` (e.g. ``> 9`` for bf16) makes the
    compiler fall back to the dense ``V(H)`` outerGroups path, which the
    CModel executes correctly.

    Transform
    ---------
    The trailing ``pads`` entry on the matched Conv is bumped by
    ``extra * stride[-1]`` so the conv naturally produces ``extra`` extra
    output positions with zero-padding (numerically equivalent to extending
    the input with zeros). A ``Slice`` node is then inserted after the conv
    that crops back to the original output width, so all downstream consumers
    see a bit-identical tensor.

    Parameters
    ----------
    bus_width_bytes:
        DEDR bus width in bytes (Torq HW: ``iram_seg_width = 72``).
    sg_groups_max:
        Maximum SIMD scatter-gather group count for DEDR (Torq HW: ``4``).
    """

    bus_width_bytes: int = 72
    sg_groups_max: int = 4

    @staticmethod
    def _element_size_bytes(dtype) -> int | None:
        if isinstance(dtype, np.dtype):
            return dtype.itemsize
        if isinstance(dtype, int):
            sizes = {
                onnx.TensorProto.FLOAT: 4,
                onnx.TensorProto.UINT8: 1,
                onnx.TensorProto.INT8: 1,
                onnx.TensorProto.UINT16: 2,
                onnx.TensorProto.INT16: 2,
                onnx.TensorProto.INT32: 4,
                onnx.TensorProto.INT64: 8,
                onnx.TensorProto.UINT32: 4,
                onnx.TensorProto.UINT64: 8,
                onnx.TensorProto.FLOAT16: 2,
                onnx.TensorProto.BFLOAT16: 2,
                onnx.TensorProto.DOUBLE: 8,
                onnx.TensorProto.BOOL: 1,
            }
            return sizes.get(int(dtype))
        return None

    def _threshold(self, dtype) -> int | None:
        elem = self._element_size_bytes(dtype)
        if not elem or self.sg_groups_max <= 0:
            return None
        return (self.bus_width_bytes // elem) // self.sg_groups_max

    @staticmethod
    def _is_static_shape(shape) -> bool:
        return shape is not None and all(
            isinstance(d, (int, np.integer)) for d in shape
        )

    def match(self, node: gs.Node) -> bool:
        if node.op != "Conv" or len(node.inputs) < 2 or not node.outputs:
            return False
        x, w = node.inputs[0], node.inputs[1]
        out = node.outputs[0]
        if not (
            self._is_static_shape(x.shape)
            and self._is_static_shape(w.shape)
            and self._is_static_shape(out.shape)
        ):
            return False
        if len(x.shape) < 3 or len(w.shape) != len(x.shape):
            return False

        # Depthwise: group == in_channels and weight C/group dim == 1.
        in_channels = int(x.shape[1])
        group = int(node.attrs.get("group", 1))
        if group != in_channels or int(w.shape[1]) != 1:
            return False

        rank = len(x.shape) - 2
        strides = list(node.attrs.get("strides", [1] * rank))
        if len(strides) != rank or int(strides[-1]) <= 1:
            return False

        threshold = self._threshold(out.dtype)
        if threshold is None or threshold <= 0:
            return False
        return int(out.shape[-1]) <= threshold

    def transform(self, node: gs.Node):
        x = node.inputs[0]
        out = node.outputs[0]
        rank = len(x.shape) - 2

        threshold = self._threshold(out.dtype)
        cur_out_w = int(out.shape[-1])
        target_out_w = threshold + 1
        extra = target_out_w - cur_out_w

        strides = list(node.attrs.get("strides", [1] * rank))
        s_w = int(strides[-1])
        extra_pad = extra * s_w

        pads = list(node.attrs.get("pads", [0] * (2 * rank)))
        if len(pads) != 2 * rank:
            raise ValueError(
                f"Conv '{node.name}' has unexpected 'pads' length {len(pads)} "
                f"(expected {2 * rank})"
            )

        # ONNX Conv pads layout: [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
        pads[2 * rank - 1] = int(pads[2 * rank - 1]) + extra_pad
        node.attrs["pads"] = pads
        # `auto_pad` would override an explicit `pads` list.
        node.attrs["auto_pad"] = "NOTSET"

        new_out_shape = list(out.shape)
        new_out_shape[-1] = target_out_w

        widened = gs.Variable(
            name=out.name + "_widened",
            dtype=out.dtype,
            shape=new_out_shape,
        )

        consumers = list(out.outputs)
        graph_output_indices = [
            i for i, g_out in enumerate(self.graph.outputs) if g_out is out
        ]

        node.outputs[0] = widened

        last_axis = len(new_out_shape) - 1
        starts = gs.Constant(
            f"{node.name}_widen_slice_starts",
            np.array([0], dtype=np.int64),
        )
        ends = gs.Constant(
            f"{node.name}_widen_slice_ends",
            np.array([cur_out_w], dtype=np.int64),
        )
        axes = gs.Constant(
            f"{node.name}_widen_slice_axes",
            np.array([last_axis], dtype=np.int64),
        )
        steps = gs.Constant(
            f"{node.name}_widen_slice_steps",
            np.array([1], dtype=np.int64),
        )
        # Preserve the original output name on the slice so downstream tensor
        # references stay stable; `out` becomes orphaned and is dropped by cleanup.
        sliced = gs.Variable(
            name=out.name,
            dtype=out.dtype,
            shape=list(out.shape),
        )
        slice_node = gs.Node(
            op="Slice",
            name=f"{node.name}_widen_slice",
            inputs=[widened, starts, ends, axes, steps],
            outputs=[sliced],
        )
        self.graph.nodes.append(slice_node)

        rewire_consumers(consumers, out, sliced)
        for i in graph_output_indices:
            self.graph.outputs[i] = sliced

        self._logger.debug(
            "Widened depthwise Conv '%s': out_w %d -> %d (pad+%d on axis %d, slice [:%d])",
            node.name, cur_out_w, target_out_w, extra_pad, last_axis, cur_out_w,
        )


class AiVadOnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):

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
    ) -> "AiVadOnnxGraphEditor":
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
        Reshapevad_dim_0: int,
        Reshapevad_dim_1: int,
        Concathidden_state_dim_1: int
    ):
        to_fix = [
            FixedDimMapping("Reshapevad_dim_0", DimMatchType.EXACT, Reshapevad_dim_0),
            FixedDimMapping("Reshapevad_dim_1", DimMatchType.EXACT, Reshapevad_dim_1),
            FixedDimMapping("Concathidden_state_dim_1", DimMatchType.EXACT, Concathidden_state_dim_1),
        ]

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

    def widen_small_strided_depthwise_conv(
        self,
        bus_width_bytes: int = 72,
        sg_groups_max: int = 4,
    ):
        """Widen narrow strided-depthwise Convs to dodge the Torq DEDR
        scatter-gather codegen path that hangs the simulator's CModel.

        See :class:`WidenSmallStridedDepthwiseConv` for the full rationale.
        """
        self.apply_edit(
            WidenSmallStridedDepthwiseConv(
                self._graph,
                self._graph_name,
                bus_width_bytes=bus_width_bytes,
                sg_groups_max=sg_groups_max,
            )
        )
        return self
