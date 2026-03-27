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


class MoonshineOnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):

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
