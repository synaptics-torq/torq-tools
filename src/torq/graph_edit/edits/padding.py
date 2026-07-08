# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers
from ._helpers import _static_int_shape


@dataclass
class RewriteNegativePads(OnnxGraphEdit):
    """
    Rewrite ``Pad`` ops with negative (crop) paddings into ``Pad(positive)+Slice``.

    IREE/Torch-MLIR handle ``Slice`` more reliably than ``Pad`` with negative
    pads. Limited to ``mode == constant`` with a constant ``pads`` initializer
    and a fully-static ranked data input.
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _pad_mode(node: gs.Node) -> str:
        raw = node.attrs.get("mode", "constant")
        if isinstance(raw, bytes):
            return raw.decode("ascii")
        return str(raw)

    def match(self, node: gs.Node) -> bool:
        if node.op != "Pad" or len(node.inputs) < 2 or self._pad_mode(node) != "constant":
            return False
        pads_inp = node.inputs[1]
        if not isinstance(pads_inp, gs.Constant):
            return False
        pads_flat = np.asarray(pads_inp.values).astype(np.int64).reshape(-1)
        if pads_flat.size % 2 != 0 or np.all(pads_flat >= 0):
            return False
        rank = pads_flat.size // 2
        shape_in = _static_int_shape(node.inputs[0])
        return shape_in is not None and len(shape_in) == rank

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Pad")
        pads_flat = (
            np.asarray(node.inputs[1].values).astype(np.int64).reshape(-1)
        )
        rank = pads_flat.size // 2
        shape_in = _static_int_shape(node.inputs[0])

        pos_begin = np.maximum(pads_flat[:rank], 0)
        neg_begin = np.minimum(pads_flat[:rank], 0)
        pos_end = np.maximum(pads_flat[rank:], 0)
        neg_end = np.minimum(pads_flat[rank:], 0)

        inter_shape = [
            int(shape_in[i] + pos_begin[i] + pos_end[i]) for i in range(rank)
        ]
        slice_starts = (-neg_begin).astype(np.int64)
        slice_ends = (np.asarray(inter_shape, dtype=np.int64) + neg_end).astype(np.int64)

        if not all(
            0 <= slice_starts[i] <= slice_ends[i] <= inter_shape[i]
            for i in range(rank)
        ):
            self._logger.debug(
                "skip Pad %s: invalid slice range starts=%s ends=%s inter=%s",
                node.name, slice_starts.tolist(), slice_ends.tolist(), inter_shape,
            )
            return

        out = node.outputs[0]
        consumers = list(out.outputs)
        graph_output_idx = [
            i for i, g in enumerate(self.graph.outputs) if g is out
        ]
        out_dtype = out.dtype
        needs_pos_pad = bool(np.any(pos_begin > 0) or np.any(pos_end > 0))

        base = node.name or out.name

        if needs_pos_pad:
            pads_pos_const = gs.Constant(
                name=f"{base}_pos_pads",
                values=np.concatenate([pos_begin, pos_end]).astype(np.int64),
            )
            pad_inputs = [node.inputs[0], pads_pos_const]
            if len(node.inputs) >= 3:
                pad_inputs.append(node.inputs[2])
            mid = gs.Variable(
                name=f"{base}_pos_pad_tmp",
                dtype=out_dtype,
                shape=inter_shape,
            )
            self.graph.nodes.append(
                gs.Node(
                    op="Pad",
                    name=f"{base}_pos_only",
                    inputs=pad_inputs,
                    outputs=[mid],
                    attrs={"mode": "constant"},
                )
            )
            slice_data = mid
        else:
            slice_data = node.inputs[0]

        starts_const = gs.Constant(
            f"{base}_slice_starts", slice_starts
        )
        ends_const = gs.Constant(
            f"{base}_slice_ends", slice_ends
        )
        sliced = gs.Variable(
            name=f"{base}_cropped", dtype=out_dtype, shape=list(out.shape) if out.shape else None,
        )
        self.graph.nodes.append(
            gs.Node(
                op="Slice",
                name=f"{base}_crop_slice",
                inputs=[slice_data, starts_const, ends_const],
                outputs=[sliced],
            )
        )
        rewire_consumers(consumers, out, sliced)
        for i in graph_output_idx:
            self.graph.outputs[i] = sliced
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "rewrote Pad %s -> %sSlice (inter_shape=%s)",
            node.name, "Pad+" if needs_pos_pad else "(no pos Pad) ", inter_shape,
        )

@dataclass
class AbsorbPadding(OnnxGraphEdit):
    """
    Fuse non-negative ``Pad -> Conv`` chains into a single ``Conv`` with merged pads.

    Pads with negative paddings are out of scope: run :class:`RewriteNegativePads`
    first to convert them to ``Pad+Slice``. Convs that set ``auto_pad`` are
    skipped (per ONNX spec ``auto_pad`` overrides explicit ``pads``).
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        if node.op != "Pad" or not node.inputs or not node.outputs:
            return False
        users = list(node.outputs[0].outputs)
        if len(users) != 1 or users[0].op != "Conv":
            return False
        pads_inp = node.inputs[1] if len(node.inputs) >= 2 else None
        if not isinstance(pads_inp, gs.Constant):
            return False
        pads_flat = np.asarray(pads_inp.values).astype(np.int64).reshape(-1)
        if np.any(pads_flat < 0):
            return False
        conv = users[0]
        if "pads" not in conv.attrs or "kernel_shape" not in conv.attrs:
            return False
        if "auto_pad" in conv.attrs:
            return False
        spatial_rank = len(conv.attrs["kernel_shape"])
        return len(conv.attrs["pads"]) == 2 * spatial_rank

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Pad")
        conv = node.outputs[0].outputs[0]
        pads_flat = (
            np.asarray(node.inputs[1].values).astype(np.int64).reshape(-1)
        )
        rank = len(pads_flat) // 2
        spatial_rank = len(conv.attrs["kernel_shape"])
        spatial_axes = list(range(rank - spatial_rank, rank))
        conv_pads = list(conv.attrs["pads"])

        pb_half = pads_flat[:rank]
        pa_half = pads_flat[rank:]
        new_conv_pads: list[int] = []
        for j in range(spatial_rank):
            new_conv_pads.append(int(conv_pads[j] + pb_half[spatial_axes[j]]))
        for j in range(spatial_rank):
            new_conv_pads.append(
                int(conv_pads[j + spatial_rank] + pa_half[spatial_axes[j]])
            )

        conv.attrs["pads"] = new_conv_pads
        conv.inputs[0] = node.inputs[0]
        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug(
            "fused Pad %s -> Conv %s (new pads=%s)",
            node.name, conv.name, new_conv_pads,
        )

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
