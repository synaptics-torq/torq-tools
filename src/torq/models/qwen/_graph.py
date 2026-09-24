# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
)
from ...graph_edit.edits import CombineKVCacheMixin, CommonGraphEditsMixin


class QwenOnnxGraphEditor(
    OnnxGraphEditor,
    CommonGraphEditsMixin,
    CombineKVCacheMixin,
):
    """ONNX graph editor for Qwen decoder models."""

    def __init__(
        self,
        graph: gs.Graph,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ):
        super().__init__(
            graph,
            "model",
            export_dtype=export_dtype,
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ) -> "QwenOnnxGraphEditor":
        """Create a Qwen graph editor from an ONNX path or model object."""

        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)

        graph = gs.import_onnx(onnx_model)

        return cls(
            graph,
            export_dtype,
        )

    def to_onnx(
        self,
        check_type: bool = True,
        strict_mode: bool = True,
        data_prop: bool = True,
        override_ir: int | None = None,
    ) -> onnx.ModelProto:
        """
        Export the edited Qwen graph without in-memory ONNX shape inference.

        Qwen3-0.6B uses multi-gigabyte external weight data. The standard
        editor path calls onnx.shape_inference.infer_shapes(), which attempts
        to serialize the complete model and fails for this model size.
        """
        del check_type, strict_mode, data_prop

        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

        onnx_model = gs.export_onnx(self._graph)

        if isinstance(override_ir, int):
            onnx_model.ir_version = override_ir

        return onnx_model

    def split_rope_qk(self) -> None:
        """Replace Qwen RoPE Q/K Slice pairs with Split operations."""

        split_name = "qwen_rope_split_sizes_64_64"

        split_sizes = self._graph.tensors().get(split_name)

        if split_sizes is None:
            split_sizes = gs.Constant(
                name=split_name,
                values=np.array([64, 64], dtype=np.int64),
            )

        layers = sorted({
            int(node.name.split("/model/layers.")[1].split("/")[0])
            for node in self._graph.nodes
            if "/model/layers." in node.name
        })

        for layer in layers:
            base = f"/model/layers.{layer}/self_attn/"

            specs = [
                ("Slice", "Slice_1", "RoPE_Q_Split"),
                ("Slice_2", "Slice_3", "RoPE_K_Split"),
            ]

            for first_name, second_name, split_name_suffix in specs:
                first = next(
                    (
                        node
                        for node in self._graph.nodes
                        if node.name == base + first_name
                    ),
                    None,
                )
                second = next(
                    (
                        node
                        for node in self._graph.nodes
                        if node.name == base + second_name
                    ),
                    None,
                )

                if first is None and second is None:
                    continue

                if first is None or second is None:
                    raise ValueError(
                        f"Incomplete RoPE Slice pair for layer {layer}: "
                        f"{first_name}, {second_name}"
                    )

                split = gs.Node(
                    op="Split",
                    name=base + split_name_suffix,
                    inputs=[
                        first.inputs[0],
                        split_sizes,
                    ],
                    outputs=[
                        first.outputs[0],
                        second.outputs[0],
                    ],
                    attrs={"axis": 3},
                )

                self._graph.nodes.append(split)

                first.outputs.clear()
                second.outputs.clear()

                self._graph.nodes.remove(first)
                self._graph.nodes.remove(second)

        self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

        self._logger.info(
            "Replaced Qwen RoPE Slice pairs with Split operations"
        )

    def split_transformer_layers(
        self,
        split_at: int = 14,
    ) -> tuple[gs.Graph, gs.Graph]:
        """Split the static Qwen transformer into two runnable graphs.

        The split is made between layers ``split_at - 1`` and ``split_at``.
        For Qwen3-0.6B, the default split is layers 0-13 and 14-27.
        """
        if not 0 < split_at < 28:
            raise ValueError(
                f"split_at must be between 1 and 27, got {split_at}"
            )

        source_graph = self._graph.copy()

        def cache_names(start: int, end: int) -> list[str]:
            return [
                f"past_key_values.{i}.key_value"
                for i in range(start, end)
            ]

        def make_partition(
            part: str,
            name: str,
        ) -> gs.Graph:
            # Each partition gets its own independent graph copy.
            graph = source_graph.copy()
            tensors = graph.tensors()

            token_embedding = tensors["token_embedding"]
            position_ids = tensors["position_ids"]
            final_hidden = tensors["last_hidden_states"]

            boundary_name = (
                f"/model/layers.{split_at - 1}/Add_1_output_0"
            )
            boundary = tensors[boundary_name]

            # Detach runtime inputs from their producers so each partition
            # has an independent graph boundary.
            position_ids.inputs.clear()
            position_ids.shape = [1, 1]

            if part == "A":
                # Keep the layer-13 producer connected. The boundary tensor is
                # an output of Part A.
                boundary.shape = [1, 1, 1024]
                boundary.dtype = np.float32

                inputs = [
                    token_embedding,
                    position_ids,
                    *[
                        tensors[f"past_key_values.{i}.key_value"]
                        for i in range(0, split_at)
                    ],
                ]

                outputs = [
                    tensors[f"present.{i}.key_value"]
                    for i in range(0, split_at)
                ] + [boundary]

            else:
                boundary.inputs.clear()
                boundary.shape = [1, 1, 1024]
                boundary.dtype = np.float32

                inputs = [
                    boundary,
                    position_ids,
                    *[
                        tensors[f"past_key_values.{i}.key_value"]
                        for i in range(split_at, 28)
                    ],
                ]

                outputs = [
                    tensors[f"present.{i}.key_value"]
                    for i in range(split_at, 28)
                ] + [final_hidden]

            graph.inputs = inputs
            graph.outputs = outputs

            return graph.cleanup(
                remove_unused_graph_inputs=True,
                remove_unused_node_outputs=False,
            ).toposort()

        part_a = make_partition(
            "A",
            "transformer_part_A",
        )

        part_b = make_partition(
            "B",
            "transformer_part_B",
        )

        self._logger.info(
            "Split Qwen transformer at layer %d: "
            "Part A layers 0-%d, Part B layers %d-27",
            split_at,
            split_at - 1,
            split_at,
        )

        return part_a, part_b

    def replace_current_len_squeeze(self) -> None:
        """
        Replace the Qwen current-length Squeeze with an equivalent Reshape.

        The static graph contains:

            position_ids [1, 1]
                -> Squeeze(axis=0)
                -> position_ids_squeezed [1]

        Torch-MLIR crashes while folding this Squeeze after dtype conversion.
        Reshape([1]) is equivalent here and avoids that compiler path.
        """

        for node in self._graph.nodes:
            if (
                node.op != "Squeeze"
                or node.name != "current_len_to_1d"
                or not node.inputs
                or not node.outputs
            ):
                continue

            inp = node.inputs[0]
            out = node.outputs[0]

            reshape_shape = gs.Constant(
                name="current_len_to_1d_shape",
                values=np.array([1], dtype=np.int64),
            )

            new_out = self._graph.layer(
                name="current_len_to_1d_reshape",
                op="Reshape",
                inputs=[
                    inp,
                    reshape_shape,
                ],
                outputs=[
                    gs.Variable(
                        name=out.name,
                        dtype=out.dtype,
                        shape=[1],
                    )
                ],
            )[0]

            consumers = list(out.outputs)

            for consumer in consumers:
                for i, consumer_input in enumerate(consumer.inputs):
                    if consumer_input is out:
                        consumer.inputs[i] = new_out

            for i, graph_out in enumerate(self._graph.outputs):
                if graph_out is out:
                    self._graph.outputs[i] = new_out

            node.inputs.clear()
            node.outputs.clear()

            self._logger.info(
                "Replaced current_len_to_1d Squeeze with Reshape([1])"
            )

            break

    def _eval_static_tensor(
        self,
        tensor: gs.Tensor,
    ) -> np.ndarray | None:
        """
        Evaluate a limited set of Qwen shape-expression tensors.

        This intentionally handles only simple ONNX operations used to build
        shape and Slice/Expand parameter tensors. If any required value cannot
        be proven static, return None instead of guessing.
        """

        if isinstance(tensor, gs.Constant):
            try:
                return np.asarray(tensor.values)
            except Exception:
                return None

        producers = getattr(tensor, "inputs", None) or []
        if len(producers) != 1:
            return None

        node = producers[0]

        if node.op == "Constant":
            value = node.attrs.get("value")

            if value is None or not hasattr(value, "values"):
                return None

            return np.asarray(value.values)

        if node.op == "Shape":
            if not node.inputs:
                return None

            source = node.inputs[0]
            source_shape = getattr(source, "shape", None)

            if (
                source_shape is not None
                and all(
                    isinstance(dim, (int, np.integer))
                    for dim in source_shape
                )
            ):
                return np.asarray(source_shape, dtype=np.int64)

            source_value = self._eval_static_tensor(source)

            if source_value is None:
                return None

            return np.asarray(
                np.asarray(source_value).shape,
                dtype=np.int64,
            )

        if node.op == "Gather":
            if len(node.inputs) < 2:
                return None

            data = self._eval_static_tensor(node.inputs[0])
            indices = self._eval_static_tensor(node.inputs[1])

            if data is None or indices is None:
                return None

            axis = int(node.attrs.get("axis", 0))

            try:
                return np.take(
                    data,
                    np.asarray(indices).astype(np.int64),
                    axis=axis,
                )
            except Exception:
                return None

        if node.op == "Unsqueeze":
            if not node.inputs:
                return None

            data = self._eval_static_tensor(node.inputs[0])

            if data is None:
                return None

            if len(node.inputs) >= 2:
                axes_value = self._eval_static_tensor(node.inputs[1])

                if axes_value is None:
                    return None

                axes = [
                    int(value)
                    for value in np.asarray(axes_value).reshape(-1)
                ]
            else:
                raw_axes = node.attrs.get("axes")

                if raw_axes is None:
                    return None

                axes = [int(value) for value in raw_axes]

            result = np.asarray(data)

            try:
                for axis in sorted(axes):
                    result = np.expand_dims(
                        result,
                        axis=axis,
                    )

                return result
            except Exception:
                return None

        if node.op == "Squeeze":
            if not node.inputs:
                return None

            data = self._eval_static_tensor(node.inputs[0])

            if data is None:
                return None

            if len(node.inputs) >= 2:
                axes_value = self._eval_static_tensor(node.inputs[1])

                if axes_value is None:
                    return None

                axes = tuple(
                    int(value)
                    for value in np.asarray(axes_value).reshape(-1)
                )
            else:
                raw_axes = node.attrs.get("axes")

                if raw_axes is None:
                    axes = None
                else:
                    axes = tuple(int(value) for value in raw_axes)

            try:
                return np.squeeze(
                    np.asarray(data),
                    axis=axes,
                )
            except Exception:
                return None

        if node.op == "Concat":
            values = []

            for inp in node.inputs:
                value = self._eval_static_tensor(inp)

                if value is None:
                    return None

                values.append(np.asarray(value))

            axis = int(node.attrs.get("axis", 0))

            try:
                return np.concatenate(
                    values,
                    axis=axis,
                )
            except Exception:
                return None

        if node.op == "Reshape":
            if len(node.inputs) < 2:
                return None

            data = self._eval_static_tensor(node.inputs[0])
            target_shape = self._eval_static_tensor(node.inputs[1])

            if data is None or target_shape is None:
                return None

            target_shape = [
                int(value)
                for value in np.asarray(target_shape).reshape(-1)
            ]

            try:
                return np.reshape(
                    np.asarray(data),
                    target_shape,
                )
            except Exception:
                return None

        if node.op == "ConstantOfShape":
            if not node.inputs:
                return None

            shape_value = self._eval_static_tensor(node.inputs[0])

            if shape_value is None:
                return None

            shape = [
                int(value)
                for value in np.asarray(shape_value).reshape(-1)
            ]

            if any(dim < 0 for dim in shape):
                return None

            fill_value = 0

            attr_value = node.attrs.get("value")

            if (
                attr_value is not None
                and hasattr(attr_value, "values")
            ):
                fill_array = np.asarray(attr_value.values)

                if fill_array.size != 1:
                    return None

                fill_value = fill_array.reshape(()).item()

            try:
                return np.full(
                    shape,
                    fill_value,
                )
            except Exception:
                return None

        if node.op in {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Equal",
        }:
            if len(node.inputs) != 2:
                return None

            left = self._eval_static_tensor(node.inputs[0])
            right = self._eval_static_tensor(node.inputs[1])

            if left is None or right is None:
                return None

            try:
                if node.op == "Add":
                    return np.add(left, right)

                if node.op == "Sub":
                    return np.subtract(left, right)

                if node.op == "Mul":
                    return np.multiply(left, right)

                if node.op == "Div":
                    return np.divide(left, right)

                return np.equal(left, right)

            except Exception:
                return None

        if node.op == "Where":
            if len(node.inputs) != 3:
                return None

            condition = self._eval_static_tensor(node.inputs[0])
            true_value = self._eval_static_tensor(node.inputs[1])
            false_value = self._eval_static_tensor(node.inputs[2])

            if (
                condition is None
                or true_value is None
                or false_value is None
            ):
                return None

            try:
                return np.where(
                    condition,
                    true_value,
                    false_value,
                )
            except Exception:
                return None

        if node.op == "Cast":
            if not node.inputs:
                return None

            value = self._eval_static_tensor(node.inputs[0])

            if value is None:
                return None

            to_dtype = node.attrs.get("to")

            dtype_map = {
                onnx.TensorProto.INT64: np.int64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.BOOL: np.bool_,
            }

            np_dtype = dtype_map.get(to_dtype)

            if np_dtype is None:
                return None

            try:
                return np.asarray(value).astype(np_dtype)
            except Exception:
                return None

        return None

    def resolve_static_shape_inputs(self) -> None:
        """
        Materialize statically-computable Expand and Slice parameters.

        Optimum-generated Qwen graphs contain shape-expression chains whose
        values become constant after Qwen I/O and KV-cache dimensions are fixed.

        This pass:
        1. replaces statically-computable Expand shape inputs with constants;
        2. replaces statically-computable Slice parameters with constants;
        3. computes Slice output-shape metadata when the Slice is fully static.
        """

        resolved_count = 0
        resolved_slice_shapes = 0

        for node in self._graph.nodes:
            if node.op == "Expand":
                if len(node.inputs) < 2:
                    continue

                shape_value = self._eval_static_tensor(
                    node.inputs[1]
                )

                if shape_value is None:
                    continue

                shape_value = np.asarray(
                    shape_value,
                    dtype=np.int64,
                ).reshape(-1)

                if (
                    shape_value.size == 0
                    or np.any(shape_value < 0)
                ):
                    continue

                node.inputs[1] = gs.Constant(
                    name=f"{node.name}_static_shape",
                    values=shape_value,
                )

                resolved_count += 1

            elif node.op == "Slice":
                # ONNX Slice inputs:
                #   0: data
                #   1: starts
                #   2: ends
                #   3: axes
                #   4: steps

                for input_index in range(
                    1,
                    min(len(node.inputs), 5),
                ):
                    value = self._eval_static_tensor(
                        node.inputs[input_index]
                    )

                    if value is None:
                        continue

                    value = np.asarray(
                        value,
                        dtype=np.int64,
                    )

                    node.inputs[input_index] = gs.Constant(
                        name=(
                            f"{node.name}_static_param_"
                            f"{input_index}"
                        ),
                        values=value,
                    )

                    resolved_count += 1

                if (
                    len(node.inputs) < 3
                    or not node.outputs
                ):
                    continue

                input_shape = getattr(
                    node.inputs[0],
                    "shape",
                    None,
                )

                if (
                    input_shape is None
                    or not all(
                        isinstance(dim, (int, np.integer))
                        and int(dim) >= 0
                        for dim in input_shape
                    )
                ):
                    continue

                starts = self._eval_static_tensor(
                    node.inputs[1]
                )
                ends = self._eval_static_tensor(
                    node.inputs[2]
                )

                if starts is None or ends is None:
                    continue

                starts = [
                    int(v)
                    for v in np.asarray(starts).reshape(-1)
                ]
                ends = [
                    int(v)
                    for v in np.asarray(ends).reshape(-1)
                ]

                if len(node.inputs) >= 4:
                    axes_value = self._eval_static_tensor(
                        node.inputs[3]
                    )

                    if axes_value is None:
                        continue

                    axes = [
                        int(v)
                        for v in np.asarray(
                            axes_value
                        ).reshape(-1)
                    ]
                else:
                    axes = list(range(len(starts)))

                if len(node.inputs) >= 5:
                    steps_value = self._eval_static_tensor(
                        node.inputs[4]
                    )

                    if steps_value is None:
                        continue

                    steps = [
                        int(v)
                        for v in np.asarray(
                            steps_value
                        ).reshape(-1)
                    ]
                else:
                    steps = [1] * len(starts)

                if not (
                    len(starts)
                    == len(ends)
                    == len(axes)
                    == len(steps)
                ):
                    continue

                output_shape = [
                    int(dim)
                    for dim in input_shape
                ]

                try:
                    for start, end, axis, step in zip(
                        starts,
                        ends,
                        axes,
                        steps,
                    ):
                        rank = len(output_shape)

                        if axis < 0:
                            axis += rank

                        if axis < 0 or axis >= rank:
                            raise ValueError

                        if step == 0:
                            raise ValueError

                        dim_size = output_shape[axis]

                        normalized_start, normalized_end, normalized_step = (
                            slice(
                                start,
                                end,
                                step,
                            ).indices(dim_size)
                        )

                        output_shape[axis] = len(
                            range(
                                normalized_start,
                                normalized_end,
                                normalized_step,
                            )
                        )

                except Exception:
                    continue

                node.outputs[0].shape = output_shape
                resolved_slice_shapes += 1

        self._logger.info(
            "Resolved %d static Qwen shape inputs and %d Slice output shapes",
            resolved_count,
            resolved_slice_shapes,
        )

    def convert_static_shape_params_to_int32(self) -> None:
        """Convert Qwen-materialized static shape parameters from INT64 to INT32."""
        converted = 0

        for node in self._graph.nodes:
            if node.op == "Expand" and len(node.inputs) >= 2:
                tensor = node.inputs[1]
                if (
                    isinstance(tensor, gs.Constant)
                    and "_static_shape" in tensor.name
                    and tensor.values.dtype == np.int64
                ):
                    node.inputs[1] = gs.Constant(
                        name=f"{tensor.name}_int32",
                        values=np.asarray(tensor.values, dtype=np.int32),
                    )
                    converted += 1

            elif node.op == "Slice":
                for input_index in range(
                    1,
                    min(len(node.inputs), 5),
                ):
                    tensor = node.inputs[input_index]
                    if (
                        isinstance(tensor, gs.Constant)
                        and "_static_param_" in tensor.name
                        and tensor.values.dtype == np.int64
                    ):
                        node.inputs[input_index] = gs.Constant(
                            name=f"{tensor.name}_int32",
                            values=np.asarray(
                                tensor.values,
                                dtype=np.int32,
                            ),
                        )
                        converted += 1

        self._logger.info(
            "Converted %d Qwen static shape parameters from INT64 to INT32",
            converted,
        )

    def fix_io(
        self,
        seq_len: int,
        dims: list[FixedDimMapping] | None = None,
        *,
        batch_dim: str = "batch_size",
        seq_len_dim: str = "sequence_length",
        past_seq_len_dim: str = "past_sequence_length",
    ) -> None:
        """Replace Qwen's dynamic input/output dimensions with fixed values."""

        to_fix = [
            FixedDimMapping(
                batch_dim,
                DimMatchType.EXACT,
                1,
            ),
            FixedDimMapping(
                seq_len_dim,
                DimMatchType.EXACT,
                1,
            ),
            FixedDimMapping(
                past_seq_len_dim,
                DimMatchType.CONTAINS,
                seq_len,
            ),
        ]

        to_fix.extend(dims or [])

        self.fix_io_dims(to_fix)
