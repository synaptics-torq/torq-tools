# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from collections import defaultdict
import os
import re

import onnx
import onnx_graphsurgeon as gs
import numpy as np

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
    rewire_consumers
)
from ...graph_edit.edits import *


class SmolLM2OnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin):

    def __init__(
        self,
        graph: gs.Graph,
        export_dtype: onnx.TensorProto.DataType | None = None
    ):
        super().__init__(
            graph,
            "model",
            export_dtype=export_dtype
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ) -> "SmolLM2OnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            export_dtype
        )

    def fix_io(
        self,
        seq_len: int,
        dims: list[FixedDimMapping] | None = None,
        *,
        batch_dim: str = "batch_size",
        seq_len_dim: str = "sequence_length",
        past_seq_len_dim: str = "past_sequence_length"
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(past_seq_len_dim, DimMatchType.CONTAINS, seq_len),
        ]
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def combine_kv_io_tensors(
        self,
        kv_tensor_shape: list[int]
    ):
        # matches "prefix....<layer_idx>.<key|value>" naming convention
        _KV_LAYER_RE = re.compile(r"\.(\d+)\.(key|value)$")
        # concatenate along H axis: [..., H, L, D] <-> [..., 2*H, L, D]
        _H_DIM_AXIS = len(kv_tensor_shape) - 3

        def _get_kv_pairs(io_coll: list[gs.Variable], prefix: str) -> list[tuple[int, gs.Variable, gs.Variable]]:
            io_dict: dict[int, dict[str, gs.Variable]] = defaultdict(dict)
            for io in io_coll:
                if not isinstance(io, gs.Variable):
                    raise TypeError(f"Expected gs.Variable, got {type(io)}")
                if list(io.shape) != kv_tensor_shape:
                    continue
                if not io.name.startswith(prefix):
                    continue
                m = _KV_LAYER_RE.search(io.name)
                if m is None:
                    raise ValueError(
                        f"Cannot extract layer index from KV tensor name '{io.name}'; "
                        "expected '<prefix>...<layer_idx>.<key|value>' naming"
                    )
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
                        f"Layer {layer} is missing {'key' if 'key' not in entry else 'value'} tensor"
                    )
                kv_pairs.append((layer, entry["key"], entry["value"]))
            return kv_pairs

        def _remove_io(tensor: gs.Variable, io_coll: list[gs.Variable]):
            for idx, io_tensor in enumerate(io_coll):
                if tensor is io_tensor:
                    io_coll.pop(idx)
                    break
        
        def _concatenate_kv_input(layer: int, key_input: gs.Variable, value_input: gs.Variable, prefix: str):
            assert key_input.dtype == value_input.dtype
            assert list(key_input.shape) == kv_tensor_shape
            assert list(value_input.shape) == kv_tensor_shape
            key_consumers: list[gs.Node] = key_input.outputs
            value_consumers: list[gs.Node] = value_input.outputs

            n_kv_heads = kv_tensor_shape[_H_DIM_AXIS]
            combined_shape = kv_tensor_shape.copy()
            combined_shape[_H_DIM_AXIS] *= 2
            combined_input = gs.Variable(
                name=f"{prefix}.{layer}.key_value",
                dtype=key_input.dtype,
                shape=combined_shape
            )
            if not (kv_concat_axis := self._graph.tensors().get("kv_concat_axis")):
                kv_concat_axis = gs.Constant(
                    "kv_concat_axis", np.array([_H_DIM_AXIS], dtype=np.int64)
                )
            if not (kv_inp_key_starts := self._graph.tensors().get("kv_inp_key_starts")):
                kv_inp_key_starts = gs.Constant(
                    "kv_inp_key_starts", np.array([0], dtype=np.int64)
                )
            if not (kv_inp_key_ends := self._graph.tensors().get("kv_inp_key_ends")):
                kv_inp_key_ends = gs.Constant(
                    "kv_inp_key_ends", np.array([n_kv_heads], dtype=np.int64)
                )
            if not (kv_inp_value_starts := self._graph.tensors().get("kv_inp_value_starts")):
                kv_inp_value_starts = gs.Constant(
                    "kv_inp_value_starts", np.array([n_kv_heads], dtype=np.int64)
                )
            if not (kv_inp_value_ends := self._graph.tensors().get("kv_inp_value_ends")):
                kv_inp_value_ends = gs.Constant(
                    "kv_inp_value_ends", np.array([2 * n_kv_heads], dtype=np.int64)
                )
            key_slice: gs.Variable = self._graph.layer(
                name=f"{prefix}.{layer}.key_slice",
                op="Slice",
                inputs=[combined_input, kv_inp_key_starts, kv_inp_key_ends, kv_concat_axis],
                outputs=[gs.Variable(
                    name=f"{prefix}.{layer}.key_from_combined",
                    dtype=key_input.dtype,
                    shape=key_input.shape
                )]
            )[0]
            value_slice: gs.Variable = self._graph.layer(
                name=f"{prefix}.{layer}.value_slice",
                op="Slice",
                inputs=[combined_input, kv_inp_value_starts, kv_inp_value_ends, kv_concat_axis],
                outputs=[gs.Variable(
                    name=f"{prefix}.{layer}.value_from_combined",
                    dtype=value_input.dtype,
                    shape=value_input.shape
                )]
            )[0]

            rewire_consumers(key_consumers, key_input, key_slice)
            rewire_consumers(value_consumers, value_input, value_slice)
            key_input.outputs.clear()
            value_input.outputs.clear()
            _remove_io(key_input, self._graph.inputs)
            _remove_io(value_input, self._graph.inputs)
            self._graph.inputs.append(combined_input)

            self._logger.debug("Combined KV input layer %d: '%s' + '%s' -> '%s'", layer, key_input.name, value_input.name, combined_input.name)

        def _concatenate_kv_output(layer: int, key_output: gs.Variable, value_output: gs.Variable, prefix: str):
            assert key_output.dtype == value_output.dtype
            assert list(key_output.shape) == kv_tensor_shape
            assert list(value_output.shape) == kv_tensor_shape

            combined_shape = kv_tensor_shape.copy()
            combined_shape[_H_DIM_AXIS] *= 2
            combined_tensor: gs.Variable = self._graph.layer(
                name=f"{prefix}.{layer}_kv_concat",
                op="Concat",
                inputs=[key_output, value_output],
                outputs=[gs.Variable(
                    name=f"{prefix}.{layer}.key_value",
                    dtype=key_output.dtype,
                    shape=combined_shape
                )],
                attrs={"axis": _H_DIM_AXIS}
            )[0]
            _remove_io(key_output, self._graph.outputs)
            _remove_io(value_output, self._graph.outputs)
            self._graph.outputs.append(combined_tensor)

            self._logger.debug("Combined KV output layer %d: '%s' + '%s' -> '%s'", layer, key_output.name, value_output.name, combined_tensor.name)

        input_pairs = _get_kv_pairs(self._graph.inputs, "past_key_values")
        output_pairs = _get_kv_pairs(self._graph.outputs, "present")
        self._logger.info("Combining KV tensors: %d input pairs, %d output pairs (axis=%d)", len(input_pairs), len(output_pairs), _H_DIM_AXIS)
        for kv_info in input_pairs:
            _concatenate_kv_input(*kv_info, "past_key_values")
        for kv_info in output_pairs:
            _concatenate_kv_output(*kv_info, "present")
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
