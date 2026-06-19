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
    OnnxGraphEditor
)
from ...graph_edit.edits import CombineKVCacheMixin, CommonGraphEditsMixin


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

