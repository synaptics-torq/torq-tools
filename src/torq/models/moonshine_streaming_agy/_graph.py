# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
)
from ...graph_edit.edits import *

# Re-use the excellent graph surgeon edits from the standard Moonshine implementation
from ..moonshine._graph import (
    MoveOutputFromConcat,
    ReplaceInt64FloatCast,
    ReplacePadWithConcat,
    DecomposeStridedConv1D,
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

    def decompose_layer_normalization(self):
        self.apply_edit(DecomposeLayerNormalization(self._graph, self._graph_name))
        return self
