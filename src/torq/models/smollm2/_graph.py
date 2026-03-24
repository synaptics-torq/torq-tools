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


class SmolLM2OnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):

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
