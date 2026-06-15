# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs
from ..moonshine_streaming._graph import MoonshineStreamingOnnxGraphEditor
from ..moonshine_streaming.onnx import FixedDimMapping, DimMatchType

class MoonshineStreaming5SplitOnnxGraphEditor(MoonshineStreamingOnnxGraphEditor):

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
    ) -> "MoonshineStreaming5SplitOnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            component,
            export_dtype
        )

    def fix_frontend_io(
        self,
        chunk_len: int,
        batch_dim: str = "batch",
        chunk_len_dim: str = "chunk_len",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(chunk_len_dim, DimMatchType.EXACT, chunk_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_encoder_io(
        self,
        seq_len: int,
        batch_dim: str = "batch",
        seq_len_dim: str = "seq_length",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_adapter_io(
        self,
        seq_len: int,
        batch_dim: str = "batch",
        seq_len_dim: str = "seq_length",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_cross_kv_io(
        self,
        seq_len: int,
        batch_dim: str = "batch",
        seq_len_dim: str = "seq_length",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_decoder_kv_io(
        self,
        dec_seq_len: int,
        enc_seq_len: int,
        batch_dim: str = "batch",
        dec_seq_dim: str = "dec_seq",
        enc_seq_dim: str = "enc_seq",
        past_seq_dim: str = "past_seq",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(dec_seq_dim, DimMatchType.EXACT, dec_seq_len),
            FixedDimMapping(enc_seq_dim, DimMatchType.EXACT, enc_seq_len),
        ]
        # past_seq can be fixed using DimMatchType.CONTAINS
        to_fix.append(FixedDimMapping(past_seq_dim, DimMatchType.CONTAINS, dec_seq_len))
        self.fix_io_dims(to_fix)
