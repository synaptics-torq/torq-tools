# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs
from ._graph_base import MoonshineStreamingOnnxGraphEditor
from .onnx import FixedDimMapping, DimMatchType


class MoonshineStreaming2SplitOnnxGraphEditor(MoonshineStreamingOnnxGraphEditor):

    def __init__(
        self,
        graph: gs.Graph,
        component: str,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ):
        super().__init__(graph, component, export_dtype=export_dtype)

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        component: str,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ) -> "MoonshineStreaming2SplitOnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(graph, component, export_dtype)

    def fix_fused_encoder_io(
        self,
        chunk_len: int,
        feat_len: int,
        batch_dim: str = "batch",
        chunk_len_dim: str = "chunk_len",
        stable_dim: str = "t_stable",
    ):
        """Fix all dynamic dims on the fused encoder (frontend + encoder + adapter + cross_kv)."""
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(chunk_len_dim, DimMatchType.EXACT, chunk_len),
            FixedDimMapping(stable_dim, DimMatchType.EXACT, feat_len),
            # feat_len may also appear as a CONTAINS match in output names
            FixedDimMapping("features_dim_1", DimMatchType.CONTAINS, feat_len),
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
            FixedDimMapping(past_seq_dim, DimMatchType.CONTAINS, dec_seq_len),
        ]
        self.fix_io_dims(to_fix)

    def make_decoder_static(self, max_tokens: int):
        """Convert the dynamic decoder to a static pre-allocated KV-cache decoder."""
        import numpy as np

        cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
        self._graph.inputs.append(cur_len_2d)
        squeeze_axes = gs.Constant("squeeze_axes_cur_len", np.array([0], dtype=np.int64))
        cur_len = self._graph.layer(
            name="current_len_to_1d",
            op="Squeeze",
            inputs=[cur_len_2d, squeeze_axes],
            outputs=[gs.Variable("current_len_squeezed", dtype=np.int64, shape=[1])],
        )[0]

        (
            self
            .replace_dynamic_kv_cache(cur_len, max_tokens)
            .mask_future_attn_scores(cur_len, max_tokens)
            .add_curr_len_input(cur_len)
        )
