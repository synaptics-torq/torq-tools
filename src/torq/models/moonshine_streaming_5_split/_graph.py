# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import onnx
import onnx_graphsurgeon as gs
from ._graph_base import MoonshineStreamingOnnxGraphEditor
from .onnx import FixedDimMapping, DimMatchType

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
        feat_len: int | None = None,
        batch_dim: str = "batch",
        chunk_len_dim: str = "chunk_len",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(chunk_len_dim, DimMatchType.EXACT, chunk_len),
        ]
        if feat_len is not None:
            # StaticStreamingFrontendWrapper uses n_frames as a Python constant so all
            # internal shapes are concrete. Only the features output seq dim may still be
            # symbolic (legacy exporter prefix varies, so use CONTAINS).
            to_fix.append(FixedDimMapping("features_dim_1", DimMatchType.CONTAINS, feat_len))
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

    def fix_streaming_encoder_io(
        self,
        stable_len: int,
        batch_dim: str = "batch",
        stable_dim: str = "t_stable",
    ):
        """Fix the dynamic stable_features sequence dim on the streaming encoder."""
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(stable_dim, DimMatchType.EXACT, stable_len),
        ]
        # right_ctx and buf_* dims are already concrete integers in the exported graph.
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

    def make_decoder_static(self, max_tokens: int):
        """
        Apply the three proven graph edits to convert the dynamic decoder to a static
        pre-allocated KV-cache decoder.

        Adds a `current_len [1,1]` graph input that drives:
          - ReplaceDynamicKVCache: Where-blend self-KV at position cur_len
          - MaskFutureAttentionScores: additive -inf mask for positions > cur_len
          - AddCurrLenInput: replaces Shape->Gather seq-length reads with cur_len
        """
        import numpy as np
        import onnx_graphsurgeon as gs

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
