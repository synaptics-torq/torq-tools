# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from __future__ import annotations

import os

import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
)
from ...graph_edit.edits import *


class LiquidOnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):
    """
    Graph editor for the LiquidAI LFM2.5 hybrid (conv + attention) model.

    LFM2.5 has 16 layers, with a mix of `conv` and `full_attention` layer
    types.  Conv layers expose a fixed-size sliding window state
    `past_conv.N` / `present_conv.N` of shape `[B, conv_dim, conv_L_cache]`
    (already static).  Attention layers expose standard dynamic KV cache
    tensors `past_key_values.X.{key,value}` that need static replacement.
    """

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
    ) -> "LiquidOnnxGraphEditor":
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
        past_seq_len_dim: str = "past_sequence_length",
        total_seq_len_dim: str = "total_sequence_length",
        num_logits_dim: str = "num_logits_to_keep",
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(past_seq_len_dim, DimMatchType.CONTAINS, seq_len),
            FixedDimMapping(total_seq_len_dim, DimMatchType.CONTAINS, seq_len),
            FixedDimMapping(num_logits_dim, DimMatchType.CONTAINS, 1),
        ]
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def fold_num_logits_to_keep(self, value: int = 1):
        """Replace the `num_logits_to_keep` scalar graph input with a constant.

        LFM2.5 takes a scalar `num_logits_to_keep` input that controls how
        many trailing logits the model returns.  For autoregressive decode
        we always want the last logit (value=1), so we fold the input out
        of the graph by wiring all consumers to a constant.
        """
        import numpy as np

        target_name = "num_logits_to_keep"
        target = None
        for inp in self._graph.inputs:
            if inp.name == target_name:
                target = inp
                break
        if target is None:
            return self

        const = gs.Constant(
            name=f"{target_name}_const",
            values=np.array(value, dtype=np.int64),
        )
        for node in self._graph.nodes:
            for i, t in enumerate(node.inputs):
                if t is target:
                    node.inputs[i] = const

        self._graph.inputs = [t for t in self._graph.inputs if t is not target]
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()
        return self
