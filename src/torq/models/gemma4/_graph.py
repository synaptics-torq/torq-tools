# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

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


class Gemma4OnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):
    """Graph editor for Gemma-4-E2B's quantized (int4) ONNX export.

    Gemma-4 mixes three attention implementations in one decoder graph (see
    ``STATIC_EXPORT_PLAN.md``): 12 sliding-window layers as a fused
    ``GroupQueryAttention`` op (rotary applied *externally*, ``do_rotary=0``),
    3 full-attention layers as hand-unrolled MatMul/Softmax attention with
    their own KV-cache ``Concat``, and 20 KV-shared layers that reuse one of
    two donor layers' already-computed KV cache without any ``Concat`` of
    their own.
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
    ) -> "Gemma4OnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            export_dtype
        )

    def fix_io(
        self,
        max_kv_len: int,
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
            FixedDimMapping(past_seq_len_dim, DimMatchType.CONTAINS, max_kv_len),
            FixedDimMapping(total_seq_len_dim, DimMatchType.CONTAINS, max_kv_len),
            FixedDimMapping(num_logits_dim, DimMatchType.CONTAINS, 1),
        ]
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def normalize_kv_concat_axis(self):
        """Rewrite ``axis: 2`` -> ``axis: -2`` on the decoder's 3 hand-unrolled
        (manual) full-attention layers' KV-cache ``Concat`` nodes.

        ``ReplaceDynamicKVCache.match()`` requires an exact ``axis == -2``.
        Gemma-4's manual layers (the ones whose attention isn't a fused
        ``GroupQueryAttention`` op) emit their own ``Concat`` with the
        positive-axis encoding instead -- same semantics (second-to-last axis
        of a 4D tensor), different attribute value. Only touches Concat nodes
        that produce a ``present.N.{key,value}`` graph output, so it can't
        accidentally normalize an unrelated Concat elsewhere in the graph.
        """
        output_names = {o.name for o in self._graph.outputs}
        n = 0
        for node in self._graph.nodes:
            if node.op != "Concat" or node.attrs.get("axis") != 2:
                continue
            if not node.outputs or node.outputs[0].name not in output_names:
                continue
            if not node.outputs[0].name.startswith("present."):
                continue
            node.attrs["axis"] = -2
            n += 1
        self._logger.info("Normalized axis on %d KV-cache Concat node(s) (2 -> -2)", n)
        return self

    def fold_num_logits_to_keep(self, value: int = 1):
        """Replace the `num_logits_to_keep` scalar graph input with a constant.

        Autoregressive decode always wants the last logit (value=1), so the
        runtime input is folded out. Required for staticness: even once
        `fix_io_dims` pins the declared I/O dim to 1, the `logits` output's
        actual `Slice` start index still needs the *value* folded for shape
        inference to resolve a concrete internal shape.
        """
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
