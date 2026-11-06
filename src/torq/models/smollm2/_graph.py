# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

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


class SmolLM2OnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin, CombineKVCacheMixin):

    def __init__(
        self,
        graph: gs.Graph,
        export_dtype: onnx.TensorProto.DataType | None = None,
        **editor_kwargs,
    ):
        super().__init__(
            graph,
            "model",
            export_dtype=export_dtype,
            **editor_kwargs,
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        export_dtype: onnx.TensorProto.DataType | None = None,
        **editor_kwargs,
    ) -> "SmolLM2OnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(
            graph,
            export_dtype,
            **editor_kwargs,
        )

    def fix_io(
        self,
        seq_len: int,
        dims: list[FixedDimMapping] | None = None,
        *,
        chunk_len: int = 1,
        batch_dim: str = "batch_size",
        seq_len_dim: str = "sequence_length",
        past_seq_len_dim: str = "past_sequence_length"
    ):
        to_fix = [
            FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
            FixedDimMapping(seq_len_dim, DimMatchType.EXACT, chunk_len),
            FixedDimMapping(past_seq_len_dim, DimMatchType.CONTAINS, seq_len),
        ]
        to_fix.extend(dims or [])
        self.fix_io_dims(to_fix)

    def fix_native_rotary_positions(self, chunk_len: int) -> bool:
        """
        Expand the shared rotary position to the chunk's consecutive positions.

        The source rotary is seq-major: ``MatMul(inv_freq [1, D/2, 1],
        Cast(Unsqueeze(position_ids)) [1, 1, S])`` -> ``[1, D/2, S]`` -> self
        Concat -> ``[1, S, D]`` cos/sin, whose per-layer Unsqueeze broadcasts
        over heads. ``fix_io(chunk_len)`` set the ``position_ids`` input's dim
        to B by name, so this restores it to the ABI's ``[1, 1]`` start
        position (the runtime ``Shape(position_ids)`` chain then sees exactly
        what decode sees) and rewires the rotary position to
        ``start + arange(B)``; the MatMul broadcasts ``[1, D/2, 1] x [1, 1, B]
        -> [1, D/2, B]`` and the chain flows through unchanged.

        Returns:
            True when the rotary position was expanded.
        """

        if chunk_len <= 1:
            return False

        matmul = next(
            (
                node
                for node in self.graph.nodes
                if node.op == "MatMul" and "rotary_emb" in node.name
            ),
            None,
        )
        if matmul is None or len(matmul.inputs) != 2:
            return False

        def derives_from_position(tensor, depth=5):
            current = tensor
            for _ in range(depth):
                if current.name == "position_ids":
                    return True
                if not current.inputs:
                    return False
                producer = current.inputs[0]
                if producer.op not in ("Cast", "Unsqueeze"):
                    return False
                current = producer.inputs[0]
            return False

        position = next(
            (
                inp
                for inp in matmul.inputs
                if derives_from_position(inp)
            ),
            None,
        )
        if position is None:
            return False

        # fix_io fixed the start-position input to [1, B] by dim name; the
        # family ABI (and the runtime Shape chain) needs the [1, 1] start.
        start = next(
            (t for t in self.graph.inputs if t.name == "position_ids"),
            None,
        )
        if start is not None:
            start.shape = [1, 1]

        dtype = position.dtype or np.float32
        chunk_positions = self.graph.layer(
            name="/model/rotary_emb/chunk_positions",
            op="Add",
            inputs=[
                position,
                gs.Constant(
                    "/model/rotary_emb/chunk_offsets",
                    np.arange(chunk_len, dtype=dtype).reshape(1, 1, chunk_len),
                ),
            ],
            outputs=[gs.Variable(
                "/model/rotary_emb/position_values",
                dtype=dtype,
                shape=[1, 1, chunk_len],
            )],
        )[0]
        matmul.inputs[1] = chunk_positions

        self._logger.info(
            "Expanded the rotary position to %d consecutive chunk positions",
            chunk_len,
        )

        return True
