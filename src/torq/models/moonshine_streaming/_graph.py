# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Graph edits and editor for Moonshine Streaming static-shape conversion.

Handles the streaming-specific graph patterns produced by torch.onnx.export(dynamo=True):
  - Dynamo Softmax nodes named ``node_Softmax_NNN`` → renamed for MaskFutureAttentionScores
  - Shape(past_self_key_*) → Squeeze seq-len pattern (vs. Shape → Gather in non-streaming)
  - Concat→output KV cache pattern (same as non-streaming, handled by ReplaceDynamicKVCache)
"""

import os
from dataclasses import dataclass

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
    rewire_consumers,
)
from ...graph_edit.edits import CommonGraphEditsMixin
from ...graph_edit.onnx import OnnxGraphEdit


@dataclass
class ReplaceShapeSeqLen(OnnxGraphEdit):
    """Replace Shape(past_self_key_*) → Squeeze seq-len extraction with cur_len.

    In the streaming decoder_with_past, the dynamo export generates:
        Shape(past_self_key_N) → Squeeze(axis=2) → [consumers]
    This dynamically extracts the KV sequence length. After making the graph
    static, the Shape always returns max_tokens which is incorrect at runtime.
    We replace the Squeeze output with the provided cur_len variable.
    """

    cur_len: gs.Variable

    def match(self, node: gs.Node) -> bool:
        if node.op != "Shape":
            return False
        if not node.inputs or "past_self_key" not in node.inputs[0].name:
            return False
        if not node.outputs or not node.outputs[0].outputs:
            return False
        consumer = node.outputs[0].outputs[0]
        return consumer.op == "Squeeze"

    def transform(self, node: gs.Node):
        squeeze_node = node.outputs[0].outputs[0]
        squeeze_out = squeeze_node.outputs[0]
        consumers = list(squeeze_out.outputs)
        rewire_consumers(consumers, squeeze_out, self.cur_len)

        # Disconnect
        node.inputs.clear()
        squeeze_node.outputs.clear()

        self._logger.debug(
            "Replaced Shape(%s) → Squeeze with cur_len", node.name
        )


class MoonshineStreamingOnnxGraphEditor(OnnxGraphEditor, CommonGraphEditsMixin):

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
    ) -> "MoonshineStreamingOnnxGraphEditor":
        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)
        graph = gs.import_onnx(onnx_model)
        return cls(graph, component, export_dtype)

    # ── I/O dimension fixers ─────────────────────────────────────────────

    def fix_preprocessor_io(self, num_samples: int, enc_seq_len: int):
        """Set preprocessor I/O to static shapes.

        Inputs:  [1, num_samples] × 2
        Outputs: [1, enc_seq_len, hidden], [1, enc_seq_len]
        """
        to_fix = [
            FixedDimMapping("batch", DimMatchType.EXACT, 1),
            FixedDimMapping("audio_length", DimMatchType.EXACT, num_samples),
            # The output dim is a complex expression containing "audio_length//80"
            FixedDimMapping("audio_length//80", DimMatchType.CONTAINS, enc_seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_encoder_io(self, enc_seq_len: int):
        """Set encoder I/O to static shapes.

        Inputs:  [1, enc_seq_len, hidden], [1, enc_seq_len]
        Outputs: [1, enc_seq_len, hidden]
        """
        to_fix = [
            FixedDimMapping("batch", DimMatchType.EXACT, 1),
            FixedDimMapping("seq_length", DimMatchType.EXACT, enc_seq_len),
        ]
        self.fix_io_dims(to_fix)

    def fix_decoder_io(
        self,
        enc_seq_len: int,
        max_tokens: int,
        with_past: bool,
    ):
        """Set decoder I/O to static shapes.

        For decoder (no past): fixes batch and enc_seq dims.
        For decoder_with_past: also fixes past_seq → max_tokens and
        past_seq + 1 → max_tokens on outputs.
        """
        to_fix = [
            FixedDimMapping("batch", DimMatchType.EXACT, 1),
            FixedDimMapping("enc_seq", DimMatchType.CONTAINS, enc_seq_len),
        ]
        if with_past:
            to_fix.append(
                FixedDimMapping("past_seq", DimMatchType.CONTAINS, max_tokens),
            )
        self.fix_io_dims(to_fix)

    # ── Softmax renaming ─────────────────────────────────────────────────

    def rename_self_attn_softmax(self):
        """Rename self-attention Softmax nodes from dynamo names to the
        ``layers.N/self_attn/Softmax`` pattern expected by
        :class:`MaskFutureAttentionScores`.

        Self-attention Softmax nodes are identified by having ``past_seq``
        in their input tensor's last dimension (before ``fix_io_dims``
        replaces symbolic dims with integers).
        """
        layer = 0
        for node in self._graph.nodes:
            if node.op != "Softmax":
                continue
            inp = node.inputs[0]
            if not hasattr(inp, "shape") or not inp.shape or len(inp.shape) < 4:
                continue
            last_dim = inp.shape[-1]
            if isinstance(last_dim, str) and "past_seq" in last_dim:
                old_name = node.name
                node.name = f"layers.{layer}/self_attn/Softmax"
                self._logger.debug("Renamed '%s' → '%s'", old_name, node.name)
                layer += 1
        return self

    def replace_shape_seq_len(self, cur_len: gs.Variable):
        """Replace Shape(past_self_key_*) → Squeeze with cur_len."""
        self.apply_edit(ReplaceShapeSeqLen(self._graph, self._graph_name, cur_len))
        return self
