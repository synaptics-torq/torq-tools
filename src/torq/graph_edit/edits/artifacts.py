# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers
from ...utils.onnx import normalize_layer_name


@dataclass
class ExtractConstantLUT(OnnxGraphEdit):

    lut_shape: tuple[int, ...]
    save_to: os.PathLike | str
    inp_name: str | None = None

    def match(self, node: gs.Node) -> bool:
        if node.op != "Gather" or len(node.inputs) < 2:
            return False
        if node.attrs.get("axis", 0) != 0:
            return False
        lut = node.inputs[0]
        if not isinstance(lut, gs.Constant):
            return False
        lut_shape = lut.values.shape
        if lut_shape == self.lut_shape:
            return True
        return False

    def transform(self, node: gs.Node):
        if not (node.op == "Gather" and len(node.inputs) >= 2 and isinstance((lut := node.inputs[0]), gs.Constant)):
            raise ValueError(f"Gather node '{node.name}' does not have a constant data input")
        if (axis := node.attrs.get("axis", 0)) != 0:
            raise ValueError(f"Only support axis = 0 for LUT, found axis = {axis} for Gather node '{node.name}'")
        
        lut_data = lut.values
        if not isinstance(lut_data, np.ndarray):
            self._logger.warning("Constant data is not NumPy array, attempting to load lazy values")
            try:
                lut_data = lut_data.load()
            except AttributeError as e:
                raise ValueError(f"Constant data for {node.name} is not loadable") from e
            if not isinstance(lut_data, np.ndarray):
                raise ValueError(f"Invalid Constant data type: {type(lut_data)}")
        
        self.save_to = Path(self.save_to)
        self.save_to.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.save_to, lut_data)

        if not self.inp_name:
            self.inp_name = f"extracted_lut_{normalize_layer_name(node.name)}_input"
        lut_out: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(lut_out.outputs)        
        lut_entry_inp = gs.Variable(
            name=self.inp_name,
            dtype=lut_out.dtype,
            shape=lut_out.shape
        )
        rewire_consumers(consumers, lut_out, lut_entry_inp)
        self.graph.inputs.append(lut_entry_inp)
        node.outputs.clear()
        self._logger.debug(
            "Extracted LUT from '%s', consumers redirected to graph input '%s'",
            node.name, self.inp_name
        )

@dataclass
class TrimLMHeadVocab(OnnxGraphEdit):
    """
    Trim LM head weight matrix to a subset of tokens.

    The weight matrix is sliced from [hidden, vocab] to [hidden, kept_count].
    If include_argmax is True, an ArgMax is appended to output the compact index.
    Otherwise the output is the trimmed logits tensor [1, 1, kept_count].

    The caller is responsible for mapping compact_idx -> original token ID via
    kept_token_ids[compact_idx].

    Args:
        kept_token_ids (np.ndarray): 1-D array of original token IDs to keep, in the
            order they should appear in the trimmed weight (sorted recommended).
        output_name (str): Name of the MatMul output to match (default: "logits").
        save_lut (Path | str | None): If provided, save kept_token_ids to this .npy path.
        include_argmax (bool): If True, append ArgMax to the graph (default: False).
    """

    kept_token_ids: np.ndarray
    output_name: str = "logits"
    save_lut: Path | str | None = None
    include_argmax: bool = False

    def __post_init__(self):
        self.kept_token_ids = np.asarray(self.kept_token_ids, dtype=np.int64)
        if self.kept_token_ids.ndim != 1 or len(self.kept_token_ids) == 0:
            raise ValueError("kept_token_ids must be a non-empty 1-D array")
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul" or not node.outputs:
            return False
        return node.outputs[0].name == self.output_name

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        weight_inp = node.inputs[1]
        if not isinstance(weight_inp, gs.Constant):
            raise ValueError(
                f"Expected constant weight for LM head MatMul, got {type(weight_inp).__name__}"
            )

        W = weight_inp.values
        if W.ndim != 2:
            raise ValueError(f"Expected 2-D weight matrix, got shape {W.shape}")

        hidden_size, vocab_size = W.shape
        if np.any(self.kept_token_ids >= vocab_size) or np.any(self.kept_token_ids < 0):
            raise ValueError(
                f"kept_token_ids contains values outside [0, {vocab_size})"
            )
        kept_count = len(self.kept_token_ids)

        W_trimmed = W[:, self.kept_token_ids]
        trimmed_weight = gs.Constant(
            name=weight_inp.name + "_trimmed",
            values=W_trimmed,
            export_dtype=getattr(weight_inp, "export_dtype", None),
        )

        node.inputs[1] = trimmed_weight

        logits_out = node.outputs[0]
        old_shape = list(logits_out.shape) if logits_out.shape else None
        if old_shape and len(old_shape) >= 1:
            new_logits_shape = old_shape[:-1] + [kept_count]
        else:
            new_logits_shape = [1, 1, kept_count]

        trimmed_logits = gs.Variable(
            name="logits",
            dtype=logits_out.dtype,
            shape=new_logits_shape,
        )
        consumers = list(logits_out.outputs)
        node.outputs[0] = trimmed_logits

        if self.include_argmax:
            final_out = self.graph.layer(
                name="lm_head_argmax",
                op="ArgMax",
                attrs={"axis": -1, "keepdims": 0},
                inputs=[trimmed_logits],
                outputs=[gs.Variable(
                    name="compact_token_idx",
                    dtype=np.int64,
                    shape=new_logits_shape[:-1],
                )],
            )[0]
        else:
            final_out = trimmed_logits

        rewire_consumers(consumers, logits_out, final_out)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is logits_out:
                self.graph.outputs[i] = final_out

        if self.save_lut is not None:
            lut_path = Path(self.save_lut)
            lut_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(lut_path, self.kept_token_ids)
            self._logger.debug("Saved token ID LUT to '%s'", lut_path)

        self._logger.debug(
            "Trimmed LM head vocab: %d -> %d tokens (argmax=%s)",
            vocab_size, kept_count, self.include_argmax,
        )

@dataclass
class SplitLMHead(OnnxGraphEdit):
    """
    Extract the final LM head into a standalone graph.

    The main graph output is replaced with the non-constant MatMul input, named
    ``hidden_states_name``. The extracted LM head graph accepts that tensor as
    its only input, preserves the original LM head output, and is saved to
    ``save_to``.
    """

    save_to: Path | str
    output_name: str = "logits"
    hidden_states_name: str = "last_hidden_states"

    def _find_lm_head_matmul(self) -> gs.Node:
        for node in self.graph.nodes:
            if node.op != "MatMul" or not node.outputs:
                continue
            output = node.outputs[0]
            if output.name != self.output_name:
                continue
            if any(graph_output is output for graph_output in self.graph.outputs):
                return node
        raise ValueError(f"Could not find final LM head MatMul feeding graph output '{self.output_name}'")

    @staticmethod
    def _select_hidden_states(node: gs.Node) -> gs.Variable:
        if len(node.inputs) != 2:
            raise ValueError(f"LM head MatMul '{node.name}' must have 2 inputs, found {len(node.inputs)}")
        if all(isinstance(inp, gs.Constant) for inp in node.inputs):
            raise ValueError(f"LM head MatMul '{node.name}' is invalid because both inputs are constant")
        hidden_states = next(
            (inp for inp in node.inputs if not isinstance(inp, gs.Constant)),
            None
        )
        if not isinstance(hidden_states, gs.Variable):
            raise ValueError(
                f"Expected LM head hidden states to be a graph variable, got {type(hidden_states).__name__}"
            )
        return hidden_states

    def _extract_lm_head_graph(self) -> gs.Graph:
        lm_head = self._find_lm_head_matmul()
        lm_head_logits = lm_head.outputs[0]
        hidden_states = self._select_hidden_states(lm_head)
        lm_head_input = gs.Variable(
            name=self.hidden_states_name,
            dtype=hidden_states.dtype,
            shape=hidden_states.shape,
        )
        for idx, inp in enumerate(lm_head.inputs):
            if inp is hidden_states:
                lm_head.inputs[idx] = lm_head_input
                break
        lm_head_graph = gs.Graph(
            name="main",
            nodes=[lm_head],
            inputs=[lm_head_input],
            outputs=[lm_head_logits],
        )
        return lm_head_graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul" or not node.outputs:
            return False
        output = node.outputs[0]
        if output.name != self.output_name:
            return False
        return any(graph_output is output for graph_output in self.graph.outputs)

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        hidden_states = self._select_hidden_states(node)
        lm_head_graph = self._extract_lm_head_graph()
        lm_head_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(lm_head_graph),
            True, True, True
        )
        save_to = Path(self.save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(lm_head_model, save_to)

        logits = node.outputs[0]
        hidden_states.name = self.hidden_states_name
        for idx, graph_output in enumerate(self.graph.outputs):
            if graph_output is logits:
                self.graph.outputs[idx] = hidden_states

        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug(
            "Split LM head MatMul '%s'; graph output '%s' now exposes '%s'",
            node.name,
            logits.name,
            hidden_states.name,
        )
        self._logger.debug("Saved split LM head to '%s'", save_to)
