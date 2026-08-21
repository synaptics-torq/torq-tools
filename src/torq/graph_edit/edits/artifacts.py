# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers, replace_node
from ...utils.onnx import normalize_layer_name


@dataclass(frozen=True)
class _LMHead:
    output_node: gs.Node
    matmul: gs.Node
    nodes: tuple[gs.Node, ...]
    hidden_states: gs.Variable
    weight: gs.Constant
    vocab_parameters: tuple[gs.Constant, ...]
    logits: gs.Variable


def _producer(tensor: gs.Tensor, op: str) -> gs.Node | None:
    if len(tensor.inputs) != 1 or tensor.inputs[0].op != op:
        return None
    return tensor.inputs[0]


def _match_lm_head(node: gs.Node, output_name: str) -> _LMHead | None:
    if not node.outputs or node.outputs[0].name != output_name:
        return None

    if node.op == "MatMul" and len(node.inputs) == 2:
        hidden_states, weight = node.inputs
        if isinstance(hidden_states, gs.Variable) and isinstance(weight, gs.Constant):
            return _LMHead(node, node, (node,), hidden_states, weight, (weight,), node.outputs[0])

    # ORT dynamic integer quantization dequantizes MatMulInteger with
    # Cast(result) * (activation_scale * weight_scale).
    if node.op != "Mul" or len(node.inputs) != 2:
        return None
    cast = None
    for inp in node.inputs:
        if producer := _producer(inp, "Cast"):
            cast = producer
            break
    if cast is None or len(cast.inputs) != 1:
        return None
    matmul = _producer(cast.inputs[0], "MatMulInteger")
    if matmul is None or not 3 <= len(matmul.inputs) <= 4:
        return None
    dql = _producer(matmul.inputs[0], "DynamicQuantizeLinear")
    if dql is None or len(dql.inputs) != 1 or len(dql.outputs) != 3:
        return None
    hidden_states = dql.inputs[0]
    weight = matmul.inputs[1]
    if not isinstance(hidden_states, gs.Variable) or not isinstance(weight, gs.Constant):
        return None
    if matmul.inputs[2] is not dql.outputs[2]:
        return None

    scale_input = next(inp for inp in node.inputs if inp is not cast.outputs[0])
    scale_mul = _producer(scale_input, "Mul")
    if scale_mul is None or len(scale_mul.inputs) != 2 or dql.outputs[1] not in scale_mul.inputs:
        return None
    weight_scale = next(inp for inp in scale_mul.inputs if inp is not dql.outputs[1])
    if not isinstance(weight_scale, gs.Constant):
        return None
    vocab_parameters = [weight, weight_scale]
    if len(matmul.inputs) == 4:
        weight_zero_point = matmul.inputs[3]
        if not isinstance(weight_zero_point, gs.Constant):
            return None
        vocab_parameters.append(weight_zero_point)
    return _LMHead(
        node,
        matmul,
        (dql, matmul, cast, scale_mul, node),
        hidden_states,
        weight,
        tuple(vocab_parameters),
        node.outputs[0],
    )


def _require_lm_head(node: gs.Node, output_name: str) -> _LMHead:
    lm_head = _match_lm_head(node, output_name)
    if lm_head is None:
        raise ValueError(f"Node '{node.name}' is not a supported LM head producing '{output_name}'")
    return lm_head


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
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is lut_out:
                self.graph.outputs[i] = lut_entry_inp
        self.graph.inputs.append(lut_entry_inp)
        node.outputs.clear()
        self._logger.debug(
            "Extracted LUT from '%s', consumers redirected to graph input '%s'",
            node.name, self.inp_name
        )

@dataclass
class ComputeDequantizedLUT(OnnxGraphEdit):
    """
    Replace integer embeddings LUT + DequantizeLinear with a constant floating point LUT.
    """

    lut_path: os.PathLike | str
    export_dtype: onnx.TensorProto.DataType

    def _is_suitable_dequant(self, node: gs.Node) -> bool:
        return (
            node.op == "DequantizeLinear"
            and 2 <= len(node.inputs) <= 3
            and node.inputs[0] in self.graph.inputs
            and all(isinstance(inp, gs.Constant) for inp in node.inputs[1:])
        )

    def _remove_matching_dequant_nodes(
        self,
        emb_inp: gs.Variable,
        new_emb_inp: gs.Variable,
        target_scale: np.ndarray,
        target_zp: np.ndarray | None = None,
        *,
        target_axis: int = 1
    ) -> set[str]:

        def _same_scale_and_zp(node: gs.Node) -> bool:
            if not self._is_same_constant_value(node.inputs[1], target_scale):
                return False
            if target_zp is None:
                return len(node.inputs) == 2
            return (
                len(node.inputs) == 3
                and self._is_same_constant_value(node.inputs[2], target_zp)
            )

        removed = set()
        for consumer in emb_inp.outputs:
            if not self._is_suitable_dequant(consumer):
                continue
            if consumer.attrs.get("axis", 1) != target_axis:
                continue
            if not _same_scale_and_zp(consumer):
                continue
            replace_node(consumer, [new_emb_inp])
            removed.add(consumer.name)
            self._logger.debug(
                "Generated dequantized LUT from '%s'", consumer.name
            )
        return removed

    def _cleanup_emb_inputs(
        self,
        emb_inp: gs.Variable,
        new_emb_inp: gs.Variable,
        new_lut: np.ndarray,
        removed_node_names: set[str]
    ):
        replace_emb_inp = not emb_inp.outputs or all(node.name in removed_node_names for node in emb_inp.outputs)
        emb_inp_idx: int | None = None
        for i, inp in enumerate(self.graph.inputs):
            if inp is emb_inp:
                emb_inp_idx = i
        assert emb_inp_idx is not None, f"Fatal: {emb_inp.name} not in graph inputs"
        if replace_emb_inp:
            emb_inp_name = emb_inp.name
            self.graph.inputs[emb_inp_idx] = new_emb_inp
            self.graph.cleanup(True, True, True, True)
            new_emb_inp.name = emb_inp_name
            np.save(self.lut_path, new_lut)
            self._logger.debug(
                "Replaced LUT @ '%s' and graph input '%s' with dequantized values",
                self.lut_path, new_emb_inp.name
            )
        else:
            tag = uuid4().hex
            new_emb_inp.name = f"{emb_inp.name}-dequantized-{tag}"
            self.graph.inputs.insert(emb_inp_idx + 1, new_emb_inp)
            self.lut_path = Path(self.lut_path)
            new_lut_path = self.lut_path.with_stem(f"{self.lut_path.stem}-dequantized-{tag}")
            np.save(new_lut_path, new_lut)
            self.graph.cleanup(True, True, True, True)
            self._logger.debug(
                "Added new LUT @ '%s' and graph input '%s' with dequantized values",
                new_lut_path, new_emb_inp.name
            )

    def match(self, node: gs.Node) -> bool:
        return self._is_suitable_dequant(node)

    def transform(self, node: gs.Node):
        if len(node.inputs) > 3:
            raise ValueError("Expected 3 or less inputs for onnx.DequantizeLinear")
        x_q: np.ndarray = np.load(self.lut_path).astype(np.float32)
        scale: np.ndarray = node.inputs[1].values.astype(np.float32)
        if len(node.inputs) > 2:
            zp: np.ndarray = node.inputs[2].values.astype(np.float32)
        else:
            zp: np.ndarray = np.zeros_like(x_q)
        x = (x_q - zp) * scale
        export_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(self.export_dtype)
        new_lut = x.astype(export_np_dtype)
        emb_inp: gs.Variable = next((i for i in self.graph.inputs if i is node.inputs[0]), None)
        if emb_inp is None:
            raise ValueError(f"Could not find DequantizeLinear op '{node.name}' input in graph inputs")
        tag = uuid4().hex
        new_emb_inp = gs.Variable(
            f"{emb_inp.name}-dequantized-{tag}",
            export_np_dtype,
            emb_inp.shape,
        )
        params = [inp.values for inp in node.inputs[1:]]
        removed = self._remove_matching_dequant_nodes(
            emb_inp,
            new_emb_inp,
            *params,
            target_axis=node.attrs.get("axis", 1)
        )
        self._cleanup_emb_inputs(
            emb_inp,
            new_emb_inp,
            new_lut,
            removed
        )

@dataclass
class TrimLMHeadVocab(OnnxGraphEdit):
    """
    Trim an LM head to a subset of tokens.

    The weight matrix is sliced from [hidden, vocab] to [hidden, kept_count],
    along with any per-vocabulary quantization scale and zero point.
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
        return _match_lm_head(node, self.output_name) is not None

    def transform(self, node: gs.Node):
        lm_head = _require_lm_head(node, self.output_name)
        W = lm_head.weight.values
        if W.ndim != 2:
            raise ValueError(f"Expected 2-D weight matrix, got shape {W.shape}")

        _, vocab_size = W.shape
        if np.any(self.kept_token_ids >= vocab_size) or np.any(self.kept_token_ids < 0):
            raise ValueError(
                f"kept_token_ids contains values outside [0, {vocab_size})"
            )
        kept_count = len(self.kept_token_ids)

        W_trimmed = W[:, self.kept_token_ids]
        replacements = []
        for parameter in lm_head.vocab_parameters:
            values = parameter.values
            if parameter is lm_head.weight:
                trimmed_values = W_trimmed
            elif values.ndim == 0 or values.shape == (1,):
                continue
            elif values.ndim == 1 and values.shape[0] == vocab_size:
                trimmed_values = values[self.kept_token_ids]
            else:
                raise ValueError(
                    f"Expected scalar or [{vocab_size}] LM head parameter '{parameter.name}', got shape {values.shape}"
                )
            replacements.append((
                parameter,
                gs.Constant(
                    name=parameter.name + "_trimmed",
                    values=trimmed_values,
                    export_dtype=getattr(parameter, "export_dtype", None),
                ),
            ))
        for lm_head_node in lm_head.nodes:
            for idx, inp in enumerate(lm_head_node.inputs):
                replacement = next((new for old, new in replacements if inp is old), None)
                if replacement is not None:
                    lm_head_node.inputs[idx] = replacement

        for lm_head_node in lm_head.nodes[lm_head.nodes.index(lm_head.matmul):]:
            for output in lm_head_node.outputs:
                if output.shape and output.shape[-1] == vocab_size:
                    output.shape = list(output.shape[:-1]) + [kept_count]

        logits_out = lm_head.logits
        old_shape = list(logits_out.shape) if logits_out.shape else None
        if old_shape and len(old_shape) >= 1:
            new_logits_shape = old_shape[:-1] + [kept_count]
        else:
            new_logits_shape = [1, 1, kept_count]

        trimmed_logits = gs.Variable(
            name=logits_out.name,
            dtype=logits_out.dtype,
            shape=new_logits_shape,
        )
        consumers = list(logits_out.outputs)
        lm_head.output_node.outputs[0] = trimmed_logits

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

    The main graph output is replaced with the LM head activation input, named
    ``hidden_states_name``. The extracted graph accepts that tensor as its only
    input and preserves either a floating-point MatMul or a dynamically
    quantized MatMulInteger head.
    """

    save_to: Path | str
    output_name: str = "logits"
    hidden_states_name: str = "last_hidden_states"

    def _extract_lm_head_graph(self, lm_head: _LMHead) -> gs.Graph:
        lm_head_input = gs.Variable(
            name=self.hidden_states_name,
            dtype=lm_head.hidden_states.dtype,
            shape=lm_head.hidden_states.shape,
        )
        for lm_head_node in lm_head.nodes:
            for idx, inp in enumerate(lm_head_node.inputs):
                if inp is lm_head.hidden_states:
                    lm_head_node.inputs[idx] = lm_head_input
        lm_head_graph = gs.Graph(
            name="main",
            nodes=list(lm_head.nodes),
            inputs=[lm_head_input],
            outputs=[lm_head.logits],
        )
        return lm_head_graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

    def match(self, node: gs.Node) -> bool:
        lm_head = _match_lm_head(node, self.output_name)
        return bool(lm_head and any(graph_output is lm_head.logits for graph_output in self.graph.outputs))

    def transform(self, node: gs.Node):
        lm_head = _require_lm_head(node, self.output_name)
        hidden_states = lm_head.hidden_states
        lm_head_graph = self._extract_lm_head_graph(lm_head)
        lm_head_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(lm_head_graph),
            True, True, True
        )
        save_to = Path(self.save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(lm_head_model, save_to)

        logits = lm_head.logits
        hidden_states.name = self.hidden_states_name
        for idx, graph_output in enumerate(self.graph.outputs):
            if graph_output is logits:
                self.graph.outputs[idx] = hidden_states

        for lm_head_node in lm_head.nodes:
            lm_head_node.inputs.clear()
            lm_head_node.outputs.clear()
        self._logger.debug(
            "Split LM head '%s'; graph output '%s' now exposes '%s'",
            lm_head.matmul.name,
            logits.name,
            hidden_states.name,
        )
        self._logger.debug("Saved split LM head to '%s'", save_to)
