# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper, shape_inference, TensorProto

from ..utils.onnx import drop_empty_name_value_info


def rewire_consumers(
    consumers: list[gs.Node],
    orig: gs.Constant | gs.Variable,
    new: gs.Constant | gs.Variable
):
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp is orig:
                consumer.inputs[i] = new


class DimMatchType(Enum):
    EXACT    = auto()
    CONTAINS = auto()


@dataclass(frozen=True)
class FixedDimMapping:
    match_name: str
    match_type: DimMatchType
    value: int

    def matches(self, dim_name: str):
        if self.match_type == DimMatchType.EXACT:
            return self.match_name == dim_name
        else:
            return self.match_name in dim_name


@dataclass
class OnnxGraphEdit(ABC):
    graph: gs.Graph
    graph_name: str
    requires_shape_inference: bool = field(default=False, init=False)

    def __post_init__(self):
        self._logger = logging.getLogger(f"{self.name}[{self.graph_name}]")

    @abstractmethod
    def match(self, node: gs.Node) -> bool: ...

    @abstractmethod
    def transform(self, node: gs.Node): ...

    def finalize(self, node: gs.Node):
        pass

    def apply_edit(self, node: gs.Node):
        if self.match(node):
            self.transform(node)
            self.finalize(node)

    def _check_node_op(self, node: gs.Node, expected_op: str):
        if not isinstance(node, gs.Node):
            self._logger.error("Expected gs.Node instance, got '%s'", str(type(node)))
            raise TypeError(f"{self.name}[{self.graph_name}]: Expected gs.Node instance, got '{type(node)}'")
        if node.op != expected_op:
            self._logger.error("Expected '%s' node, got '%s'", expected_op, node.op)
            raise ValueError(f"{self.name}[{self.graph_name}]: Expected '{expected_op}' node, got '{node.op}'")

    def __call__(self, node: gs.Node):
        self.apply_edit(node)

    @property
    def name(self) -> str:
        return str(self.__class__.__name__)


class OnnxGraphEditor:

    def __init__(
        self,
        graph: gs.Graph,
        graph_name: str,
        edits: Iterable[OnnxGraphEdit] | None = None,
        export_dtype: onnx.TensorProto.DataType | None = None,
        is_rnn: bool = False,
    ):
        self._export_dtype = export_dtype
        self._graph = graph
        self._graph_name = graph_name
        self._graph_bak = self._graph.copy()
        self._logger = logging.getLogger(str(self))
        self._edits: dict[str, OnnxGraphEdit] = {}
        # Restoring omitted RNN outputs is only relevant for graphs that
        # actually contain RNN/GRU/LSTM nodes; opt-in via ``is_rnn`` so we
        # don't run the post-pass on every unrelated graph.
        self._is_rnn = bool(is_rnn)
        if isinstance(edits, Iterable):
            self.register_edits(edits)

    @property
    def edits(self) -> list[str]:
        return sorted(self._edits.keys())

    @property
    def graph(self) -> gs.Graph:
        return self._graph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._graph_name}]"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._logger.error(
                "Exception during graph edits; restored original graph: %s",
                exc,
            )
            self._graph = self._graph_bak
        return False

    @staticmethod
    def restore_rnn_output_arity(graph: gs.Graph) -> None:
        from .edits import DecomposeBidirectionalRnn

        DecomposeBidirectionalRnn.restore_output_arity(graph)

    def to_onnx(
        self,
        check_type: bool = True,
        strict_mode: bool = True,
        data_prop: bool = True,
        override_ir: int | None = None
    ) -> onnx.ModelProto:
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
        if self._is_rnn:
            self.restore_rnn_output_arity(self._graph)
        self._clear_intermediate_shapes()
        onnx_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(self._graph),
            check_type=check_type,
            strict_mode=strict_mode,
            data_prop=data_prop,
        )
        drop_empty_name_value_info(onnx_model)
        if isinstance(override_ir, int):
            self._graph.producer_version
            onnx_model.ir_version = override_ir
        return onnx_model

    def register_edit(self, edit: OnnxGraphEdit, edit_name: str | None = None):
        edit_name = edit_name or edit.name
        self._edits[edit_name] = edit

    def register_edits(self, edits: Mapping[str, OnnxGraphEdit] | Iterable[OnnxGraphEdit]):
        if not edits:
            self._logger.warning("No graph edit functions to register")
            return

        if isinstance(edits, Mapping):
            pairs = edits.items()
        elif isinstance(edits, Iterable):
            pairs = ((edit.name, edit) for edit in edits)
        else:
            raise TypeError(
                f"Expected Mapping[str, OnnxGraphEdit] or Iterable[OnnxGraphEdit], got {type(edits).__name__}"
            )
        for name, edit in pairs:
            self.register_edit(edit, name)

    def apply_edit(self, edit: OnnxGraphEdit | str):
        if isinstance(edit, str):
            edit = self._edits[edit]
        # Shape inference replaces self._graph with a freshly imported gs.Graph,
        # so re-bind the edit to the current graph before walking it.
        edit.graph = self._graph
        for node in self._graph.nodes:
            edit(node)
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
        if self._is_rnn:
            self.restore_rnn_output_arity(self._graph)
        if edit.requires_shape_inference:
            self._infer_shapes()
        return self

    def apply_edits(self, edits: Sequence[OnnxGraphEdit | str]):
        for edit in edits:
            self.apply_edit(edit)
        return self

    def _clear_intermediate_shapes(self):
        inputs_outputs_names = {t.name for t in self._graph.inputs + self._graph.outputs}
        for name, tensor in self._graph.tensors().items():
            if name in inputs_outputs_names:
                continue
            if isinstance(tensor, gs.Variable):
                tensor.shape = None

    def _infer_shapes(self):
        self._clear_intermediate_shapes()
        model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(self._graph),
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )
        drop_empty_name_value_info(model)
        self._graph = gs.import_onnx(model)

    def fix_io_dims(self, to_fix: list[FixedDimMapping] | None = None):
        to_fix = tuple(to_fix or [])
        
        # 1. Fix shapes of all inputs/outputs, raising ValueError if any dynamic dimension is unmapped.
        for tensor in self._graph.inputs + self._graph.outputs:
            old_shape = list(tensor.shape) if tensor.shape is not None else None
            if tensor.shape is not None:
                for i, dim in enumerate(tensor.shape):
                    if not isinstance(dim, str):
                        continue
                    if dim.isdigit():
                        tensor.shape[i] = int(dim)
                        continue
                    for fixed_dim in to_fix:
                        if fixed_dim.matches(dim):
                            tensor.shape[i] = fixed_dim.value
                            break
                    else: # no match found, unexpected dim!
                        raise ValueError(
                            f"Unexpected dynamic dimension '{dim}' in tensor '{tensor.name}'"
                        )
                if tensor.shape != old_shape:
                    self._logger.debug(
                        "Fixing IO '%s': %s -> %s",
                        tensor.name,
                        str(old_shape),
                        str(tensor.shape)
                    )



    def _reorder_graph_io(
        self,
        io_type: str,
        io_name: str,
        new_pos: int
    ):
        assert io_type in ("input", "output")
        io_tensors = self._graph.inputs if io_type == "input" else self._graph.outputs

        # Normalize new_pos to a concrete [0, len] insertion index
        n = len(io_tensors)
        if new_pos < 0:
            new_pos = n + new_pos + 1  # so -1 means "end"
        new_pos = max(0, min(new_pos, n))

        for i, t in enumerate(io_tensors):
            if not isinstance(t, gs.Variable):
                self._logger.warning("Found non variable graph %s '%s', (%s)", io_type, getattr(t, "name", "<unnamed>"), type(t))
                continue
            if t.name == io_name:
                io_tensors.pop(i)
                if new_pos > i:
                    new_pos -= 1
                io_tensors.insert(new_pos, t)
                return

        self._logger.warning("%s '%s' not found in graph; no reordering performed", io_type.capitalize(), io_name)


    def reorder_graph_input(
        self,
        input_name: str,
        new_pos: int
    ):
        self._reorder_graph_io("input", input_name, new_pos)

    def reorder_graph_output(
        self,
        output_name: str,
        new_pos: int
    ):
        self._reorder_graph_io("output", output_name, new_pos)

    def apply_fixed_input_shapes(
        self,
        input_shapes: Mapping[str, Sequence[int]]
    ):
        """Stamp explicit static shapes onto matching ``graph.input`` tensors.

        Output shapes are left for shape inference to derive from the now-fixed
        inputs.
        """
        if not input_shapes:
            return self
        for tensor in self._graph.inputs:
            if tensor.name not in input_shapes:
                continue
            old_shape = list(tensor.shape) if tensor.shape else None
            tensor.shape = [int(d) for d in input_shapes[tensor.name]]
            self._logger.debug(
                "input '%s' shape %s -> %s", tensor.name, old_shape, tensor.shape
            )
        return self

    def freeze_shape_seeds(
        self,
        input_shapes: Mapping[str, Sequence[int]]
    ):
        """Constant-fold shape-arithmetic subgraphs feeding Reshape/Expand/Slice/Pad.

        Evaluates each shape-control input under onnxruntime with zero-valued
        feeds (only shape arithmetic matters), replaces consumers with frozen
        initializers, and prunes the now-orphan shape-computation nodes.
        """
        if not input_shapes:
            self._logger.debug("no input_shapes given; freeze_shape_seeds is a no-op")
            return self

        model = gs.export_onnx(self._graph)
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            self._logger.debug("pre-pass shape_inference skipped: %s", exc)

        seeds = _find_shape_seed_tensors(model)
        producer, initializers, graph_inputs = _build_value_maps(model)
        seen: set[str] = set()
        candidates: list[str] = []
        for s in seeds:
            if not s or s in seen or s in initializers or s in graph_inputs:
                continue
            seen.add(s)
            p = producer.get(s)
            if p is not None and p.op_type in _SHAPE_OPS:
                candidates.append(s)
        if not candidates:
            return self

        feeds = {
            name: np.zeros(tuple(shape), dtype=np.float32)
            for name, shape in input_shapes.items()
        }
        for s in candidates:
            p = producer[s]
            self._logger.debug("freeze seed %s <- %s (%s)", s, p.name, p.op_type)

        # _evaluate_shape_seed_tensors deepcopies internally, so it's safe to
        # mutate `model` afterwards.
        frozen = _evaluate_shape_seed_tensors(model, candidates, feeds)
        _replace_consumers_with_frozen_initializers(model, frozen)
        _prune_dead_shape_nodes(model)

        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            self._logger.debug("post-pass shape_inference skipped: %s", exc)
        self._logger.debug("froze %d shape seed(s)", len(frozen))
        self._graph = gs.import_onnx(model)
        return self


# -----------------------------------------------------------------------------
# Module-level helpers used by OnnxGraphEditor.freeze_shape_seeds.
# -----------------------------------------------------------------------------

_SHAPE_OPS: frozenset[str] = frozenset(
    {
        "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat",
        "Equal", "Where", "ConstantOfShape",
        "Mul", "Div", "Add", "Sub",
        "Cast", "Reshape", "Slice", "Expand", "Transpose",
        "Constant", "Identity",
    }
)


def _build_value_maps(model: onnx.ModelProto):
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}
    return producer, initializers, graph_inputs


def _shape_control_input_indices(node: onnx.NodeProto) -> list[int]:
    if node.op_type in ("Expand", "Reshape") and len(node.input) >= 2:
        return [1]
    if node.op_type == "Slice":
        return list(range(1, len(node.input)))
    if node.op_type == "Pad" and len(node.input) >= 2:
        # pads (idx 1) and axes (idx 3, opset 18+) are INT64 shape arithmetic.
        # constant_value (idx 2) matches data dtype and is virtually always a
        # literal initializer in practice; skip it to keep the probe INT64.
        idxs = [1]
        if len(node.input) >= 4 and node.input[3]:
            idxs.append(3)
        return idxs
    return []


def _find_shape_seed_tensors(model: onnx.ModelProto) -> list[str]:
    seeds: list[str] = []
    for node in model.graph.node:
        for idx in _shape_control_input_indices(node):
            if idx < len(node.input):
                seeds.append(node.input[idx])
    return seeds


_PROBE_MAX_IR_VERSION = 11


def _evaluate_shape_seed_tensors(
    model: onnx.ModelProto,
    tensor_names: list[str],
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the model with the named tensors exposed via Identity probes."""
    # Imported locally to avoid a hard dependency at module import time.
    from ..utils.ort import make_cpu_session  # noqa: PLC0415

    probe = copy.deepcopy(model)
    del probe.graph.output[:]
    out_names: list[str] = []
    for i, tname in enumerate(tensor_names):
        out_name = f"__probe_out_{i}"
        out_names.append(out_name)
        probe.graph.node.append(
            helper.make_node(
                "Identity", inputs=[tname], outputs=[out_name],
                name=f"_ProbeIdentity_{i}",
            )
        )
        probe.graph.output.append(
            helper.make_tensor_value_info(out_name, TensorProto.INT64, None)
        )
    probe = shape_inference.infer_shapes(probe)
    if probe.ir_version > _PROBE_MAX_IR_VERSION:
        probe.ir_version = _PROBE_MAX_IR_VERSION
    sess = make_cpu_session(probe.SerializeToString())
    vals = sess.run(out_names, feeds)
    return dict(zip(tensor_names, vals))


def _replace_consumers_with_frozen_initializers(
    model: onnx.ModelProto, frozen: dict[str, np.ndarray]
) -> None:
    rename_map: dict[str, str] = {}
    for old, value in frozen.items():
        new = f"{old}__frozen"
        rename_map[old] = new
        model.graph.initializer.append(numpy_helper.from_array(value, name=new))
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]
    for out in model.graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]


def _prune_dead_shape_nodes(model: onnx.ModelProto) -> None:
    needed = {o.name for o in model.graph.output}
    kept: list[onnx.NodeProto] = []
    changed = True
    while changed:
        changed = False
        for node in reversed(model.graph.node):
            if any(o in needed for o in node.output) and node not in kept:
                kept.append(node)
                for inp in node.input:
                    if inp and inp not in needed:
                        needed.add(inp)
                        changed = True
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)

    used = (
        {i.name for i in model.graph.input}
        | {o.name for o in model.graph.output}
        | {init.name for init in model.graph.initializer}
    )
    for n in model.graph.node:
        used.update(x for x in n.input if x)
        used.update(x for x in n.output if x)
    keep_vi = [vi for vi in model.graph.value_info if vi.name in used]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)
