# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from collections.abc import Iterable, Mapping, Sequence

import onnx
import onnx_graphsurgeon as gs


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

    def __post_init__(self):
        self._logger = logging.getLogger(f"{self.name}[{self.graph_name}]")

    @staticmethod
    def rewire_consumers(
        consumers: list[gs.Node],
        orig: gs.Constant | gs.Variable,
        new: gs.Constant | gs.Variable
    ):
        for consumer in consumers:
            for i, inp in enumerate(consumer.inputs):
                if inp is orig:
                    consumer.inputs[i] = new

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
        export_dtype: onnx.TensorProto.DataType | None = None
    ):
        self._export_dtype = export_dtype
        self._graph = graph
        self._graph_name = graph_name
        self._graph_bak = self._graph.copy()
        self._logger = logging.getLogger(str(self))
        self._edits: dict[str, OnnxGraphEdit] = {}
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
        onnx.save(gs.export_onnx(self._graph), "temp.onnx")
        onnx_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(self._graph),
            check_type=check_type,
            strict_mode=strict_mode,
            data_prop=data_prop
        )
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
        for node in self._graph.nodes:
            if isinstance(edit, str):
                edit = self._edits[edit]
            edit(node)
        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
        return self

    def apply_edits(self, edits: Sequence[OnnxGraphEdit | str]):
        for edit in edits:
            self.apply_edit(edit)
        return self

    def fix_io_dims(self, to_fix: list[FixedDimMapping] | None = None):
        to_fix = tuple(to_fix or [])
        for tensor in self._graph.inputs + self._graph.outputs:
            old_shape = list(tensor.shape)
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
            if tuple(tensor.shape) != tuple(old_shape):
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

        self._logger.warning("%s '%s' (type: %s) not found in graph; no reordering performed", io_type.capitalize(), io_name)


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
