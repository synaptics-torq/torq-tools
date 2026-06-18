# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers
from ._helpers import (
    _const_array,
    _const_int_list,
    _consumers_in_graph,
    _static_int_shape,
)


@dataclass
class EliminateExpand(OnnxGraphEdit):
    """
    Eliminate Expand ops at inputs for specific ops and let the compiler handle implicit broadcasting.
    """

    ops: list[str]

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _iter_producers(node):
        for tensor in node.inputs or []:
            if tensor is None:
                continue
            for producer in getattr(tensor, "inputs", []) or []:
                if producer is not None:
                    yield producer

    @staticmethod
    def _clear_downstream_shapes(node: gs.Node, graph_output_ids: set[int]):
        """BFS forward from node, clearing .shape on all intermediate variables."""
        queue = [node]
        visited = set()
        while queue:
            n = queue.pop(0)
            if id(n) in visited:
                continue
            visited.add(id(n))
            for out in n.outputs:
                if not isinstance(out, gs.Variable):
                    continue
                if id(out) not in graph_output_ids:
                    out.shape = None
                for consumer in out.outputs:
                    if id(consumer) not in visited:
                        queue.append(consumer)

    def match(self, node: gs.Node) -> bool:
        if node.op not in self.ops:
            return False
        return any(
            producer.op == "Expand"
            for producer in self._iter_producers(node)
        )

    def transform(self, node: gs.Node):
        if node.op not in self.ops:
            raise ValueError(
                f"Expected node op in {self.ops} for Expand elimination, got '{node.op}'"
            )
        graph_output_ids = {id(o) for o in self.graph.outputs}
        for prod in self._iter_producers(node):
            if prod.op != "Expand":
                continue
            inp_tensor = prod.inputs[0]
            expand_out = prod.outputs[0]
            rewire_consumers([node], expand_out, inp_tensor)
            # Clear stale shape metadata on all downstream variables so ONNX
            # shape inference can recompute them from the pre-expand input.
            self._clear_downstream_shapes(node, graph_output_ids)
            if not expand_out.outputs and not any(id(out) == id(expand_out) for out in self.graph.outputs):
                prod.inputs.clear()
                prod.outputs.clear()
            self._logger.debug(
                "Eliminated Expand '%s' feeding %s node '%s'",
                prod.name, node.op, node.name
            )

@dataclass
class EliminateTranspose(OnnxGraphEdit):
    """
    Eliminate Transpose ops that don't rearrange data in memory.

    Handles two cases:
    1. Identity: ``perm == [0, 1, ..., n-1]``. The Transpose is bypassed entirely.
    2. Data-preserving: the flat element order produced by ``Transpose(x, perm)``
       equals that of ``Reshape(x, permuted_shape)`` (e.g. ``[1,1,H,D]`` with
       ``perm=[0,2,1,3]`` when one of the swapped dims is 1). The Transpose is
       replaced with an equivalent Reshape.

    Note: ``input_shape == permuted_shape`` is *not* sufficient on its own --
    e.g. ``shape=[2,2] perm=[1,0]`` has equal in/out shapes but really
    rearranges data.
    """

    @staticmethod
    def _is_data_preserving_perm(perm: list[int], shape: list[int]) -> bool:
        """Cycle heuristic: True when at most one dim > 1 per perm cycle.

        This is a *necessary* condition for ``Transpose(x, perm)`` to be
        equivalent to a Reshape, but not sufficient (e.g. shape
        ``[1, 32, 16]`` with ``perm=[2, 1, 0]``). Used as a fast pre-filter.
        """
        visited = [False] * len(perm)
        for i in range(len(perm)):
            if visited[i] or perm[i] == i:
                visited[i] = True
                continue
            cycle_dims: list[int] = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle_dims.append(shape[j])
                j = perm[j]
            if sum(1 for d in cycle_dims if d > 1) > 1:
                return False
        return True

    @staticmethod
    def _transpose_equals_reshape(shape: list[int], perm: list[int]) -> bool:
        """True when ``Reshape(x, permuted_shape)`` matches ``Transpose(x, perm)``.

        Hybrid check: first apply a cheap cycle heuristic that rules out
        permutations that cannot be data-preserving, then confirm with an
        exact element-comparison on a synthetic index tensor. The heuristic
        alone admits false positives (e.g. ``shape=[1, 4, N] perm=[2, 1, 0]``
        passes the cycle test but is a real transpose), so for tensors too
        large to materialize the probe (numel > ~1M) we conservatively return
        False -- a missed elimination is only a speed loss, whereas an
        incorrect elimination can silently miscompile.
        """
        if len(shape) != len(perm):
            return False
        # True identity is always safe regardless of tensor size; equal in/out
        # shapes alone are not (e.g. [2,2] perm=[1,0] really swaps data).
        if all(perm[i] == i for i in range(len(perm))):
            return True
        if not EliminateTranspose._is_data_preserving_perm(perm, shape):
            return False
        numel = 1
        for d in shape:
            numel *= int(d)
        if numel > 1_000_000:
            return False
        out_shape = [shape[p] for p in perm]
        x = np.arange(numel, dtype=np.int64).reshape(shape)
        return bool(
            np.array_equal(np.transpose(x, perm), x.reshape(out_shape))
        )

    def match(self, node: gs.Node) -> bool:
        if node.op != "Transpose" or not node.inputs or not node.outputs:
            return False
        perm = node.attrs.get("perm", None)
        if perm is None:
            return False
        inp_shape = getattr(node.inputs[0], "shape", None)
        if inp_shape is None or not all(isinstance(d, (int, np.integer)) for d in inp_shape):
            return False

        perm = [int(p) for p in perm]
        shape = [int(d) for d in inp_shape]

        return self._transpose_equals_reshape(shape, perm)

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Transpose")
        inp = node.inputs[0]
        out = node.outputs[0]
        inp_shape = [int(d) for d in inp.shape]
        out_shape = [int(d) for d in out.shape]
        consumers: list[gs.Node] = list(out.outputs)

        if inp_shape == out_shape:
            rewire_consumers(consumers, out, inp)
            for i, graph_out in enumerate(self.graph.outputs):
                if graph_out is out:
                    self.graph.outputs[i] = inp
        else:
            shape_const = gs.Constant(
                name=node.name + "_fold_shape",
                values=np.array(out_shape, dtype=np.int64)
            )
            reshape_out: gs.Variable = self.graph.layer(
                name=node.name + "_fold_reshape",
                op="Reshape",
                inputs=[inp, shape_const],
                outputs=[gs.Variable(
                    name=out.name + "_reshaped",
                    dtype=out.dtype,
                    shape=out_shape
                )]
            )[0]
            rewire_consumers(consumers, out, reshape_out)
            for i, graph_out in enumerate(self.graph.outputs):
                if graph_out is out:
                    self.graph.outputs[i] = reshape_out

        node.inputs.clear()
        node.outputs.clear()
        if inp_shape == out_shape:
            self._logger.debug(
                "Eliminated Transpose '%s': %s -> %s", node.name, inp_shape, out_shape
            )
        else:
            self._logger.debug(
                "Folded Transpose '%s' into Reshape '%s'", node.name, node.name + "_fold_reshape"
            )

@dataclass
class CollapseReshapeChain(OnnxGraphEdit):
    """
    Collapse consecutive Reshape ops into a single Reshape.

    Matches a Reshape node whose only consumer is another Reshape,
    and replaces the chain with a single Reshape from the first input
    to the last output shape.
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "Reshape" or not node.inputs or not node.outputs:
            return False
        out = node.outputs[0]
        if len(out.outputs) != 1:
            return False
        consumer: gs.Node = out.outputs[0]
        return consumer.op == "Reshape"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Reshape")
        data_inp = node.inputs[0]

        # walk forward through all consecutive Reshapes
        current = node
        collapsed: list[str] = [node.name]
        while True:
            out = current.outputs[0]
            if len(out.outputs) != 1 or out.outputs[0].op != "Reshape":
                break
            next_node: gs.Node = out.outputs[0]
            current.inputs.clear()
            current.outputs.clear()
            collapsed.append(next_node.name)
            current = next_node

        # wire original data input into the final Reshape
        current.inputs[0] = data_inp
        self._logger.debug(
            "Collapsed %d Reshapes into '%s'", len(collapsed), current.name
        )

class ConstantBroadcastPolicy(Enum):
    """
    Strategy for handling broadcastable constants during graph edits.

    - `DEFER_RUNTIME`: Insert `Expand` nodes so constants broadcast at runtime (lower memory, slower inference).
    - `MATERIALIZE`: Pre-broadcast constants and store the expanded tensor (faster inference, higher memory).
    - `SKIP`: Leave constants untouched and let downstream tools handle broadcasting.
    """
    DEFER_RUNTIME = auto()
    MATERIALIZE = auto()
    SKIP = auto()

@dataclass
class BroadcastOpInputs(OnnxGraphEdit):
    """
    Add explicit `Expand` nodes for broadcasting op inputs to output shape.

    Args:
        ops (list[str]): Ops to apply explicit input broadcasting, will apply to all ops if list is empty.
        out_idx (int): Index of output to use as broadcast target shape (default: 0).
        inp_idx (list[int]): Only broadcast inputs at these indices (default: None, broadcast all inputs).
        constants_policy (ConstantBroadcastPolicy): How to treat constant inputs (default: skip).
    """

    ops: list[str]
    out_idx: int = 0
    inp_idx: list[int] | None = None
    constants_policy: ConstantBroadcastPolicy = ConstantBroadcastPolicy.SKIP

    def __post_init__(self):
        self.inp_idx = self.inp_idx or []
        return super().__post_init__()

    @staticmethod
    def _has_valid_shape(tensor: gs.Constant | gs.Variable) -> bool:
        try:
            shape = getattr(tensor, "shape", None)
            return shape is not None and all(isinstance(d, (int, np.integer)) for d in shape)
        except TypeError:
            raise ValueError(f"{tensor.name}, {tensor.shape}")

    @staticmethod
    def _unique_tensor_id(tensor: gs.Constant | gs.Variable, hash_length: int = 8) -> str:
        inputs = [getattr(n, "name", str(n)) for n in tensor.inputs]
        outputs = [getattr(n, "name", str(n)) for n in tensor.outputs]
        id_str = tensor.name + ":" + "|".join(inputs) + ">>" + "|".join(outputs)
        return hashlib.sha256(id_str.encode()).hexdigest()[:hash_length]

    def _add_broadcast_to_tensor(self, tensor: gs.Constant | gs.Variable, bcast_shape: list[int]):
        # create copy of initial consumers to prevent cycle later
        consumers: list[gs.Node] = tensor.outputs.copy()
        bcast_shape_const: gs.Constant = gs.Constant(
            name=tensor.name + "_bcast_shape",
            values=np.array(bcast_shape).astype(np.int64)
        )
        bcast_out: gs.Variable = self.graph.layer(
            name=tensor.name + "_bcast",
            op="Expand",
            inputs=[tensor, bcast_shape_const],
            outputs=[gs.Variable(name=tensor.name + "_expanded", dtype=tensor.dtype, shape=bcast_shape)]
        )[0]
        rewire_consumers(consumers, tensor, bcast_out)

    def match(self, node: gs.Node) -> bool:
        if self.ops and node.op not in self.ops:
            return False
        if not node.inputs or not node.outputs:
            return False

        if not (0 <= self.out_idx < len(node.outputs)):
            self._logger.warning(
                "Received invalid output index; valid: %s, received: %s",
                list(range(len(node.outputs))), self.out_idx
            )
            return False
        if not self._has_valid_shape(node.outputs[self.out_idx]):
            return False

        target_inp_idxs = self.inp_idx or list(range(len(node.inputs)))
        if any(i < 0 or i >= len(node.inputs) for i in target_inp_idxs):
            self._logger.warning(
                "Received invalid input indices; valid: %s, received: %s",
                list(range(len(node.inputs))), self.inp_idx
            )
            return False
        return all(self._has_valid_shape(node.inputs[i]) for i in target_inp_idxs)

    def transform(self, node: gs.Node):
        target_out: gs.Variable = node.outputs[self.out_idx]
        assert isinstance(target_out, gs.Variable), "Node output must be `gs.Variable`"
        if not self._has_valid_shape(target_out):
            raise ValueError(
                "Missing valid integer shape info for output '%s' (node: %s, '%s')",
                target_out.name, node.op, node.name
            )
        bcast_shape: list[int] = list(target_out.shape)
        target_inp_idxs = self.inp_idx or list(range(len(node.inputs)))
        bcast_done: set[str] = set()

        for i in target_inp_idxs:
            inp = node.inputs[i]
            if self._unique_tensor_id(inp) in bcast_done:
                continue

            if not self._has_valid_shape(inp):
                self._logger.warning(
                    "Broadcasting input '%s' with no valid integer shape info (node: %s, '%s')",
                    inp.name, node.op, node.name
                )
            
            if list(inp.shape) == bcast_shape:
                continue
            
            if isinstance(inp, gs.Variable):
                self._add_broadcast_to_tensor(inp, bcast_shape)
            elif isinstance(inp, gs.Constant):
                if getattr(inp, "dtype", None) is None:
                    self._logger.warning(
                        "Skipping broadcast of initializer '%s' due to missing dtype info",
                        inp.name
                    )
                    continue
                if self.constants_policy == ConstantBroadcastPolicy.SKIP:
                    continue
                if self.constants_policy == ConstantBroadcastPolicy.DEFER_RUNTIME:
                    self._add_broadcast_to_tensor(inp, bcast_shape)
                elif self.constants_policy == ConstantBroadcastPolicy.MATERIALIZE:
                    export_dtype = inp.export_dtype
                    if inp.dtype == onnx.TensorProto.BFLOAT16:
                        dtype = np.float32
                        export_dtype = onnx.TensorProto.BFLOAT16
                    else:
                        dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.dtype) \
                            if isinstance(inp.dtype, int) else inp.dtype
                    bcast_values = np.broadcast_to(inp.values, bcast_shape).astype(dtype)
                    bcast_const = gs.Constant(
                        name=inp.name + "_bcast",
                        values=bcast_values,
                        export_dtype=export_dtype
                    )
                    bcast_const.outputs = inp.outputs
                    inp.outputs.clear()
                else:
                    raise ValueError(f"Invalid constant broadcast policy '{self.constants_policy}'")
            else:
                raise ValueError(f"Invalid input tensor type '{type(inp)}'")
            
            bcast_done.add(self._unique_tensor_id(inp))
            self._logger.debug(
                "Broadcasted input '%s' of %s node '%s' to %s",
                inp.name, node.op, node.name, bcast_shape
            )

@dataclass
class EliminateRank0Gather(OnnxGraphEdit):
    """
    Rewrite ``Gather(rank-0) -> Unsqueeze(axes=[0])`` chains to a rank-1 Gather.

    IREE's ``FoldCollapseShape{,FullSlice}IntoInterfaceTensorStore`` codegen
    patterns trip a ranks-don't-match verifier when a tensor of rank > 0 is
    collapsed to rank 0 inside a dispatch; making the Gather produce ``[1]``
    directly avoids the rank-0 intermediate. Only fires when every consumer of
    the rank-0 Gather is ``Unsqueeze(axes=[0])``.
    """

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _output_rank(var) -> int | None:
        shape = getattr(var, "shape", None)
        return None if shape is None else len(shape)

    @staticmethod
    def _unsqueeze_axes_is_zero_only(node: gs.Node) -> bool:
        if node.op != "Unsqueeze":
            return False
        if len(node.inputs) >= 2:
            return _const_int_list(node.inputs[1]) == [0]
        axes_attr = node.attrs.get("axes")
        if axes_attr is None:
            return False
        return list(axes_attr) == [0]

    def match(self, node: gs.Node) -> bool:
        if node.op != "Gather" or len(node.inputs) < 2 or len(node.outputs) != 1:
            return False
        data, indices = node.inputs[0], node.inputs[1]
        if (
            self._output_rank(node.outputs[0]) != 0
            or self._output_rank(data) != 1
            or self._output_rank(indices) != 0
        ):
            return False
        consumers = _consumers_in_graph(self.graph, node.outputs[0].name)
        if not consumers:
            return False
        return all(self._unsqueeze_axes_is_zero_only(c) for c in consumers)

    def _make_rank1_indices(self, idx, name_prefix: str):
        if isinstance(idx, gs.Constant):
            arr = np.asarray(idx.values, dtype=np.int64).reshape(1)
            return gs.Constant(name=f"{name_prefix}_idx1d", values=arr)
        shape_const = gs.Constant(
            name=f"{name_prefix}_idx_shape",
            values=np.array([1], dtype=np.int64),
        )
        out = gs.Variable(
            name=f"{name_prefix}_idx1d",
            dtype=getattr(idx, "dtype", np.int64),
            shape=(1,),
        )
        self.graph.nodes.append(
            gs.Node(
                op="Reshape",
                name=f"{name_prefix}_idx_reshape",
                inputs=[idx, shape_const],
                outputs=[out],
            )
        )
        return out

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Gather")
        out = node.outputs[0]
        consumers = _consumers_in_graph(self.graph, out.name)

        base = node.name or "rank0_gather"
        new_idx = self._make_rank1_indices(node.inputs[1], base)
        new_out = gs.Variable(name=f"{out.name}_rank1", dtype=out.dtype, shape=(1,))
        node.inputs[1] = new_idx
        node.outputs = [new_out]

        for unsq in consumers:
            unsq_out = unsq.outputs[0]
            for n in self.graph.nodes:
                n.inputs = [
                    new_out if getattr(i, "name", None) == unsq_out.name else i
                    for i in n.inputs
                ]
            for i, go in enumerate(self.graph.outputs):
                if getattr(go, "name", None) == unsq_out.name:
                    self.graph.outputs[i] = new_out
            unsq.inputs = []
            unsq.outputs = []

        self._logger.debug(
            "rewrote rank-0 Gather %r (consumed by %d Unsqueeze(s)) to rank-1 path",
            node.name, len(consumers),
        )

@dataclass
class EliminateSingletonGatherUnsqueeze(OnnxGraphEdit):
    """
    Fold ``Gather(axis=k, indices=0) -> elementwise_unary -> Unsqueeze(axes=[k])``
    when ``data.shape[k] == 1`` and the final shape equals the original data
    shape.

    Once shapes are static, the squeeze/unsqueeze around an elementwise unary
    is a no-op and can be removed by feeding the unary directly.
    """

    _ELEMENTWISE_UNARY_OPS = frozenset(
        {
            "Abs", "Ceil", "Cos", "Erf", "Exp", "Floor", "Log", "Neg",
            "Relu", "Sigmoid", "Sin", "Sqrt", "Tanh",
        }
    )

    def __post_init__(self):
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _normalize_axis(axis: int, rank: int) -> int | None:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        return axis

    @staticmethod
    def _is_scalar_zero(t) -> bool:
        arr = _const_array(t)
        if arr is None or arr.shape != ():
            return False
        return int(arr.reshape(())) == 0

    @staticmethod
    def _unsqueeze_axes(node: gs.Node, out_rank: int) -> list[int] | None:
        if node.op != "Unsqueeze":
            return None
        if len(node.inputs) >= 2:
            axes = _const_int_list(node.inputs[1])
        else:
            raw = node.attrs.get("axes")
            axes = [int(v) for v in raw] if raw is not None else None
        if axes is None:
            return None
        normalized: list[int] = []
        for axis in axes:
            if axis < 0:
                axis += out_rank
            if axis < 0 or axis >= out_rank:
                return None
            normalized.append(axis)
        return normalized

    def _resolve_chain(self, gather: gs.Node):
        """Return (data_shape, unary, unsqueeze) or None if pattern doesn't match."""
        if (
            gather.op != "Gather"
            or len(gather.inputs) < 2
            or len(gather.outputs) != 1
        ):
            return None
        data, indices = gather.inputs[0], gather.inputs[1]
        data_shape = _static_int_shape(data)
        gather_shape = _static_int_shape(gather.outputs[0])
        if data_shape is None or gather_shape is None:
            return None

        axis = self._normalize_axis(
            int(gather.attrs.get("axis", 0)), len(data_shape)
        )
        if axis is None or data_shape[axis] != 1 or not self._is_scalar_zero(indices):
            return None
        if gather_shape != data_shape[:axis] + data_shape[axis + 1:]:
            return None

        gather_consumers = _consumers_in_graph(self.graph, gather.outputs[0].name)
        if len(gather_consumers) != 1:
            return None
        unary = gather_consumers[0]
        if (
            unary.op not in self._ELEMENTWISE_UNARY_OPS
            or len(unary.inputs) != 1
            or len(unary.outputs) != 1
        ):
            return None

        unary_consumers = _consumers_in_graph(self.graph, unary.outputs[0].name)
        if len(unary_consumers) != 1:
            return None
        unsqueeze = unary_consumers[0]
        out_shape = (
            _static_int_shape(unsqueeze.outputs[0])
            if len(unsqueeze.outputs) == 1
            else None
        )
        if out_shape != data_shape:
            return None
        if self._unsqueeze_axes(unsqueeze, len(out_shape)) is None:
            return None
        return data_shape, unary, unsqueeze

    def match(self, node: gs.Node) -> bool:
        return self._resolve_chain(node) is not None

    def transform(self, node: gs.Node):
        chain = self._resolve_chain(node)
        if chain is None:
            return
        data_shape, unary, unsqueeze = chain

        unary.inputs[0] = node.inputs[0]
        unary.outputs[0] = unsqueeze.outputs[0]
        node.inputs.clear()
        node.outputs.clear()
        unsqueeze.inputs.clear()
        unsqueeze.outputs.clear()

        self._logger.debug(
            "removed singleton Gather->%s->Unsqueeze shim at %r shape=%s",
            unary.op, node.name, data_shape,
        )
