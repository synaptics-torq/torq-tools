# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Constant-fold the ``pads`` (and optional ``constant_value``) inputs of every ``Pad``.

Many ONNX exports compute Pad's ``pads`` input through a ``Shape``/``Gather``/
``Concat`` shape-arithmetic subgraph. After :class:`ApplyFixedShapes` makes
the model's I/O shapes static, those subgraphs are fully evaluable at compile
time -- this pass walks the graph backward from each Pad's control inputs,
evaluates them with a small numpy interpreter (no onnxruntime required), and
replaces the input with a frozen initializer.

After folding the orphaned shape-computation nodes are removed by
:func:`_remove_dead_nodes_and_initializers`.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Optional

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, TensorProto

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.replace_pad_inputs_with_constants")


_TENSORPROTO_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.BFLOAT16: np.float32,  # approximate; only used for shape arithmetic
    TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32,
    TensorProto.INT16: np.int16,
    TensorProto.INT8: np.int8,
    TensorProto.UINT64: np.uint64,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT8: np.uint8,
    TensorProto.BOOL: np.bool_,
}


def _get_attr(node: onnx.NodeProto, name: str):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _get_attr_int(node: onnx.NodeProto, name: str, default=None):
    a = _get_attr(node, name)
    return a.i if a is not None else default


def _get_attr_ints(node: onnx.NodeProto, name: str, default=None):
    a = _get_attr(node, name)
    return list(a.ints) if a is not None else default


def _get_attr_tensor(node: onnx.NodeProto, name: str):
    a = _get_attr(node, name)
    return None if a is None else numpy_helper.to_array(a.t)


def _value_info_shape(vi) -> Optional[list[int]]:
    if not vi.type.HasField("tensor_type"):
        return None
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims: list[int] = []
    for d in t.shape.dim:
        if not d.HasField("dim_value"):
            return None
        dims.append(int(d.dim_value))
    return dims


class _EvalContext:
    """Tiny constant-folder for ONNX shape-arithmetic ops (Shape/Gather/Concat/...)."""

    def __init__(self, graph: onnx.GraphProto):
        self.graph = graph
        self.initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        self.producers: dict[str, onnx.NodeProto] = {}
        for n in graph.node:
            for o in n.output:
                if o:
                    self.producers[o] = n
        self.known_shapes: dict[str, list[int]] = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            shape = _value_info_shape(vi)
            if shape is not None:
                self.known_shapes[vi.name] = shape
        for init in graph.initializer:
            self.known_shapes[init.name] = list(self.initializers[init.name].shape)
        self.cache: dict[str, np.ndarray] = {}

    def eval_value(self, name: str) -> Optional[np.ndarray]:
        if not name:
            return None
        if name in self.cache:
            return self.cache[name]
        if name in self.initializers:
            self.cache[name] = self.initializers[name]
            return self.cache[name]
        node = self.producers.get(name)
        if node is None:
            return None
        vals = self._eval_node(node)
        if vals is None:
            return None
        for out_name, arr in zip(node.output, vals):
            if out_name:
                self.cache[out_name] = arr
        return self.cache.get(name)

    def _eval_node(self, node: onnx.NodeProto) -> Optional[list[np.ndarray]]:
        op = node.op_type
        if op == "Constant":
            t = _get_attr_tensor(node, "value")
            return None if t is None else [np.asarray(t)]
        if op == "Identity":
            x = self.eval_value(node.input[0])
            return None if x is None else [x]
        if op == "Shape":
            shape = self.known_shapes.get(node.input[0])
            if shape is None:
                return None
            s = np.asarray(shape, dtype=np.int64)
            start = _get_attr_int(node, "start", None)
            end = _get_attr_int(node, "end", None)
            if start is not None or end is not None:
                rank = s.shape[0]
                st = 0 if start is None else (start + rank if start < 0 else start)
                ed = rank if end is None else (end + rank if end < 0 else end)
                s = s[st:ed]
            return [s.astype(np.int64)]
        if op == "ConstantOfShape":
            shape_arr = self.eval_value(node.input[0])
            if shape_arr is None:
                return None
            shape_tuple = tuple(
                int(x) for x in np.asarray(shape_arr).astype(np.int64).reshape(-1).tolist()
            )
            value_attr = _get_attr(node, "value")
            if value_attr is not None:
                arr = numpy_helper.to_array(value_attr.t)
                return [np.full(shape_tuple, arr.reshape(-1)[0], dtype=arr.dtype)]
            return [np.zeros(shape_tuple, dtype=np.float32)]
        if op == "Cast":
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            to_dtype = _get_attr_int(node, "to")
            return [x.astype(_TENSORPROTO_TO_NUMPY[to_dtype])]
        if op == "Reshape":
            x = self.eval_value(node.input[0])
            shape = self.eval_value(node.input[1])
            if x is None or shape is None:
                return None
            try:
                return [np.reshape(x, np.asarray(shape).astype(np.int64).reshape(-1).tolist())]
            except Exception:
                return None
        if op == "Concat":
            axis = _get_attr_int(node, "axis")
            xs = []
            for inp in node.input:
                x = self.eval_value(inp)
                if x is None:
                    return None
                xs.append(np.asarray(x))
            try:
                return [np.concatenate(xs, axis=axis)]
            except Exception:
                return None
        if op in ("Unsqueeze", "Squeeze"):
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            if len(node.input) >= 2 and node.input[1]:
                axes_arr = self.eval_value(node.input[1])
                if axes_arr is None:
                    return None
                axes = [int(a) for a in np.asarray(axes_arr).reshape(-1).tolist()]
            else:
                axes = _get_attr_ints(node, "axes", None)
            try:
                if op == "Unsqueeze":
                    y = x
                    for ax in sorted(axes or []):
                        y = np.expand_dims(y, axis=ax)
                else:
                    if axes is None:
                        y = np.squeeze(x)
                    else:
                        y = x
                        for ax in sorted(axes, reverse=True):
                            y = np.squeeze(y, axis=ax)
                return [y]
            except Exception:
                return None
        if op == "Transpose":
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            return [np.transpose(x, axes=_get_attr_ints(node, "perm", None))]
        if op == "Gather":
            data = self.eval_value(node.input[0])
            indices = self.eval_value(node.input[1])
            if data is None or indices is None:
                return None
            try:
                return [
                    np.take(
                        data,
                        indices.astype(np.int64),
                        axis=_get_attr_int(node, "axis", 0),
                    )
                ]
            except Exception:
                return None
        if op == "Slice":
            data = self.eval_value(node.input[0])
            starts = self.eval_value(node.input[1])
            ends = self.eval_value(node.input[2])
            if data is None or starts is None or ends is None:
                return None
            if len(node.input) >= 4 and node.input[3]:
                axes_arr = self.eval_value(node.input[3])
                if axes_arr is None:
                    return None
                axes = axes_arr.astype(np.int64).reshape(-1).tolist()
            else:
                axes = list(range(len(np.asarray(starts).reshape(-1))))
            if len(node.input) >= 5 and node.input[4]:
                steps_arr = self.eval_value(node.input[4])
                if steps_arr is None:
                    return None
                steps = steps_arr.astype(np.int64).reshape(-1).tolist()
            else:
                steps = [1] * len(axes)
            starts = np.asarray(starts).astype(np.int64).reshape(-1).tolist()
            ends = np.asarray(ends).astype(np.int64).reshape(-1).tolist()
            slc = [slice(None)] * data.ndim
            for ax, st, ed, step in zip(axes, starts, ends, steps):
                slc[int(ax)] = slice(int(st), int(ed), int(step))
            try:
                return [data[tuple(slc)]]
            except Exception:
                return None
        if op == "Expand":
            x = self.eval_value(node.input[0])
            shape = self.eval_value(node.input[1])
            if x is None or shape is None:
                return None
            try:
                return [
                    np.asarray(
                        np.broadcast_to(
                            x,
                            tuple(int(v) for v in np.asarray(shape).reshape(-1).tolist()),
                        )
                    )
                ]
            except Exception:
                return None
        return None


def _all_value_names(graph: onnx.GraphProto) -> set[str]:
    names: set[str] = {init.name for init in graph.initializer}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(vi.name)
    for n in graph.node:
        names.update(x for x in n.input if x)
        names.update(x for x in n.output if x)
    return names


def _unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _replace_pad_inputs(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    out = copy.deepcopy(model)
    graph = out.graph
    ctx = _EvalContext(graph)
    existing = _all_value_names(graph)
    replaced = 0

    for node in graph.node:
        if node.op_type != "Pad":
            continue

        if len(node.input) >= 2 and node.input[1]:
            val = ctx.eval_value(node.input[1])
            if val is not None:
                arr = np.asarray(val).astype(np.int64).reshape(-1)
                new_name = _unique_name(f"{node.name or 'pad'}_pads_const", existing)
                existing.add(new_name)
                graph.initializer.append(numpy_helper.from_array(arr, name=new_name))
                node.input[1] = new_name
                replaced += 1
                logger.info("Pad %s: pads <- %s", node.name or "(unnamed)", arr.tolist())

        if len(node.input) >= 3 and node.input[2]:
            val = ctx.eval_value(node.input[2])
            if val is not None:
                arr = np.asarray(val)
                if arr.size == 1:
                    new_name = _unique_name(
                        f"{node.name or 'pad'}_value_const", existing
                    )
                    existing.add(new_name)
                    graph.initializer.append(
                        numpy_helper.from_array(arr.reshape(()), name=new_name)
                    )
                    node.input[2] = new_name
                    logger.info(
                        "Pad %s: constant_value <- %s",
                        node.name or "(unnamed)", arr.reshape(-1).tolist(),
                    )

    return out, replaced


def _remove_dead_nodes_and_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    out = copy.deepcopy(model)
    graph = out.graph

    producers: dict[str, onnx.NodeProto] = {
        o: n for n in graph.node for o in n.output if o
    }

    needed_values = {o.name for o in graph.output if o.name}
    needed_node_ids: set[int] = set()
    queue = deque(needed_values)
    while queue:
        val = queue.popleft()
        node = producers.get(val)
        if node is None or id(node) in needed_node_ids:
            continue
        needed_node_ids.add(id(node))
        for inp in node.input:
            if inp:
                queue.append(inp)

    new_nodes = [n for n in graph.node if id(n) in needed_node_ids]
    del graph.node[:]
    graph.node.extend(new_nodes)

    used: set[str] = set()
    for n in graph.node:
        used.update(x for x in n.input if x)
    used.update(o.name for o in graph.output)

    keep = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return out


class ReplacePadInputsWithConstants:
    """Pass: constant-fold every ``Pad``'s ``pads`` (and ``constant_value``) input.

    Always followed internally by dead-node + initializer pruning so the
    orphaned shape-computation subgraphs are removed in the same step.
    """

    name = "replace_pad_inputs_with_constants"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        out, replaced = _replace_pad_inputs(model)
        if replaced == 0:
            return model

        out = _remove_dead_nodes_and_initializers(out)
        try:
            out = shape_inference.infer_shapes(out)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        logger.info("folded %d Pad input(s)", replaced)
        return out
