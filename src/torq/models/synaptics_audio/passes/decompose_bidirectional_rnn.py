# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Decompose ONNX bidirectional recurrent layers into two unidirectional layers.

torch-mlir's ONNX importer cannot legalize ``onnx.GRU`` (or ``onnx.LSTM`` /
``onnx.RNN``) when ``direction == "bidirectional"`` and only supports
``direction == "forward"`` for the unidirectional case. This pass rewrites
every bidirectional recurrent op in a graph using only forward-direction
ops:

* a ``forward`` op fed with the index-0 slice of the per-direction inputs
  (``W``, ``R``, ``B``, ``initial_h``, and for LSTM ``initial_c``, plus the
  per-direction ``activations`` / ``activation_alpha`` / ``activation_beta``
  if specified);
* the time-reversed input ``X`` (flipped along the seq axis with a
  ``Gather``) is fed into a second ``forward`` op with the index-1 slice;
  its sequence output is flipped back so that ``Y_rev[t]`` aligns with the
  original timestep ``t``;
* a ``Concat`` (axis = num_directions axis) re-stacks the two ``[..., 1, ...]``
  Y outputs back to ``[..., 2, ...]``;
* (when present) a ``Concat`` along axis 0 (or axis 1 for ``layout == 1``)
  for the optional ``Y_h`` (and ``Y_c`` for LSTM) outputs to recreate the
  packed per-direction tensor.

Constant inputs (initializers / ``gs.Constant``) are split at graph-build
time into two new constants. Static ``ConstantOfShape`` initial states are
materialized and split the same way. Other runtime inputs get an inserted
``Split`` along axis 0.

The transformation preserves the original output names so downstream consumers
do not need to be updated.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.decompose_bidirectional_rnn")


_RNN_OPS: tuple[str, ...] = ("GRU", "LSTM", "RNN")


def _attr_str(attrs: dict, key: str, default: str) -> str:
    val = attrs.get(key, default)
    if isinstance(val, bytes):
        val = val.decode()
    return val


def _is_empty_input(v) -> bool:
    return v is None or (isinstance(v, gs.Variable) and v.name == "")


def _split_constant_axis0(
    c: gs.Constant, name_prefix: str
) -> tuple[gs.Constant, gs.Constant]:
    arr = np.asarray(c.values)
    if arr.shape[0] != 2:
        raise ValueError(
            f"expected leading dim 2 for {c.name!r}, got shape {tuple(arr.shape)}"
        )
    fwd = gs.Constant(name=f"{name_prefix}_fwd", values=np.ascontiguousarray(arr[0:1]))
    rev = gs.Constant(name=f"{name_prefix}_rev", values=np.ascontiguousarray(arr[1:2]))
    return fwd, rev


def _constant_of_shape_fill(var) -> np.ndarray | None:
    producers = getattr(var, "inputs", None) or []
    if len(producers) != 1 or producers[0].op != "ConstantOfShape":
        return None

    value = producers[0].attrs.get("value")
    if value is None:
        dtype = getattr(var, "dtype", None) or np.float32
        return np.array(0, dtype=dtype)
    if hasattr(value, "values"):
        arr = np.asarray(value.values)
    else:
        arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            f"ConstantOfShape fill for {getattr(var, 'name', '<unnamed>')!r} "
            f"must be scalar, got shape {arr.shape}"
        )
    return arr.reshape(())


def _materialize_constant_of_shape(
    var,
    shape: tuple[int, ...] | None,
    name: str,
) -> gs.Constant | None:
    if shape is None or any(int(d) <= 0 for d in shape):
        return None

    fill = _constant_of_shape_fill(var)
    if fill is None:
        return None

    values = np.full(shape, fill.item(), dtype=fill.dtype)
    return gs.Constant(name=name, values=np.ascontiguousarray(values))


def _split_runtime_axis0(
    graph: gs.Graph,
    var: gs.Variable,
    name_prefix: str,
    half_shape: tuple | None,
) -> tuple[gs.Variable, gs.Variable]:
    fwd = gs.Variable(name=f"{name_prefix}_fwd", dtype=var.dtype, shape=half_shape)
    rev = gs.Variable(name=f"{name_prefix}_rev", dtype=var.dtype, shape=half_shape)
    sizes = gs.Constant(
        name=f"{name_prefix}_split_sizes",
        values=np.array([1, 1], dtype=np.int64),
    )
    graph.nodes.append(
        gs.Node(
            op="Split",
            name=f"{name_prefix}_split",
            inputs=[var, sizes],
            outputs=[fwd, rev],
            attrs={"axis": 0},
        )
    )
    return fwd, rev


def _split_input(
    graph: gs.Graph,
    var,
    name_prefix: str,
    materialized_shape: tuple[int, ...] | None = None,
) -> tuple[object, object]:
    if isinstance(var, gs.Constant):
        return _split_constant_axis0(var, name_prefix)
    materialized = _materialize_constant_of_shape(
        var, materialized_shape, f"{name_prefix}_constant"
    )
    if materialized is not None:
        return _split_constant_axis0(materialized, name_prefix)
    half_shape = None
    if getattr(var, "shape", None):
        full = list(var.shape)
        if len(full) >= 1:
            half_shape = tuple([1, *full[1:]])
    return _split_runtime_axis0(graph, var, name_prefix, half_shape)


def _split_per_direction_attr(attrs: dict, key: str) -> tuple[list | None, list | None]:
    if key not in attrs:
        return None, None
    raw = list(attrs[key])
    if len(raw) % 2 != 0:
        raise ValueError(
            f"per-direction attribute {key!r} has odd length {len(raw)}"
        )
    half = len(raw) // 2
    return raw[:half], raw[half:]


def _per_direction_size(op: str) -> int:
    return {"RNN": 5, "GRU": 6, "LSTM": 7}[op]


def _full_output_count(op: str) -> int:
    return {"RNN": 2, "GRU": 2, "LSTM": 3}[op]


def _per_direction_shape_for_output(out_pos: int, original_out, layout: int):
    if not getattr(original_out, "shape", None):
        return None
    full = list(original_out.shape)
    if out_pos == 0:
        ax = 1 if layout == 0 else 2
    else:
        ax = 0 if layout == 0 else 1
    if ax >= len(full):
        return None
    return tuple(1 if d == ax else full[d] for d in range(len(full)))


def _per_direction_shape_for_unused(
    op: str, out_pos: int, X, hidden_size: int | None, layout: int
):
    if not getattr(X, "shape", None) or hidden_size is None:
        return None
    xshape = list(X.shape)
    if len(xshape) < 3:
        return None
    seq_len = xshape[0] if layout == 0 else xshape[1]
    batch = xshape[1] if layout == 0 else xshape[0]
    if out_pos == 0:
        return (
            (seq_len, 1, batch, hidden_size)
            if layout == 0
            else (batch, seq_len, 1, hidden_size)
        )
    return (1, batch, hidden_size) if layout == 0 else (batch, 1, hidden_size)


def _initial_state_shape(
    input_pos: int, X, hidden_size: int | None
) -> tuple[int, ...] | None:
    if input_pos not in (5, 6) or hidden_size is None:
        return None
    if not getattr(X, "shape", None) or len(X.shape) < 2:
        return None
    try:
        batch = int(X.shape[1])
    except (TypeError, ValueError):
        return None
    if batch <= 0:
        return None
    return (2, batch, hidden_size)


def _gather_flip(
    graph: gs.Graph,
    src,
    axis: int,
    length: int,
    name_prefix: str,
) -> gs.Variable:
    idx = gs.Constant(
        name=f"{name_prefix}_flip_idx",
        values=np.arange(length - 1, -1, -1, dtype=np.int64),
    )
    out_shape = tuple(src.shape) if getattr(src, "shape", None) else None
    out = gs.Variable(name=f"{name_prefix}_flipped", dtype=src.dtype, shape=out_shape)
    graph.nodes.append(
        gs.Node(
            op="Gather",
            name=f"{name_prefix}_flip",
            inputs=[src, idx],
            outputs=[out],
            attrs={"axis": axis},
        )
    )
    return out


def _decompose_one(graph: gs.Graph, node: gs.Node) -> bool:
    attrs = dict(node.attrs)
    if _attr_str(attrs, "direction", "forward") != "bidirectional":
        return False

    op = node.op
    if op not in _RNN_OPS:
        return False

    layout = int(attrs.get("layout", 0))
    if layout != 0:
        # Audio recipes only exercise layout=0; the GRU/LSTM/RNN axis math
        # below assumes it. Raise so we notice instead of silently miscompiling.
        raise NotImplementedError(
            f"{op}: bidirectional decomposition only supports layout=0, got {layout}"
        )

    n_per_dir = _per_direction_size(op)
    raw_inputs = list(node.inputs)
    while len(raw_inputs) < n_per_dir:
        raw_inputs.append(gs.Variable.empty())

    X = raw_inputs[0]
    seq_axis = 0
    yh_axis = 0
    y_dir_axis = 1

    if not getattr(X, "shape", None) or len(X.shape) <= seq_axis:
        raise ValueError(
            f"{op}: cannot determine sequence length for X={X.name!r}; "
            "shape inference is required before decomposition"
        )
    seq_len = int(X.shape[seq_axis])
    if seq_len <= 0:
        raise ValueError(
            f"{op}: non-static sequence length {seq_len} on X={X.name!r}"
        )

    base = node.name or f"{op.lower()}_bidir"
    hidden_size_attr = attrs.get("hidden_size")
    hidden_size = int(hidden_size_attr) if hidden_size_attr is not None else None

    pd_indices = [i for i in range(1, n_per_dir) if i != 4]
    fwd_inputs: list[object] = list(raw_inputs)
    rev_inputs: list[object] = list(raw_inputs)
    for i in pd_indices:
        var = raw_inputs[i]
        if _is_empty_input(var):
            continue
        fwd_var, rev_var = _split_input(
            graph,
            var,
            f"{base}_in{i}",
            _initial_state_shape(i, X, hidden_size),
        )
        fwd_inputs[i] = fwd_var
        rev_inputs[i] = rev_var

    rev_inputs[0] = _gather_flip(graph, X, seq_axis, seq_len, f"{base}_x")

    fwd_attrs: dict = {}
    rev_attrs: dict = {}
    for key in ("activations", "activation_alpha", "activation_beta"):
        fwd_part, rev_part = _split_per_direction_attr(attrs, key)
        if fwd_part is not None:
            fwd_attrs[key] = fwd_part
            rev_attrs[key] = rev_part
    for key, val in attrs.items():
        if key in ("direction", "activations", "activation_alpha", "activation_beta"):
            continue
        fwd_attrs[key] = val
        rev_attrs[key] = val
    fwd_attrs["direction"] = "forward"
    rev_attrs["direction"] = "forward"

    raw_outputs = list(node.outputs)
    if not raw_outputs:
        return False

    full_out_count = _full_output_count(op)

    # Pad raw_outputs out to the schema's full output count so the per-direction
    # nodes always carry every output the upstream importer expects. Adding
    # placeholder, unconnected outputs is a no-op for ONNX consumers but keeps
    # the imported torch.operator two-result, which works around an upstream
    # torch-mlir limitation in the unidirectional GRU expander.
    original_out_count = len(raw_outputs)
    while len(raw_outputs) < full_out_count:
        raw_outputs.append(gs.Variable.empty())

    new_nodes: list[gs.Node] = []
    fwd_outs: list[gs.Variable] = []
    rev_outs: list[gs.Variable] = []

    for out_pos, out in enumerate(raw_outputs):
        if _is_empty_input(out):
            per_shape = _per_direction_shape_for_unused(
                op, out_pos, X, hidden_size, layout
            )
            out_dtype = X.dtype
        else:
            per_shape = _per_direction_shape_for_output(out_pos, out, layout)
            out_dtype = out.dtype
        f = gs.Variable(name=f"{base}_out{out_pos}_fwd", dtype=out_dtype, shape=per_shape)
        r = gs.Variable(name=f"{base}_out{out_pos}_rev", dtype=out_dtype, shape=per_shape)
        fwd_outs.append(f)
        rev_outs.append(r)

    new_nodes.append(
        gs.Node(op=op, name=f"{base}_fwd", inputs=fwd_inputs, outputs=fwd_outs, attrs=fwd_attrs)
    )
    new_nodes.append(
        gs.Node(op=op, name=f"{base}_rev", inputs=rev_inputs, outputs=rev_outs, attrs=rev_attrs)
    )

    for out_pos in range(original_out_count):
        out = raw_outputs[out_pos]
        if _is_empty_input(out):
            continue
        ax = y_dir_axis if out_pos == 0 else yh_axis
        rev_branch = rev_outs[out_pos]
        if out_pos == 0:
            rev_branch = _gather_flip(
                graph, rev_branch, seq_axis, seq_len, f"{base}_out{out_pos}_rev"
            )
        out.inputs.clear()
        new_nodes.append(
            gs.Node(
                op="Concat",
                name=f"{base}_out{out_pos}_concat",
                inputs=[fwd_outs[out_pos], rev_branch],
                outputs=[out],
                attrs={"axis": ax},
            )
        )

    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.extend(new_nodes)

    logger.info("decomposed bidirectional %s %r at layout=%d", op, base, layout)
    return True


class DecomposeBidirectionalRnn:
    """Pass: split bidirectional GRU/LSTM/RNN into two forward-direction ops."""

    name = "decompose_bidirectional_rnn"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        graph = gs.import_onnx(model)
        graph.name = graph.name or "main"

        decomposed = 0
        for node in list(graph.nodes):
            if node.op not in _RNN_OPS:
                continue
            if _decompose_one(graph, node):
                decomposed += 1

        if not decomposed:
            return model

        graph.cleanup().toposort()
        out_model = gs.export_onnx(graph)
        try:
            out_model = shape_inference.infer_shapes(out_model)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        return out_model
