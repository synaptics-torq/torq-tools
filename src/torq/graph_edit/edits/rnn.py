# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit


@dataclass
class DecomposeBidirectionalRnn(OnnxGraphEdit):
    """
    Normalize ONNX ``GRU`` / ``LSTM`` / ``RNN`` ops for the torq backend.

    Two behaviors, controlled together so a single edit can fully prepare
    recurrent ops for the torq import + lowering pipeline:

    1. **Bidirectional decomposition** (always on). torch-mlir's ONNX
       importer only supports ``direction == "forward"``. Bidirectional
       ops are split into two forward ops -- one on ``X``, one on the
       time-reversed ``X`` -- with the reversed branch's sequence output
       flipped back before ``Concat``. Only ``layout == 0`` is supported;
       ``layout == 1`` raises ``NotImplementedError`` rather than silently
       miscompiling.

    2. **Long-sequence chunking** (opt-in via ``max_chunk_len``).
       *Originally-forward* ops whose statically-known sequence length
       exceeds ``K = max_chunk_len`` are rewritten into ``ceil(T / K)``
       shorter forward ops chained along the sequence axis:
       ``Split(axis=0)`` slices ``X``, each chunk threads its ``Y_h``
       (and ``Y_c`` for LSTM) into the next chunk's ``initial_h`` (and
       ``initial_c``), and per-step ``Y`` outputs are stitched back with
       ``Concat(axis=0)``. Weights (``W``/``R``/``B``, and ``P`` for
       LSTM) are shared by reference -- no duplication.

       Forward branches synthesized by the bidirectional decomposition
       above are intentionally **not** chunked: splitting the
       time-reversed branch along the sequence axis broke per-step Y
       stitching on real Synaptics bidirectional GRUs (e.g.
       ``voice_filter``), so bidirectional sources keep the full
       contiguous sequence in each direction.

       Why: torq's executable lowering fully unrolls the recurrent loop
       (the backend cannot consume dynamic dispatch parameters) and
       produces one giant dispatch per RNN op. That scales poorly through
       ``TileAndFusePass``'s memory-fit simulation; splitting into
       ``K``-step chunks yields several modest dispatches that compile in
       seconds while preserving semantics exactly. ``K = 4`` matches the
       longest configuration that currently compiles within CI timeouts
       for the NNNR3 fGRU stack on torq.

       Chunking constraints (silently skipped when not met): ``layout ==
       0``, empty ``sequence_lens``, statically-known sequence length
       strictly greater than ``K``.

    Args:
        max_chunk_len (int | None): When set, chunk forward ops whose
            sequence length exceeds this value. ``None`` (default)
            disables chunking, preserving the original
            decompose-only behavior.
    """

    max_chunk_len: int | None = None

    _RNN_OUTPUT_SLOT_SUFFIX: ClassVar[dict[str, tuple[str, ...]]] = {
        "GRU": ("Y", "Y_h"),
        "RNN": ("Y", "Y_h"),
        "LSTM": ("Y", "Y_h", "Y_c"),
    }
    _RNN_OPS: tuple[str, ...] = ("GRU", "LSTM", "RNN")
    _STATE_INPUT_POSITIONS: ClassVar[dict[str, tuple[int, ...]]] = {
        "GRU": (5,), "RNN": (5,), "LSTM": (5, 6),
    }
    _STATE_OUTPUT_POSITIONS: ClassVar[dict[str, tuple[int, ...]]] = {
        "GRU": (1,), "RNN": (1,), "LSTM": (1, 2),
    }
    # Full ONNX input arity, including optional LSTM peepholes at index 7.
    _FULL_INPUT_COUNT: ClassVar[dict[str, int]] = {"GRU": 6, "RNN": 6, "LSTM": 8}

    def __post_init__(self):
        if self.max_chunk_len is not None and self.max_chunk_len <= 0:
            raise ValueError(
                f"max_chunk_len must be >= 1 when set, got {self.max_chunk_len}"
            )
        self.requires_shape_inference = True
        # ``apply_edit`` iterates the live node list, so any forward branch
        # we synthesize during bidirectional decomposition gets re-visited
        # by the outer loop. Track those branch identities so the
        # forward-chunking path explicitly skips them; bidirectional
        # sources must run each direction's full contiguous sequence.
        self._decompose_emitted: set[int] = set()
        return super().__post_init__()

    @staticmethod
    def _attr_str(attrs: dict, key: str, default: str) -> str:
        val = attrs.get(key, default)
        if isinstance(val, bytes):
            val = val.decode()
        return val

    @staticmethod
    def _is_empty_input(v) -> bool:
        return v is None or (isinstance(v, gs.Variable) and v.name == "")

    @classmethod
    def _unused_output_shape(
        cls,
        op: str,
        out_pos: int,
        X,
        hidden_size: int | None,
        layout: int = 0,
    ) -> tuple[int, ...] | None:
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
        if op == "LSTM" and out_pos == 2:
            return (1, batch, hidden_size) if layout == 0 else (batch, 1, hidden_size)
        return (1, batch, hidden_size) if layout == 0 else (batch, 1, hidden_size)

    @classmethod
    def _typed_rnn_output(
        cls,
        node_name: str,
        op: str,
        out_pos: int,
        *,
        dtype,
        shape: tuple[int, ...] | None,
    ) -> gs.Variable:
        suffix = cls._RNN_OUTPUT_SLOT_SUFFIX[op][out_pos]
        return gs.Variable(name=f"{node_name}:{suffix}", dtype=dtype, shape=shape)

    @classmethod
    def _unused_slot(
        cls,
        node_name: str,
        op: str,
        out_pos: int,
        X,
        hidden_size: int | None,
        layout: int,
        dtype,
    ) -> gs.Variable:
        return cls._typed_rnn_output(
            node_name,
            op,
            out_pos,
            dtype=dtype,
            shape=cls._unused_output_shape(op, out_pos, X, hidden_size, layout),
        )

    @classmethod
    def _bidir_branch_output(
        cls,
        base: str,
        branch: str,
        op: str,
        out_pos: int,
        out,
        X,
        hidden_size: int | None,
        layout: int,
    ) -> gs.Variable:
        if cls._is_empty_input(out):
            return cls._unused_slot(
                f"{base}_{branch}", op, out_pos, X, hidden_size, layout, X.dtype
            )
        return gs.Variable(
            name=f"{base}_out{out_pos}_{branch}",
            dtype=out.dtype,
            shape=cls._per_direction_shape_for_output(out_pos, out, layout),
        )

    @classmethod
    def restore_output_arity(cls, graph: gs.Graph) -> None:
        """Re-append typed omitted RNN outputs dropped by GraphSurgeon cleanup."""
        for node in graph.nodes:
            op = node.op
            if op not in cls._RNN_OUTPUT_SLOT_SUFFIX:
                continue
            X = node.inputs[0] if node.inputs else None
            hidden = node.attrs.get("hidden_size")
            hidden_size = int(hidden) if hidden is not None else None
            layout = int(node.attrs.get("layout", 0))
            dtype = getattr(X, "dtype", None) or np.float32
            while len(node.outputs) < cls._full_output_count(op):
                node.outputs.append(
                    cls._unused_slot(
                        node.name or op.lower(),
                        op,
                        len(node.outputs),
                        X,
                        hidden_size,
                        layout,
                        dtype,
                    )
                )

    @staticmethod
    def _per_direction_size(op: str) -> int:
        return {"RNN": 5, "GRU": 6, "LSTM": 7}[op]

    @staticmethod
    def _full_output_count(op: str) -> int:
        return {"RNN": 2, "GRU": 2, "LSTM": 3}[op]

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _materialize_constant_of_shape(
        cls, var, shape: tuple[int, ...] | None, name: str
    ) -> gs.Constant | None:
        if shape is None or any(int(d) <= 0 for d in shape):
            return None
        fill = cls._constant_of_shape_fill(var)
        if fill is None:
            return None
        values = np.full(shape, fill.item(), dtype=fill.dtype)
        return gs.Constant(name=name, values=np.ascontiguousarray(values))

    def _split_runtime_axis0(
        self,
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
        self.graph.nodes.append(
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
        self, var, name_prefix: str, materialized_shape: tuple[int, ...] | None = None
    ) -> tuple[object, object]:
        if isinstance(var, gs.Constant):
            return self._split_constant_axis0(var, name_prefix)
        materialized = self._materialize_constant_of_shape(
            var, materialized_shape, f"{name_prefix}_constant"
        )
        if materialized is not None:
            return self._split_constant_axis0(materialized, name_prefix)
        half_shape = None
        if getattr(var, "shape", None):
            full = list(var.shape)
            if len(full) >= 1:
                half_shape = tuple([1, *full[1:]])
        return self._split_runtime_axis0(var, name_prefix, half_shape)

    @staticmethod
    def _split_per_direction_attr(
        attrs: dict, key: str
    ) -> tuple[list | None, list | None]:
        if key not in attrs:
            return None, None
        raw = list(attrs[key])
        if len(raw) % 2 != 0:
            raise ValueError(
                f"per-direction attribute {key!r} has odd length {len(raw)}"
            )
        half = len(raw) // 2
        return raw[:half], raw[half:]

    @staticmethod
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

    @staticmethod
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
        self, src, axis: int, length: int, name_prefix: str
    ) -> gs.Variable:
        idx = gs.Constant(
            name=f"{name_prefix}_flip_idx",
            values=np.arange(length - 1, -1, -1, dtype=np.int64),
        )
        out_shape = tuple(src.shape) if getattr(src, "shape", None) else None
        out = gs.Variable(
            name=f"{name_prefix}_flipped", dtype=src.dtype, shape=out_shape,
        )
        self.graph.nodes.append(
            gs.Node(
                op="Gather",
                name=f"{name_prefix}_flip",
                inputs=[src, idx],
                outputs=[out],
                attrs={"axis": axis},
            )
        )
        return out

    def _needs_chunking(self, node: gs.Node) -> bool:
        """Whether ``node`` is a forward RNN that the chunking pass should split."""
        if self.max_chunk_len is None:
            return False
        if id(node) in self._decompose_emitted:
            return False
        attrs = dict(node.attrs)
        if self._attr_str(attrs, "direction", "forward") != "forward":
            return False
        if int(attrs.get("layout", 0)) != 0:
            return False
        if not node.inputs or self._is_empty_input(node.inputs[0]):
            return False
        X = node.inputs[0]
        xshape = getattr(X, "shape", None)
        if not xshape or len(xshape) < 1:
            return False
        try:
            seq_len = int(xshape[0])
        except (TypeError, ValueError):
            return False
        if seq_len <= self.max_chunk_len:
            return False
        # sequence_lens (input 4), if present, must be empty; ragged batches
        # can't be naively chunked along the sequence axis.
        if len(node.inputs) >= 5 and not self._is_empty_input(node.inputs[4]):
            return False
        return True

    def match(self, node: gs.Node) -> bool:
        if node.op not in self._RNN_OPS:
            return False
        direction = self._attr_str(dict(node.attrs), "direction", "forward")
        if direction == "bidirectional":
            return True
        return self._needs_chunking(node)

    def transform(self, node: gs.Node):
        attrs = dict(node.attrs)
        op = node.op
        direction = self._attr_str(attrs, "direction", "forward")

        if direction != "bidirectional":
            # Forward op flagged by _needs_chunking; rewrite it in place.
            self._chunk_in_place(node)
            return

        layout = int(attrs.get("layout", 0))
        if layout != 0:
            raise NotImplementedError(
                f"{op}: bidirectional decomposition only supports layout=0, got {layout}"
            )

        n_per_dir = self._per_direction_size(op)
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
            if self._is_empty_input(var):
                continue
            fwd_var, rev_var = self._split_input(
                var,
                f"{base}_in{i}",
                self._initial_state_shape(i, X, hidden_size),
            )
            fwd_inputs[i] = fwd_var
            rev_inputs[i] = rev_var

        rev_inputs[0] = self._gather_flip(X, seq_axis, seq_len, f"{base}_x")

        fwd_attrs: dict = {}
        rev_attrs: dict = {}
        for key in ("activations", "activation_alpha", "activation_beta"):
            fwd_part, rev_part = self._split_per_direction_attr(attrs, key)
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
            return

        full_out_count = self._full_output_count(op)
        original_out_count = len(raw_outputs)
        while len(raw_outputs) < full_out_count:
            raw_outputs.append(gs.Variable.empty())

        new_nodes: list[gs.Node] = []
        fwd_outs: list[gs.Variable] = []
        rev_outs: list[gs.Variable] = []

        for out_pos, out in enumerate(raw_outputs):
            fwd_outs.append(
                self._bidir_branch_output(
                    base, "fwd", op, out_pos, out, X, hidden_size, layout
                )
            )
            rev_outs.append(
                self._bidir_branch_output(
                    base, "rev", op, out_pos, out, X, hidden_size, layout
                )
            )

        fwd_node = gs.Node(
            op=op, name=f"{base}_fwd", inputs=fwd_inputs, outputs=fwd_outs, attrs=fwd_attrs,
        )
        rev_node = gs.Node(
            op=op, name=f"{base}_rev", inputs=rev_inputs, outputs=rev_outs, attrs=rev_attrs,
        )
        new_nodes.append(fwd_node)
        new_nodes.append(rev_node)
        self._decompose_emitted.add(id(fwd_node))
        self._decompose_emitted.add(id(rev_node))

        for out_pos in range(original_out_count):
            out = raw_outputs[out_pos]
            if self._is_empty_input(out):
                continue
            ax = y_dir_axis if out_pos == 0 else yh_axis
            rev_branch = rev_outs[out_pos]
            if out_pos == 0:
                rev_branch = self._gather_flip(
                    rev_branch, seq_axis, seq_len, f"{base}_out{out_pos}_rev"
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
        self.graph.nodes.extend(new_nodes)
        self._logger.debug(
            "decomposed bidirectional %s %r at layout=%d", op, base, layout
        )
        # Note: the forward branches synthesized here are intentionally not
        # passed through ``_chunk_in_place``. Splitting the time-reversed
        # branch along the sequence axis broke per-step Y stitching on
        # real Synaptics bidirectional GRUs (e.g. voice_filter); preserve
        # the original full-sequence semantics for bidirectional sources.

    def _chunk_in_place(self, node: gs.Node) -> None:
        """
        Rewrite a forward ``GRU`` / ``LSTM`` / ``RNN`` node into a chain of
        shorter equivalent ops along the sequence axis, preserving the
        identity of its output tensors so consumers stay wired.

        Caller must have already confirmed via :meth:`_needs_chunking` that
        ``node`` is a chunkable forward op.
        """
        op = node.op
        attrs = dict(node.attrs)
        X = node.inputs[0]
        seq_len = int(X.shape[0])
        K = self.max_chunk_len
        n_chunks = (seq_len + K - 1) // K
        chunk_sizes = [K] * (n_chunks - 1) + [seq_len - K * (n_chunks - 1)]

        base = node.name or f"{op.lower()}_chunk"

        hidden_size_attr = attrs.get("hidden_size")
        hidden_size = int(hidden_size_attr) if hidden_size_attr is not None else None
        batch: int | None = None
        if len(X.shape) >= 2:
            try:
                batch = int(X.shape[1])
            except (TypeError, ValueError):
                batch = None
        state_shape = (
            (1, batch, hidden_size)
            if (hidden_size is not None and batch is not None)
            else None
        )

        n_inputs = self._FULL_INPUT_COUNT[op]
        n_outputs = self._full_output_count(op)
        raw_inputs = list(node.inputs)
        while len(raw_inputs) < n_inputs:
            raw_inputs.append(gs.Variable.empty())
        raw_outputs = list(node.outputs)
        while len(raw_outputs) < n_outputs:
            raw_outputs.append(gs.Variable.empty())

        split_sizes = gs.Constant(
            name=f"{base}_split_sizes",
            values=np.asarray(chunk_sizes, dtype=np.int64),
        )
        x_chunks: list[gs.Variable] = []
        for i, size in enumerate(chunk_sizes):
            chunk_shape = (
                (size, *tuple(X.shape[1:])) if X.shape and len(X.shape) >= 1 else None
            )
            x_chunks.append(
                gs.Variable(name=f"{base}_x{i}", dtype=X.dtype, shape=chunk_shape)
            )
        new_nodes: list[gs.Node] = [
            gs.Node(
                op="Split",
                name=f"{base}_x_split",
                inputs=[X, split_sizes],
                outputs=x_chunks,
                attrs={"axis": 0},
            )
        ]

        state_in_positions = self._STATE_INPUT_POSITIONS[op]
        state_out_positions = self._STATE_OUTPUT_POSITIONS[op]
        emit_y = not self._is_empty_input(raw_outputs[0])

        # Running state vars threaded between chunks. Initial values are
        # whatever the original op had (may be empty -- ONNX defaults to zero).
        state_vars: list[object] = [raw_inputs[p] for p in state_in_positions]
        y_chunks: list[gs.Variable] = []
        y_dtype = raw_outputs[0].dtype if emit_y else X.dtype

        for i, size in enumerate(chunk_sizes):
            is_last = (i == n_chunks - 1)
            chunk_inputs: list[object] = list(raw_inputs)
            chunk_inputs[0] = x_chunks[i]
            for slot, in_pos in enumerate(state_in_positions):
                chunk_inputs[in_pos] = state_vars[slot]

            chunk_outputs: list[object] = [gs.Variable.empty()] * n_outputs

            if emit_y:
                y_shape = (size, 1, batch, hidden_size) if state_shape else None
                y_chunk = gs.Variable(
                    name=f"{base}_y{i}", dtype=y_dtype, shape=y_shape,
                )
                chunk_outputs[0] = y_chunk
                y_chunks.append(y_chunk)

            next_state_vars: list[object] = []
            for slot, out_pos in enumerate(state_out_positions):
                orig_state_out = raw_outputs[out_pos]
                if is_last:
                    if self._is_empty_input(orig_state_out):
                        state_var = self._unused_slot(
                            f"{base}_chunk{i}",
                            op,
                            out_pos,
                            X,
                            hidden_size,
                            int(attrs.get("layout", 0)),
                            X.dtype,
                        )
                    else:
                        # Reuse the original tensor so any graph-output /
                        # consumer wiring is preserved.
                        orig_state_out.inputs.clear()
                        state_var = orig_state_out
                else:
                    dtype_for_state = (
                        orig_state_out.dtype
                        if not self._is_empty_input(orig_state_out)
                        else X.dtype
                    )
                    state_var = gs.Variable(
                        name=f"{base}_state{slot}_chunk{i}",
                        dtype=dtype_for_state,
                        shape=state_shape,
                    )
                chunk_outputs[out_pos] = state_var
                next_state_vars.append(state_var)

            while chunk_outputs and self._is_empty_input(chunk_outputs[-1]):
                chunk_outputs.pop()

            new_nodes.append(
                gs.Node(
                    op=op,
                    name=f"{base}_chunk{i}",
                    inputs=chunk_inputs,
                    outputs=chunk_outputs,
                    attrs=dict(attrs),
                )
            )
            state_vars = next_state_vars

        if emit_y:
            orig_y = raw_outputs[0]
            orig_y.inputs.clear()
            new_nodes.append(
                gs.Node(
                    op="Concat",
                    name=f"{base}_y_concat",
                    inputs=y_chunks,
                    outputs=[orig_y],
                    attrs={"axis": 0},
                )
            )

        node.inputs.clear()
        node.outputs.clear()
        self.graph.nodes.extend(new_nodes)
        self._logger.debug(
            "chunked %s %r: seq_len=%d -> %d chunk(s) of size %s",
            op, base, seq_len, n_chunks, chunk_sizes,
        )
