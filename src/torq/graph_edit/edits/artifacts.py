# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter

from ..onnx import OnnxGraphEdit, rewire_consumers
from ...utils.onnx import normalize_layer_name


@dataclass
class ExtractGatherBlockQuantizedLUT(OnnxGraphEdit):
    """
    Lift a `com.microsoft.GatherBlockQuantized` embedding table (`bits=4`,
    blockwise scale/zero_point) out of the graph *still packed/quantized* --
    saved as a **directory** of `data_quant.npy`/`scales.npy`/
    `zero_points.npy` plus a `meta.json` carrying the
    `bits`/`block_size`/`per_row_shape` needed to dequantize a row
    host-side -- and replace the *graph output* it eventually feeds -- not
    just the Gather node's own output -- with a graph input of the same
    name.

    Deliberately does NOT eagerly dequantize the whole table: only 1-2 rows
    are ever gathered per inference step, so a full fp32 dequant would be
    pure waste -- for gemma4's two tables, ~1.76GB packed would balloon to
    ~11GB dequantized (confirmed: an earlier version of this edit did
    exactly that, and needed a chunked/memmap'd write + an OOM-debugging
    detour to make the write itself survive a memory cap -- switching to
    packed extraction sidesteps needing any of that, since the packed data
    is roughly the same size as the source graph's own weights and doesn't
    need chunking to write out). The (cheap, host-side) block-dequant math
    to reconstruct a row lives next to whatever reads this file (see
    `_dequant_row`/`_PackedEmbeddingLUT` in `models/gemma4/export_int4.py`
    for the reference implementation and its validation against the
    original op's onnxruntime output).

    Unlike `ExtractConstantLUT` (which lifts a plain float `Gather`'s
    constant table straight out), real usage here has a downstream
    elementwise `Mul` (embedding scale) and optionally a `Reshape` between
    the Gather and the actual graph output; the `Mul`'s scalar is folded
    directly into the saved `scales` array (cheap: shape is `[rows,
    n_blocks]`, not `[rows, k]`) and the `Reshape`'s per-row shape is saved
    as `per_row_shape` metadata, both pruned from the graph the same way any
    other now-dead subgraph is (via the normal `graph.cleanup()` that
    already runs after every edit) -- so the resulting graph has *no nodes
    left* for this embedding, not just a smaller one. Anything upstream of
    the Gather (e.g. an index-redirect `Where` chain for out-of-vocab
    special tokens) is pruned the same way, since nothing consumes it once
    the Gather node is disconnected -- that logic, if present, must be
    replicated by whatever host-side code drives the extracted lookup at
    inference time.

    Args:
        save_to: destination **directory** for the packed table. Written as
            individual `.npy` arrays + `meta.json` rather than one `.npz`
            so the large arrays can be memory-mapped by the reader (numpy
            cannot mmap a `.npz` -- it is a zip container).
        inp_name: name for the new graph input. Defaults to the traced
            graph output's own name (i.e. the input directly replaces the
            output -- literally zero nodes needed).
        node_name: exact `GatherBlockQuantized` node name to target. Graphs
            can contain more than one such node (e.g. gemma4's main
            embedding + per-layer embedding), and `OnnxGraphEditor.apply_edit`
            runs one edit instance across *every* matching node in a single
            pass -- without this filter, one `save_to`/`inp_name` would be
            forced onto all of them. Leave `None` only when the graph is
            known to contain exactly one `GatherBlockQuantized` node.

    Raises:
        ValueError: if the node's `bits != 4`, or if the path from the
            Gather node to a graph output contains anything other than a
            single elementwise `Mul` by a scalar/broadcastable constant
            and/or a single `Reshape` -- this edit only folds *that*
            specific, verified real-usage pattern; anything else needs a
            human to check the math before it's silently baked into the
            saved `scales`/metadata.
    """

    save_to: os.PathLike | str
    inp_name: str | None = None
    node_name: str | None = None

    def match(self, node: gs.Node) -> bool:
        if node.op != "GatherBlockQuantized":
            return False
        return self.node_name is None or node.name == self.node_name

    def _trace_to_graph_output(self, node: gs.Node):
        """Follow the single-consumer chain from `node`'s output forward,
        folding any Mul (by a constant) / Reshape encountered, until a
        tensor that is itself a graph output is reached. Returns
        (graph_output_tensor, post_scale: float | None, post_reshape_dims:
        tuple | None).
        """
        var = node.outputs[0]
        post_scale = None
        post_reshape = None
        visited = 0
        while var not in self.graph.outputs:
            visited += 1
            if visited > 4:
                raise ValueError(
                    f"'{node.name}': path to a graph output is too long / "
                    "not the expected Gather->[Mul]->[Reshape]->output pattern"
                )
            consumers = list(var.outputs)
            if len(consumers) != 1:
                raise ValueError(
                    f"'{node.name}': expected exactly one consumer between "
                    f"the Gather and the graph output, found {len(consumers)}"
                )
            consumer = consumers[0]
            if consumer.op == "Mul":
                const = next((i for i in consumer.inputs if isinstance(i, gs.Constant)), None)
                if const is None:
                    raise ValueError(f"'{consumer.name}': Mul has no constant operand to fold")
                if post_scale is not None:
                    raise ValueError(f"'{node.name}': more than one Mul in the path, unsupported")
                post_scale = float(np.asarray(const.values).reshape(-1)[0])
            elif consumer.op == "Reshape":
                shape_const = next((i for i in consumer.inputs if isinstance(i, gs.Constant)), None)
                if shape_const is None:
                    raise ValueError(f"'{consumer.name}': Reshape has no constant target shape")
                if post_reshape is not None:
                    raise ValueError(f"'{node.name}': more than one Reshape in the path, unsupported")
                post_reshape = tuple(int(d) for d in np.asarray(shape_const.values))
            else:
                raise ValueError(
                    f"'{node.name}': unsupported op '{consumer.op}' between Gather and graph output "
                    "-- only a single Mul and/or Reshape are folded"
                )
            var = consumer.outputs[0]
        return var, post_scale, post_reshape

    def transform(self, node: gs.Node):
        self._check_node_op(node, "GatherBlockQuantized")
        bits = int(node.attrs.get("bits", 4))
        if bits != 4:
            raise ValueError(f"'{node.name}': only bits=4 is handled, got bits={bits}")
        block_size = int(node.attrs.get("block_size", 32))
        quantize_axis = int(node.attrs.get("quantize_axis", 1))
        if quantize_axis != 1:
            raise ValueError(f"'{node.name}': only quantize_axis=1 is handled, got {quantize_axis}")

        data_quant, _indices, scales, zero_points = node.inputs[:4]
        for t, label in ((data_quant, "data"), (scales, "scales"), (zero_points, "zero_points")):
            if not isinstance(t, gs.Constant):
                raise ValueError(f"'{node.name}': '{label}' input must be a constant to extract")

        graph_out, post_scale, post_reshape = self._trace_to_graph_output(node)

        rows, packed_k = data_quant.values.shape
        k = packed_k * 2
        n_blocks = k // block_size

        per_row_shape = ()
        if post_reshape is not None:
            # post_reshape includes the (dynamic, now-fixed) leading dims;
            # only the LUT's own per-row shape (everything after the row
            # dim) is meaningful once extracted -- e.g. real usage reshapes
            # [batch,seq,K] -> [batch,seq,35,256]; the saved table becomes
            # [rows, 35, 256] and the caller reshapes to [1,1,35,256] after
            # indexing a single row.
            candidate = tuple(d for d in post_reshape if d not in (0, -1))[-(len(post_reshape) - 2):] \
                if len(post_reshape) > 2 else ()
            if candidate and int(np.prod(candidate)) == k:
                per_row_shape = candidate

        # Save the table still packed/quantized -- weights, scales,
        # zero_points -- rather than dequantizing all `rows` up front (see
        # class docstring for why: only 1-2 rows are ever gathered per
        # inference step, so a full dequant is pure size blowup). The
        # post-Gather `Mul` scale (if any) is folded directly into `scales`
        # here since that array is tiny (`[rows, n_blocks]`, not `[rows,
        # k]`) -- cheap regardless of table size, so host code doesn't need
        # to separately track it.
        w_packed = np.asarray(data_quant.values)
        zp_packed = np.asarray(zero_points.values)
        scale_vals = np.asarray(scales.values).astype(np.float32)
        if post_scale is not None:
            scale_vals = scale_vals * np.float32(post_scale)

        # Saved as a directory of individual `.npy` arrays plus a small
        # `meta.json`, NOT a single `.npz`: only 1-2 rows are ever read per
        # inference step, and numpy cannot memory-map a `.npz` (it is a zip
        # container, so `mmap_mode` silently has no effect on the members).
        # Keeping these tables resident is a large, pure waste -- measured
        # ~314MB + ~1435MB of RSS for gemma4's two tables before this
        # change. See `_PackedEmbeddingLUT` in `models/gemma4/export_int4.py`
        # for the memory-mapped reader.
        self.save_to = Path(self.save_to)
        self.save_to.mkdir(parents=True, exist_ok=True)
        np.save(self.save_to / "data_quant.npy", w_packed)
        np.save(self.save_to / "scales.npy", scale_vals)
        np.save(self.save_to / "zero_points.npy", zp_packed)
        (self.save_to / "meta.json").write_text(json.dumps({
            "bits": int(bits),
            "block_size": int(block_size),
            "per_row_shape": [int(d) for d in per_row_shape],
        }))

        inp_name = self.inp_name or graph_out.name
        new_inp = gs.Variable(name=inp_name, dtype=graph_out.dtype, shape=graph_out.shape)
        consumers = list(graph_out.outputs)
        rewire_consumers(consumers, graph_out, new_inp)
        for i, go in enumerate(self.graph.outputs):
            if go is graph_out:
                self.graph.outputs[i] = new_inp
        self.graph.inputs.append(new_inp)

        node.inputs.clear()
        node.outputs.clear()
        self._logger.info(
            "Extracted packed GatherBlockQuantized LUT from '%s' -> '%s' (rows=%d, k=%d, "
            "block_size=%d, per_row_shape=%s); graph output '%s' replaced by graph input '%s'",
            node.name, str(self.save_to), rows, k, block_size, per_row_shape, graph_out.name, inp_name,
        )


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
class TrimLMHeadVocab(OnnxGraphEdit):
    """
    Trim LM head weight matrix to a subset of tokens.

    The weight matrix is sliced from [hidden, vocab] to [hidden, kept_count].
    If include_argmax is True, an ArgMax is appended to output the compact index.
    Otherwise the output is the trimmed logits tensor [1, 1, kept_count].

    The caller is responsible for mapping compact_idx -> original token ID via
    kept_token_ids[compact_idx].

    Handles two weight forms for the matched `MatMul`'s second operand:

    - A dense `gs.Constant` (e.g. gemma3's fp32/bf16 lm_head): sliced directly.
    - A `DequantizeLinear -> MatMul` pair (e.g. gemma4's int4-packed lm_head,
      post `DecomposeMatMulNBits`): the `DequantizeLinear`'s three constant
      inputs (packed data / scale / zero_point) are sliced instead, along
      whichever axis its `axis` attribute does *not* block-quantize -- the
      vocab dimension is never the blocked axis (block-quantization blocks
      the hidden/reduction dimension), so this is always a safe, block-
      aligned slice, never splitting a block across kept/dropped tokens.

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
        self.requires_shape_inference = True
        return super().__post_init__()

    @staticmethod
    def _dequant_producer(weight_inp) -> "gs.Node | None":
        if not isinstance(weight_inp, gs.Variable) or not weight_inp.inputs:
            return None
        producer = weight_inp.inputs[0]
        if producer.op != "DequantizeLinear" or len(producer.inputs) < 3:
            return None
        if not all(isinstance(i, gs.Constant) for i in producer.inputs[:3]):
            return None
        return producer

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul" or not node.outputs or len(node.inputs) < 2:
            return False
        if node.outputs[0].name != self.output_name:
            return False
        weight_inp = node.inputs[1]
        return isinstance(weight_inp, gs.Constant) or self._dequant_producer(weight_inp) is not None

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        weight_inp = node.inputs[1]
        dql_node = self._dequant_producer(weight_inp)

        if dql_node is None:
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
        else:
            data_c, scale_c, zp_c = dql_node.inputs[:3]
            if data_c.values.ndim != 2:
                raise ValueError(f"Expected 2-D packed weight, got shape {data_c.values.shape}")
            block_axis = int(dql_node.attrs.get("axis", 0))
            vocab_axis = 1 - block_axis  # the non-block-quantized axis

            hidden_size, vocab_size = data_c.values.shape if vocab_axis == 1 else data_c.values.shape[::-1]
            if np.any(self.kept_token_ids >= vocab_size) or np.any(self.kept_token_ids < 0):
                raise ValueError(
                    f"kept_token_ids contains values outside [0, {vocab_size})"
                )
            kept_count = len(self.kept_token_ids)

            def _trim(c: gs.Constant) -> gs.Constant:
                idx = [slice(None)] * c.values.ndim
                idx[vocab_axis] = self.kept_token_ids
                sliced = c.values[tuple(idx)]
                # Not `gs.Constant(..., values=sliced, ...)`: onnx_graphsurgeon's
                # own export path for a *newly constructed* Constant just does
                # `values.tobytes()` (see `constant_to_onnx_tensor`), which for
                # sub-byte dtypes (int4/uint4) writes the unpacked (1
                # byte/element) buffer instead of ONNX's packed (2/byte) raw
                # format -- verified to corrupt-or-fail on round-trip (`onnx.
                # helper.make_tensor` rejects it: "Raw data size does not
                # match"). It only works for *unmodified* Constants because
                # those keep a `LazyValues` reference to the original,
                # already-correctly-packed `TensorProto` and never touch this
                # path at all. Building through `onnx.numpy_helper.from_array`
                # (which packs correctly) and re-importing via `OnnxImporter.
                # import_tensor` gives the new Constant that same `LazyValues`
                # wrapping, sidestepping the bug entirely. Verified round-trip
                # byte-exact for both int4 and bf16.
                tensor_proto = numpy_helper.from_array(sliced, name=c.name + "_trimmed")
                return OnnxImporter.import_tensor(tensor_proto)

            dql_node.inputs[0] = _trim(data_c)
            dql_node.inputs[1] = _trim(scale_c)
            dql_node.inputs[2] = _trim(zp_c)
            if weight_inp.shape:
                new_shape = list(weight_inp.shape)
                new_shape[vocab_axis] = kept_count
                weight_inp.shape = new_shape

        logits_out = node.outputs[0]
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

        # Every tensor between the trimmed matmul and the graph's actual
        # output(s) still carries its *stale* pre-trim declared shape (e.g.
        # gemma4's logit-softcap `Div`/`Tanh`/`Mul` chain, downstream of
        # `output_name` here rather than *being* it) -- elementwise ops, so
        # the values compute correctly regardless, but the graph's shape
        # metadata is now wrong until cleared and re-inferred. Same fix as
        # `EliminateExpand._clear_downstream_shapes` in shape.py, except
        # graph outputs are cleared too (there the output shape doesn't
        # change; here it does).
        queue = list({id(c): c for c in final_out.outputs}.values())
        visited: set[int] = set()
        while queue:
            n = queue.pop(0)
            if id(n) in visited:
                continue
            visited.add(id(n))
            for out in n.outputs:
                if isinstance(out, gs.Variable):
                    out.shape = None
                    for consumer in out.outputs:
                        if id(consumer) not in visited:
                            queue.append(consumer)

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
