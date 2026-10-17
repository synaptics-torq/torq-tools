# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
)
from ...graph_edit.edits import CombineKVCacheMixin, CommonGraphEditsMixin


class QwenOnnxGraphEditor(
    OnnxGraphEditor,
    CommonGraphEditsMixin,
    CombineKVCacheMixin,
):
    """ONNX graph editor for Qwen decoder models."""

    def __init__(
        self,
        graph: gs.Graph,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ):
        super().__init__(
            graph,
            "model",
            export_dtype=export_dtype,
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_model: str | os.PathLike | onnx.ModelProto,
        export_dtype: onnx.TensorProto.DataType | None = None,
    ) -> "QwenOnnxGraphEditor":
        """Create a Qwen graph editor from an ONNX path or model object."""

        if not isinstance(onnx_model, onnx.ModelProto):
            onnx_model = onnx.load(onnx_model)

        graph = gs.import_onnx(onnx_model)

        return cls(
            graph,
            export_dtype,
        )

    def to_onnx(
        self,
        check_type: bool = True,
        strict_mode: bool = True,
        data_prop: bool = True,
        override_ir: int | None = None,
    ) -> onnx.ModelProto:
        """
        Export the edited Qwen graph without in-memory ONNX shape inference.

        Qwen3-0.6B uses multi-gigabyte external weight data. The standard
        editor path calls onnx.shape_inference.infer_shapes(), which attempts
        to serialize the complete model and fails for this model size.
        """
        del check_type, strict_mode, data_prop

        self._graph = self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

        onnx_model = gs.export_onnx(self._graph)

        if isinstance(override_ir, int):
            onnx_model.ir_version = override_ir

        return onnx_model

    def fold_legacy_rotary_cache(self, max_len: int) -> int:
        """
        Replace per-layer ``rotary_emb`` cos/sin cache slices with a static table.

        Older Qwen2/Qwen2.5 exports (transformers <4.45) keep a
        ``[max_position_embeddings, head_dim]`` cos/sin cache in every layer and
        slice it to ``Min(max_pos, past_len + seq_len)`` before gathering by
        ``position_ids``. In the static graph ``position_ids < max_len``, so the
        slice is equivalent to the constant ``table[:max_len]``. Identical tables
        are shared across layers.

        Returns:
            Number of Slice nodes folded.
        """

        shared: dict[bytes, gs.Constant] = {}
        folded = 0

        for node in list(self.graph.nodes):
            if node.op != "Slice" or "/rotary_emb/" not in node.name:
                continue

            table = node.inputs[0]
            starts = node.inputs[1] if len(node.inputs) > 1 else None

            if not isinstance(table, gs.Constant) or table.values.ndim != 2:
                continue
            if not isinstance(starts, gs.Constant) or np.any(starts.values != 0):
                continue

            if table.values.shape[0] < max_len:
                raise ValueError(
                    f"Rotary cache '{table.name}' has {table.values.shape[0]} "
                    f"positions, fewer than max_len={max_len}"
                )

            values = np.ascontiguousarray(table.values[:max_len])
            key = values.tobytes()

            if key not in shared:
                shared[key] = gs.Constant(
                    f"rotary_emb_cache_{len(shared)}",
                    values,
                )

            # Consumers (Gather by position_ids) now read the static table.
            for consumer in list(node.outputs[0].outputs):
                for idx, inp in enumerate(consumer.inputs):
                    if inp is node.outputs[0]:
                        consumer.inputs[idx] = shared[key]

            node.outputs.clear()
            folded += 1

        self.graph.cleanup()

        self._logger.info(
            "Folded %d legacy rotary_emb cache slices into %d static tables "
            "of %d positions",
            folded,
            len(shared),
            max_len,
        )

        return folded

    def blend_combined_kv_in_place(self) -> int:
        """
        Update the combined ``key_value`` cache with one Where per layer.

        ``combine_kv_io_tensors`` splits each ``[1, 2 * kv, S, D]`` cache input into
        K and V, blends the new token into each with a Where, and concatenates the
        results back into the cache output. Every step this copies the whole cache
        three times. This rewrites each layer to::

            present = Where(mask, Concat(new_k, new_v), past)   # one cache write
            K, V    = Slice(present, heads [0, kv)), Slice(present, heads [kv, 2 kv))

        Only the new ``[1, 2 * kv, 1, D]`` rows are concatenated. K and V are
        contiguous head ranges of the cache, so the slices need no reshuffle.

        Returns:
            Number of layers rewritten.
        """

        tensors = self.graph.tensors()
        rewritten = 0
        layer = 0
        while (combined_out := tensors.get(f"present.{layer}.key_value")) is not None:
            concat = combined_out.inputs[0] if combined_out.inputs else None
            past = tensors.get(f"past_key_values.{layer}.key_value")
            key, value = tensors.get(f"present.{layer}.key"), tensors.get(f"present.{layer}.value")
            if concat is None or concat.op != "Concat" or past is None or key is None or value is None:
                raise ValueError(f"Layer {layer} does not have the combined KV pattern")

            key_blend, value_blend = key.inputs[0], value.inputs[0]
            if key_blend.op != "Where" or value_blend.op != "Where":
                raise ValueError(f"Layer {layer} KV update is not a Where")
            mask, new_key = key_blend.inputs[0], key_blend.inputs[1]
            new_value = value_blend.inputs[1]

            kv_heads = key.shape[1]
            dtype = past.dtype
            new_rows = self.graph.layer(
                name=f"present.{layer}_new_kv_concat",
                op="Concat",
                attrs={"axis": 1},
                inputs=[new_key, new_value],
                outputs=[gs.Variable(f"present.{layer}.new_key_value", dtype=dtype,
                                     shape=[1, 2 * kv_heads, 1, key.shape[3]])],
            )[0]

            # The combined Where writes the cache output directly.
            concat.outputs.clear()
            self.graph.layer(
                name=f"present.{layer}_blend_kv",
                op="Where",
                inputs=[mask, new_rows, past],
                outputs=[combined_out],
            )

            # Attention reads K and V as head ranges of the updated cache.
            for tensor, blend, start in ((key, key_blend, 0), (value, value_blend, kv_heads)):
                blend.outputs.clear()
                self.graph.layer(
                    name=f"{tensor.name}_from_present",
                    op="Slice",
                    inputs=[
                        combined_out,
                        np.array([start], dtype=np.int64),
                        np.array([start + kv_heads], dtype=np.int64),
                        np.array([1], dtype=np.int64),
                    ],
                    outputs=[tensor],
                )

            rewritten += 1
            layer += 1

        self.graph.cleanup().toposort()

        self._logger.info("Blended %d combined KV caches in place", rewritten)

        return rewritten

    def grouped_query_attention(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        """
        Attend against the un-expanded KV caches instead of ``repeat_kv`` copies.

        Exported GQA attention expands each ``[1, kv, S, D]`` cache to
        ``[1, heads, S, D]`` (Unsqueeze -> Expand -> Reshape) and applies half of
        the softmax scale to the expanded K. For single-token decode this rewrites
        each layer to::

            Qg = Reshape(Q * (q_scale * k_scale), [1, kv, heads / kv, D])
            O  = Softmax(Qg @ Transpose(K) + mask) @ V      # [1, kv, heads/kv, D]
            O  = Reshape(O, [1, heads, 1, D])

        Query head ``h`` maps to KV group ``h // (heads / kv)``, matching HF's
        ``repeat_kv``. This removes the per-step copy and scale of the full cache.

        Returns:
            Number of layers rewritten.
        """

        group = num_heads // num_kv_heads
        nodes = {node.name: node for node in self.graph.nodes}

        def producer(tensor: gs.Tensor, op: str) -> gs.Node:
            node = tensor.inputs[0] if tensor.inputs else None
            if node is None or node.op != op:
                raise ValueError(f"Expected {op} producing '{tensor.name}'")
            return node

        def scale_of(mul: gs.Node) -> tuple[gs.Tensor, float]:
            data, const = mul.inputs
            values = None
            if isinstance(const, gs.Constant):
                values = const.values
            elif const.inputs and const.inputs[0].op == "Constant":
                # Newer exports (e.g. Qwen3) emit the scale as a Constant node.
                values = const.inputs[0].attrs["value"].values
            if values is None or values.size != 1:
                raise ValueError(f"'{mul.name}' is not a scalar scale")
            return data, float(values.ravel()[0])

        rewritten = 0
        layer = 0
        while (scores := nodes.get(f"/model/layers.{layer}/self_attn/MatMul")) is not None:
            base = f"/model/layers.{layer}/self_attn/"
            context = nodes[base + "MatMul_1"]

            # Q side: Mul(q, q_scale). K side: Mul(Transpose(Reshape(Expand(Unsqueeze(K)))), k_scale).
            q, q_scale = scale_of(producer(scores.inputs[0], "Mul"))
            k_t, k_scale = scale_of(producer(scores.inputs[1], "Mul"))
            k_cache = producer(
                producer(producer(producer(k_t, "Transpose").inputs[0], "Reshape").inputs[0], "Expand").inputs[0],
                "Unsqueeze",
            ).inputs[0]
            v_cache = producer(
                producer(producer(context.inputs[1], "Reshape").inputs[0], "Expand").inputs[0],
                "Unsqueeze",
            ).inputs[0]
            dtype = q.dtype or np.float32

            q_grouped = self.graph.layer(
                name=base + "GQA/QReshape",
                op="Reshape",
                inputs=[
                    self.graph.layer(
                        name=base + "GQA/QScale",
                        op="Mul",
                        inputs=[q, np.array(q_scale * k_scale, dtype=np.float32)],
                        outputs=[gs.Variable(base + "GQA/q_scaled", dtype=dtype, shape=[1, num_heads, 1, head_dim])],
                    )[0],
                    np.array([1, num_kv_heads, group, head_dim], dtype=np.int64),
                ],
                outputs=[gs.Variable(base + "GQA/q", dtype=dtype, shape=[1, num_kv_heads, group, head_dim])],
            )[0]
            k_transposed = self.graph.layer(
                name=base + "GQA/KTranspose",
                op="Transpose",
                attrs={"perm": [0, 1, 3, 2]},
                inputs=[k_cache],
                outputs=[gs.Variable(base + "GQA/k_t", dtype=dtype)],
            )[0]

            # Scores [1, kv, group, S]; the [1, 1, 1, S] mask Add still broadcasts.
            scores.inputs = [q_grouped, k_transposed]
            scores.outputs[0].shape = None
            score_users = list(scores.outputs[0].outputs)
            for user in score_users:
                user.outputs[0].shape = None
            softmax_out = next(u for u in score_users if u.op == "Add").outputs[0].outputs[0].outputs[0]
            softmax_out.shape = None

            # Context [1, kv, group, D] -> [1, heads, 1, D] for the unchanged head merge.
            context_out = context.outputs[0]
            grouped_out = gs.Variable(base + "GQA/context", dtype=dtype, shape=[1, num_kv_heads, group, head_dim])
            context.inputs = [context.inputs[0], v_cache]
            context.outputs = [grouped_out]
            self.graph.layer(
                name=base + "GQA/ContextReshape",
                op="Reshape",
                inputs=[grouped_out, np.array([1, num_heads, 1, head_dim], dtype=np.int64)],
                outputs=[context_out],
            )

            rewritten += 1
            layer += 1

        self.graph.cleanup().toposort()

        self._logger.info(
            "Rewrote %d layers to grouped-query attention (%d heads over %d KV heads)",
            rewritten,
            num_heads,
            num_kv_heads,
        )

        return rewritten

    def rotary_gather_to_onehot_matmul(self) -> int:
        """
        Replace ``Gather(rotary_table, position_ids)`` with a one-hot MatMul.

        Torq has no NPU lowering for a data-dependent row Gather, so with
        ``--torq-disable-host --torq-disable-css`` the static rotary-table
        lookup from :meth:`fold_legacy_rotary_cache` cannot be compiled.
        ``onehot(position_ids) @ table`` selects the same row on the NPU and is
        exact (a single non-zero product). Gathers of the same table share one
        MatMul, and all tables share one one-hot vector.

        Returns:
            Number of Gather nodes replaced.
        """

        gathers = [
            node
            for node in self.graph.nodes
            if node.op == "Gather"
            and isinstance(node.inputs[0], gs.Constant)
            and node.inputs[0].name.startswith("rotary_emb_cache_")
            and node.attrs.get("axis", 0) == 0
        ]
        if not gathers:
            return 0

        indices = gathers[0].inputs[1]
        if any(g.inputs[1] is not indices for g in gathers):
            raise ValueError("Rotary Gathers use different index tensors")

        num_positions = gathers[0].inputs[0].values.shape[0]

        # [1, 1] positions -> [1, 1, 1] == arange(P) -> [1, 1, P] one-hot.
        positions = self.graph.layer(
            name="rotary_onehot/Unsqueeze",
            op="Unsqueeze",
            inputs=[indices, np.array([-1], dtype=np.int64)],
            outputs=[gs.Variable("rotary_onehot/positions", dtype=indices.dtype)],
        )[0]
        is_position = self.graph.layer(
            name="rotary_onehot/Equal",
            op="Equal",
            inputs=[
                positions,
                gs.Constant(
                    "rotary_onehot/arange",
                    np.arange(num_positions, dtype=indices.dtype or np.int64),
                ),
            ],
            outputs=[gs.Variable("rotary_onehot/mask", dtype=np.bool_)],
        )[0]
        onehot = self.graph.layer(
            name="rotary_onehot/Cast",
            op="Cast",
            attrs={"to": onnx.TensorProto.FLOAT},
            inputs=[is_position],
            outputs=[
                gs.Variable(
                    "rotary_onehot",
                    dtype=np.float32,
                    shape=[1, 1, num_positions],
                )
            ],
        )[0]

        selected: dict[str, gs.Variable] = {}
        for gather in gathers:
            table = gather.inputs[0]
            if table.name not in selected:
                selected[table.name] = self.graph.layer(
                    name=f"{table.name}/OneHotMatMul",
                    op="MatMul",
                    inputs=[onehot, table],
                    outputs=[
                        gs.Variable(
                            f"{table.name}/selected",
                            dtype=np.float32,
                            shape=[1, 1, table.values.shape[1]],
                        )
                    ],
                )[0]

            out = gather.outputs[0]
            for consumer in list(out.outputs):
                for idx, inp in enumerate(consumer.inputs):
                    if inp is out:
                        consumer.inputs[idx] = selected[table.name]
            gather.outputs.clear()

        self.graph.cleanup().toposort()

        self._logger.info(
            "Replaced %d rotary table Gathers with %d one-hot MatMuls",
            len(gathers),
            len(selected),
        )

        return len(gathers)

    def materialize_tied_lm_head(self) -> bool:
        """
        Replace a ``Transpose(embed_tokens.weight)`` LM-head operand with a constant.

        Exports of tied-embedding models (e.g. Qwen2.5-0.5B) feed the LM head
        from a runtime Transpose of the embedding table. Vocabulary trimming
        and weight quantization expect a constant ``[hidden, vocab]`` weight.

        Returns:
            True if the LM head weight was materialized.
        """

        lm_head = next(
            (node for node in self.graph.nodes if node.name == "/lm_head/MatMul"),
            None,
        )
        if lm_head is None or isinstance(lm_head.inputs[1], gs.Constant):
            return False

        weight = lm_head.inputs[1]
        transpose = weight.inputs[0] if weight.inputs else None

        if (
            transpose is None
            or transpose.op != "Transpose"
            or not isinstance(transpose.inputs[0], gs.Constant)
            or list(transpose.attrs.get("perm", [1, 0])) != [1, 0]
        ):
            raise ValueError(
                "LM head weight is neither a constant nor Transpose(constant)"
            )

        source = transpose.inputs[0]
        lm_head.inputs[1] = gs.Constant(
            "lm_head.weight",
            np.ascontiguousarray(source.values.T),
        )
        self.graph.cleanup()

        self._logger.info(
            "Materialized tied LM head weight from '%s' %s",
            source.name,
            list(lm_head.inputs[1].shape),
        )

        return True

    def split_rope_qk(self, head_dim: int = 128) -> None:
        """Replace Qwen RoPE Q/K Slice pairs with Split operations."""

        half = head_dim // 2
        split_name = f"qwen_rope_split_sizes_{half}_{half}"

        split_sizes = self._graph.tensors().get(split_name)

        if split_sizes is None:
            split_sizes = gs.Constant(
                name=split_name,
                values=np.array([half, half], dtype=np.int64),
            )

        layers = sorted({
            int(node.name.split("/model/layers.")[1].split("/")[0])
            for node in self._graph.nodes
            if "/model/layers." in node.name
        })

        for layer in layers:
            base = f"/model/layers.{layer}/self_attn/"

            specs = [
                ("Slice", "Slice_1", "RoPE_Q_Split"),
                ("Slice_2", "Slice_3", "RoPE_K_Split"),
            ]

            for first_name, second_name, split_name_suffix in specs:
                first = next(
                    (
                        node
                        for node in self._graph.nodes
                        if node.name == base + first_name
                    ),
                    None,
                )
                second = next(
                    (
                        node
                        for node in self._graph.nodes
                        if node.name == base + second_name
                    ),
                    None,
                )

                if first is None and second is None:
                    continue

                if first is None or second is None:
                    raise ValueError(
                        f"Incomplete RoPE Slice pair for layer {layer}: "
                        f"{first_name}, {second_name}"
                    )

                split = gs.Node(
                    op="Split",
                    name=base + split_name_suffix,
                    inputs=[
                        first.inputs[0],
                        split_sizes,
                    ],
                    outputs=[
                        first.outputs[0],
                        second.outputs[0],
                    ],
                    attrs={"axis": 3},
                )

                self._graph.nodes.append(split)

                first.outputs.clear()
                second.outputs.clear()

                self._graph.nodes.remove(first)
                self._graph.nodes.remove(second)

        self._graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

        self._logger.info(
            "Replaced Qwen RoPE Slice pairs with Split operations"
        )

    def split_transformer_layers(
        self,
        split_at: int = 14,
        num_layers: int = 28,
        hidden_size: int = 1024,
    ) -> tuple[gs.Graph, gs.Graph]:
        """Split the static Qwen transformer into two runnable graphs.

        The split is made between layers ``split_at - 1`` and ``split_at``.
        For Qwen3-0.6B, the default split is layers 0-13 and 14-27.
        """
        if not 0 < split_at < num_layers:
            raise ValueError(
                f"split_at must be between 1 and {num_layers - 1}, got {split_at}"
            )

        source_graph = self._graph.copy()

        def cache_names(start: int, end: int) -> list[str]:
            return [
                f"past_key_values.{i}.key_value"
                for i in range(start, end)
            ]

        def make_partition(
            part: str,
            name: str,
        ) -> gs.Graph:
            # Each partition gets its own independent graph copy.
            graph = source_graph.copy()
            tensors = graph.tensors()

            token_embedding = tensors["token_embedding"]
            position_ids = tensors["position_ids"]
            final_hidden = tensors["last_hidden_states"]

            boundary_name = (
                f"/model/layers.{split_at - 1}/Add_1_output_0"
            )
            boundary = tensors[boundary_name]

            # Detach runtime inputs from their producers so each partition
            # has an independent graph boundary.
            position_ids.inputs.clear()
            position_ids.shape = [1, 1]

            if part == "A":
                # Keep the boundary-layer producer connected. The boundary tensor is
                # an output of Part A.
                boundary.shape = [1, 1, hidden_size]
                boundary.dtype = np.float32

                inputs = [
                    token_embedding,
                    position_ids,
                    *[
                        tensors[f"past_key_values.{i}.key_value"]
                        for i in range(0, split_at)
                    ],
                ]

                outputs = [
                    tensors[f"present.{i}.key_value"]
                    for i in range(0, split_at)
                ] + [boundary]

            else:
                boundary.inputs.clear()
                boundary.shape = [1, 1, hidden_size]
                boundary.dtype = np.float32

                inputs = [
                    boundary,
                    position_ids,
                    *[
                        tensors[f"past_key_values.{i}.key_value"]
                        for i in range(split_at, num_layers)
                    ],
                ]

                outputs = [
                    tensors[f"present.{i}.key_value"]
                    for i in range(split_at, num_layers)
                ] + [final_hidden]

            graph.inputs = inputs
            graph.outputs = outputs

            return graph.cleanup(
                remove_unused_graph_inputs=True,
                remove_unused_node_outputs=False,
            ).toposort()

        part_a = make_partition(
            "A",
            "transformer_part_A",
        )

        part_b = make_partition(
            "B",
            "transformer_part_B",
        )

        self._logger.info(
            "Split Qwen transformer at layer %d: "
            "Part A layers 0-%d, Part B layers %d-%d",
            split_at,
            split_at - 1,
            split_at,
            num_layers - 1,
        )

        return part_a, part_b

    def replace_current_len_squeeze(self) -> None:
        """
        Replace the Qwen current-length Squeeze with an equivalent Reshape.

        The static graph contains:

            position_ids [1, 1]
                -> Squeeze(axis=0)
                -> position_ids_squeezed [1]

        Torch-MLIR crashes while folding this Squeeze after dtype conversion.
        Reshape([1]) is equivalent here and avoids that compiler path.
        """

        for node in self._graph.nodes:
            if (
                node.op != "Squeeze"
                or node.name != "current_len_to_1d"
                or not node.inputs
                or not node.outputs
            ):
                continue

            inp = node.inputs[0]
            out = node.outputs[0]

            reshape_shape = gs.Constant(
                name="current_len_to_1d_shape",
                values=np.array([1], dtype=np.int64),
            )

            new_out = self._graph.layer(
                name="current_len_to_1d_reshape",
                op="Reshape",
                inputs=[
                    inp,
                    reshape_shape,
                ],
                outputs=[
                    gs.Variable(
                        name=out.name,
                        dtype=out.dtype,
                        shape=[1],
                    )
                ],
            )[0]

            consumers = list(out.outputs)

            for consumer in consumers:
                for i, consumer_input in enumerate(consumer.inputs):
                    if consumer_input is out:
                        consumer.inputs[i] = new_out

            for i, graph_out in enumerate(self._graph.outputs):
                if graph_out is out:
                    self._graph.outputs[i] = new_out

            node.inputs.clear()
            node.outputs.clear()

            self._logger.info(
                "Replaced current_len_to_1d Squeeze with Reshape([1])"
            )

            break

    def _eval_static_tensor(
        self,
        tensor: gs.Tensor,
    ) -> np.ndarray | None:
        """
        Evaluate a limited set of Qwen shape-expression tensors.

        This intentionally handles only simple ONNX operations used to build
        shape and Slice/Expand parameter tensors. If any required value cannot
        be proven static, return None instead of guessing.
        """

        if isinstance(tensor, gs.Constant):
            try:
                return np.asarray(tensor.values)
            except Exception:
                return None

        producers = getattr(tensor, "inputs", None) or []
        if len(producers) != 1:
            return None

        node = producers[0]

        if node.op == "Constant":
            value = node.attrs.get("value")

            if value is None or not hasattr(value, "values"):
                return None

            return np.asarray(value.values)

        if node.op == "Shape":
            if not node.inputs:
                return None

            source = node.inputs[0]
            source_shape = getattr(source, "shape", None)

            if (
                source_shape is not None
                and all(
                    isinstance(dim, (int, np.integer))
                    for dim in source_shape
                )
            ):
                return np.asarray(source_shape, dtype=np.int64)

            source_value = self._eval_static_tensor(source)

            if source_value is None:
                return None

            return np.asarray(
                np.asarray(source_value).shape,
                dtype=np.int64,
            )

        if node.op == "Gather":
            if len(node.inputs) < 2:
                return None

            data = self._eval_static_tensor(node.inputs[0])
            indices = self._eval_static_tensor(node.inputs[1])

            if data is None or indices is None:
                return None

            axis = int(node.attrs.get("axis", 0))

            try:
                return np.take(
                    data,
                    np.asarray(indices).astype(np.int64),
                    axis=axis,
                )
            except Exception:
                return None

        if node.op == "Unsqueeze":
            if not node.inputs:
                return None

            data = self._eval_static_tensor(node.inputs[0])

            if data is None:
                return None

            if len(node.inputs) >= 2:
                axes_value = self._eval_static_tensor(node.inputs[1])

                if axes_value is None:
                    return None

                axes = [
                    int(value)
                    for value in np.asarray(axes_value).reshape(-1)
                ]
            else:
                raw_axes = node.attrs.get("axes")

                if raw_axes is None:
                    return None

                axes = [int(value) for value in raw_axes]

            result = np.asarray(data)

            try:
                for axis in sorted(axes):
                    result = np.expand_dims(
                        result,
                        axis=axis,
                    )

                return result
            except Exception:
                return None

        if node.op == "Squeeze":
            if not node.inputs:
                return None

            data = self._eval_static_tensor(node.inputs[0])

            if data is None:
                return None

            if len(node.inputs) >= 2:
                axes_value = self._eval_static_tensor(node.inputs[1])

                if axes_value is None:
                    return None

                axes = tuple(
                    int(value)
                    for value in np.asarray(axes_value).reshape(-1)
                )
            else:
                raw_axes = node.attrs.get("axes")

                if raw_axes is None:
                    axes = None
                else:
                    axes = tuple(int(value) for value in raw_axes)

            try:
                return np.squeeze(
                    np.asarray(data),
                    axis=axes,
                )
            except Exception:
                return None

        if node.op == "Concat":
            values = []

            for inp in node.inputs:
                value = self._eval_static_tensor(inp)

                if value is None:
                    return None

                values.append(np.asarray(value))

            axis = int(node.attrs.get("axis", 0))

            try:
                return np.concatenate(
                    values,
                    axis=axis,
                )
            except Exception:
                return None

        if node.op == "Reshape":
            if len(node.inputs) < 2:
                return None

            data = self._eval_static_tensor(node.inputs[0])
            target_shape = self._eval_static_tensor(node.inputs[1])

            if data is None or target_shape is None:
                return None

            target_shape = [
                int(value)
                for value in np.asarray(target_shape).reshape(-1)
            ]

            try:
                return np.reshape(
                    np.asarray(data),
                    target_shape,
                )
            except Exception:
                return None

        if node.op == "ConstantOfShape":
            if not node.inputs:
                return None

            shape_value = self._eval_static_tensor(node.inputs[0])

            if shape_value is None:
                return None

            shape = [
                int(value)
                for value in np.asarray(shape_value).reshape(-1)
            ]

            if any(dim < 0 for dim in shape):
                return None

            fill_value = 0

            attr_value = node.attrs.get("value")

            if (
                attr_value is not None
                and hasattr(attr_value, "values")
            ):
                fill_array = np.asarray(attr_value.values)

                if fill_array.size != 1:
                    return None

                fill_value = fill_array.reshape(()).item()

            try:
                return np.full(
                    shape,
                    fill_value,
                )
            except Exception:
                return None

        if node.op in {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Equal",
        }:
            if len(node.inputs) != 2:
                return None

            left = self._eval_static_tensor(node.inputs[0])
            right = self._eval_static_tensor(node.inputs[1])

            if left is None or right is None:
                return None

            try:
                if node.op == "Add":
                    return np.add(left, right)

                if node.op == "Sub":
                    return np.subtract(left, right)

                if node.op == "Mul":
                    return np.multiply(left, right)

                if node.op == "Div":
                    return np.divide(left, right)

                return np.equal(left, right)

            except Exception:
                return None

        if node.op == "Where":
            if len(node.inputs) != 3:
                return None

            condition = self._eval_static_tensor(node.inputs[0])
            true_value = self._eval_static_tensor(node.inputs[1])
            false_value = self._eval_static_tensor(node.inputs[2])

            if (
                condition is None
                or true_value is None
                or false_value is None
            ):
                return None

            try:
                return np.where(
                    condition,
                    true_value,
                    false_value,
                )
            except Exception:
                return None

        if node.op == "Cast":
            if not node.inputs:
                return None

            value = self._eval_static_tensor(node.inputs[0])

            if value is None:
                return None

            to_dtype = node.attrs.get("to")

            dtype_map = {
                onnx.TensorProto.INT64: np.int64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.BOOL: np.bool_,
            }

            np_dtype = dtype_map.get(to_dtype)

            if np_dtype is None:
                return None

            try:
                return np.asarray(value).astype(np_dtype)
            except Exception:
                return None

        return None

    def resolve_static_shape_inputs(self) -> None:
        """
        Materialize statically-computable Expand and Slice parameters.

        Optimum-generated Qwen graphs contain shape-expression chains whose
        values become constant after Qwen I/O and KV-cache dimensions are fixed.

        This pass:
        1. replaces statically-computable Expand shape inputs with constants;
        2. replaces statically-computable Slice parameters with constants;
        3. computes Slice output-shape metadata when the Slice is fully static.
        """

        resolved_count = 0
        resolved_slice_shapes = 0

        for node in self._graph.nodes:
            if node.op == "Expand":
                if len(node.inputs) < 2:
                    continue

                shape_value = self._eval_static_tensor(
                    node.inputs[1]
                )

                if shape_value is None:
                    continue

                shape_value = np.asarray(
                    shape_value,
                    dtype=np.int64,
                ).reshape(-1)

                if (
                    shape_value.size == 0
                    or np.any(shape_value < 0)
                ):
                    continue

                node.inputs[1] = gs.Constant(
                    name=f"{node.name}_static_shape",
                    values=shape_value,
                )

                resolved_count += 1

            elif node.op == "Slice":
                # ONNX Slice inputs:
                #   0: data
                #   1: starts
                #   2: ends
                #   3: axes
                #   4: steps

                for input_index in range(
                    1,
                    min(len(node.inputs), 5),
                ):
                    value = self._eval_static_tensor(
                        node.inputs[input_index]
                    )

                    if value is None:
                        continue

                    value = np.asarray(
                        value,
                        dtype=np.int64,
                    )

                    node.inputs[input_index] = gs.Constant(
                        name=(
                            f"{node.name}_static_param_"
                            f"{input_index}"
                        ),
                        values=value,
                    )

                    resolved_count += 1

                if (
                    len(node.inputs) < 3
                    or not node.outputs
                ):
                    continue

                input_shape = getattr(
                    node.inputs[0],
                    "shape",
                    None,
                )

                if (
                    input_shape is None
                    or not all(
                        isinstance(dim, (int, np.integer))
                        and int(dim) >= 0
                        for dim in input_shape
                    )
                ):
                    continue

                starts = self._eval_static_tensor(
                    node.inputs[1]
                )
                ends = self._eval_static_tensor(
                    node.inputs[2]
                )

                if starts is None or ends is None:
                    continue

                starts = [
                    int(v)
                    for v in np.asarray(starts).reshape(-1)
                ]
                ends = [
                    int(v)
                    for v in np.asarray(ends).reshape(-1)
                ]

                if len(node.inputs) >= 4:
                    axes_value = self._eval_static_tensor(
                        node.inputs[3]
                    )

                    if axes_value is None:
                        continue

                    axes = [
                        int(v)
                        for v in np.asarray(
                            axes_value
                        ).reshape(-1)
                    ]
                else:
                    axes = list(range(len(starts)))

                if len(node.inputs) >= 5:
                    steps_value = self._eval_static_tensor(
                        node.inputs[4]
                    )

                    if steps_value is None:
                        continue

                    steps = [
                        int(v)
                        for v in np.asarray(
                            steps_value
                        ).reshape(-1)
                    ]
                else:
                    steps = [1] * len(starts)

                if not (
                    len(starts)
                    == len(ends)
                    == len(axes)
                    == len(steps)
                ):
                    continue

                output_shape = [
                    int(dim)
                    for dim in input_shape
                ]

                try:
                    for start, end, axis, step in zip(
                        starts,
                        ends,
                        axes,
                        steps,
                    ):
                        rank = len(output_shape)

                        if axis < 0:
                            axis += rank

                        if axis < 0 or axis >= rank:
                            raise ValueError

                        if step == 0:
                            raise ValueError

                        dim_size = output_shape[axis]

                        normalized_start, normalized_end, normalized_step = (
                            slice(
                                start,
                                end,
                                step,
                            ).indices(dim_size)
                        )

                        output_shape[axis] = len(
                            range(
                                normalized_start,
                                normalized_end,
                                normalized_step,
                            )
                        )

                except Exception:
                    continue

                node.outputs[0].shape = output_shape
                resolved_slice_shapes += 1

        self._logger.info(
            "Resolved %d static Qwen shape inputs and %d Slice output shapes",
            resolved_count,
            resolved_slice_shapes,
        )

    def convert_static_shape_params_to_int32(self) -> None:
        """Convert Qwen-materialized static shape parameters from INT64 to INT32."""
        converted = 0

        for node in self._graph.nodes:
            if node.op == "Expand" and len(node.inputs) >= 2:
                tensor = node.inputs[1]
                if (
                    isinstance(tensor, gs.Constant)
                    and "_static_shape" in tensor.name
                    and tensor.values.dtype == np.int64
                ):
                    node.inputs[1] = gs.Constant(
                        name=f"{tensor.name}_int32",
                        values=np.asarray(tensor.values, dtype=np.int32),
                    )
                    converted += 1

            elif node.op == "Slice":
                for input_index in range(
                    1,
                    min(len(node.inputs), 5),
                ):
                    tensor = node.inputs[input_index]
                    if (
                        isinstance(tensor, gs.Constant)
                        and "_static_param_" in tensor.name
                        and tensor.values.dtype == np.int64
                    ):
                        node.inputs[input_index] = gs.Constant(
                            name=f"{tensor.name}_int32",
                            values=np.asarray(
                                tensor.values,
                                dtype=np.int32,
                            ),
                        )
                        converted += 1

        self._logger.info(
            "Converted %d Qwen static shape parameters from INT64 to INT32",
            converted,
        )

    def fix_io(
        self,
        seq_len: int,
        dims: list[FixedDimMapping] | None = None,
        *,
        batch_dim: str = "batch_size",
        seq_len_dim: str = "sequence_length",
        past_seq_len_dim: str = "past_sequence_length",
    ) -> None:
        """Replace Qwen's dynamic input/output dimensions with fixed values."""

        to_fix = [
            FixedDimMapping(
                batch_dim,
                DimMatchType.EXACT,
                1,
            ),
            FixedDimMapping(
                seq_len_dim,
                DimMatchType.EXACT,
                1,
            ),
            FixedDimMapping(
                past_seq_len_dim,
                DimMatchType.CONTAINS,
                seq_len,
            ),
        ]

        to_fix.extend(dims or [])

        self.fix_io_dims(to_fix)
