# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs

from ...graph_edit.edits import DecoderGraphEditor


class QwenOnnxGraphEditor(DecoderGraphEditor):
    """ONNX graph editor for Qwen decoder models."""

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
