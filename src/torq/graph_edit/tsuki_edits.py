# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from dataclasses import dataclass

import onnx_graphsurgeon as gs
import numpy as np

from .onnx import OnnxGraphEdit


def _create_norm_subgraph(graph: gs.Graph, node: gs.Node, axis: int = -1, epsilon: float = 1e-5):
    """Build the LayerNormalization decomposition into `graph`.

    The terminal op writes directly into `node.outputs[0]` so existing consumers
    pick up the new value without rewiring.
    """
    x = node.inputs[0]
    scale = node.inputs[1]
    bias = node.inputs[2] if len(node.inputs) > 2 else None
    y = node.outputs[0]
    prefix = f"_norm_decomp_{y.name}_"

    x_dtype = getattr(x, "dtype", None)

    def _var(name: str) -> gs.Variable:
        return gs.Variable(name=prefix + name, dtype=x_dtype, shape=None)

    axes_const = gs.Constant(
        name=prefix + "axes",
        values=np.array([axis], dtype=np.int64),
    )
    eps_const = gs.Constant(
        name=prefix + "eps",
        values=np.array(epsilon, dtype=np.float32),
    )

    mean = graph.layer(
        name=prefix + "mean",
        op="ReduceMean",
        inputs=[x, axes_const],
        attrs={"keepdims": 1},
        outputs=[_var("mean_out")],
    )[0]
    diff = graph.layer(
        name=prefix + "diff",
        op="Sub",
        inputs=[x, mean],
        outputs=[_var("diff_out")],
    )[0]
    diff_sq = graph.layer(
        name=prefix + "diff_sq",
        op="Mul",
        inputs=[diff, diff],
        outputs=[_var("diff_sq_out")],
    )[0]
    var = graph.layer(
        name=prefix + "var",
        op="ReduceMean",
        inputs=[diff_sq, axes_const],
        attrs={"keepdims": 1},
        outputs=[_var("var_out")],
    )[0]
    var_eps = graph.layer(
        name=prefix + "var_eps",
        op="Add",
        inputs=[var, eps_const],
        outputs=[_var("var_eps_out")],
    )[0]
    std = graph.layer(
        name=prefix + "std",
        op="Sqrt",
        inputs=[var_eps],
        outputs=[_var("std_out")],
    )[0]
    inv_std = graph.layer(
        name=prefix + "inv_std",
        op="Reciprocal",
        inputs=[std],
        outputs=[_var("inv_std_out")],
    )[0]
    norm = graph.layer(
        name=prefix + "norm",
        op="Mul",
        inputs=[diff, inv_std],
        outputs=[_var("norm_out")],
    )[0]

    if bias is not None:
        scaled = graph.layer(
            name=prefix + "scaled",
            op="Mul",
            inputs=[norm, scale],
            outputs=[_var("scaled_out")],
        )[0]
        graph.layer(
            name=prefix + "biased",
            op="Add",
            inputs=[scaled, bias],
            outputs=[y],
        )
    else:
        graph.layer(
            name=prefix + "scaled",
            op="Mul",
            inputs=[norm, scale],
            outputs=[y],
        )


@dataclass
class DecomposeLayerNorm(OnnxGraphEdit):
    """
    Replace onnx.LayerNormalization with its primitive op decomposition.

    Decomposes ``y = scale * (x - mean) / sqrt(var + eps) + bias`` into
    ReduceMean / Sub / Mul / Add / Sqrt / Reciprocal ops so downstream
    compilers that don't support fused LayerNormalization can lower the graph.

    Notes:
        - Reads ``axis`` and ``epsilon`` from node attributes (defaults: -1, 1e-5)
        - Terminal Add (or scale Mul when no bias) reuses the original output tensor,
          so graph outputs and downstream consumers don't need rewiring.
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "LayerNormalization"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "LayerNormalization")
        axis = node.attrs.get("axis", -1)
        epsilon = node.attrs.get("epsilon", 1e-5)

        _create_norm_subgraph(self.graph, node, axis=axis, epsilon=epsilon)

        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "Decomposed LayerNormalization '%s' (axis=%s, epsilon=%s)",
            node.name, axis, epsilon,
        )

class NormalizationPatches:
    def decompose_layer_norm(self):
        self.apply_edit(DecomposeLayerNorm(self._graph, self._graph_name))
        return self