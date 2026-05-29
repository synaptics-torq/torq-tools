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
    x_shape = getattr(x, "shape", None)
    reduced_shape = None
    if x_shape is not None:
        norm_axis = axis if axis >= 0 else len(x_shape) + axis
        reduced_shape = list(x_shape)
        reduced_shape[norm_axis] = 1

    def _var(name: str, shape=x_shape) -> gs.Variable:
        return gs.Variable(name=prefix + name, dtype=x_dtype, shape=shape)

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
        outputs=[_var("mean_out", shape=reduced_shape)],
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
        outputs=[_var("var_out", shape=reduced_shape)],
    )[0]
    var_eps = graph.layer(
        name=prefix + "var_eps",
        op="Add",
        inputs=[var, eps_const],
        outputs=[_var("var_eps_out", shape=reduced_shape)],
    )[0]
    std = graph.layer(
        name=prefix + "std",
        op="Sqrt",
        inputs=[var_eps],
        outputs=[_var("std_out", shape=reduced_shape)],
    )[0]
    inv_std = graph.layer(
        name=prefix + "inv_std",
        op="Reciprocal",
        inputs=[std],
        outputs=[_var("inv_std_out", shape=reduced_shape)],
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

def _create_instance_norm_subgraph(
    graph: gs.Graph,
    node: gs.Node,
    epsilon: float = 1e-5,
    N: gs.Variable | None = None,
):
    """Build the InstanceNormalization decomposition into `graph`.

    Mean/var are computed over spatial axes (2..rank-1) per (N, C) slice.
    Scale/bias of shape (C,) are unsqueezed to (1, C, 1, ..., 1) before broadcast.

    When `N` is provided, mean/var are computed as ``ReduceSum(x) / N`` using
    the caller-supplied runtime divisor instead of ``ReduceMean(x)``.
    """
    x = node.inputs[0]
    scale = node.inputs[1]
    bias = node.inputs[2] if len(node.inputs) > 2 else None
    y = node.outputs[0]
    prefix = f"_in_decomp_{y.name}_"

    x_dtype = getattr(x, "dtype", None)
    x_shape = getattr(x, "shape", None)
    if x_shape is None:
        raise ValueError(
            f"InstanceNormalization decomposition requires shape info on input '{x.name}'"
        )
    rank = len(x_shape)
    if rank < 3:
        raise ValueError(
            f"InstanceNormalization expects rank >= 3 input, got rank {rank} for '{x.name}'"
        )
    spatial_axes = list(range(2, rank))
    unsqueeze_axes = [0] + spatial_axes
    reduced_shape = [x_shape[0], x_shape[1]] + [1] * len(spatial_axes)
    channel_shape = [1, x_shape[1]] + [1] * len(spatial_axes)

    def _var(name: str, shape=x_shape) -> gs.Variable:
        return gs.Variable(name=prefix + name, dtype=x_dtype, shape=shape)

    axes_const = gs.Constant(
        name=prefix + "axes",
        values=np.array(spatial_axes, dtype=np.int64),
    )
    eps_const = gs.Constant(
        name=prefix + "eps",
        values=np.array(epsilon, dtype=np.float32),
    )
    unsqueeze_axes_const = gs.Constant(
        name=prefix + "unsqueeze_axes",
        values=np.array(unsqueeze_axes, dtype=np.int64),
    )

    reduce_op = "ReduceSum" if N is not None else "ReduceMean"

    def _reduce(in_tensor: gs.Variable, label: str) -> gs.Variable:
        reduced = graph.layer(
            name=prefix + label,
            op=reduce_op,
            inputs=[in_tensor, axes_const],
            attrs={"keepdims": 1},
            outputs=[_var(label + "_out", shape=reduced_shape)],
        )[0]
        if N is not None:
            reduced = graph.layer(
                name=prefix + label + "_div_n",
                op="Div",
                inputs=[reduced, N],
                outputs=[_var(label + "_mean", shape=reduced_shape)],
            )[0]
        return reduced

    mean = _reduce(x, "mean")
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
    var = _reduce(diff_sq, "var")
    var_eps = graph.layer(
        name=prefix + "var_eps",
        op="Add",
        inputs=[var, eps_const],
        outputs=[_var("var_eps_out", shape=reduced_shape)],
    )[0]
    std = graph.layer(
        name=prefix + "std",
        op="Sqrt",
        inputs=[var_eps],
        outputs=[_var("std_out", shape=reduced_shape)],
    )[0]
    inv_std = graph.layer(
        name=prefix + "inv_std",
        op="Reciprocal",
        inputs=[std],
        outputs=[_var("inv_std_out", shape=reduced_shape)],
    )[0]
    norm = graph.layer(
        name=prefix + "norm",
        op="Mul",
        inputs=[diff, inv_std],
        outputs=[_var("norm_out")],
    )[0]

    scale_reshaped = graph.layer(
        name=prefix + "scale_unsqueeze",
        op="Unsqueeze",
        inputs=[scale, unsqueeze_axes_const],
        outputs=[_var("scale_reshaped", shape=channel_shape)],
    )[0]

    if bias is not None:
        scaled = graph.layer(
            name=prefix + "scaled",
            op="Mul",
            inputs=[norm, scale_reshaped],
            outputs=[_var("scaled_out")],
        )[0]
        bias_reshaped = graph.layer(
            name=prefix + "bias_unsqueeze",
            op="Unsqueeze",
            inputs=[bias, unsqueeze_axes_const],
            outputs=[_var("bias_reshaped", shape=channel_shape)],
        )[0]
        graph.layer(
            name=prefix + "biased",
            op="Add",
            inputs=[scaled, bias_reshaped],
            outputs=[y],
        )
    else:
        graph.layer(
            name=prefix + "scaled",
            op="Mul",
            inputs=[norm, scale_reshaped],
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

@dataclass
class DecomposeInstanceNorm(OnnxGraphEdit):
    """
    Replace onnx.InstanceNormalization with its primitive op decomposition.

    Decomposes ``y = scale * (x - mean) / sqrt(var + eps) + bias`` where mean
    and var are computed over the spatial axes (``2..rank-1``) per ``(N, C)``
    slice. Scale and bias (shape ``(C,)``) are unsqueezed to
    ``(1, C, 1, ..., 1)`` before broadcasting.

    Args:
        N (gs.Variable | None): Optional shared graph input providing the runtime
            spatial element count. When supplied, mean/var are computed as
            ``ReduceSum(x) / N`` instead of ``ReduceMean(x)`` — useful for targets
            that don't lower ``ReduceMean``'s implicit divisor. The caller is
            responsible for appending the variable to ``graph.inputs``.
        dynamic (bool | None): Selects the reduction path. ``True`` -> ReduceMean,
            ``False`` -> ReduceSum + Div by ``N``. When ``None`` (default), inferred
            from ``N`` (``dynamic = (N is None)``).

    Raises:
        ValueError: If ``dynamic=False`` but no ``N`` is supplied.
        ValueError: If the input lacks shape info or has rank < 3.

    Notes:
        - Reads ``epsilon`` from node attributes (default 1e-5)
        - Terminal Add (or scale Mul when no bias) reuses the original output tensor,
          so graph outputs and downstream consumers don't need rewiring.
    """

    N: gs.Variable | None = None
    dynamic: bool | None = None

    def __post_init__(self):
        if self.dynamic is None:
            self.dynamic = self.N is None
        if not self.dynamic and self.N is None:
            raise ValueError(
                "DecomposeInstanceNorm: dynamic=False requires a graph input `N`"
            )
        return super().__post_init__()

    def match(self, node: gs.Node) -> bool:
        return node.op == "InstanceNormalization"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "InstanceNormalization")
        epsilon = node.attrs.get("epsilon", 1e-5)

        _create_instance_norm_subgraph(
            self.graph, node,
            epsilon=epsilon,
            N=None if self.dynamic else self.N,
        )

        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "Decomposed InstanceNormalization '%s' (epsilon=%s, dynamic=%s)",
            node.name, epsilon, self.dynamic,
        )

@dataclass
class Convert1x1Conv1DToGeMM(OnnxGraphEdit):
    """
    Replace onnx.Conv1D with GeMM where appropriate

    Decomposes ``Conv1D where (kernel=1, stride=1, dilation=1, groups=1 with GeMM``

    Notes: Requires constant weights
        
    """
    def match(self, node: gs.Node) -> bool:
        if node.op == "Conv" \
            and node.attrs["strides"][0]==1 \
            and len(node.attrs["strides"]) == 1 \
            and node.inputs[1].shape[-1] == 1 \
            and len(node.inputs[1].shape) == 3:

            if hasattr(node.attrs, "dilations"):
                if node.attrs["dilations"][0]==1 and len(node.attrs["dilations"]) == 1:
                    return False
            if hasattr(node.attrs, "group"):
                if node.attrs["group"][0]==1:
                    return False
            return True
        return False

    def transform(self, node: gs.Node):
        input_node = node.inputs[0]
        output_node = node.outputs[0]
        weights = node.inputs[1]
        bias = node.inputs[2]
        if not isinstance(weights, gs.Constant) or not isinstance(bias, gs.Constant):
            raise ValueError(f"Cannot convert non constant weight: {node.name}\nPlease see: python3 -m src.torq.tools.fold_constants -i <in_model.onnx> -o <out_model.onnx>")

        prefix = f"_conv1d_converted_to_gemm_{output_node.name}_"
        weights.values = weights.values.squeeze(-1).T

        self.graph.layer(
            name=prefix + "gemm",
            op="Gemm",
            inputs=[input_node, weights, bias],
            outputs=[output_node],
        )


        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "Replaced Conv '%s' with GeMM",
            node.name,
        )

@dataclass
class ConvertConstantMatmulToGeMM(OnnxGraphEdit):
    """
    Replace Matmul where weights are constant with GeMM

    Replaces ``Matmul where weights are constant``

    Notes: Requires constant weights
        
    """
    def match(self, node: gs.Node) -> bool:
        if node.op == "MatMul" \
            and len(node.inputs) >= 2\
            and isinstance(node.inputs[1], gs.Constant):
            return True
        return False

    def transform(self, node: gs.Node):
        input_node = node.inputs[0]
        weights = node.inputs[1]
        output_node = node.outputs[0]

        bias = None
        if node.o().op == "Add" and isinstance(node.o().inputs[1], gs.Constant):
            bias = node.o().inputs[1]
            output_node = node.o().outputs[0]
        prefix = f"_matmul_converted_to_gemm_{output_node.name}_"
        new_inputs = [input_node, weights]
        if not (bias is None):
            new_inputs+=[bias]

        self.graph.layer(
            name=prefix + "gemm",
            op="Gemm",
            inputs=new_inputs,
            outputs=[output_node],
        )

        if not (bias is None):
            node.o().outputs.clear()
            node.o().inputs.clear()

        node.inputs.clear()
        node.outputs.clear()

        if bias is None:
            self._logger.debug(
                "Replaced MatMul '%s' with GeMM",
                node.name,
            )
        else:
            self._logger.debug(
                "Replaced MatMul -> Add '%s' with GeMM",
                node.name,
            )

@dataclass
class ConvertReduceSumToConvertMean(OnnxGraphEdit):
    """
    Converts ReduceSum(x) into ReduceMean(x) * len(x)

    Notes:
        
    """
    def match(self, node: gs.Node) -> bool:
        return node.op == "ReduceSum"

    def transform(self, node: gs.Node):
        input_node = node.inputs[0]
        output_node = node.outputs[0]
        prefix = f"_reducesum_as_reducemean_{output_node.name}_"
        axis = node.inputs[1]
        attrs = node.attrs
        input_node_dtype = getattr(input_node, "dtype", None)
        axis_lens = [input_node.shape[int(axis.values[i])] for i in range(len(axis.values))]
        axis_len = np.cumprod(axis_lens)[-1]

        def _var(name: str) -> gs.Variable:
            return gs.Variable(name=prefix + name, dtype=input_node_dtype, shape=output_node.shape)

        axis_len_const = gs.Constant(
            name=prefix + "axis_len",
            values=np.array([axis_len], dtype=np.float32),
        )

        mean = self.graph.layer(
            name=prefix + "reducemean",
            op="ReduceMean",
            inputs=[input_node, axis],
            outputs=[_var("mean_out")],
            attrs=attrs,
        )[0]

        self.graph.layer(
            name=prefix + "mul",
            op="Mul",
            inputs=[mean, axis_len_const],
            outputs=[output_node]
        )

        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug(
            "Replaced ReduceSum(x) '%s' with ReduceMean(x) * len(x)",
            node.name,
        )

@dataclass
class ConvertReciprocalMulToDiv(OnnxGraphEdit):
    """
    Converts Reciprocal -> Mul INTO Div

    Notes:
        
    """
    def match(self, node: gs.Node) -> bool:
        #TODO: needs to be tightened for 
        return node.op == "Reciprocal" and node.o().op=="Mul"

    def transform(self, node: gs.Node):
        reciprocal_node = node
        mul_node = node.o()
        input_nodes = reciprocal_node.inputs + mul_node.inputs[1:]
        output_nodes = mul_node.outputs

        prefix = f"_reducesum_as_reducemean_{output_nodes[0].name}_"

        self.graph.layer(
            name=prefix + "div",
            op="Div",
            inputs=input_nodes,
            outputs=output_nodes
        )

        reciprocal_node.inputs.clear()
        reciprocal_node.outputs.clear()

        mul_node.inputs.clear()
        mul_node.outputs.clear()

        self._logger.debug(
            "Replaced Reciprocal -> Mul '%s' with Div",
            node.name,
        )


class NormalizationPatches:
    def decompose_layer_norm(self):
        self.apply_edit(DecomposeLayerNorm(self._graph, self._graph_name))
        return self

    def decompose_instance_norm(self, N: gs.Variable | None = None, dynamic: bool | None = None):
        self.apply_edit(DecomposeInstanceNorm(self._graph, self._graph_name, N=N, dynamic=dynamic))
        return self
    
    def convert_reduce_sum(self):
        self.apply_edit(ConvertReduceSumToConvertMean(self._graph, self._graph_name))

    def convert_reciprocal_mul(self):
        self.apply_edit(ConvertReciprocalMulToDiv(self._graph, self._graph_name))
    #def tile_reduce_operators(self):

    
class MiscTsukiPatches:
    def convert_1x1_conv1d_to_gemm(self):
        self.apply_edit(Convert1x1Conv1DToGeMM(self._graph, self._graph_name))

    def convert_weight_mamtul_to_gemm(self):
        self.apply_edit(ConvertConstantMatmulToGeMM(self._graph, self._graph_name))