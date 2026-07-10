# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from dataclasses import dataclass

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ..onnx import OnnxGraphEdit, rewire_consumers


@dataclass
class DequantizeProjectionsMatMul(OnnxGraphEdit):
    """
    Manually dequantize projection scores MatMul producer to prevent MLIR warnings.

    Args:
        hidden_size (int): Model hidden KV dims size
        vocab_size (int): Model vocabulary size
        export_dtype (onnx.TensorProto.DataType): ONNX export data type for tensors

    Raises:
        ValueError: If MatMul producer is not a `DequantizeLinear` op
        ValueError: If weights are not correctly formatted
        ValueError: If dequantization params are not correctly formatted
    """

    hidden_size: int
    vocab_size: int
    export_dtype: onnx.TensorProto.DataType

    def __post_init__(self):
        if self.export_dtype not in onnx.TensorProto.DataType.values():
            raise RuntimeError(f"A valid export dtype is required for this edit, received {type(self.export_dtype)}")
        return super().__post_init__()

    def match(self, node: gs.Node):
        if node.op == "MatMul" and node.outputs[0].name == "logits":
            return isinstance(node.i(1), gs.Node) and node.i(1).op == "DequantizeLinear"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        dequant_node: gs.Node = node.i(1)
        try:
            transpose_node: gs.Node = dequant_node.i()
        except IndexError:
            self._logger.debug("Dequantize node does not have Transpose input, looking in inputs for const weight")
            quant_weights: gs.Constant = dequant_node.inputs[0]
        else:
            quant_weights: gs.Constant = transpose_node.inputs[0]
        if not isinstance(quant_weights, gs.Constant):
            self._logger.warning("Dequantization weights not found, skipping")
            return

        self._check_node_op(dequant_node, "DequantizeLinear")

        W_q: np.ndarray = quant_weights.values
        if W_q.shape == (self.vocab_size, self.hidden_size):
            W_q = W_q.T
        if W_q.shape != (self.hidden_size, self.vocab_size):
            raise ValueError(f"Expected weight shape of {(self.vocab_size, self.hidden_size)} or {(self.hidden_size, self.vocab_size)}, got {W_q.shape}")
        if W_q.dtype != np.uint8:
            raise ValueError(f"Expected uint8 weights, got {W_q.dtype}")

        if len(dequant_node.inputs) < 3:
            raise ValueError(f"Expected 3 inputs (x, scale, zp) for DequantizeLinear node, got {len(dequant_node.inputs)}")
        scale_inp, zp_inp = dequant_node.inputs[1], dequant_node.inputs[2]
        if not isinstance(scale_inp, gs.Constant):
            raise ValueError(f"Expected constant scale, got {type(scale_inp)}")
        if not isinstance(zp_inp, gs.Constant):
            raise ValueError(f"Expected constant zp, got {type(scale_inp)}")
        scale = scale_inp.values.item()
        zp = zp_inp.values.item()
        node.inputs[1] = gs.Constant(
            node.inputs[1].name + "_float_folded",
            (W_q.astype(np.int32) - np.int32(zp)).astype(np.float32) * np.float32(scale),
            export_dtype=self.export_dtype
        )

        dequant_node.outputs.clear()

        self._logger.debug("Dequantized projection scores producer")

@dataclass
class DecomposeLayerNormalization(OnnxGraphEdit):
    """
    Decompose ONNX LayerNormalization into basic arithmetic operations using Pow and Div.
    Closely matches standard PyTorch export/decomposition.

    Requires the input's shape to be fully static (run shape inference / fix
    dynamic I/O dims before applying this edit).

    Args:
        enabled (bool): Gate to disable this edit without removing call sites.
            Intended as a temporary workaround; flip to False once the target
            compiler natively supports LayerNormalization.
    """
    enabled: bool = True

    def match(self, node: gs.Node) -> bool:
        return self.enabled and node.op == "LayerNormalization"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "LayerNormalization")

        X = node.inputs[0]
        scale = node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None

        axis = node.attrs.get("axis", -1)
        epsilon = node.attrs.get("epsilon", 1e-05)

        # Normalize axis and calculate reduction axes
        rank = len(X.shape) if X.shape is not None else 0
        if rank > 0:
            axis = axis % rank
            reduce_axes = list(range(axis, rank))
        else:
            reduce_axes = [axis]

        Y = node.outputs[0]

        target_shape = list(X.shape)
        target_shape_const = gs.Constant(
            name=node.name + "_target_shape",
            values=np.array(target_shape, dtype=np.int64)
        )

        def expand_tensor(tensor, name_suffix):
            if tensor.shape is not None and list(tensor.shape) == list(target_shape):
                return tensor
            return self.graph.layer(
                name=node.name + "_" + name_suffix + "_expand",
                op="Expand",
                inputs=[tensor, target_shape_const],
                outputs=[gs.Variable(name=node.name + "_" + name_suffix + "_expanded", dtype=tensor.dtype, shape=target_shape)]
            )[0]

        # Create axes constant for ReduceMean
        axes_const = gs.Constant(
            name=node.name + "_axes",
            values=np.array(reduce_axes, dtype=np.int64)
        )

        # 1. Compute mean: mean = ReduceMean(X, axes)
        mean = self.graph.layer(
            name=node.name + "_mean",
            op="ReduceMean",
            inputs=[X, axes_const],
            outputs=[gs.Variable(name=node.name + "_mean_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]

        # 2. Subtract mean: X_diff = Sub(X, mean_expanded)
        mean_expanded = expand_tensor(mean, "mean")
        x_diff = self.graph.layer(
            name=node.name + "_sub_mean",
            op="Sub",
            inputs=[X, mean_expanded],
            outputs=[gs.Variable(name=node.name + "_diff", dtype=X.dtype)]
        )[0]

        # 3. Compute variance: var = ReduceMean((X - mean)^2, axes) using Pow
        # Ensure we use a valid NumPy dtype matching X.dtype
        if isinstance(X.dtype, int):
            try:
                import onnx.helper
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(X.dtype)
            except Exception:
                np_dtype = np.float32
        else:
            np_dtype = X.dtype if X.dtype is not None else np.float32

        pow_exp = gs.Constant(
            name=node.name + "_pow_exp",
            values=np.array(2.0, dtype=np_dtype)
        )
        x_diff_sq = self.graph.layer(
            name=node.name + "_diff_sq",
            op="Pow",
            inputs=[x_diff, pow_exp],
            outputs=[gs.Variable(name=node.name + "_diff_sq_val", dtype=X.dtype)]
        )[0]

        var = self.graph.layer(
            name=node.name + "_var",
            op="ReduceMean",
            inputs=[x_diff_sq, axes_const],
            outputs=[gs.Variable(name=node.name + "_var_val", dtype=X.dtype)],
            attrs={"keepdims": 1}
        )[0]

        # 4. Standard deviation: stddev = Sqrt(var + eps)
        eps_const = gs.Constant(
            name=node.name + "_eps",
            values=np.array(epsilon, dtype=np_dtype),
        )

        var_eps = self.graph.layer(
            name=node.name + "_var_eps",
            op="Add",
            inputs=[var, eps_const],
            outputs=[gs.Variable(name=node.name + "_var_eps_val", dtype=X.dtype)]
        )[0]

        stddev = self.graph.layer(
            name=node.name + "_stddev",
            op="Sqrt",
            inputs=[var_eps],
            outputs=[gs.Variable(name=node.name + "_stddev_val", dtype=X.dtype)]
        )[0]

        # 5. Normalize: X_norm = Div(X_diff, stddev_expanded)
        stddev_expanded = expand_tensor(stddev, "stddev")

        x_norm = self.graph.layer(
            name=node.name + "_div_stddev",
            op="Div",
            inputs=[x_diff, stddev_expanded],
            outputs=[gs.Variable(name=node.name + "_norm", dtype=X.dtype)]
        )[0]

        # 6. Apply scale and optional bias
        scale_expanded = expand_tensor(scale, "scale")
        if bias is not None:
            bias_expanded = expand_tensor(bias, "bias")
            x_scaled = self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[gs.Variable(name=node.name + "_scaled", dtype=X.dtype)]
            )[0]
            self.graph.layer(
                name=node.name + "_add_bias",
                op="Add",
                inputs=[x_scaled, bias_expanded],
                outputs=[Y]
            )
        else:
            self.graph.layer(
                name=node.name + "_mul_scale",
                op="Mul",
                inputs=[x_norm, scale_expanded],
                outputs=[Y]
            )

        # Disconnect node
        node.inputs.clear()
        node.outputs.clear()

        self._logger.debug("Decomposed LayerNormalization node '%s'", node.name)

@dataclass
class RemoveIsNaN(OnnxGraphEdit):
    """
    Remove unsupported IsNaN operations.

    Raises:
        ValueError: If IsNaN is not consumed by a `Where` op
    """

    def match(self, node: gs.Node) -> bool:
        return node.op == "IsNaN"

    def transform(self, node: gs.Node):
        self._check_node_op(node, "IsNaN")
        producer: gs.Tensor = node.inputs[0]
        where_node: gs.Node = node.o()
        if where_node.op != "Where":
            raise ValueError(
                f"Expected Where node consumer, got {where_node.op} for IsNaN replacement"
            )
        where_out: gs.Variable = where_node.outputs[0]
        consumers: list[gs.Node] = list(where_out.outputs)
        rewire_consumers(consumers, where_out, producer)

        # disconnect IsNaN -> Where chain
        node.inputs.clear()
        where_node.inputs.clear()
        where_node.outputs.clear()

        self._logger.debug("Removed unsupported IsNaN op '%s'", node.name)

@dataclass
class RemoveRedundantCasts(OnnxGraphEdit):
    """
    Remove redundant Cast ops where input dtype == output dtype
    """

    @staticmethod
    def _to_onnx_dtype(dtype: np.dtype | int | None) -> int | None:
        if dtype is None:
            return None
        if isinstance(dtype, int):
            return dtype
        try:
            return onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        except Exception:
            return None

    def match(self, node: gs.Node) -> bool:
        if node.op != "Cast" or not node.inputs or not node.outputs:
            return False
        inp_dtype = self._to_onnx_dtype(getattr(node.inputs[0], "dtype", None))
        if inp_dtype is None:
            return False
        cast_to = node.attrs.get("to", None)
        if isinstance(cast_to, int) and inp_dtype == cast_to:
            return True
        out_dtype = self._to_onnx_dtype(getattr(node.outputs[0], "dtype", None))
        return out_dtype is not None and inp_dtype == out_dtype

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Cast")
        inp = node.inputs[0]
        out = node.outputs[0]
        consumers: list[gs.Node] = list(out.outputs)
        rewire_consumers(consumers, out, inp)
        for i, graph_out in enumerate(self.graph.outputs):
            if graph_out is out:
                self.graph.outputs[i] = inp
        node.inputs.clear()
        node.outputs.clear()
        self._logger.debug("Removed redundant Cast node '%s'", node.name)

@dataclass
class FoldScalarMatMul(OnnxGraphEdit):
    """
    Fold `MatMul A @ B`, where B is a batched scalar, into Mul.

    Raises:
        ValueError: If MatMul operand shapes are incompatible
    """

    def match(self, node: gs.Node) -> bool:
        if node.op != "MatMul":
            return False

        a, b = node.inputs
        a_shape = getattr(a, "shape", None)
        b_shape = getattr(b, "shape", None)
        if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
            return a_shape[-1] == 1 and b_shape[-2] == 1 and b_shape[-1] == 1
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "MatMul")
        a, b = node.inputs
        a_shape = getattr(a, "shape", None)
        b_shape = getattr(b, "shape", None)
        y = node.outputs[0]

        if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("Invalid MatMul operand shapes for scalar scale matmul replacement")
        if not (a_shape[-1] == 1 and b_shape[-2] == 1 and b_shape[-1] == 1):
            raise ValueError(f"Expected scalar-compatible MatMul shapes, got A={a_shape}, B={b_shape}")
        
        self.graph.layer(
            name=node.name + "_mul_fold",
            op="Mul",
            inputs=[a, b],
            outputs=[y]
        )
        node.outputs.clear()

        self._logger.debug("Folded scalar MatMul node '%s' into Mul", node.name)

@dataclass
class ReplaceConstantDivWithMul(OnnxGraphEdit):
    """
    Replaces x/C with x * C' where C' = 1/C is a newly computed constant.

    Args:
        export_dtype (onnx.TensorProto.DataType): ONNX export data type for tensors

    Raises:
        TypeError: If divisor is not a constant tensor
    """
    export_dtype: onnx.TensorProto.DataType

    def match(self, node: gs.Node) -> bool:
        if node.op == "Div" and len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
            return True
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Div")
        if not len(node.inputs) > 1 or not isinstance(node.inputs[1], gs.Constant):
            raise TypeError("Expected second operand of Div to be a `gs.Constant`")

        # x/C -> x * C' where C' = 1/C
        divisor: gs.Constant = node.inputs[1]
        if not (reciprocal := self.graph.tensors().get(divisor.name + "_reciprocal")):
            reciprocal = gs.Constant(
                name=divisor.name + "_reciprocal",
                values=np.array(np.float32(1.0) / divisor.values.astype(np.float32)),
                export_dtype=self.export_dtype,
            )
        node.op = "Mul"
        node.inputs[1] = reciprocal

        self._logger.debug("Replaced Div @ '%s' by constant '%s' with Mul by reciprocal", node.name, divisor.name)

@dataclass
class ReplaceInt64FloatCast(OnnxGraphEdit):
    """
    Replace int64 -> float casts with a look-up table: `output<fp32|fp16|bf16> = LUT[input<int64>]`
    """

    max_int: int

    @staticmethod
    def _is_integer_type(dtype: np.dtype | int) -> bool:
        if isinstance(dtype, np.dtype):
            return np.issubdtype(dtype, np.integer)
        elif isinstance(dtype, int):
            return dtype in {
                onnx.TensorProto.INT8, onnx.TensorProto.INT16, onnx.TensorProto.INT32, onnx.TensorProto.INT64,
                onnx.TensorProto.UINT8, onnx.TensorProto.UINT16, onnx.TensorProto.UINT32, onnx.TensorProto.UINT64,
            }
        else:
            return False

    def match(self, node: gs.Node) -> bool:
        if node.op == "Cast":
            if self._is_integer_type(node.inputs[0].dtype) and node.attrs["to"] in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.BFLOAT16,
            ):
                int_inp: gs.Variable | gs.Constant = node.inputs[0]
                cast_dtype = node.attrs["to"]
                if cast_dtype not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as target is not float",
                        node.name
                    )
                    return False
                if not int_inp.shape or not all(isinstance(d, (int, np.integer)) for d in int_inp.shape):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as input is not static",
                        node.name
                    )
                    return False
                if not all(i == 1 for i in list(int_inp.shape)[:-1]):
                    self._logger.debug(
                        "Skipping int -> float Cast replacement for node '%s' as input has non-batch dims",
                        node.name
                    )
                    return False
                return True 
        return False

    def transform(self, node: gs.Node):
        self._check_node_op(node, "Cast")
        cast_dtype_str = onnx.helper.tensor_dtype_to_string(node.attrs["to"])
        self._logger.warning("Replacing int -> %s cast with lookup table, disable if env supports this casting", cast_dtype_str)

        int_inp: gs.Variable | gs.Constant = node.inputs[0]
        inp_dtype = int_inp.dtype
        if not self._is_integer_type(inp_dtype):
            raise ValueError(
                f"Cast input must be integer, found {int_inp.dtype} for int -> float cast replacement"
            )
        cast_dtype = node.attrs["to"]
        if cast_dtype not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
            raise ValueError(
                f"Cast output must be float, found {onnx.helper.tensor_dtype_to_string(cast_dtype)} for int -> float cast replacement"
            )

        if not int_inp.shape or not all(isinstance(d, (int, np.integer)) for d in int_inp.shape):
            raise ValueError(
                f"Cast input must have static shape, found {int_inp.shape} for int -> float cast replacement"
            )
        if not all(i == 1 for i in list(int_inp.shape)[:-1]):
            raise ValueError(
                f"Cast input must be batched, found non-batch dims in shape {int_inp.shape} for int -> float cast replacement"
            )
        tensors = self.graph.tensors()
        if not (shape_scalar := tensors.get(int_inp.name + "_shape_scalar")):
            shape_scalar = gs.Constant(int_inp.name + "_shape_scalar", np.array([], dtype=np.int64))
        if not (shape_batched := tensors.get(int_inp.name + "_shape_batched")):
            shape_batched = gs.Constant(int_inp.name + "_shape_batched", np.asarray(int_inp.shape, dtype=np.int64))
        float_lut_name = int_inp.name + f"_lut_{self.max_int}_{onnx.helper.tensor_dtype_to_string(cast_dtype)}"
        if not (lookup_table := tensors.get(float_lut_name)):
            lookup_table = gs.Constant(float_lut_name, np.arange(self.max_int, dtype=np.float32), export_dtype=cast_dtype)
        float_out: gs.Variable = node.outputs[0]
        consumers: list[gs.Node] = list(float_out.outputs)
        assert float_out.shape == int_inp.shape, f"Cast node '{node.name}': input shape {int_inp.shape} != output shape {float_out.shape}"

        int_inp_scalar: gs.Variable = self.graph.layer(
            name=int_inp.name + "_to_scalar",
            op="Reshape",
            inputs=[int_inp, shape_scalar],
            outputs=[gs.Variable(name=int_inp.name + "_scalar", dtype=int_inp.dtype, shape=[])]
        )[0]
        lookup_output: gs.Variable = self.graph.layer(
            name=int_inp.name + "_lookup",
            op="Gather",
            inputs=[lookup_table, int_inp_scalar],
            outputs=[gs.Variable(name=int_inp.name + "_float_value", dtype=cast_dtype, shape=[])]
        )[0]
        lookup_output_batched: gs.Variable = self.graph.layer(
            name=int_inp.name + "_lookup_batch",
            op="Reshape",
            inputs=[lookup_output, shape_batched],
            outputs=[gs.Variable(name=lookup_output.name + "_batched", dtype=cast_dtype, shape=int_inp.shape)]
        )[0]
        rewire_consumers(consumers, float_out, lookup_output_batched)
        node.outputs.clear()

        self._logger.debug("Replaced int -> %s Cast node '%s' with look-up table", cast_dtype_str, node.name)
