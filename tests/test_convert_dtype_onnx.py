# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import unittest

from onnx import TensorProto, helper

from torq.tools.convert_dtype.onnx import FP32Converter, Int64Converter


def _tensor_type(value_info) -> int:
    return value_info.type.tensor_type.elem_type


def _attr_i(node, name: str) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    raise AssertionError(f"Missing attribute {name!r} on node {node.name!r}")


def _make_einsum_model():
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4]),
        helper.make_tensor_value_info("z", TensorProto.FLOAT, [4, 5]),
    ]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 5])
    einsum = helper.make_node(
        "Einsum",
        ["x", "y", "z"],
        ["out"],
        name="einsum",
        equation="ab,bc,cd->ad",
    )
    graph = helper.make_graph([einsum], "einsum_graph", inputs, [output])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 22)],
    )


class FP32ConverterEnforcedIoTests(unittest.TestCase):
    def test_bf16_conversion_updates_constant_of_shape_default_value_dtype(self):
        shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
        output = helper.make_tensor_value_info("filled", TensorProto.FLOAT, [2, 3])
        constant_of_shape = helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["filled"],
            name="fill",
        )
        graph = helper.make_graph([constant_of_shape], "fill_graph", [shape], [output])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 22)])

        converted = FP32Converter("bf16", convert_io=True).convert_model(model)

        converted_fill = next(node for node in converted.graph.node if node.op_type == "ConstantOfShape")
        value = next(attr for attr in converted_fill.attribute if attr.name == "value").t

        self.assertEqual(value.data_type, TensorProto.BFLOAT16)
        self.assertEqual(_tensor_type(converted.graph.output[0]), TensorProto.BFLOAT16)

    def test_bf16_conversion_updates_random_like_dtype_attrs(self):
        for op_type in ("RandomUniformLike", "RandomNormalLike"):
            with self.subTest(op_type=op_type):
                inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [2, 3])
                output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
                node = helper.make_node(
                    op_type,
                    ["inp"],
                    ["out"],
                    name="rand",
                    dtype=TensorProto.FLOAT,
                )
                graph = helper.make_graph([node], "rand_graph", [inp], [output])
                model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 22)])

                converted = FP32Converter("bf16", convert_io=True).convert_model(model)

                converted_rand = next(node for node in converted.graph.node if node.op_type == op_type)
                self.assertEqual(_attr_i(converted_rand, "dtype"), TensorProto.BFLOAT16)
                self.assertEqual(_tensor_type(converted.graph.output[0]), TensorProto.BFLOAT16)

    def test_bf16_conversion_casts_all_variadic_einsum_inputs_to_fp32(self):
        converted = FP32Converter("bf16", convert_io=True).convert_model(_make_einsum_model())

        einsum = next(node for node in converted.graph.node if node.op_type == "Einsum")
        cast_by_output = {
            output: node
            for node in converted.graph.node
            if node.op_type == "Cast"
            for output in node.output
        }

        self.assertEqual(len(einsum.input), 3)
        for input_name in einsum.input:
            cast = cast_by_output[input_name]
            self.assertEqual(_attr_i(cast, "to"), TensorProto.FLOAT)

        output_casts = [
            node
            for node in converted.graph.node
            if node.op_type == "Cast" and node.input == list(einsum.output)
        ]
        self.assertEqual(len(output_casts), 1)
        self.assertEqual(_attr_i(output_casts[0], "to"), TensorProto.BFLOAT16)
        self.assertEqual([_tensor_type(inp) for inp in converted.graph.input], [TensorProto.BFLOAT16] * 3)
        self.assertEqual(_tensor_type(converted.graph.output[0]), TensorProto.BFLOAT16)

    def test_fp16_conversion_does_not_cast_einsum_inputs_back_to_fp32(self):
        converted = FP32Converter("fp16", convert_io=True).convert_model(_make_einsum_model())

        einsum = next(node for node in converted.graph.node if node.op_type == "Einsum")

        self.assertEqual(list(einsum.input), ["x", "y", "z"])
        self.assertEqual([_tensor_type(inp) for inp in converted.graph.input], [TensorProto.FLOAT16] * 3)
        self.assertEqual(_tensor_type(converted.graph.output[0]), TensorProto.FLOAT16)


class Int64ConverterEnforcedIoTests(unittest.TestCase):
    def test_enforced_reshape_shape_input_preserves_shape_producer_output(self):
        data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [6])
        shape_source = helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [2, 3])
        output = helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])
        shape = helper.make_node("Shape", ["shape_source"], ["shape"], name="shape")
        reshape = helper.make_node("Reshape", ["data", "shape"], ["reshaped"], name="reshape")
        graph = helper.make_graph(
            [shape, reshape],
            "reshape_graph",
            [data, shape_source],
            [output],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 22)])

        converted = Int64Converter("int32", convert_io=True).convert_model(model)

        shape = next(node for node in converted.graph.node if node.name == "shape")
        reshape = next(node for node in converted.graph.node if node.name == "reshape")
        value_info = {info.name: info for info in converted.graph.value_info}

        self.assertEqual(reshape.input[1], shape.output[0])
        self.assertEqual(_tensor_type(value_info[shape.output[0]]), TensorProto.INT64)

    def test_enforced_topk_indices_output_stays_int64_and_casts_for_consumers(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
        k = helper.make_tensor("k", TensorProto.INT64, [1], [2])
        values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [2])
        indices_out = helper.make_tensor_value_info("indices_out", TensorProto.INT64, [2])
        topk = helper.make_node(
            "TopK",
            ["x", "k"],
            ["values", "indices"],
            name="topk",
            axis=0,
        )
        identity = helper.make_node("Identity", ["indices"], ["indices_out"], name="use_indices")
        graph = helper.make_graph(
            [topk, identity],
            "topk_graph",
            [x],
            [values, indices_out],
            [k],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 22)])

        converted = Int64Converter("int32", convert_io=True).convert_model(model)

        topk = next(node for node in converted.graph.node if node.op_type == "TopK")
        identity = next(node for node in converted.graph.node if node.name == "use_indices")
        index_cast = next(
            node
            for node in converted.graph.node
            if node.op_type == "Cast" and node.input == [topk.output[1]]
        )

        self.assertEqual(_attr_i(index_cast, "to"), TensorProto.INT32)
        self.assertEqual(identity.input[0], index_cast.output[0])
        self.assertEqual(converted.graph.initializer[0].data_type, TensorProto.INT64)
        self.assertEqual(_tensor_type(converted.graph.output[1]), TensorProto.INT32)


if __name__ == "__main__":
    unittest.main()
