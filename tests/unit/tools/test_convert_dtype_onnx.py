# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.tools.convert_dtype.onnx import FP32Converter, Int64Converter


def _tensor_type(value_info) -> int:
    return value_info.type.tensor_type.elem_type


def _attr_i(node, name: str) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    raise AssertionError(f"Missing attribute {name!r} on node {node.name!r}")


def _make_model(nodes, name: str, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 22)])


def _node_by_name(model, name: str):
    return next(node for node in model.graph.node if node.name == name)


def _initializer_by_name(model, name: str):
    return next(init for init in model.graph.initializer if init.name == name)


def _value_info_by_name(model) -> dict:
    return {
        value_info.name: value_info
        for value_info in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
    }


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
    return _make_model([einsum], "einsum_graph", inputs, [output])


def test_bf16_conversion_updates_constant_of_shape_default_value_dtype():
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    output = helper.make_tensor_value_info("filled", TensorProto.FLOAT, [2, 3])
    constant_of_shape = helper.make_node(
        "ConstantOfShape",
        ["shape"],
        ["filled"],
        name="fill",
    )
    model = _make_model([constant_of_shape], "fill_graph", [shape], [output])

    converted = FP32Converter("bf16", convert_io=True).convert_model(model)

    converted_fill = next(node for node in converted.graph.node if node.op_type == "ConstantOfShape")
    value = next(attr for attr in converted_fill.attribute if attr.name == "value").t

    assert value.data_type == TensorProto.BFLOAT16
    assert _tensor_type(converted.graph.output[0]) == TensorProto.BFLOAT16


@pytest.mark.parametrize("op_type", ["RandomUniformLike", "RandomNormalLike"])
def test_bf16_conversion_updates_random_like_dtype_attrs(op_type):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        op_type,
        ["inp"],
        ["out"],
        name="rand",
        dtype=TensorProto.FLOAT,
    )
    model = _make_model([node], "rand_graph", [inp], [output])

    converted = FP32Converter("bf16", convert_io=True).convert_model(model)

    converted_rand = next(node for node in converted.graph.node if node.op_type == op_type)
    assert _attr_i(converted_rand, "dtype") == TensorProto.BFLOAT16
    assert _tensor_type(converted.graph.output[0]) == TensorProto.BFLOAT16


def test_bf16_conversion_casts_all_variadic_einsum_inputs_to_fp32():
    converted = FP32Converter("bf16", convert_io=True).convert_model(_make_einsum_model())

    einsum = next(node for node in converted.graph.node if node.op_type == "Einsum")
    cast_by_output = {
        output: node
        for node in converted.graph.node
        if node.op_type == "Cast"
        for output in node.output
    }

    assert len(einsum.input) == 3
    for input_name in einsum.input:
        cast = cast_by_output[input_name]
        assert _attr_i(cast, "to") == TensorProto.FLOAT

    output_casts = [
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.input == list(einsum.output)
    ]
    assert len(output_casts) == 1
    assert _attr_i(output_casts[0], "to") == TensorProto.BFLOAT16
    assert [_tensor_type(inp) for inp in converted.graph.input] == [TensorProto.BFLOAT16] * 3
    assert _tensor_type(converted.graph.output[0]) == TensorProto.BFLOAT16


def test_fp16_conversion_does_not_cast_einsum_inputs_back_to_fp32():
    converted = FP32Converter("fp16", convert_io=True).convert_model(_make_einsum_model())

    einsum = next(node for node in converted.graph.node if node.op_type == "Einsum")

    assert list(einsum.input) == ["x", "y", "z"]
    assert [_tensor_type(inp) for inp in converted.graph.input] == [TensorProto.FLOAT16] * 3
    assert _tensor_type(converted.graph.output[0]) == TensorProto.FLOAT16


def test_bf16_conversion_truncates_initializer_values_when_requested():
    values = np.array([1.0039064, -1.0039064], dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2], values)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
    model = _make_model([add], "initializer_graph", [x], [output], [weight])

    converted = FP32Converter(
        "bf16",
        convert_io=True,
        bf16_rounding="truncate",
    ).convert_model(model)

    converted_weight = converted.graph.initializer[0]
    expected_bits = (values.view(np.uint32) >> 16).astype(np.uint16)

    assert converted_weight.data_type == TensorProto.BFLOAT16
    np.testing.assert_array_equal(
        numpy_helper.to_array(converted_weight).view(np.uint16),
        expected_bits,
    )


def test_bf16_conversion_keeps_dynamic_quantize_linear_enforced_fp32_edges():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    quantized = helper.make_tensor_value_info("quantized", TensorProto.UINT8, [2])
    scale_out = helper.make_tensor_value_info("scale_out", TensorProto.FLOAT, [])
    zero = helper.make_tensor_value_info("zero", TensorProto.UINT8, [])
    dql = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["quantized", "scale", "zero"],
        name="dql",
    )
    use_scale = helper.make_node("Identity", ["scale"], ["scale_out"], name="use_scale")
    model = _make_model([dql, use_scale], "dql_graph", [x], [quantized, scale_out, zero])

    converted = FP32Converter("bf16", convert_io=True).convert_model(model)

    dql = _node_by_name(converted, "dql")
    dql_input_cast = _node_by_name(converted, "dql_inp_cast_f32")
    scale_cast = _node_by_name(converted, "dql_scale_cast_bf16")
    use_scale = _node_by_name(converted, "use_scale")
    value_info = _value_info_by_name(converted)

    assert dql.input == [dql_input_cast.output[0]]
    assert _attr_i(dql_input_cast, "to") == TensorProto.FLOAT
    assert _tensor_type(value_info[dql.output[1]]) == TensorProto.FLOAT
    assert scale_cast.input == [dql.output[1]]
    assert _attr_i(scale_cast, "to") == TensorProto.BFLOAT16
    assert use_scale.input == [scale_cast.output[0]]
    assert [_tensor_type(out) for out in converted.graph.output] == [
        TensorProto.UINT8,
        TensorProto.BFLOAT16,
        TensorProto.UINT8,
    ]


def test_bf16_conversion_keeps_resize_roi_and_scales_fp32_when_data_is_graph_input():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 4, 4])
    roi = helper.make_tensor("roi", TensorProto.FLOAT, [0], [])
    scales = helper.make_tensor(
        "scales",
        TensorProto.FLOAT,
        [4],
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    )
    resize = helper.make_node(
        "Resize",
        ["x", "roi", "scales"],
        ["out"],
        name="resize",
        mode="nearest",
    )
    model = _make_model([resize], "resize_graph", [x], [output], [roi, scales])

    converted = FP32Converter("bf16", convert_io=True).convert_model(model)

    resize = _node_by_name(converted, "resize")

    assert resize.input == ["x", "roi", "scales"]
    assert _initializer_by_name(converted, "roi").data_type == TensorProto.FLOAT
    assert _initializer_by_name(converted, "scales").data_type == TensorProto.FLOAT
    assert _tensor_type(converted.graph.input[0]) == TensorProto.BFLOAT16
    assert _tensor_type(converted.graph.output[0]) == TensorProto.BFLOAT16


def test_enforced_reduce_axes_cast_from_exported_constant():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    output = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [2, 1])
    reduce = helper.make_node("ReduceL2", ["data", "axes"], ["reduced"], name="reduce")
    model = _make_model([reduce], "reduce_graph", [data], [output], [axes])

    converted = Int64Converter("int32", convert_io=True, enforce_io_casts=True).convert_model(model)

    reduce = _node_by_name(converted, "reduce")
    axes_cast = next(
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.output == [reduce.input[1]]
    )

    assert _attr_i(axes_cast, "to") == TensorProto.INT64
    assert converted.graph.initializer[0].data_type == TensorProto.INT32


def test_enforced_reshape_shape_input_casts_after_concat_shape_builder():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [6])
    shape_source = helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [2, 3])
    unsqueeze_source = helper.make_tensor_value_info("unsqueeze_source", TensorProto.FLOAT, [1, 1, 1])
    dim = helper.make_tensor("dim", TensorProto.INT64, [1], [3])
    output = helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])
    unsqueezed = helper.make_tensor_value_info("unsqueezed", TensorProto.FLOAT, [1, 1, 1, 1])
    unsqueeze = helper.make_node(
        "Unsqueeze",
        ["unsqueeze_source", "dim"],
        ["unsqueezed"],
        name="unsqueeze",
    )
    shape = helper.make_node(
        "Shape",
        ["shape_source"],
        ["leading_dim"],
        name="shape",
        start=0,
        end=1,
    )
    concat = helper.make_node("Concat", ["leading_dim", "dim"], ["target_shape"], name="concat", axis=0)
    reshape = helper.make_node("Reshape", ["data", "target_shape"], ["reshaped"], name="reshape")
    model = _make_model(
        [unsqueeze, shape, concat, reshape],
        "concat_shape_graph",
        [data, shape_source, unsqueeze_source],
        [output, unsqueezed],
        [dim],
    )

    converted = Int64Converter("int32", convert_io=True, enforce_io_casts=True).convert_model(model)

    shape = _node_by_name(converted, "shape")
    concat = _node_by_name(converted, "concat")
    reshape = _node_by_name(converted, "reshape")
    shape_cast = next(
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.input == list(shape.output)
    )
    reshape_cast = next(
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.input == list(concat.output)
    )

    assert _attr_i(shape_cast, "to") == TensorProto.INT32
    assert concat.input[0] == shape_cast.output[0]
    assert _attr_i(reshape_cast, "to") == TensorProto.INT64
    assert reshape.input[1] == reshape_cast.output[0]


def test_enforced_reshape_shape_input_preserves_shape_producer_output():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [6])
    shape_source = helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])
    shape = helper.make_node("Shape", ["shape_source"], ["shape"], name="shape")
    reshape = helper.make_node("Reshape", ["data", "shape"], ["reshaped"], name="reshape")
    model = _make_model(
        [shape, reshape],
        "reshape_graph",
        [data, shape_source],
        [output],
    )

    converted = Int64Converter("int32", convert_io=True).convert_model(model)

    shape = _node_by_name(converted, "shape")
    reshape = _node_by_name(converted, "reshape")
    value_info = _value_info_by_name(converted)

    assert reshape.input[1] == shape.output[0]
    assert _tensor_type(value_info[shape.output[0]]) == TensorProto.INT64


def test_enforced_topk_indices_output_stays_int64_and_casts_for_consumers():
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
    model = _make_model(
        [topk, identity],
        "topk_graph",
        [x],
        [values, indices_out],
        [k],
    )

    converted = Int64Converter("int32", convert_io=True, enforce_io_casts=True).convert_model(model)

    topk = next(node for node in converted.graph.node if node.op_type == "TopK")
    identity = _node_by_name(converted, "use_indices")
    index_cast = next(
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.input == [topk.output[1]]
    )
    k_cast = next(
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.output == [topk.input[1]]
    )

    assert _attr_i(k_cast, "to") == TensorProto.INT64
    assert _attr_i(index_cast, "to") == TensorProto.INT32
    assert identity.input[0] == index_cast.output[0]
    assert converted.graph.initializer[0].data_type == TensorProto.INT32
    assert _tensor_type(converted.graph.output[1]) == TensorProto.INT32


def test_int64_conversion_casts_all_slice_index_inputs_back_to_int64():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [2])
    initializers = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [1]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [3]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [1]),
    ]
    slice_node = helper.make_node(
        "Slice",
        ["data", "starts", "ends", "axes", "steps"],
        ["sliced"],
        name="slice",
    )
    model = _make_model([slice_node], "slice_graph", [data], [output], initializers)

    converted = Int64Converter("int32", convert_io=True, enforce_io_casts=True).convert_model(model)

    slice_node = _node_by_name(converted, "slice")
    cast_by_output = {
        node.output[0]: node
        for node in converted.graph.node
        if node.op_type == "Cast"
    }

    for input_name in slice_node.input[1:]:
        assert _attr_i(cast_by_output[input_name], "to") == TensorProto.INT64
    assert {init.data_type for init in converted.graph.initializer} == {TensorProto.INT32}


def test_int64_conversion_updates_uint64_tensors_to_uint32():
    x = helper.make_tensor_value_info("x", TensorProto.UINT64, [2])
    output = helper.make_tensor_value_info("out", TensorProto.UINT64, [2])
    identity = helper.make_node("Identity", ["x"], ["out"], name="identity")
    model = _make_model([identity], "uint_graph", [x], [output])

    converted = Int64Converter("int32", convert_io=True).convert_model(model)

    assert _tensor_type(converted.graph.input[0]) == TensorProto.UINT32
    assert _tensor_type(converted.graph.output[0]) == TensorProto.UINT32


def test_int64_conversion_default_preserves_slice_index_initializers_without_casts():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [2])
    initializers = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [1]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [3]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [1]),
    ]
    slice_node = helper.make_node(
        "Slice",
        ["data", "starts", "ends", "axes", "steps"],
        ["sliced"],
        name="slice",
    )
    model = _make_model([slice_node], "slice_graph", [data], [output], initializers)

    converted = Int64Converter("int32", convert_io=True).convert_model(model)

    slice_node = _node_by_name(converted, "slice")

    assert not any(node.op_type == "Cast" for node in converted.graph.node)
    assert list(slice_node.input) == ["data", "starts", "ends", "axes", "steps"]
    assert {init.data_type for init in converted.graph.initializer} == {TensorProto.INT64}


def test_int64_conversion_default_keeps_shape_output_int64_without_casts():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [6])
    shape_source = helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])
    shape = helper.make_node("Shape", ["shape_source"], ["shape"], name="shape")
    reshape = helper.make_node("Reshape", ["data", "shape"], ["reshaped"], name="reshape")
    model = _make_model(
        [shape, reshape],
        "reshape_graph",
        [data, shape_source],
        [output],
    )

    converted = Int64Converter("int32", convert_io=True).convert_model(model)

    shape = _node_by_name(converted, "shape")
    reshape = _node_by_name(converted, "reshape")
    value_info = _value_info_by_name(converted)

    assert not any(node.op_type == "Cast" for node in converted.graph.node)
    assert reshape.input[1] == shape.output[0]
    assert _tensor_type(value_info[shape.output[0]]) == TensorProto.INT64
