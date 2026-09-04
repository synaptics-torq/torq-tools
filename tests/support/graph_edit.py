# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
from numpy.testing import assert_allclose

from torq.utils.ort import make_cpu_session


DEFAULT_OPSET = 17
ORT_MAX_IR_VERSION = 11


def graph(
    *,
    nodes: Sequence[gs.Node],
    inputs: Sequence[gs.Variable],
    outputs: Sequence[gs.Variable],
    name: str = "test",
    opset: int = DEFAULT_OPSET,
) -> gs.Graph:
    return gs.Graph(
        name=name,
        nodes=list(nodes),
        inputs=list(inputs),
        outputs=list(outputs),
        opset=opset,
    )


def quantized_lm_head_graph() -> gs.Graph:
    hidden = gs.Variable("hidden", dtype=np.float32, shape=[1, 1, 2])
    quantized = gs.Variable("hidden_quantized", dtype=np.uint8, shape=[1, 1, 2])
    activation_scale = gs.Variable("hidden_scale", dtype=np.float32, shape=[])
    activation_zero_point = gs.Variable("hidden_zero_point", dtype=np.uint8, shape=[])
    dql = gs.Node(
        "DynamicQuantizeLinear",
        "lm_head_quantize",
        inputs=[hidden],
        outputs=[quantized, activation_scale, activation_zero_point],
    )
    weight = gs.Constant(
        "weight_quantized",
        np.array([[2, -3, 4, 5, -6], [7, 8, -9, 10, 11]], dtype=np.int8),
    )
    weight_scale = gs.Constant(
        "weight_scale",
        np.array([0.25, 0.5, 0.125, 0.75, 0.375], dtype=np.float32),
    )
    weight_zero_point = gs.Constant(
        "weight_zero_point",
        np.array([1, -1, 2, 0, -2], dtype=np.int8),
    )
    quantized_logits = gs.Variable("logits_quantized", dtype=np.int32, shape=[1, 1, 5])
    matmul = gs.Node(
        "MatMulInteger",
        "lm_head_matmul",
        inputs=[quantized, weight, activation_zero_point, weight_zero_point],
        outputs=[quantized_logits],
    )
    cast_logits = gs.Variable("logits_cast", dtype=np.float32, shape=[1, 1, 5])
    cast = gs.Node(
        "Cast",
        "lm_head_cast",
        attrs={"to": onnx.TensorProto.FLOAT},
        inputs=[quantized_logits],
        outputs=[cast_logits],
    )
    output_scale = gs.Variable("output_scale", dtype=np.float32, shape=[5])
    scale_mul = gs.Node(
        "Mul",
        "lm_head_scales",
        inputs=[activation_scale, weight_scale],
        outputs=[output_scale],
    )
    logits = gs.Variable("logits", dtype=np.float32, shape=[1, 1, 5])
    dequant_mul = gs.Node(
        "Mul",
        "lm_head_dequant",
        inputs=[cast_logits, output_scale],
        outputs=[logits],
    )
    return graph(
        nodes=[dql, matmul, cast, scale_mul, dequant_mul],
        inputs=[hidden],
        outputs=[logits],
    )


def clone_graph(g: gs.Graph) -> gs.Graph:
    return gs.import_onnx(to_model(g))


def cap_model_for_ort(model: onnx.ModelProto) -> onnx.ModelProto:
    if model.ir_version > ORT_MAX_IR_VERSION:
        model.ir_version = ORT_MAX_IR_VERSION
    for opset in model.opset_import:
        if opset.domain == "" and opset.version > DEFAULT_OPSET:
            opset.version = DEFAULT_OPSET
    return model


def to_model(g: gs.Graph, *, infer_shapes: bool = False) -> onnx.ModelProto:
    exported = gs.export_onnx(
        g.copy().cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()
    )
    exported = cap_model_for_ort(exported)
    if infer_shapes:
        exported = onnx.shape_inference.infer_shapes(exported)
        exported = cap_model_for_ort(exported)
    onnx.checker.check_model(exported)
    return exported


def run_model(
    model_or_graph: onnx.ModelProto | gs.Graph,
    feeds: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    model = (
        to_model(model_or_graph, infer_shapes=True)
        if isinstance(model_or_graph, gs.Graph)
        else cap_model_for_ort(model_or_graph)
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = make_cpu_session(model.SerializeToString(), sess_options=sess_options)
    names = [out.name for out in sess.get_outputs()]
    return dict(zip(names, sess.run(names, dict(feeds))))


def assert_model_outputs_close(
    before: onnx.ModelProto | gs.Graph,
    after: onnx.ModelProto | gs.Graph,
    feeds: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    expected = list(run_model(before, feeds).values())
    actual = list(run_model(after, feeds).values())
    assert len(actual) == len(expected)
    for exp, act in zip(expected, actual):
        if exp.dtype.kind in {"b", "i", "u"}:
            np.testing.assert_array_equal(act, exp)
        else:
            assert_allclose(act, exp, rtol=rtol, atol=atol)


def node_ops(g: gs.Graph) -> list[str]:
    return [node.op for node in g.nodes if node.outputs]


def only_node(g: gs.Graph, op: str) -> gs.Node:
    matches = [node for node in g.nodes if node.op == op and node.outputs]
    assert len(matches) == 1
    return matches[0]


def const(name: str, values, *, dtype=None, export_dtype=None) -> gs.Constant:
    arr = np.asarray(values, dtype=dtype)
    return gs.Constant(name=name, values=arr, export_dtype=export_dtype)


def empty_optional() -> gs.Variable:
    return gs.Variable.empty()


def conv_bn_graph(w_values, scale_values, shift_values, bias_values=None, extra_consumer=False):
    """Conv -> Mul -> Add chain (exported eval-mode BatchNorm), the pattern
    FoldConvBatchNorm targets. Returns (graph, conv, mul, add)."""
    cin = w_values.shape[1]
    spatial = [4] * (w_values.ndim - 2)
    cout = w_values.shape[0]
    x = gs.Variable("x", dtype=np.float32, shape=[1, cin] + spatial)
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, cout] + spatial)
    mul_out = gs.Variable("mul_out", dtype=np.float32, shape=[1, cout] + spatial)
    add_out = gs.Variable("add_out", dtype=np.float32, shape=[1, cout] + spatial)
    conv_inputs = [x, gs.Constant("w", w_values)]
    if bias_values is not None:
        conv_inputs.append(gs.Constant("c", bias_values))
    conv = gs.Node("Conv", "conv", inputs=conv_inputs, outputs=[conv_out],
                   attrs={"kernel_shape": list(w_values.shape[2:]), "pads": [1] * (2 * len(spatial))})
    mul = gs.Node("Mul", "mul", inputs=[conv_out, gs.Constant("s", scale_values)], outputs=[mul_out])
    add = gs.Node("Add", "add", inputs=[mul_out, gs.Constant("b", shift_values)], outputs=[add_out])
    nodes = [conv, mul, add]
    outputs = [add_out]
    if extra_consumer:
        relu_out = gs.Variable("relu_out", dtype=np.float32, shape=[1, cout] + spatial)
        nodes.append(gs.Node("Relu", "relu", inputs=[conv_out], outputs=[relu_out]))
        outputs.append(relu_out)
    g = graph(nodes=nodes, inputs=[x], outputs=outputs)
    return g, conv, mul, add


def unit_slice_chain(v, i, axis, slice_shape):
    """Build Slice(v,[i:i+1],axis) -> Squeeze(axis) -> Unsqueeze(axis), one
    element of an unrolled stack/unbind. Returns (nodes, tip_tensor)."""
    sliced = gs.Variable(f"{v.name}_sl{i}", dtype=np.float32, shape=slice_shape)
    squeezed_shape = [d for j, d in enumerate(slice_shape) if j != axis]
    squeezed = gs.Variable(f"{v.name}_sq{i}", dtype=np.float32, shape=squeezed_shape)
    unsqueezed = gs.Variable(f"{v.name}_un{i}", dtype=np.float32, shape=slice_shape)
    nodes = [
        gs.Node("Slice", f"{v.name}_slice{i}", inputs=[
            v,
            gs.Constant(f"{v.name}_s{i}", np.array([i], dtype=np.int64)),
            gs.Constant(f"{v.name}_e{i}", np.array([i + 1], dtype=np.int64)),
            gs.Constant(f"{v.name}_a{i}", np.array([axis], dtype=np.int64)),
        ], outputs=[sliced]),
        gs.Node("Squeeze", f"{v.name}_squeeze{i}", inputs=[
            sliced, gs.Constant(f"{v.name}_sqax{i}", np.array([axis], dtype=np.int64)),
        ], outputs=[squeezed]),
        gs.Node("Unsqueeze", f"{v.name}_unsqueeze{i}", inputs=[
            squeezed, gs.Constant(f"{v.name}_unax{i}", np.array([axis], dtype=np.int64)),
        ], outputs=[unsqueezed]),
    ]
    return nodes, unsqueezed


def unrolled_concat_graph(v_len, axis=1, extra_front=0):
    """Concat over per-element unit_slice_chains of v (an "unrolled
    stack/unbind"), optionally with extra_front non-slice inputs prepended.
    Returns (graph, concat, v)."""
    v = gs.Variable("v", dtype=np.float32, shape=[1, v_len, 8])
    out_len = v_len + extra_front
    out = gs.Variable("out", dtype=np.float32, shape=[1, out_len, 8])
    nodes = []
    cat_inputs = []
    graph_inputs = [v]
    for i in range(extra_front):
        tok = gs.Variable(f"tok{i}", dtype=np.float32, shape=[1, 1, 8])
        graph_inputs.append(tok)
        cat_inputs.append(tok)
    for i in range(v_len):
        chain, tip = unit_slice_chain(v, i, axis, [1, 1, 8])
        nodes += chain
        cat_inputs.append(tip)
    concat = gs.Node("Concat", "cat", inputs=cat_inputs, outputs=[out], attrs={"axis": axis})
    nodes.append(concat)
    return graph(nodes=nodes, inputs=graph_inputs, outputs=[out]), concat, v
