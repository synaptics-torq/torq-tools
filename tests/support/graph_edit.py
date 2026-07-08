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
