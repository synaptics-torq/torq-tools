# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The base exporter's opt-in ONNX cleanup hook (export_onnx(cleanup=True))."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from support.graph_edit import graph, to_model
from torq.model_export.onnx import OnnxModelExporterBase


pytestmark = pytest.mark.ort


def _conv_bn_model():
    """Conv -> Mul -> Add (eval-mode BatchNorm), foldable by the cleanup."""
    x = gs.Variable("x", dtype=np.float32, shape=[1, 3, 4, 4])
    w = gs.Constant("w", np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 27.0)
    conv_out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 2, 4, 4])
    mul_out = gs.Variable("mul_out", dtype=np.float32, shape=[1, 2, 4, 4])
    bn_out = gs.Variable("bn_out", dtype=np.float32, shape=[1, 2, 4, 4])
    nodes = [
        gs.Node("Conv", "conv", inputs=[x, w], outputs=[conv_out],
                attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}),
        gs.Node("Mul", "mul", inputs=[
            conv_out,
            gs.Constant("gamma", np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1)),
        ], outputs=[mul_out]),
        gs.Node("Add", "add", inputs=[
            mul_out,
            gs.Constant("beta", np.array([10.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1)),
        ], outputs=[bn_out]),
    ]
    return to_model(graph(nodes=nodes, inputs=[x], outputs=[bn_out]))


class _StubExporter(OnnxModelExporterBase):
    """Minimal concrete exporter serving one in-memory component."""

    def __init__(self, base_dir):
        self._base_dir = base_dir
        super().__init__("fp32", False, {}, base_dir, opt_configs={})

    def _setup_dirs(self):
        dirs = [self._base_dir / name for name in ("onnx", "export", "convert", "torq")]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _load_onnx(self):
        return {"model": _conv_bn_model()}

    def make_static(self): ...

    def apply_post_static_patches(self, model_path, component): ...

    def validate_onnx(self, n_iters: int = 5): ...


@pytest.mark.parametrize("cleanup", [False, True])
def test_export_onnx_cleanup_flag(tmp_path, cleanup):
    exporter = _StubExporter(tmp_path)
    exporter.export_onnx(validate=False, cleanup=cleanup)

    exported = onnx.load(exporter.export_dir / "model.onnx")
    ops = [node.op_type for node in exported.graph.node]
    if cleanup:
        assert ops == ["Conv"]  # BN chain folded into the conv
    else:
        assert ops == ["Conv", "Mul", "Add"]
