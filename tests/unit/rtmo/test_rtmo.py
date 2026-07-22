# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from torq.models.rtmo._pos_enc import build_2d_sincos_position_embedding
from torq.models.rtmo._surgery import build_stripped_model

# The source RTMO ONNX is a large binary that does not live in the repo; tests
# that need it are skipped when it is absent.
_SOURCE = Path("models/rtmo/model.onnx")
_requires_source = pytest.mark.skipif(
    not _SOURCE.exists(), reason=f"RTMO source model not found at {_SOURCE}"
)

_EXPECTED_OUTPUTS = {
    "cls_scores_s16": 1, "cls_scores_s32": 1,
    "bbox_preds_s16": 4, "bbox_preds_s32": 4,
    "kpt_vis_s16": 17, "kpt_vis_s32": 17,
    "pose_feats_s16": 192, "pose_feats_s32": 192,
}


def test_pos_enc_shape_and_origin():
    emb = build_2d_sincos_position_embedding(10, 10, embed_dim=256)
    assert emb.shape == (1, 100, 256)
    assert emb.dtype == np.float32
    # Position (0, 0): cos quarters are 1, sin quarters are 0.
    row0 = emb[0, 0]
    assert np.allclose(row0[0:64], 1.0)      # cos_w
    assert np.allclose(row0[64:128], 0.0)    # sin_w
    assert np.allclose(row0[128:192], 1.0)   # cos_h
    assert np.allclose(row0[192:256], 0.0)   # sin_h


def test_pos_enc_requires_divisible_by_four():
    with pytest.raises(ValueError):
        build_2d_sincos_position_embedding(4, 4, embed_dim=250)


@_requires_source
def test_pos_enc_reproduces_baked_constant():
    """The regenerated 13x13 grid must match the baked neck.pos_enc_0."""
    model = onnx.load(str(_SOURCE), load_external_data=True)
    graph = gs.import_onnx(model)
    baked = graph.tensors()["neck.pos_enc_0"].values  # [1, 169, 256]
    regen = build_2d_sincos_position_embedding(13, 13, embed_dim=baked.shape[-1])
    # Below bf16 rounding — the model is deployed in bf16 anyway.
    assert np.abs(regen - baked).max() < 2e-3


@_requires_source
@pytest.mark.parametrize("input_size", [320, 416])
def test_strip_outputs_shapes(input_size):
    model = onnx.load(str(_SOURCE), load_external_data=True)
    stripped = build_stripped_model(model, input_size=input_size, batch=1)
    onnx.checker.check_model(stripped)

    outs = {o.name: o for o in stripped.graph.output}
    assert set(outs) == set(_EXPECTED_OUTPUTS)
    for name, ch in _EXPECTED_OUTPUTS.items():
        stride = 16 if name.endswith("s16") else 32
        side = input_size // stride
        shape = [d.dim_value for d in outs[name].type.tensor_type.shape.dim]
        assert shape == [1, ch, side, side], name

    # No dynamic-shape post-processing ops survive the cut.
    ops = {n.op_type for n in stripped.graph.node}
    assert not ({"NonMaxSuppression", "TopK", "NonZero", "Range"} & ops)


@_requires_source
def test_strip_matches_original_head(tmp_path):
    """Stripped@416 must reproduce the original model's head conv taps."""
    ort = pytest.importorskip("onnxruntime")
    taps = {
        "onnx::Shape_971": "cls_scores_s16", "onnx::Transpose_972": "bbox_preds_s16",
        "onnx::Transpose_973": "kpt_vis_s16", "onnx::Transpose_974": "pose_feats_s16",
        "onnx::Transpose_1001": "cls_scores_s32", "onnx::Transpose_1002": "bbox_preds_s32",
        "onnx::Transpose_1003": "kpt_vis_s32", "onnx::Transpose_1004": "pose_feats_s32",
    }
    orig = onnx.load(str(_SOURCE), load_external_data=True)
    for internal in taps:
        orig.graph.output.add().name = internal
    sess_o = ort.InferenceSession(orig.SerializeToString(), providers=["CPUExecutionProvider"])

    stripped = build_stripped_model(
        onnx.load(str(_SOURCE), load_external_data=True), input_size=416, batch=1
    )
    sess_s = ort.InferenceSession(stripped.SerializeToString(), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 416, 416), dtype=np.float32)
    ref = dict(zip(taps.values(), sess_o.run(list(taps), {"input": x})))
    got = {o.name: v for o, v in zip(sess_s.get_outputs(), sess_s.run(None, {"input": x}))}

    for name in _EXPECTED_OUTPUTS:
        assert np.abs(ref[name] - got[name]).max() < 5e-3, name
