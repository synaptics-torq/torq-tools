# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""RTMO hybrid runtime.

The deployable RTMO variant is a three-part hybrid — an int8 conv backbone, a
bf16 AIFI transformer neck, and an int8 detection head — compiled to three
NSS-only vmfbs (see :mod:`torq.models.rtmo._hybrid` / ``build_hybrid.sh``). This
keeps the int8 speed while the higher-precision transformer removes the
full-int8 false positives.

:class:`RTMOHybrid` chains the three vmfbs, requantizing at the seams exactly as
it would on-device, dequantizes the eight int8 head outputs, and runs the
host-side decode (:func:`torq.models.rtmo._postprocess.model_postprocess`) to
return ``dets`` / ``keypoints``.

The seam and head (scale, zero_point) constants below correspond to the
reference hybrid build (100-image COCO calibration). A rebuild with different
calibration produces different scales; regenerate them from the TFLite parts'
I/O quantization if you rebuild.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a runtime dep of the demo
    cv2 = None

import ml_dtypes

from ...inference.runners import VMFBInferenceRunner
from ._postprocess import model_postprocess

BF16 = ml_dtypes.bfloat16
DEFAULT_INPUT_SIZE = 320

# The three chained parts (filenames expected in the model directory).
BACKBONE_VMFB = "rtmo_hyb_backbone_int8.vmfb"
TRANSFORMER_VMFB = "rtmo_hyb_transformer_bf16.vmfb"
HEAD_VMFB = "rtmo_hyb_head_int8.vmfb"

# NCHW head shape -> canonical output name (outputs are matched by shape).
HEAD_SHAPES = {
    (1, 1, 20, 20): "cls_scores_s16",   (1, 1, 10, 10): "cls_scores_s32",
    (1, 4, 20, 20): "bbox_preds_s16",   (1, 4, 10, 10): "bbox_preds_s32",
    (1, 17, 20, 20): "kpt_vis_s16",     (1, 17, 10, 10): "kpt_vis_s32",
    (1, 192, 20, 20): "pose_feats_s16", (1, 192, 10, 10): "pose_feats_s32",
}

# Backbone int8 image input quantization.
IN_SCALE, IN_ZP = 1.0, -128

# Seam (scale, zero_point): backbone int8 outputs -> {dequant} -> transformer
# (bf16) -> {requant} -> head int8 inputs. P3/P4 are skip connections whose
# backbone-output and head-input scales are identical, so they pass through
# unchanged; only P5 (transformed by the neck) is requantized.
SEAMS = {
    "p3_shape": (1, 40, 40, 96), "p4_shape": (1, 20, 20, 192), "p5_shape": (1, 10, 10, 256),
    "bb_p3": (0.026187874376773834, -117), "hd_p3": (0.026187879964709282, -117),
    "bb_p4": (0.048588335514068604, -122), "hd_p4": (0.048588335514068604, -122),
    "bb_p5": (0.037300221621990204, -7),   "hd_p5": (0.06488647311925888, 2),
}

# NHWC head-output shape -> (name, scale, zero_point) for dequantizing the int8 head.
HEAD_QUANT = {
    (1, 20, 20, 1): ("cls_scores_s16", 0.102855027, 95),
    (1, 10, 10, 1): ("cls_scores_s32", 0.152402967, 100),
    (1, 20, 20, 4): ("bbox_preds_s16", 0.0596383289, 3),
    (1, 10, 10, 4): ("bbox_preds_s32", 0.0294422898, 0),
    (1, 20, 20, 17): ("kpt_vis_s16", 0.0980607048, 21),
    (1, 10, 10, 17): ("kpt_vis_s32", 0.103052862, 22),
    (1, 20, 20, 192): ("pose_feats_s16", 0.01536687, -5),
    (1, 10, 10, 192): ("pose_feats_s32", 0.0186605658, 10),
}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def letterbox(img, size):
    """Aspect-preserving resize with padding; returns (canvas, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img_bgr, size=DEFAULT_INPUT_SIZE):
    """BGR uint8 HWC -> (1,3,S,S) float32 NCHW RGB in [0,255] + letterbox meta."""
    proc, scale, pad_x, pad_y = letterbox(img_bgr, size)
    rgb = proc[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[None, ...]
    return tensor, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y}


# --------------------------------------------------------------------------- #
# Hybrid runner
# --------------------------------------------------------------------------- #
def _q(x, sz, dt=np.int8):
    scale, zp = sz
    info = np.iinfo(dt)
    return np.clip(np.rint(x / scale) + zp, info.min, info.max).astype(dt)


def _dq(x, sz):
    scale, zp = sz
    return (x.astype(np.float32) - zp) * scale


def _by_shape(arrs, shape):
    return next(a for a in arrs if tuple(np.asarray(a).shape) == shape)


def _scales_match(a, b, rtol=1e-4):
    return abs(a[0] - b[0]) <= rtol * abs(b[0]) and a[1] == b[1]


class RTMOHybrid:
    """Chain the three RTMO hybrid vmfbs behind a single ``infer(image)`` call."""

    def __init__(self, model_dir, *, device_uri="torq", function="main", n_threads=None):
        d = Path(model_dir)
        common = dict(function=function, device_uri=device_uri, n_threads=n_threads)
        self._bb = VMFBInferenceRunner(d / BACKBONE_VMFB, **common)
        self._tf = VMFBInferenceRunner(d / TRANSFORMER_VMFB, **common)
        self._hd = VMFBInferenceRunner(d / HEAD_VMFB, **common)
        self._infer_time_ms = 0.0
        self._p3_pass = _scales_match(SEAMS["bb_p3"], SEAMS["hd_p3"])
        self._p4_pass = _scales_match(SEAMS["bb_p4"], SEAMS["hd_p4"])

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    def _run_chain(self, x_int8):
        s = SEAMS
        bb_out = self._bb.infer([x_int8])                       # 3 int8 NHWC feature maps
        p3 = _by_shape(bb_out, s["p3_shape"])
        p4 = _by_shape(bb_out, s["p4_shape"])
        p5 = _by_shape(bb_out, s["p5_shape"])

        # backbone P5 -> bf16 -> transformer -> refined P5 (NCHW)
        p5_bf16 = _dq(p5, s["bb_p5"]).astype(BF16)
        p5t = np.asarray(self._tf.infer([p5_bf16])[0]).astype(np.float32)
        p5t = np.transpose(p5t, (0, 2, 3, 1))                   # NCHW -> NHWC

        # requant seams into the head's int8 input scales (P3/P4 pass through)
        p5_h = _q(p5t, s["hd_p5"])
        p3_h = p3 if self._p3_pass else _q(_dq(p3, s["bb_p3"]), s["hd_p3"])
        p4_h = p4 if self._p4_pass else _q(_dq(p4, s["bb_p4"]), s["hd_p4"])
        return self._hd.infer([p3_h, p4_h, p5_h])               # 8 int8 NHWC heads

    def _dequant_heads(self, hd_out):
        heads = {}
        for a in hd_out:
            a = np.asarray(a)
            name, scale, zp = HEAD_QUANT[tuple(a.shape)]
            real = (a.astype(np.float32) - zp) * scale
            heads[name] = np.transpose(real, (0, 3, 1, 2))      # NHWC -> NCHW
        return heads

    def infer(self, img_bgr, size=DEFAULT_INPUT_SIZE):
        """Run the hybrid on a BGR image; return ``(dets, keypoints, meta)``."""
        tensor, meta = preprocess(img_bgr, size)
        nhwc = np.transpose(tensor, (0, 2, 3, 1)).astype(np.float32)
        x = _q(nhwc, (IN_SCALE, IN_ZP))
        st = perf_counter_ns()
        hd_out = self._run_chain(x)
        self._infer_time_ms = (perf_counter_ns() - st) / 1e6
        heads = self._dequant_heads(hd_out)
        dets, keypoints = model_postprocess(heads)
        return dets, keypoints, meta


def load_rtmo(model_dir, *, device_uri="torq", n_threads=None) -> RTMOHybrid:
    """Load the three hybrid vmfbs from ``model_dir`` into an :class:`RTMOHybrid`."""
    return RTMOHybrid(model_dir, device_uri=device_uri, n_threads=n_threads)
