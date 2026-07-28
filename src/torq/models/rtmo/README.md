# RTMO tiny pose — Torq

RTMO is a one-stage multi-person pose estimator. The upstream ONNX
([`Synaptics/RTMO_pose`](https://huggingface.co/Synaptics/RTMO_pose) on Hugging
Face) bakes the mmdeploy decode/NMS tail in (`TopK`/`NonMaxSuppression`/`NonZero`
+ the DCC pose decoder → data-dependent shapes, not NPU-friendly). This package:

1. **Strips the post-processing** — cuts at the eight dense head convolutions,
   exposing fixed-shape outputs; the decode/NMS runs host-side.
2. **Builds the deployable hybrid** — int8 conv backbone + **bf16** AIFI
   transformer neck + int8 head, compiled to three NSS-only vmfbs. This keeps
   int8 speed while the bf16 transformer removes the full-int8 false positives
   (~56 ms on the SL2619 board, matching fp32/bf16 detections).

## Quickstart

The pipeline needs heavy TensorFlow-side deps beyond the torq-tools core
(`onnx2tf`, `tensorflow`, `tf_keras`, `onnxsim`, `opencv-python`) — install
[`requirements.txt`](./requirements.txt) into a dedicated venv.

```sh
# 0. deps (dedicated venv recommended)
python -m pip install -r src/torq/models/rtmo/requirements.txt

# 1. source model + calib images from HF (Synaptics/RTMO_pose)
python -m torq.models.rtmo.download_source -o models/rtmo

# 2. strip post-processing + retarget -> model_nopost_fp32.onnx
torq-export-model rtmo -i models/rtmo/model.onnx -o models/rtmo/export

# 3. build: ONNX -> 3 TFLite parts -> 3 NSS-only vmfbs
TORQC=/path/to/torq-compile-built-from-main \
ONNX=models/rtmo/export/model_nopost_fp32.onnx \
IMAGES=models/rtmo/calib \
PY=~/torq/.venv-rtmo-quant/bin/python \
src/torq/models/rtmo/build_hybrid.sh ./hybrid_out
#  -> hybrid_out/tflite/rtmo_hybrid_{backbone_int8,transformer_bf16,head_int8}.tflite
#  -> hybrid_out/rtmo_hyb_{backbone_int8,transformer_bf16,head_int8}.vmfb

# 4. run the pose demo (boxes + 17-keypoint skeletons drawn on the image)
python -m torq.models.infer_model rtmo person.jpg -m ./hybrid_out
```

`build_hybrid.sh` does both stages (`[1/2]` ONNX→TFLite via `_hybrid.py`, `[2/2]`
TFLite→vmfb via `tosa-converter` + `torq-compile`). Required env: **`TORQC`** (a
`torq-compile` built from `main`), **`ONNX`** (the nopost fp32), **`IMAGES`**
(calib dir). Optional: `PY` (default `python`, needs the deps), `TOSA` (default
`tosa-converter-for-tflite` on PATH), `N_CALIB` (default 100). Steps 1–2 can be
folded in with `torq-export-model rtmo --download` (auto-fetches the source).

## Outputs (the eight head tensors)

For a square input of side `S` (default 320), grouped by branch, level ascending
(stride 16 then 32). Raw logits/features — no on-chip `Sigmoid`.

| Output | Shape s16 / s32 | Meaning |
|---|---|---|
| `cls_scores`  | `[B,1,20,20]` / `[B,1,10,10]`     | person logits |
| `bbox_preds`  | `[B,4,20,20]` / `[B,4,10,10]`     | bbox regression (l,t,r,b) |
| `kpt_vis`     | `[B,17,20,20]` / `[B,17,10,10]`   | per-keypoint visibility |
| `pose_feats`  | `[B,192,20,20]` / `[B,192,10,10]` | pose features (DCC input) |

The host-side decode (NMS + the DCC GAU/SimCC pose classifier) is in
[`_postprocess.py`](./_postprocess.py) and runs automatically in the demo.

## How the hybrid runs ([`_inference.py`](./_inference.py))

`RTMOHybrid` chains the three vmfbs — **backbone (int8) → transformer (bf16) →
head (int8)** — requantizing at the seams as it would on-device, dequantizes the
eight int8 head outputs, and runs `_postprocess` to boxes + keypoints. The FPN
P3/P4 skip connections pass straight through (identical backbone-out/head-in
scales); only the neck-transformed P5 is requantized.

> The seam/head dequant scales in `_inference.py` are baked to the reference
> 100-image calibration. A rebuild with different calibration produces different
> scales — regenerate them from the TFLite parts' I/O quantization if you rebuild.

## Reference

**Export options** — `--input-size` (must be ÷32, default 320), `--bf16-convert-io`
(cast I/O to bf16 too), `--no-bf16` (stop at fp32), `--download` (fetch source
from HF if missing). Retargeting rewrites the two input-size-tied neck constants
(the AIFI positional encoding + the encoder unflatten `Reshape`) via
[`_surgery.py`](./_surgery.py) / [`_pos_enc.py`](./_pos_enc.py).

**Why bf16 for the transformer** — full int8 keeps detection accurate (cls/bbox
cosine ≈ 0.97) but drops the neck + keypoint head to ≈ 0.90, producing spurious
low-confidence boxes. Higher precision *only* on the small transformer removes
them; int16 there is near-lossless but its NSS ACT-LUT is not numerically
functional, so the deployed path uses **bf16** (fp32-level, compiles clean).

**Modules**

| File | Role |
|---|---|
| [`download_source.py`](./download_source.py) | fetch `model.onnx` + `calib/` from HF |
| [`export.py`](./export.py) | strip post-processing + retarget → `model_nopost_*.onnx` |
| [`_hybrid.py`](./_hybrid.py) | split at the transformer boundary + per-part PTQ → 3 TFLite |
| [`quantize.py`](./quantize.py) | whole-model int8 / int16x8 TFLite (non-hybrid) |
| [`build_hybrid.sh`](./build_hybrid.sh) | ONNX → 3 TFLite → 3 NSS-only vmfbs |
| [`_inference.py`](./_inference.py), [`_postprocess.py`](./_postprocess.py), [`infer.py`](./infer.py) | chained runtime + host decode + demo |
