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

The build runs **entirely through the Torq compiler Python API** — no
`torq-compile` / `tosa-converter` binaries. On top of the torq-tools core
requirements, one env needs:

- the **Torq compiler wheel** (`torq.compiler`) — build it from `torq-compiler-dev`
  with `scripts/build_compiler_wheel.sh <host-build-dir>`, then
  `pip install torq-compiler-dev/dist/torq_compiler-*.whl`;
- **`tosa-converter-for-tflite`** — the self-contained TFLite→TOSA MLIR importer
  (already pinned in the torq-tools core `requirements.txt`);
- the heavy TensorFlow-side quantization deps (`onnx2tf`, `tensorflow`,
  `tf_keras`, `onnxsim`, `opencv-python`) from this dir's
  [`requirements.txt`](./requirements.txt).

```sh
# 0. TF-side quant deps (into the torq-tools env that has the compiler wheel installed)
python -m pip install -r src/torq/models/rtmo/requirements.txt

# 1. source model + calib images from HF (Synaptics/RTMO_pose)
python -m torq.models.rtmo.download_source -o models/rtmo

# 2. strip post-processing + retarget -> model_nopost_fp32.onnx
torq-export-model rtmo -i models/rtmo/model.onnx -o models/rtmo/export

# 3. hybrid: ONNX -> 3 TFLite parts -> 3 NSS-only vmfbs  (Python API, no binaries)
python -m torq.models.rtmo._hybrid \
    -i models/rtmo/export/model_nopost_fp32.onnx \
    -o hybrid_out --images-dir models/rtmo/calib \
    --transformer-scheme bf16 --compile
#  -> hybrid_out/rtmo_hybrid_{backbone_int8,transformer_bf16,head_int8}.tflite
#  -> hybrid_out/rtmo_hyb_{backbone_int8,transformer_bf16,head_int8}.vmfb
```

`_hybrid.py` quantizes the three parts (int8 backbone/head + bf16 transformer)
and, with `--compile`, compiles each to an NSS-only vmfb through the **Torq
compiler Python API** (`torq.utils.compile`): TFLite→TOSA MLIR in-process via
`tosa-converter-for-tflite`, then MLIR→vmfb via the `torq.compiler` wheel — the
same path as the other models, no `torq-compile`/`tosa-converter` binaries. It
compiles the int8 parts unsliced and the bf16 transformer sliced. Standard
`--use-binary` / `--compiler-path` / `--compile-flags` (from `add_torq_args`)
force the binary fallback. Steps 1–2 fold into the export with
`torq-export-model rtmo --download`.

## Outputs (the eight head tensors)

For a square input of side `S` (default 320), grouped by branch, level ascending
(stride 16 then 32). Raw logits/features — no on-chip `Sigmoid`.

| Output | Shape s16 / s32 | Meaning |
|---|---|---|
| `cls_scores`  | `[B,1,20,20]` / `[B,1,10,10]`     | person logits |
| `bbox_preds`  | `[B,4,20,20]` / `[B,4,10,10]`     | bbox regression (l,t,r,b) |
| `kpt_vis`     | `[B,17,20,20]` / `[B,17,10,10]`   | per-keypoint visibility |
| `pose_feats`  | `[B,192,20,20]` / `[B,192,10,10]` | pose features (DCC input) |

These are raw tensors. The host-side decode (NMS + the DCC GAU/SimCC pose
classifier) in [`_postprocess.py`](./_postprocess.py) consumes them off-device —
it is not part of the vmfbs. The build also runs it during verification: the
`[6/6]` step decodes both the fp32-ONNX heads and the quantized-hybrid heads and
compares the results *after* post-processing (detection-count agreement, matched
box IoU, keypoint pixel error, score MAE) — not just the per-head cosine.

## Deployment

This package builds the artifacts; deployment (chaining the vmfbs + host decode)
lives with the on-device runner. The three vmfbs are **NSS-only** (compiled
`--torq-disable-css --torq-disable-host`), so they run on a Torq device via the
Torq HAL — not on the host CPU. The runner chains them **backbone (int8) →
transformer (bf16) → head (int8)**, requantizing at the seams as on-device (the
FPN P3/P4 skips pass straight through — identical backbone-out/head-in scales —
only the neck-transformed P5 is requantized), then dequantizes the eight int8
heads and runs the host-side decode.

> The seam and head (scale, zero_point) constants belong to a specific
> calibration (the reference 100-image set). A rebuild with different calibration
> produces different scales — read them back from the TFLite parts' I/O
> quantization (`tf.lite.Interpreter(...).get_{input,output}_details()`).

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
| [`_hybrid.py`](./_hybrid.py) | split at the transformer + per-part PTQ → 3 TFLite; `--compile` → 3 NSS-only vmfbs (`compile_hybrid`, via the compiler Python API) |
| [`_compile_worker.py`](./_compile_worker.py) | TF-free subprocess that compiles the parts (LLVM isolation from TensorFlow) |
| [`_postprocess.py`](./_postprocess.py) | host-side decode (NMS + DCC GAU/SimCC pose); used by the build's post-processing comparison |
| [`quantize.py`](./quantize.py) | whole-model int8 / int16x8 TFLite (non-hybrid) |
| [`_surgery.py`](./_surgery.py), [`_pos_enc.py`](./_pos_enc.py) | input-size retargeting of the neck constants |
