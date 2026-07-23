# RTMO tiny (pose) export for Torq

RTMO is a one-stage multi-person pose estimator. The upstream ONNX
(`models/rtmo/model.onnx`) is an mmdeploy export with the whole detection tail
baked in: it takes an image and returns `dets` / `keypoints` with
**data-dependent shapes**, produced by `TopK`, `NonMaxSuppression`, `NonZero`,
`Range`, dynamic `Reshape`/`Gather`, and the DCC pose decoder. None of that is
NPU-friendly.

This package does two things:

1. **Removes the post-processing** — cuts the graph at the eight dense head
   convolutions and exposes them as fixed-shape outputs.
2. **Re-targets the input size** (default **320×320**, was 416×416) and converts
   the result to **bf16**.

Everything after the cut (bbox decode, DCC keypoint decode, NMS) is expected to
run **host-side** on the eight dense outputs.

## Outputs

For a square input of side `S` (default 320), grouped by branch, feature level
ascending (stride 16 then stride 32) — mirroring mmpose `head_module.forward`:

| Output           | Shape (S=320)      | Meaning                                  |
|------------------|--------------------|------------------------------------------|
| `cls_scores_s16` | `[B, 1,   20, 20]` | person logits, stride 16                 |
| `cls_scores_s32` | `[B, 1,   10, 10]` | person logits, stride 32                 |
| `bbox_preds_s16` | `[B, 4,   20, 20]` | bbox regression (l,t,r,b), stride 16     |
| `bbox_preds_s32` | `[B, 4,   10, 10]` | bbox regression, stride 32               |
| `kpt_vis_s16`    | `[B, 17,  20, 20]` | per-keypoint visibility logits, stride 16|
| `kpt_vis_s32`    | `[B, 17,  10, 10]` | per-keypoint visibility logits, stride 32|
| `pose_feats_s16` | `[B, 192, 20, 20]` | pose features (DCC input), stride 16     |
| `pose_feats_s32` | `[B, 192, 10, 10]` | pose features (DCC input), stride 32     |

All outputs are logits/raw features — no `Sigmoid` is applied on-chip. By
default I/O stays fp32 (the bf16 conversion only touches the weights, and the
compile flow converts I/O with `--torq-convert-io-dtype`). Pass
`--bf16-convert-io` for a variant whose input + eight outputs are bf16 too.

## Artifacts

| File                        | Weights | I/O   | Produced by            |
|-----------------------------|---------|-------|------------------------|
| `model_nopost_fp32.onnx`    | fp32    | fp32  | always                 |
| `model_nopost_bf16.onnx`    | bf16    | fp32  | default                |
| `model_nopost_bf16_io.onnx` | bf16    | bf16  | `--bf16-convert-io`    |

## Usage

```sh
# CLI (default: source models/rtmo/model.onnx, 320x320, bf16 weights / fp32 I/O)
torq-export-model rtmo -i models/rtmo/model.onnx -o models/rtmo/export

# bf16 weights AND bf16 I/O
torq-export-model rtmo --bf16-convert-io

# fp32 only, custom size
torq-export-model rtmo --input-size 416 --no-bf16
```

```python
from torq.models.rtmo import export_rtmo

export_rtmo("models/rtmo/model.onnx", "models/rtmo/export", input_size=320)
# -> model_nopost_fp32.onnx, model_nopost_bf16.onnx

export_rtmo("models/rtmo/model.onnx", "models/rtmo/export", bf16_convert_io=True)
# -> model_nopost_fp32.onnx, model_nopost_bf16_io.onnx
```

## Host-side post-processing (what was removed)

To turn the eight outputs back into detections + keypoints:

1. **Scores**: `sigmoid(cls_scores)`; flatten each level to `[H*W]` and
   concatenate → per-anchor person confidence.
2. **Boxes**: decode `bbox_preds` against the anchor-point grid and the level
   stride (`ltrb` distances scaled by stride, offset by the cell centre).
3. **DCC keypoints**: gather `pose_feats` for the surviving instances and run
   the DCC (a small GAU + SimCC-style x/y coordinate classifier). The DCC
   weights (`head.dcc.*`) live in the original ONNX and were dropped by the cut,
   so re-implement it host-side or extract it as a separate small graph.
4. **Visibility**: `sigmoid(kpt_vis)`, gathered per instance.
5. **NMS** over the decoded boxes/scores.

## How the re-targeting works

The backbone + neck + head is fully convolutional except for the neck
transformer (AIFI), which runs only on the smallest (stride-32) level. Two
constants there are tied to the input size and are rewritten by
[`_surgery.py`](./_surgery.py):

- `neck.pos_enc_0` — the 2D sin-cos positional encoding, regenerated for the new
  stride-32 grid ([`_pos_enc.py`](./_pos_enc.py); validated to reproduce the
  baked 13×13 constant to ~1e-3, below bf16 rounding).
- the encoder "unflatten" `Reshape` target `[-1, 256, s32, s32]`.

`--input-size` must be divisible by 32. Correctness of the strip + re-target is
covered by `tests/unit/rtmo/test_rtmo.py`, which checks the 320/416 output
shapes and that the stripped model at 416 reproduces the original head taps.

## int8 TFLite quantization ([`quantize.py`](./quantize.py))

Takes the stripped fp32 ONNX (`model_nopost_fp32.onnx`) to a full-integer int8
TFLite for the NPU (int8 TFLite → TOSA → vmfb). Steps, each verified against the
source ONNX:

1. **Prepare** — onnx-simplifier folds the neck's dynamic-shape reshape guards
   (otherwise onnx2tf emits `Shape`/`Equal`/`Select` clusters that block
   full-integer quantization), and the FFN's exact GELU `0.5·y·(1+erf(y/√2))` is
   replaced with the int8-friendly quick-GELU `y·sigmoid(1.702·y)` — only
   Mul+Sigmoid, no Flex `Erf`, and it lowers as a scaled SiLU on-chip.
2. **Convert** ONNX → TF (SavedModel + float32 TFLite) via `onnx2tf` with the
   `tf_converter` backend (NCHW→NHWC). The default flatbuffer_direct backend's
   strict quantizer can't handle the attention's activation×activation MatMul.
3. **PTQ** — TFLite `TFLiteConverter` calibrated on ~200 COCO images →
   full-integer int8 (int8 I/O, per-channel weights).

### Accuracy (cosine vs fp32 ONNX, 16 held-out images)

| Output branch | float TFLite | int8 | int16-act/int8-wt |
|---|---|---|---|
| cls_scores / bbox_preds | ~1.0000 | 0.96–0.98 | 0.998 |
| kpt_vis / pose_feats    | ~1.0000 | **0.90** | 0.991 |

The float conversion is essentially exact. Full int8 holds up on detection
(cls/bbox ≈ 0.97) but the transformer neck + keypoint head drop to ≈ 0.90 — a
known int8 limit for attention. `--scheme int16x8` (int8 weights, int16
activations) recovers to ≈ 0.99 where the backend supports int16 activations.

### Running it (needs a dedicated venv)

The pipeline needs `onnx2tf` + `tensorflow` + `onnxsim`, which are **not** in the
main tools venv (they'd downgrade shared deps). Use a dedicated venv:

```sh
python3 -m venv ~/torq/.venv-rtmo-quant
~/torq/.venv-rtmo-quant/bin/pip install onnx2tf tensorflow-cpu onnxsim \
    onnxruntime opencv-python-headless tf_keras

# run as a standalone script (the package __init__ pulls LLM deps not in this venv)
~/torq/.venv-rtmo-quant/bin/python src/torq/models/rtmo/quantize.py \
    -i models/rtmo/export/rtmo_nopost_fp32.onnx \
    -o models/rtmo/export/int8 --images-dir models/rtmo/calib
# -> rtmo_prepared.onnx, tf/<...>_float32.tflite, rtmo_int8.tflite
```

Input preprocessing: resize to `input_size`, RGB, `[0,255]` (`--mean/--std` to
override); the same tensors are fed to ONNX and TFLite so the comparisons are
apples-to-apples. The calibration set is any directory of natural images
(`--images-dir`); COCO val images work well.
