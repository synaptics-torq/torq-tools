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
covered by `tests/unit/models/test_rtmo.py`, which checks the 320/416 output
shapes and that the stripped model at 416 reproduces the original head taps.
