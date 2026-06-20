<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
-->

# Moonshine Streaming → TFLite (int8)

Export the Moonshine Streaming **2-split** architecture (`fused_encoder`, `decoder_kv`)
to **TFLite**, with optional **weight-only int8** quantization, as a path toward
lowering to `.vmfb` for the Torq backend.

Pipeline:

```
PyTorch wrappers ──litert-torch──► .tflite (fp32) ──[weight-only int8]──► .tflite (int8)
                                                                              │
                                          iree-import-tflite ──► .mlir ──► .vmfb   (next step)
```

This directory is a standalone fork of `moonshine_streaming_2_split/`. The ONNX
graph-surgery layer is **not** used: `litert-torch` emits TFLite builtins directly, so
there is no `_graph*.py` / `edits.py` / `onnx.py`.

---

## Components

| Model | Purpose | I/O (tiny) |
|---|---|---|
| `fused_encoder` | frontend + stateful encoder + adapter + cross-KV generation, one dispatch per audio chunk | `audio[1,1280]`, conv/feature/window buffers → `k_cross/v_cross[6,1,8,4,40]` + updated buffers |
| `decoder_kv` | single-token decoder with a **fixed-size** self-KV cache and cross-KV from the encoder | `inputs_embeds[1,1,320]`, self/cross KV, masks, `position_ids` → `logits[1,1,32768]` + updated self-KV |

Host-side responsibilities (kept out of the graphs): token + position embedding
lookups, encoder warmup, and the cross-KV ring buffer.

---

## Requirements

Use the project venv (has `litert_torch`, `ai_edge_litert`, `torch`, `transformers`):

```bash
../venv/bin/python      # i.e. core/venv
```

Key deps: `litert-torch` (torch→tflite converter), `ai_edge_litert` (tflite runtime),
`torchao` (PT2E quantization). Model weights for `UsefulSensors/moonshine-streaming-tiny`
are downloaded to `models/.../weights/<size>/` on first run (or reused if cached).

All commands below are run from the repo root (`core/torq-tools-dev/`) with
`PYTHONPATH=src`.

---

## Export the `.tflite` models

```bash
# fp32, both components
PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_tflite.tflite_export

# weight-only int8 (per-channel dynamic-range; no calibration set needed)
PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_tflite.tflite_export --int8

# a single component
PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_tflite.tflite_export \
    --components decoder_kv --int8
```

### Useful flags (`tflite_export`)

| Flag | Default | Meaning |
|---|---|---|
| `-s, --model-size` | `tiny` | `tiny` or `small` |
| `--components` | `decoder_kv fused_encoder` | which models to export |
| `--int8` | off | weight-only int8 quantization (adds `_int8` suffix to filenames) |
| `--chunk-len` | `1280` | audio chunk size in samples (1280 = 80 ms @ 16 kHz) |
| `--input-seconds` | `5` | sizes `max_memory_len`/`max_tokens` for the decoder |
| `--tokens-per-sec` | `6` | decode budget per second |
| `--in-graph-embeddings` | off | keep `embed_tokens` in the decoder graph (token input). **Incompatible with `--int8`** — see Notes. |
| `--out-dir` | `models/export/tflite/2split_streaming_static` | output directory |
| `--hf-repo`, `--models-dir` | — | model source overrides |

### Output layout

```
models/export/tflite/2split_streaming_static/
├── fused_encoder.tflite          fused_encoder_int8.tflite
├── decoder_kv.tflite             decoder_kv_int8.tflite
├── decoder_token_embeddings.npy  # host-side token embedding table
├── adapter_pos_emb.npy           # host-side position embedding table
├── streaming_config.json         # chunk_len, feature_stride (F), total_lookahead, warmup, max_tokens, max_memory_len
├── tokenizer.json  config.json
```

Approximate sizes (tiny): decoder 86.7 MB → **22.8 MB** int8; encoder 43.1 MB → **11.3 MB** int8 (~3.8×).

---

## Validate on the sample audio

Runs the full streaming pipeline on the bundled OSR clip and prints the transcription.
The export is sized for `max_memory_len` (~5 s), so — exactly like the ONNX validation —
the output covers roughly the first few sentences.

```bash
# fp32
PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_tflite.validate \
    -m models/export/tflite/2split_streaming_static -s tiny

# int8
PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_tflite.validate \
    -m models/export/tflite/2split_streaming_static -s tiny --int8

# custom audio
... validate -m <dir> -s tiny --wav path/to/clip.wav
```

### Results (tiny, OSR clip)

| Config | Transcription |
|---|---|
| torch reference | `The birch canoes slid on the smooth planks. Glue the sheet…` |
| **fp32 tflite** | ✅ `The birch canoes slid on the smooth planks. Glue the sheet.` |
| int8 decoder + fp32 encoder | ✅ `The birch canoes slid on the smooth planks. Glue the sheet.` |
| fp32 decoder + int8 encoder | ⚠️ `The birds can be seen on the surface of the plants.` |
| **int8 (both)** | ⚠️ `The birds can be afraid of their plants.` |

**Takeaway:** the **decoder int8 is essentially lossless** (logit cos ≈ 0.9992), while
**weight-only int8 on the encoder is the source of degradation** (cross-KV cos ≈ 0.97 on
this tiny model). Recommended deployment: **int8 decoder + fp32 (or float-fallback)
encoder**, which already yields full-quality transcription above.

---

## Quantization

`--int8` applies **per-channel dynamic-range PT2E** quantization
(`get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)`): Linear/Conv
weights are stored int8, activations are quantized at runtime. No calibration corpus is
required. Activations and sensitive ops (LayerNorm, softmax, the asinh polynomial) stay
float.

---

## Implementation notes

These differ from the ONNX 2-split and are what make the TFLite path work:

- **Host-side embeddings (decoder).** The decoder takes float `inputs_embeds`, not a
  token id. An in-graph `EMBEDDING_LOOKUP` cannot be int8-quantized (TFLite requires
  `zero_point == 0`), so the lookup is done on the host via `decoder_token_embeddings.npy`.
  `--in-graph-embeddings` restores the token-input variant but is incompatible with `--int8`.

- **Static self-KV cache (decoder).** `DecoderKVWrapper(static_self_cache=True)` uses
  `_StaticSelfCache`, which `index_copy`s the new token's K/V into a fixed
  `[1, heads, max_tokens, dim]` buffer at `position_ids` instead of concatenating. This
  keeps the graph a reusable fixed-point (self-KV stays size 30, not 30→31) — the
  torch-level equivalent of the old `make_decoder_static` graph surgery. A 4D
  `self_attn_bias` input masks unfilled positions (mirrors `cross_attn_bias`).

- **Precomputed encoder masks.** `StatefulEncoderWrapper.precompute_masks` bakes the
  per-layer sliding-window masks as constant buffers. `create_bidirectional_mask` uses
  `torch.vmap` internally, which survives a single fp32 export but breaks the int8
  PT2E export→convert→re-export; precomputing removes vmap from the traced graph
  (numerically identical, fp32 still cos = 1.0).

---

## Next step: lower to `.vmfb`

The back half already exists in `torq/utils/compile.py` (it accepts `.tflite`):

```bash
PYTHONPATH=src ../venv/bin/python -m torq.utils.compile \
    models/export/tflite/2split_streaming_static/decoder_kv_int8.tflite \
    -o decoder_kv_int8.vmfb
```

`tools/convert_static/tflite.py` (`convert_model`) is available to strip any residual
dynamic shape signatures before import if needed (the current exports are already fully
static).

See `plan.md` for the full design and milestone list.
