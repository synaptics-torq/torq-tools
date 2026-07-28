# Gemma-4-E2B (int4) — export, lower, and run

End-to-end recipe for turning the pre-quantized int4 Gemma-4-E2B ONNX into 9
compiled `.vmfb` components plus the host-side lookup tables needed to run
inference on an SL2610.

Design rationale and history live in `STATIC_EXPORT_PLAN.md`; this file is
just the procedure.

---

## What you end up with

| Artifact | Size | Runs on |
|---|---:|---|
| 7 × layer-group `.vmfb` | 118–197 MB each | NPU |
| `final_norm.vmfb` | 27 KB | NPU |
| `lm_head.vmfb` | 232 MB | NPU |
| `token_embeddings/` | 247 MB | host (mmap) |
| `per_layer_embeddings/` | 1.5 GB | host (mmap) |
| `rope_caches/` | 769 MB | host (mmap) |

`.vmfb` total ≈ **1.34 GB**; lookup tables ≈ **2.5 GB** on disk but
memory-mapped, so resident cost is only the pages actually touched.

`embed_tokens` needs no compilation at all — it is extracted entirely to the
host-side tables.

---

## Prerequisites

**The compiler must include the sub-byte tile-size fix.** Without it, every
component carrying INT4 weights fails to compile.

```bash
cd /path/to/torq-compiler-dev
git checkout wip/fix-gemma-edge-case      # or main, once merged
cd /path/to/iree-build
ninja third_party/iree/tools/torq-compile
```

Source model (downloaded automatically unless `--onnx-source-dir` is given):
`tss-deposium/gemma-4-E2B-text-only-onnx-int4`.

---

## Step 1 — Export

```bash
torq-export-model gemma4-int4 \
  --onnx-source-dir /path/to/gemma4_onnx_q4 \
  --models-dir /path/to/out \
  --max-kv-len 256 \
  --convert-dtypes
```

Produces, under `<out>/gemma4-e2b-int4/export/onnx/`:

```
static/                     fp32, ORT-runnable
  decoder.onnx
  embed_tokens.onnx         0-node passthrough
  token_embeddings/         data_quant.npy scales.npy zero_points.npy meta.json
  per_layer_embeddings/     (same layout)
  rope_caches/              cos_full_local.npy sin_full_local.npy
                            cos_full_global.npy sin_full_global.npy
bf16/static/                bf16 + int32, what gets compiled
  decoder.onnx
```

Validation runs by default (`--skip-validation` to disable): schema checks,
a one-step ORT smoke test, and a teacher-forced greedy-decode comparison
against the original dynamic decoder.

> **Never point a second exporter at an existing export directory.**
> `OnnxModelExporterBase.__init__` unconditionally `rmtree`s the export dir,
> so constructing an exporter just to re-run validation deletes the
> artifacts. Validate in the same run that produces them.

> The generation check **skips silently** (warning only) if `tokenizer.json`
> or the original `decoder_model_merged_q4.onnx` are missing from the source
> dir. Grep the log for `generation check` to confirm it actually ran.

---

## Step 2 — Split into components and decompose

```bash
python models/gemma4-e2b-int4/export/onnx/split_decoder.py \
  <out>/gemma4-e2b-int4/export/onnx/bf16/static/decoder.onnx \
  /path/to/components \
  --decompose
```

Splits the decoder into 9 standalone components and rewrites every
`MatMulNBits` into `DequantizeLinear` + `MatMul` with native packed INT4
weights. Each component's decomposition runs in its own subprocess to keep
peak memory bounded.

Components: `group_00-04`, `group_05-09`, `group_10-14`, `group_15-19`,
`group_20-24`, `group_25-29`, `group_30-34`, `final_norm`, `lm_head`.

---

## Step 3 — Compile to `.vmfb`

```bash
TORQ_COMPILER_PATH=/path/to/torq-compile \
PYTHON=/path/to/.venv/bin/python3 \
models/gemma4-e2b-int4/export/onnx/compile_components.sh \
  /path/to/components /path/to/vmfb_out
```

Prints a pass/fail table and writes per-component logs. Expect **9/9**.

Flags used (all required):

```
--torq-hw=SL2610
--torq-enable-split-constants-optimization
--torq-enable-annotate-tied-operands
--torq-enable-transpose-optimization
--torq-disable-slicing
```

> **`--torq-disable-slicing` is load-bearing.** Without it the compile takes
> a different path that fails on an unrelated bug. Keep it.

Per component this takes ~3–5 minutes and peaks around 8 GB; the script caps
memory via `systemd-run` where available (`MEM_MAX`, default `8G`).

---

## Step 4 — Run inference

Each forward pass, for token `t` at position `p`:

**1. Host-side lookups** (`export_int4.py` has the reference implementation):

```python
from torq.models.gemma4.export_int4 import (
    _PackedEmbeddingLUT, _RopeCacheLUT, _lookup_embeddings,
)

tok  = _PackedEmbeddingLUT("static/token_embeddings")       # mmap'd
per  = _PackedEmbeddingLUT("static/per_layer_embeddings")   # mmap'd
rope = _RopeCacheLUT("static/rope_caches")                  # mmap'd

inputs_embeds, per_layer_inputs = _lookup_embeddings(tok, per, t)
rope_inputs = rope.lookup(p)   # {cos_full_local: [1,1,256], sin_full_local: ..., ...}
```

**2. Invoke the components in order**, threading `hidden_states` through:

```
group_00-04 → group_05-09 → group_10-14 → group_15-19
            → group_20-24 → group_25-29 → group_30-34
            → final_norm  → lm_head → logits
```

**I/O contract:**

| Component | Extra inputs | Outputs |
|---|---|---|
| `group_00-04` | `past_key_values.{0..4}.{key,value}` | `hidden_out`, `present.{0..4}.{key,value}` |
| `group_05-09` | `hidden_in`, `past_key_values.{5..9}.*` | `hidden_out`, `present.{5..9}.*` |
| `group_10-14` | `hidden_in`, `past_key_values.{10..14}.*` | `hidden_out`, `present.{10..14}.*` |
| `group_15-19` … `group_30-34` | `hidden_in`, `present.13.*`, `present.14.*` | `hidden_out` |
| `final_norm` | `hidden_in` | `hidden_out` |
| `lm_head` | `hidden_in` | `logits` `[1,1,262144]` |

Every layer group also takes `inputs_embeds`, `position_ids`,
`per_layer_inputs`, and all four `cos_full_*`/`sin_full_*` inputs.

**KV cache:** only 15 layers own cache slots (0–14). Layers 15–34 read from
donor layers **13** and **14** — so feed `present.13.*` and `present.14.*`
(produced by `group_10-14`) into all four later groups. Feed each step's
`present.N.*` back as the next step's `past_key_values.N.*`.

**Greedy decode:** `next_token = argmax(logits[0, -1])`.

---

## RAM budget

Measured with fp32 ONNX on CPU; NPU deployment differs but the table split
is the same.

| | Resident |
|---|---:|
| Lookup tables (mmap'd) | ~0 (pages on demand) |
| Model programs | 1.34 GB `.vmfb` |
| KV cache (`max_kv_len=256`, bf16) | 4.2 MB |

KV scales linearly with context — 1024 tokens ≈ 17 MB.

Before the tables were memory-mapped they cost **2.5 GB resident**; keep the
directory-of-`.npy` layout (numpy cannot mmap a `.npz`).

---

## Known limitations

- **The compiled `.vmfb`s have not been numerically validated.** Correctness
  is established on the fp32 ONNX path only. bf16 components cannot run
  under onnxruntime (no bf16 `Pow` kernel), so verifying the compiled
  artifacts requires executing them through IREE's runtime and comparing
  against the reference decoder. This is not done yet.
- **Sliding-window masking gap** for the 12 GQA-derived layers: causality is
  restored but the 512-token sliding bound is not. Only matters beyond 512
  tokens of context.
- `max_kv_len` is fixed at export time; changing it requires a re-export.
