# GPTQ → ONNX QDQ Pipeline (in/out_feature)

A minimal pipeline that takes the weights of Google's **Gemma-3 270M Instruct**
(`google/gemma-3-270m-it`), **quantizes them to INT4 with GPTQ**, saves the result
as an **ONNX QDQ graph**, and validates accuracy. (The input `model.onnx` is this
HF model exported to ONNX — see §3.2.)

The quantization group axis can be chosen with `--grouping`, either
**in_feature (default) or out_feature**. §1 explains the difference between the two.
(The default is in_feature; the best-of-8 accuracy of the two axes is
`cos_sim ≈ 0.9519 vs 0.9559`, i.e. effectively equivalent — see §6.)

---

## Table of Contents
1. [Background Concepts (read first)](#1-background-concepts-read-first)
2. [Big Picture](#2-big-picture)
3. [Prerequisites](#3-prerequisites)
4. [Quick Start (run it all at once)](#4-quick-start-run-it-all-at-once)
5. [Step-by-Step Detail](#5-step-by-step-detail)
6. [Interpreting Accuracy](#6-interpreting-accuracy)
7. [File Structure](#7-file-structure)
8. [Common Problems](#8-common-problems)
9. [Sources](#9-sources)

---

## 1. Background Concepts (read first)

### 1.1 The size of this model
Gemma-3 270M is a small LLM with hidden dim **640**, **18** layers, and vocab
**262,144**. The weight of each Linear layer is a matrix of shape
`W[out_features(N), in_features(K)]`.

### 1.2 What weight-only INT4 quantization is
Storing a weight's float values by approximating them as **4-bit integers
(16 levels)**. To reduce approximation error, the weights are split into small
**groups**, and each group gets its own `scale` and `zero_point`. This group size
is **group size = 32**.

Dequantize formula:
```
W ≈ (q - zero_point) * scale        # q is a 4-bit integer
```
- **asymmetric** (used here): keeps a separate zero_point → good for skewed distributions.
- symmetric: zero_point fixed at 0.

### 1.3 Which "axis" the group is cut along — selectable via argument
| Mode | Group direction | scale shape | Description |
| --- | --- | --- | --- |
| **in_feature** (Common) | K axis, 32 at a time | `[N, K/32]` | Groups the inputs of one output neuron |
| **out_feature** | **N axis, 32 at a time** | `[N/32, K]` | Groups 32 output neurons |

The favorable form can differ depending on the architecture.

### 1.4 What GPTQ is
A PTQ algorithm smarter than plain rounding (RTN). It runs calibration data through
the model to obtain the input statistics (Hessian) of each layer, and **corrects the
error introduced by rounding a weight to 4 bits using the remaining weights**. It is
more accurate than RTN (hqq/awq perform much worse on this model and are not used).

> ⚠️ **GPTQ produces different results on each run.** Due to GPU non-determinism plus
> this 270M model's sensitivity to 4-bit rounding, accuracy fluctuates significantly
> **even with the same seed and settings** (measured cos_sim 0.61~0.96). → **Run it
> several times and pick the best.** `select_best_gptq.py` automates this selection.

### 1.5 ONNX QDQ representation
In an ONNX graph, quantized weights are represented as **QDQ (Quantize-DeQuantize)**.
The output graph of this repo, for each Linear, looks like:
```
DequantizeLinear(q, scale, zp, axis=0) → Reshape([K, N]) → Cast(bf16) → MatMul
```
- **Does not use** the `block_size` attribute, and uses **no** runtime `Transpose`
  (NPU-friendly).
- The out_feature groups are pre-rearranged offline so the structure above runs directly.

### 1.6 The `-emb` form and tied embedding
This model's ONNX is in **embedding-separated (`-emb`) form**:
- The input embedding lookup is done **outside** the graph, and the `token_embedding`
  tensor is fed in as an input.
- The final `lm_head` (logits computation) is **inside** the graph and is **quantized**.

In Gemma, `embed_tokens` and `lm_head` are **tied** (same weight). GPTQ quantizes only
the Linear `lm_head`, leaving the Embedding `embed_tokens` in full precision. For what to
feed as the input embedding during `-emb` inference, see §5.5.

---

## 2. Big Picture

```
                 [required input]
              onnx/model.onnx  (fp32 base graph, ~536MB)
                     │
     ┌───────────────┼─────────────────────────────────────────┐
     │ Stage 0       │ (if no template, Stage 2 runs it automatically)
     ▼               │                                           │
 quantize_matmul.py  │                                           │
  (q4_k template)    │                                           │
     │               ▼                                           ▼
     │        run_gptq.py                          test_snr.py
     │         (Stage 1: GPTQ)                        compare_onnx_generation.py
     │               │                                     (Stage 3: accuracy)
     ▼               ▼                                           ▲
 model_q4k.onnx  compressed safetensors                         │
   (template) ──────┐   │                                        │
                    ▼   ▼                                        │
          safetensors_to_onnx_qdq.py  ──────────────────────────┘
                (Stage 2: → in_feature QDQ ONNX)
                            │
                            ▼
              onnx/gptq_int4_in_feature.onnx   ← final artifact
```

`select_best_gptq.py` repeats Stage 1+2+3 N times and **automatically selects the single
best result** (§4).

---

## 3. Prerequisites

### 3.1 Python environment
```bash
# Create and activate a fresh environment (any name), then:
pip install -r requirements.txt
```
Versions used for validation: llmcompressor 0.10.1, transformers 4.57, onnx 1.19,
onnxruntime 1.27 (dev), torch 2.11.

**Stage 1 (GPTQ) device** — selected via `run_gptq.py --device` (default `auto` →
cuda if present, else cpu):
- `cuda` — NVIDIA GPU. By far the fastest; the intended path.
- `cpu` — CUDA-free path. Works everywhere but much slower (fine for this 270M model).
- Apple Silicon (**MPS**) is **not** offered: the `compressed_tensors` offload backend
  has no MPS implementation (both calibration and save hit a hard `NotImplementedError`),
  so a Mac runs Stage 1 on `cpu`.

**HuggingFace access (one-time):** Stage 1 downloads the **gated** HF model
`google/gemma-3-270m-it` (see §3.2 for why). On a machine that has not cached it yet,
accept the license on the model page and log in once:
```bash
huggingface-cli login        # or: export HF_TOKEN=<your token>
```
After the first download it is cached, so later runs need no network (`HF_HUB_OFFLINE=1`).

### 3.2 Required input file: `onnx/model.onnx`
Large binaries (`*.onnx`, `*.safetensors`, `*.npy`) are not committed to git
(handled by `.gitignore`). **The only file you must place yourself is the single base
`onnx/model.onnx`.** (Stage 1 additionally pulls the HF PyTorch model — §3.1 — but that
is downloaded/cached automatically, not a file you provide.)

> **`model.onnx` and the HF model are the same Gemma model in two formats.**
> `model.onnx` is HuggingFace's Gemma-3 270M Instruct **exported to ONNX** — same weights
> and structure, except the embedding lookup is pulled outside the graph (**`-emb` form**:
> the input is a `token_embedding` tensor, not token ids). The two play different roles:
>
> | | `onnx/model.onnx` (ONNX) | `google/gemma-3-270m-it` (PyTorch) |
> | --- | --- | --- |
> | Role | the **graph skeleton** quantized weights are written into (Stage 0/2) + the fp32 **accuracy baseline** (Stage 3) | the weight source GPTQ actually **quantizes** (Stage 1) |
>
> So **Stage 1 never reads `model.onnx`** — it quantizes the PyTorch weights and emits
> safetensors; Stage 2 then writes those values into the ONNX skeleton. Stage 1 is
> mandatory (it produces the real GPTQ values); Stage 0 only builds an empty skeleton
> whose values Stage 2 discards. They are **not** alternatives.

| File | What it is |
| --- | --- |
| `onnx/model.onnx` (~536MB) | The **fp32 base** graph (`-emb` form), `google/gemma-3-270m-it` exported to ONNX. **The only required input.** |
| `onnx/model_q4k.onnx` (~310MB) | A q4_k QDQ **template** (graph skeleton). |

---

## 4. Quick Start (run it all at once)

With just `onnx/model.onnx`, the single line below automatically does
**GPTQ ×N → ONNX conversion → accuracy measurement → best-model selection**
(~5 min on 8 GPUs):

```bash
python select_best_gptq.py \
    --num-runs 8 --gpus 0,1,2,3,4,5,6,7 \
    --base onnx/model.onnx \
    --template onnx/model_q4k.onnx \
    --out onnx/gptq_int4_in_feature.onnx     # default grouping = in_feature
```

Example output:
```
  run0: cos_sim=0.9393
  run6: cos_sim=0.9519
  ...
=== ranking ===
  run6   cos_sim=0.9519   <- BEST
  ...
N=8  min=0.7326  mean=0.9086  max=0.9519
best run6 (cos_sim=0.9519) -> onnx/gptq_int4_in_feature.onnx
```
- If the `--template` file is missing, it is generated automatically from `--base`.
- With a single GPU, use `--gpus 0` (sequential); set `--num-runs` as high as you like.
- **No CUDA GPU?** Add `--device cpu` (runs sequentially, `--gpus` ignored). Much slower
  but works; on a Mac this is the only option (see §3.1).
- **Quantization axis selection**: `--grouping in` (default, in_feature `group_size=32`,
  native `block_size` DQ form) / `--grouping out` (out_feature `block_structure='32x1'`).
  To produce out_feature, use `--grouping out --out onnx/gptq_int4_out_feature.onnx`.
  (Measured: in best-of-8 `cos_sim ≈ 0.9519`, out best-of-8 `≈ 0.9559` — effectively
  equivalent, see §6.)

To understand the details, follow §5 step by step.

---

## 5. Step-by-Step Detail

Each stage is described in terms of **what it takes as input and what it produces**.

### Stage 0 — Build the template (`quantize_matmul.py`) · *usually skipped*
- **Input**: `onnx/model.onnx`
- **Output**: `onnx/model_q4k.onnx` (a template with QDQ structure)
- **Role**: Converts each MatMul weight of the base graph to q4_k (asymmetric 4-bit,
  block-wise `DequantizeLinear`) to build a **QDQ skeleton**. Stage 2 reuses only the
  structure of this skeleton and overwrites the values with the GPTQ results, so the
  template's actual quantized values do not matter.

```bash
python quantize_matmul.py onnx/model.onnx \
    --bits 4 --granularity q4_k --block-size 32 \
    --out onnx/model_q4k.onnx
```
> If you pass `--base-onnx` to Stage 2, this step runs automatically, so **you don't
> need to run it directly.** (Note: `--granularity q4_0` is symmetric 4-bit.)

### Stage 1 — GPTQ quantization (`run_gptq.py`) · *CUDA or CPU (§3.1)*
- **Input**: HF `google/gemma-3-270m-it` (auto-downloaded PyTorch model — **not** `model.onnx`),
  the ultrachat dataset (calibration)
- **Output**: `gemma-3-270m-it-W4A16-G32-infeat-gptq-512/` (compressed safetensors directory;
  `-outfeat-` if `--grouping out`)

```bash
python run_gptq.py                  # in_feature (default), --device auto
python run_gptq.py --grouping out   # out_feature
python run_gptq.py --device cpu     # CUDA-free (required on a Mac)
```
Core recipe (inside the script, branching on `--grouping`):
```python
# in_feature (default): idiomatic group quant, scale [N, K//32]
{"num_bits": 4, "type": "int", "symmetric": False,
 "strategy": "group", "group_size": 32}
# out_feature (--grouping out): based on W[n:n+32, k], scale [N//32, K]
{"num_bits": 4, "type": "int", "symmetric": False,
 "strategy": "block", "block_structure": "32x1"}
```
- Select the quantization axis with `--grouping {out,in}` (default `in`). The output
  directory is also distinguished as `...-G32-infeat-...` / `...-G32-outfeat-...`.
- `NUM_CALIBRATION_SAMPLES` (default 512), seed (5436), etc. are adjusted at the top of the script.
- At the end of the run, a generation demo with a few prompts is printed so you can
  eyeball quality.
- ⚠️ As in §1.4, **run-to-run variance is large.** Do not rely on a single result →
  using `select_best_gptq.py` from §4 is recommended.

### Stage 2 — safetensors → QDQ ONNX (`safetensors_to_onnx_qdq.py`)
- **Input**: Stage 1's `model.safetensors` + template (or `--base-onnx`)
- **Output**: 1 QDQ ONNX (in/out_feature — per `--grouping`)

```bash
python safetensors_to_onnx_qdq.py \
    --onnx-template   onnx/model_q4k.onnx \
    --base-onnx       onnx/model.onnx \
    --safetensors     gemma-3-270m-it-W4A16-G32-infeat-gptq-512/model.safetensors \
    --out             onnx/model_infeat_qdq.onnx \
    --grouping        in \
    --expected-group-size 32
```
- `--onnx-template`: path to the QDQ template. **If absent, generated automatically from `--base-onnx`.**
- `--base-onnx`: base ONNX for auto-generating the template (so the single base `model.onnx` suffices).
- `--expected-group-size 32`: verifies that the safetensors group size is 32 (a safeguard).
- `--skip-lm-head`: specify to leave `lm_head` unreplaced.
- `--grouping {out,in}` (default `in`): **must match** Stage 1's `--grouping`.
  - `in` (default): `DequantizeLinear(axis=0, block_size=32)` → Cast → MatMul
    (q `[K,N]` uint8, scale/zp `[K//32,N]`). It is the native block DQ form and has the
    same structure as `onnx/gptq_int4_in_feature_blocksize_example.onnx`.
  - `out`: `DequantizeLinear(axis=0)` → Reshape[K,N] → Cast → MatMul

On success it reports that 127 Linears were replaced, e.g. `converted=127 skipped=0`.

### Stage 3 — Accuracy test
**(a) logit metric** — 6 metrics vs. base (interpretation in §6):
```bash
python test_snr.py --orig onnx/model.onnx --quant onnx/model_outfeat_qdq.onnx
```
Example output (measured on a best-of-N selection):
```
{'mean_abs': 1.843, 'max_abs': 11.725, 'mean_rel': 2.111,
 'max_rel': 25985.5, 'cos_sim': 0.9559, 'snr_db': 10.278}
```

**(b) Actual generation comparison** — generate from base vs quant on the same prompt:
```bash
python compare_onnx_generation.py \
    --onnx-a  onnx/model.onnx \
    --onnx-b  onnx/model_outfeat_qdq.onnx \
    --model-dir google/gemma-3-270m-it \
    --prompt "What causes rainbows?"
```
→ Shows the `decoded_a` (base) and `decoded_b` (quant) sentences side by side.

### 5.5 (optional) Extract quantized embedding (`extract_lm_head_embeddings.py`)
When you want to feed the input embedding in `-emb` inference **consistently with the
quantized `lm_head`** (tied embedding), dequantize the quantized `lm_head` and extract it
as an embedding table:
```bash
python extract_lm_head_embeddings.py \
    --onnx onnx/gptq_int4_out_feature.onnx \
    --output onnx/token_embeddings.npy --dtype bf16
# → (262144, 640): the embedding table from dequantizing the quantized lm_head
```
- This way both the input emb and output `lm_head` are **quantized** (the 260526 evaluation approach).
- To use a full-precision embedding, just use `model.embed_tokens.weight` from the
  safetensors (bf16, not quantized) as is.

> `test_snr.py` feeds a random embedding, so it works even without this file.
> Create `token_embeddings.npy` when you need a real embedding for generation/deployment.

---

## 6. Interpreting Accuracy

The 6 metrics `test_snr.py` computes from base logits `b` and quant logits `q`:

| Metric | Meaning | Direction |
| --- | --- | --- |
| `mean_abs` | mean absolute error | lower is better |
| `max_abs` | worst absolute error | lower is better |
| `mean_rel` | mean relative error | lower is better |
| `max_rel` | worst relative error | **always very large (normal)** |
| `cos_sim` | direction similarity of the two logit vectors | **closer to 1 is better** ← the practical key |
| `snr_db` | signal-to-noise ratio (dB) | higher is better |

- **Mainly look at `cos_sim` and `snr_db`.** Read cos_sim as a proxy for "the probability
  that argmax (next token) does not change."
- `max_rel` being in the thousands~tens of thousands is normal: where the base logit is
  near 0, even a small absolute error explodes into a huge relative error. Do not judge
  quality by this value alone.
- **Expected level**: a good run of this out_feature INT4 GPTQ is `cos_sim ≈ 0.95`,
  `snr_db ≈ 10`. (Reference baseline: plain q4_k RTN is cos_sim ≈ 0.914.)

### Run-to-run variance (must be aware)
Measured distribution over 8 repetitions with the same seed (5436):

| | min | mean | max |
| --- | --- | --- | --- |
| cos_sim | 0.61 | 0.86 | **0.96** |

A single low result is not a failure — **run several times and take the max.**
`select_best_gptq.py` from §4 automates this selection.

---

## 7. File Structure

```
int4quant/
├── README.md
├── requirements.txt
├── .gitignore                         # excludes *.onnx, *.safetensors, *.npy, quant dirs
├── quantize_matmul.py                 # Stage 0: build q4_k template
├── run_gptq.py                # Stage 1: GPTQ (in_feature default / out_feature)
├── safetensors_to_onnx_qdq.py # Stage 2: safetensors → QDQ ONNX (+auto template)
├── quant_utils.py                     # shared utils (bf16→fp32, etc.), used by test_snr
├── test_snr.py                        # Stage 3a: logit 6-metric
├── compare_onnx_generation.py         # Stage 3b: generation comparison
├── select_best_gptq.py                # Stage 1+2+3 automation + best selection
├── extract_lm_head_embeddings.py      # (optional) extract quantized emb
└── onnx/                              # (gitignore) location of binaries such as model.onnx
```

| File | Stage | One-line description |
| --- | --- | --- |
| `quantize_matmul.py` | 0 | base ONNX → q4_k QDQ template |
| `run_gptq.py` | 1 | GPTQ INT4 (asymmetric) in/out_feature → safetensors |
| `safetensors_to_onnx_qdq.py` | 2 | safetensors + template → QDQ ONNX (in/out_feature) |
| `test_snr.py` | 3a | logit 6-metric vs base |
| `compare_onnx_generation.py` | 3b | base vs quant generation comparison |
| `select_best_gptq.py` | 1-3 | repeat N times → pick highest cos_sim |
| `extract_lm_head_embeddings.py` | (optional) | quantized lm_head → token_embeddings.npy |
| `quant_utils.py` | shared | bf16→fp32 conversion, etc. |

---

## 8. Common Problems

- **`ONNX template not found`**: no template. Pass `--base-onnx onnx/model.onnx` to Stage 2
  and it is generated automatically.
- **`No DequantizeLinear -> ... -> MatMul entries found`**: the `--onnx-template` is a graph
  without QDQ structure (e.g. a plain fp32 model.onnx). Use the q4_k template, or let it
  auto-generate via `--base-onnx`.
- **Low accuracy (cos_sim < 0.9)**: normal variance (§6). Run several times with
  `select_best_gptq.py` and pick the best.
- **`bf16`-related ORT error**: the onnxruntime CPU EP cannot run bf16. `test_snr.py` and
  `compare_onnx_generation.py` internally convert to fp32 via
  `quant_utils.convert_model_to_fp32()` before running, so this is usually not a problem.
- **No CUDA GPU / running on a Mac**: use `--device cpu` (§3.1). `--device auto` already
  picks cpu when no CUDA GPU is present. To pick a specific CUDA GPU, use `--gpu <idx>`
  (`run_gptq.py`) or `select_best_gptq.py --gpus`.
- **`NotImplementedError: Offload of type mps ...`**: the `compressed_tensors` offload
  backend has no MPS path. The scripts already hide MPS and run on cpu, so use
  `--device cpu` (or `auto`) on Apple Silicon — do not force mps.
- **`gemma-3-270m-it` gated / 401 on download**: accept the license on the HF model page
  and `huggingface-cli login` once (§3.1). Once cached, `HF_HUB_OFFLINE=1` runs offline.

---

## 9. Sources

- Original Stage 1/2/3 scripts: adapted from the
  [llm-compressor](https://github.com/vllm-project/llm-compressor) library's
  `examples/quantization_w4a16/` (`gemma3.py`, `safetensors_to_onnx_qdq.py`,
  `quant_utils.py`, `test_q.py`, `compare_onnx_generation.py`).
- `select_best_gptq.py` was newly written in this repo to automate the "run several times,
  pick the best" workflow (the recipe is identical to `run_gptq.py`).
