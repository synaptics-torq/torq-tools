# Gemma-3 (270M-IT) export for Torq

Export pipeline for **Gemma-3 270M** (and 1B), converting the HuggingFace model
to a static **bf16** VMFB for the Torq board — with optional **weight
quantization** (int4 / int8) via the separate `torq-quantize-model` tool.

## Source model

The source is downloaded automatically from HuggingFace via Optimum on first run:

| `--model-size` / `--instruct-model` | HF repo |
|---|---|
| `270m` | `google/gemma-3-270m` |
| `270m --instruct-model` | `google/gemma-3-270m-it` |
| `1b`   | `google/gemma-3-1b` |

Override the source with `--hf-repo <repo>` (optionally `--hf-repo-subdir`) to
point at a pre-exported ONNX repo, or `--onnx-source-dir <dir>` to use a local
ONNX and skip the download entirely.

> Note: this module exports the **fp32** model and converts it to bf16. It does
> **not** consume pre-quantized (MatMulNBits) ONNX — for int4/int8 use the
> [Weight quantization](#weight-quantization-int4--int8) step below, which
> operates on the fp32 ONNX this pipeline produces.

## Pipeline overview

```
HuggingFace (google/gemma-3-*)
  → Optimum ONNX export (fp32)
  → make static (fixed KV cache, seq_len = --max-gen-tokens, default 256)
  → post-static patches (fuse ops, combine KV I/O, optional vocab trim / lm-head split)
  → [optional] weight quantization (int4/int8) — torq-quantize-model
  → convert to bf16 (--convert-dtypes)
  → compile to VMFB (IREE + Torq backend)
```

## Quick start — export a bf16 VMFB

```sh
cd torq-tools-dev
source .venv/bin/activate    # or: source /home/kshanmug/torq/.venv-torq-tools/bin/activate

torq-export-model gemma3 \
    --instruct-model \
    --extract-embeddings \
    --convert-dtypes
```

This downloads the source, builds the static model, converts to bf16, and
compiles to a VMFB. Equivalent module form: `python -m torq.models.gemma3.export …`.

If the Torq compiler Python API isn't installed in the venv, point the fallback
at the binary first:

```sh
export TORQ_COMPILER_PATH=/home/kshanmug/torq/iree-build/third_party/iree/tools/torq-compile
```

## Key export flags

| Flag | Description |
|---|---|
| `-s, --model-size {270m,1b}` | Model size (default: `270m`) |
| `--instruct-model` | Use the instruct-tuned (`-it`) variant |
| `-t, --max-gen-tokens N` | Static sequence length / KV-cache size (default: 256) |
| `--convert-dtypes` | Convert the export to the dtypes Torq supports: float → bf16 **and** int64 → int32 |
| `--extract-embeddings` | Extract the token-embedding table to `token_embeddings.npy` (model input becomes an embedding vector) |
| `--trim-vocab` | Trim the static-export vocab to selected token groups (+ safety tokens); emits `token_id_lut.npy` |
| `--trim-vocab-groups {latin,punct,digits,digits-non-latin,other}` | Groups to keep with `--trim-vocab` (default: `latin punct digits`) |
| `--split-lm-head` | Emit the LM head as a separate `lm_head.onnx`; the main model is then written as **`transformer.onnx`** and outputs hidden states instead of logits |
| `--keep-individual-kv-io` | Keep separate key/value tensors instead of combining KV I/O |
| `--hf-repo / --hf-repo-subdir` | Override the HuggingFace source repo |
| `--onnx-source-dir DIR` | Use a local source ONNX (skips download) |
| `--models-dir DIR` | Base directory for source + export models (default: `models`) |
| `--dynamic-models` | Export dynamic (CPU) models instead of static |
| `--skip-torq` | Stop after ONNX; skip the VMFB compile |
| `--skip-validation` | Skip ORT validation of edited ONNX |
| `--compile-flags …` | Extra flags forwarded verbatim to `torq-compile` |

`--trim-vocab` and `--split-lm-head` change the output path (see below).

## Output layout

```
models/<repo>/export/<full|trim>/<unified|split_lm_head>/onnx/
    <dtype>/static/model.onnx           ← fp32 static ONNX (quantization input)
    converted/static/model.onnx         ← bf16 model (--convert-dtypes)
    …/token_embeddings.npy              ← with --extract-embeddings
    …/token_id_lut.npy                  ← with --trim-vocab
models/<repo>/export/<full|trim>/<unified|split_lm_head>/torq/
    converted/static/model.vmfb         ← compiled VMFB
```

`<full|trim>` follows `--trim-vocab`; `<unified|split_lm_head>` follows
`--split-lm-head`. The exact paths are printed at the end of the run.

With `--split-lm-head` the model file is **not** `model.onnx` — each directory
above holds `transformer.onnx` (hidden-states output) plus `lm_head.onnx`
(hidden states → logits) instead:

```
models/<repo>/export/<full|trim>/split_lm_head/onnx/<dtype>/static/
    transformer.onnx                    ← the decoder, outputs last_hidden_states
    lm_head.onnx                        ← standalone LM head
```

Inference and export validation pick the `lm_head` up automatically when it sits
next to the transformer, so `torq-infer-model gemma3 -m …/transformer.onnx`
works unchanged.

## Weight quantization (int4 / int8)

Quantization is a **separate step** applied to the fp32 static ONNX, using the
`torq-quantize-model` tool (full docs:
[`torq/tools/quantization/weight_quantization/README.md`](../../tools/quantization/weight_quantization/README.md)).
It quantizes MatMul weights to int8 / int4 / bf16 and can emit either a
DequantizeLinear (DQL) model or a single dequantized-bf16 model ready to compile.

### 1. Export the fp32 static ONNX (quantization input)

Skip the bf16 conversion and the compile so you keep the fp32 static graph
(plus embeddings / vocab LUT used by sensitivity analysis):

```sh
torq-export-model gemma3 \
    --instruct-model \
    --extract-embeddings \
    --trim-vocab \
    --skip-torq
# fp32 ONNX: models/<repo>/export/trim/unified/onnx/<dtype>/static/model.onnx
```

### 2. Quantize

```sh
# int8 (ORT-matching asymmetric uint8, block_size 32) as a DQL model
torq-quantize-model quantize -i model.onnx -o model_int8_dql.onnx --bits 8

# int4 (signed [-8,7], block_size 32)
torq-quantize-model quantize -i model.onnx -o model_int4_dql.onnx --bits 4

# Dequantized bf16 (quant error baked in, single bf16 model ready to compile).
# Scales are truncated to bf16 BEFORE dequant so weights match runtime compute.
torq-quantize-model quantize -i model.onnx -o model_int8_bf16.onnx \
    --bits 8 --dequantize-weights

# Keep the lm_head at full precision
torq-quantize-model quantize -i model.onnx -o out.onnx --bits 4 --skip-layers lm_head
```

Output modes: `--bits N` → DQL model; `--bits N --dequantize-weights` →
dequantized bf16 model; `--bits 16` → pure fp32→bf16 (no quantization).

### 3. Per-layer sensitivity → mixed int8/int4 config (optional)

`analyze` measures each MatMul's quantization impact (KL divergence vs fp32) and
writes a per-layer config (this is the real equivalent of the "layer sensitivity"
workflow — it lives in `torq-quantize-model`, not a standalone script). It needs
the `token_embeddings.npy` (and `token_id_lut.npy` for trimmed-vocab models) from
the export:

```sh
torq-quantize-model analyze -i model.onnx -o sensitivity.json \
    --config-output quant_config.json \
    --embeddings token_embeddings.npy \
    --token-lut token_id_lut.npy \
    --bits 4 8

# then quantize from the config (mixed precision), dequantized to bf16
torq-quantize-model quantize -i model.onnx -o model_mixed_bf16.onnx \
    --config quant_config.json --dequantize-weights
```

### 4. Compile the quantized model

Both DQL and dequantized-bf16 outputs are directly importable by IREE. Compile
the quantized ONNX to a VMFB (from `torq-compiler-dev`):

```sh
./compile_v1.5.sh /path/to/model_mixed_bf16.onnx
```

Or via the tools' compile helper:

```sh
python -m torq.utils.compile model_mixed_bf16.onnx -o model.vmfb \
    --compile-flags --torq-hw=SL2610 --torq-disable-slicing \
    --torq-enable-annotate-tied-operands \
    --torq-enable-split-constants-optimization \
    --iree-flow-inline-constants-max-byte-length=300000000
```

> Do **not** pass `--torq-enable-torq-hl-tiling` — that flag was removed from
> `torq-compile` (tile-and-fuse is now default) and the build fails with it.

### 5. Benchmark (optional)

Compare quantized variants over a standard question set:

```sh
python -m torq.tools.quantization.weight_quantization.benchmark run \
    -m model_int8.vmfb --instruct-model -o results_int8.json
python -m torq.tools.quantization.weight_quantization.benchmark compare \
    -a results_int8.json -b results_mixed.json \
    --name-a "int8" --name-b "mixed int8/int4" -o comparison.md
```

## Inference

Run the exported model (ONNX or VMFB) with a prompt:

```sh
torq-infer-model gemma3 -m /path/to/model.vmfb --instruct-model "Capital of France?"
# module form: python -m torq.models.gemma3.infer …
```

bf16 ONNX cannot run through onnxruntime (no CPU bf16 MatMul kernel) — validate
correctness on the **fp32** ONNX, and use the VMFB (or the DQL/dequantized model)
on the board.
