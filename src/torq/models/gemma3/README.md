# Gemma-3 (270M-IT) export for Torq

Export pipeline for **Gemma-3 270M** (and 1B), converting the HuggingFace model
to a static VMFB for the Torq board. By default the MatMul weights are **INT4**
(block size 32, bf16 activations): GPTQ weights for the 270M instruct model,
round-to-nearest for the other sources. The model is compiled with the
stream-schedule prefetch (`--torq-prefetch-algorithm=stream-schedule`). Native FP32 families with legacy full-cache I/O use constant-load prefetch to retain their shared weight tiling.

## Source model

The source is downloaded automatically from HuggingFace via Optimum on first run:

| `--model-size` / `--instruct-model` | HF repo |
|---|---|
| `270m` | `google/gemma-3-270m` |
| `270m --instruct-model` | `google/gemma-3-270m-it` |
| `1b`   | `google/gemma-3-1b-pt` |
| `1b --instruct-model` | `google/gemma-3-1b-it` |

> Upstream publishes the base 1B checkpoint as `-pt`, unlike 270M which has a
> bare repo, so plain `-s 1b` resolves to `google/gemma-3-1b-pt`.

Override the source with `--hf-repo <repo>` (optionally `--hf-repo-subdir`) to
point at a pre-exported ONNX repo, or `--onnx-source-dir <dir>` to use a local
ONNX and skip the download entirely.

> Note: this module exports the **fp32** model and quantizes it itself. It does
> **not** consume pre-quantized (MatMulNBits) ONNX.

## FunctionGemma

FunctionGemma uses the Gemma 3 architecture and shares this exporter. Select it with `--functiongemma` (270M only):

```sh
# Published checkpoint
torq-export-model gemma3 --functiongemma

# Your fine-tuned checkpoint
torq-export-model gemma3 --functiongemma --hf-repo org/my-functiongemma

# Local ONNX export of your checkpoint
torq-export-model gemma3 --functiongemma --onnx-source-dir ./functiongemma-onnx
```

The default source is `google/functiongemma-270m-it` (`--instruct-model` is implied). A source override is optional. Google recommends fine-tuning FunctionGemma for your function-calling task for best accuracy; see the [model card](https://huggingface.co/google/functiongemma-270m-it). Local ONNX sources must include `config.json`, `tokenizer.json`, and the source tokenizer metadata and chat template (`tokenizer_config.json` with an embedded template or a separate `chat_template.jinja`).

Export validation compares a tool-call prompt, including special function-call markers, against the original ONNX using the source checkpoint's chat template, a developer instruction, and a sample tool definition. It checks export equivalence, not task accuracy. The full prompt can span multiple prefill chunks; an insufficient context limit raises an error instead of truncating the tool definition. Runners honor every stop token in the source configuration, including FunctionGemma's start-of-function-response token. Tokenizer metadata and chat templates are preserved in exported, quantized, converted, and compiled directories. The read-only KV cache and other Gemma export defaults also apply.

INT4 quantization uses round-to-nearest unless you supply `--gptq-checkpoint` for this checkpoint. The hosted ordinary Gemma GPTQ weights are never selected for FunctionGemma. Programmatic `Gemma3Static` and `Gemma3Dynamic` runners accept `tools=[...]` with Hugging Face tool schemas and use the local chat template. The existing text-only inference CLI does not accept tool definitions or execute function calls.

## Pipeline overview

```
HuggingFace (google/gemma-3-*)
  → Optimum ONNX export (fp32)
  → make static (fixed KV cache, seq_len = --max-gen-tokens, default 256)
  → post-static patches (fuse ops, combine KV I/O, optional vocab trim / lm-head split)
  → weight quantization (default int4, GPTQ for 270m instruct) and conversion to bf16
  → compile to VMFB (IREE + Torq backend, stream-schedule prefetch)
```

## Quick start

From the torq-tools virtual environment:

```sh
torq-export-model gemma3 --instruct-model --trim-vocab
```

This is the model of the torq-examples `gemma3` demo
(`Synaptics/gemma-3-270-GPTQ-int4-it-torq`). The command downloads the source
and the GPTQ INT4 weights (the `gptq/` folder of that repository), builds the
static model with a trimmed vocabulary and a separate LM head, quantizes the
weights to INT4, and compiles `transformer.vmfb`, `transformer_prefill.vmfb`
and `lm_head.vmfb.trim` (the trimmed-vocabulary LM head). Equivalent module form: `python -m torq.models.gemma3.export …`.

The `.vmfb` files and the host files (`token_embeddings.npy` in bf16,
`token_id_lut.npy` in int32, `tokenizer.json`, `config.json`) are in
`models/google/gemma-3-270m-it/export/trim/split_lm_head/quantized/static/compiled/`.
Copy this directory to the board and pass its `transformer.vmfb` with `-m` to
the torq-examples `gemma3` demo.

Other variants:

```sh
# bf16 weights (no quantization)
torq-export-model gemma3 --instruct-model --trim-vocab --weight-quantization none --convert-dtypes

# INT8 weights
torq-export-model gemma3 --instruct-model --trim-vocab --weight-quantization int8
```

To compile the float variant, use `torq-export-model gemma3 --instruct-model --trim-vocab --weight-quantization none`. A run compiles only its selected variant: the default INT4 run regenerates the intermediate `fp32/static` ONNX files, but preserves any `fp32/static/compiled` files from earlier float runs. Those compiled files can therefore have older input shapes and KV layouts than the adjacent ONNX files. The selected variant emits separate `transformer.vmfb` and `transformer_prefill.vmfb` files for decoding and prefill.

If the Torq compiler Python API is not installed in the venv, point the fallback
at a `torq-compile` binary first:

```sh
export TORQ_COMPILER_PATH=/path/to/torq-compile
```

## Key export flags

| Flag | Description |
|---|---|
| `-s, --model-size {270m,1b}` | Model size (default: `270m`) |
| `--instruct-model` | Use the instruct-tuned (`-it`) variant |
| `--functiongemma` | Export FunctionGemma 270M using its tool chat template; source overrides remain optional |
| `-t, --max-gen-tokens N` | Static sequence length / KV-cache size (default: 256) |
| `--convert-dtypes` | Convert the export to the dtypes Torq supports: float → bf16 **and** int64 → int32 |
| `--extract-embeddings` | Extract the token-embedding table to `token_embeddings.npy` (model input becomes an embedding vector). **On by default** — opt out with `--no-extract-embeddings` |
| `--trim-vocab` | Trim the static-export vocab to selected token groups (+ safety tokens); emits `token_id_lut.npy` |
| `--trim-vocab-groups {latin,punct,digits,digits-non-latin,other}` | Groups to keep with `--trim-vocab` (default: `latin punct digits`) |
| `--split-lm-head` | Emit the LM head as a separate `lm_head.onnx`; the main model is then written as **`transformer.onnx`** and outputs hidden states instead of logits. **On by default** — opt out with `--no-split-lm-head` |
| `--batch-prefill N` | Also emit a fixed-shape `transformer_prefill.onnx` that processes N tokens (default: `64`; pass `0` to disable). Requires `--split-lm-head` (on by default) |
| `--keep-individual-kv-io` | Keep separate key/value tensors instead of combining KV I/O |
| `--no-read-only-kv-cache` | Export the pre-2.4 KV caches (head-major, written back whole every step). By default the time-major caches are read-only inputs and the models output only the new rows; see the [Qwen README](../qwen/README.md#read-only-kv-cache) |
| `--hf-repo / --hf-repo-subdir` | Override the HuggingFace source repo |
| `--onnx-source-dir DIR` | Use a local source ONNX (skips download) |
| `--models-dir DIR` | Base directory for source + export models (default: `models`) |
| `--dynamic-models` | Export dynamic (CPU) models instead of static. The static-only options above are on by default, so a dynamic run also needs `--no-split-lm-head --batch-prefill 0` |
| `--dynamic-quantize` | Dynamically quantize the model to 8-bit integer in-pipeline, before compiling. Turns off `--weight-quantization` |
| `--weight-quantization {int4,int8,none}` | Quantize the MatMul weights (block size 32) in-pipeline to a DQL model, before compiling (default: `int4`). See [Weight quantization](#weight-quantization-int4--int8) |
| `--gptq / --no-gptq` | With `int4` and the default 270m instruct source, take the INT4 weights from the hosted GPTQ checkpoint (default: on). Other sources use round-to-nearest |
| `--gptq-checkpoint DIR` | Use the GPTQ INT4 weights in DIR (compressed-tensors safetensors) instead of the hosted ones |
| `--skip-torq` | Stop after ONNX; skip the VMFB compile |
| `--skip-validation` | Skip ORT validation of edited ONNX |
| `--compile-flags …` | Extra flags forwarded verbatim to `torq-compile` (must be last). Takes precedence over values set by [`compile_flags.json`](compile_flags.json) |

`--trim-vocab` and `--split-lm-head` change the output path (see below).

## Output layout

```
models/<repo>/export/<full|trim>/<unified|split_lm_head>/
    <dtype>/static/model.onnx           ← fp32 static ONNX (quantization input)
    quantized/static/model.onnx         ← DQL model (--weight-quantization, the default, or --dynamic-quantize)
    converted/static/model.onnx         ← bf16 model (--weight-quantization none --convert-dtypes)
    …/static/token_embeddings.npy       ← with --extract-embeddings
    …/static/token_id_lut.npy           ← with --trim-vocab
    <variant>/static/compiled/model.vmfb ← compiled VMFB, co-located in the
                                            variant dir that is compiled
```

The compiled artifacts (`.vmfb` + `.mlir`) live in a `compiled/` subdirectory
of the variant that is actually compiled — `quantized/` with weight quantization
(the default) or `--dynamic-quantize`, `converted/` with `--convert-dtypes`, else
the base `<dtype>/` dir. Each
variant dir is self-contained (ONNX + runtime assets + `compiled/`), so it can
be deployed as-is; regenerating a variant's ONNX wipes its stale `compiled/`
alongside it, so a compiled model can never outlive the source it was built
from.

`<full|trim>` follows `--trim-vocab`; `<unified|split_lm_head>` follows
`--split-lm-head` (default: `split_lm_head`). The exact paths are printed at
the end of the run.

With `--split-lm-head` the model file is **not** `model.onnx` — each directory
above holds `transformer.onnx` (hidden-states output) plus `lm_head.onnx`
(hidden states → logits) instead:

```
models/<repo>/export/<full|trim>/split_lm_head/<dtype>/static/
    transformer.onnx                    ← the decoder, outputs last_hidden_states
    transformer_prefill.onnx            ← with --batch-prefill; processes N tokens and outputs the last hidden state
    lm_head.onnx                        ← standalone LM head
```

Inference and export validation pick up both companion models automatically when
they sit next to the transformer. Full `N`-token prompt chunks run through
`transformer_prefill.onnx`; any remainder and generated tokens use
`transformer.onnx`, with both paths sharing `lm_head.onnx`.

## Weight quantization (int4 / int8)

### INT4 with GPTQ (one command, the default)

The exporter quantizes the MatMul weights in the same run. For the 270M
instruct source it uses GPTQ INT4 weights (asymmetric, group size 32 along the
input features) from the `gptq/` folder of `Synaptics/gemma-3-270-GPTQ-int4-it-torq`
instead of round-to-nearest weights:

```sh
torq-export-model gemma3 --instruct-model --trim-vocab
```

- The LM head is split (default). The vocabulary is trimmed and the exporter
  also quantizes the trimmed LM head rows with the GPTQ weights.
- For each model, the log shows a "Quantization summary" line: KL divergence,
  cosine similarity and max absolute error of the logits against the fp32 model.
- Output: `models/<repo>/export/trim/split_lm_head/quantized/static/`
  (`transformer.onnx`, `transformer_prefill.onnx`, `lm_head.onnx`) and
  `compiled/` (the `.vmfb` files and the host files: `.npy` in the runtime
  dtype, `tokenizer.json`, `config.json`).
- The batch-prefill model (`transformer_prefill.vmfb`, `--batch-prefill 64` by
  default) halves the time to the first token for long prompts. Its
  "Quantization summary" line shows a high value: the check feeds it inputs that
  do not match a prefill chunk, so it does not measure its accuracy.

To use your own GPTQ checkpoint (compressed-tensors `pack-quantized` safetensors,
group size 32), pass `--gptq-checkpoint DIR`. Other sources (another `--hf-repo`,
the base model, `-s 1b`) and `--no-gptq` use round-to-nearest INT4 weights.
`--dynamic-quantize` turns the weight quantization off.

### INT4 / INT8 with torq-quantize-model

Quantization can also be a **separate step** applied to the fp32 static ONNX, using the
`torq-quantize-model` tool from the `torq-compiler` dependency
(`torq.lab.quantization.onnx`; see the torq-compiler user manual for the full
flag reference). It quantizes MatMul weights to int8 / int4 / bf16 and can
either emit a DequantizeLinear (DQL) model or a single dequantized-bf16
model ready to compile.

### 1. Export the fp32 static ONNX (quantization input)

Skip the bf16 conversion and the compile so you keep the fp32 static graph
(plus embeddings / vocab LUT used by sensitivity analysis):

```sh
torq-export-model gemma3 \
    --instruct-model \
    --trim-vocab \
    --weight-quantization none \
    --skip-torq
# fp32 ONNX: models/<repo>/export/trim/split_lm_head/<dtype>/static/ (transformer.onnx, lm_head.onnx, ...)
```

### 2. Quantize

Quantize each model of the export directory (`transformer.onnx`,
`transformer_prefill.onnx`, `lm_head.onnx`) the same way:

```sh
# int8 (ORT-matching asymmetric uint8, block_size 32) as a DQL model
torq-quantize-model weights -i transformer.onnx -o transformer_int8_dql.onnx --bits 8

# int4 (signed [-8,7], block_size 32)
torq-quantize-model weights -i transformer.onnx -o transformer_int4_dql.onnx --bits 4

# Dequantized bf16 (quant error baked in, single bf16 model ready to compile).
# Scales are truncated to bf16 BEFORE dequant so weights match runtime compute.
torq-quantize-model weights -i transformer.onnx -o transformer_int8_bf16.onnx \
    --bits 8 --dequantize-weights

# Keep the lm_head at full precision
torq-quantize-model weights -i transformer.onnx -o out.onnx --bits 4 --skip-layers lm_head
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
torq-quantize-model analyze weights -i transformer.onnx -o sensitivity.json \
    --config-output quant_config.json \
    --embeddings token_embeddings.npy \
    --token-lut token_id_lut.npy \
    --bits 4 8

# then quantize from the config (mixed precision), dequantized to bf16
torq-quantize-model weights -i transformer.onnx -o transformer_mixed_bf16.onnx \
    --config quant_config.json --dequantize-weights
```

### 4. Compile the quantized model

For a standalone compile (outside the pipeline), both DQL and dequantized-bf16
outputs are directly importable by IREE.

Every export dir carries its validated `torq-compile` flags: the export writes
the flags of [`compile_flags.json`](compile_flags.json) for the installed
compiler version into the export dir. The compile helper loads them:

```sh
python -m torq.utils.compile transformer_mixed_bf16.onnx -o transformer.vmfb
```

In a torq-compiler-only environment (no torq-tools), invoke `torq-compile` with
each flag from `compile_flags.json` in the export dir.

The flags include `--torq-convert-io-dtype`, so the compiled model takes bf16
embeddings and int32 token IDs. The one-command export writes the host files in
these dtypes. A standalone compile keeps them in fp32 and int64; the demo still
runs with them (the runner casts the embedding row), but you can convert them
(optional). This halves the size of `token_embeddings.npy` and makes the files
match the hosted ones:

```sh
python -c "import numpy as np, ml_dtypes; np.save('token_embeddings.npy', np.load('token_embeddings.npy').astype(ml_dtypes.bfloat16)); np.save('token_id_lut.npy', np.load('token_id_lut.npy').astype(np.int32))"
```

### 5. Benchmark (optional)

Compare quantized variants over a standard question set:

```sh
python -m torq.utils.benchmark run \
    -m model_int8.vmfb --instruct-model -o results_int8.json
python -m torq.utils.benchmark compare \
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

### Experimental native batch prefill

Gemma3 270M and FunctionGemma support explicit static families with `--native-batch-prefill 4,12,29,51,92,106,124 --extract-embeddings --split-lm-head`. Use an FP32 source and keep the context within the model’s sliding attention window. The default Gemma3-270M IT export uses INT4 GPTQ weights and the updated time-major, read-only KV cache, returning only the new rows for each chunk. Other checkpoints retain their normal GPTQ selection rules. Use `--weight-quantization none` for FP32 weights, and `--no-read-only-kv-cache` for the legacy full-cache layout. INT8, dynamic quantization, explicit dtype conversion, dynamic shapes and layer splitting are unsupported in this native path; ordinary separate-prefill exports remain available.

See the shared [native batch-prefill guide](../../model_export/batch_prefill.md) for export commands, compiler/runtime requirements, artifact layout, optional size probing, chunk selection and cache handling. FunctionGemma additionally needs checkpoint-specific validation.

The full FP32 Gemma3-270M IT family at context 256 with sizes 4,12,29,51,92,106,124 passes sequential ONNX hidden/cache comparisons and compiles into one eight-entry version-3 VMFB using the compiler core/runtime stack. BF16 device accuracy and performance remain separate validation gates. Fixed RoPE frequencies remain verified literals alongside position controls.
