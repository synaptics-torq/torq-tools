# Model export guide

Common setup, flags and artifact conventions for `torq-export-model`. Exporters expose different subsets of these options; run `torq-export-model <model> --help` for the supported flags and defaults.

Model recipes: [Qwen](qwen/README.md), [Gemma 3 / FunctionGemma](gemma3/README.md), [LFM2.5](liquid/README.md), and [LFM2-VL](liquid/README_VL.md).

## Setup and pipeline

Install the project in a virtual environment as described in [CONTRIBUTING.md](../../../CONTRIBUTING.md#development-environment). Authenticate with `hf auth login` or set `HF_TOKEN` for private or gated Hugging Face repositories. Source models are downloaded on first use unless a local source is supplied.

A normal run loads or exports source ONNX, makes the model static, applies graph edits, optionally quantizes or converts dtypes, then compiles to Torq VMFB. Exporters that support validation compare the edited model with the source. See the model guides for supported validation, defaults and limitations.

## Common export flags

| Flag | Meaning |
|---|---|
| `-s, --model-size` | Select a checkpoint; choices and defaults depend on the model |
| `-t, --max-gen-tokens N` | Static decoder context / KV-cache capacity (normally 256) |
| `--instruct-model` | Select an instruct-tuned checkpoint, where available |
| `--hf-repo`, `--hf-repo-subdir` | Override the source repository and optional subdirectory |
| `--onnx-source-dir DIR` | Use local source ONNX and skip downloading; include configuration, tokenizer metadata and chat templates required by the model |
| `--models-dir DIR` | Source and export base directory (default: `models`); see model guides for path conventions |
| `--convert-dtypes` | Convert ONNX float tensors to bf16 and int64 to int32 before compilation; standard compiler flag sets already convert dtypes, so this is optional |
| `--extract-embeddings` / `--no-extract-embeddings` | Move the embedding table to host-side `token_embeddings.npy`; enabled by default where offered |
| `--trim-vocab` | Keep selected token groups and safety tokens; emit `token_id_lut.npy` |
| `--trim-vocab-groups` | Groups to keep: `latin`, `punct`, `digits`, `digits-non-latin`, `other` (default: `latin punct digits`) |
| `--split-lm-head` / `--no-split-lm-head` | Write `transformer.onnx` (hidden states) and `lm_head.onnx` (hidden states → logits) separately; enabled by default where offered |
| `--batch-prefill N` | Also emit fixed-shape `transformer_prefill.onnx` (default: 64, `0` disables); requires a split LM head |
| `--keep-individual-kv-io` | Keep separate K and V tensors instead of combined caches; target dispatch-binding limits may prevent this |
| `--no-read-only-kv-cache` | Use the earlier head-major caches written back whole each step; see [Read-only KV cache](#read-only-kv-cache) |
| `--dynamic-models` | Export dynamic CPU models; disable static-only options such as split LM head and batched prefill when required |
| `--weight-quantization {int4,int8,none}` | Block-size-32 DequantizeLinear MatMul weight quantization; Qwen and Gemma default to INT4 |
| `--gptq` / `--no-gptq` | Use hosted GPTQ INT4 weights for a supported default source, or round-to-nearest weights |
| `--gptq-checkpoint DIR` | Use a local compressed-tensors GPTQ INT4 checkpoint (group size 32); it must belong to the source checkpoint |
| `--dynamic-quantize` | Dynamic INT8 quantization before compilation; disables weight-only quantization |
| `--dynamic-quantization-skip-model COMPONENT…` | Copy selected components through without dynamic quantization |
| `--skip-torq` | Stop after ONNX processing |
| `--skip-validation` | Skip ONNX Runtime validation of edited models |
| `--compile-flags …` | Extra compiler flags, placed last on the command line; override the validated model flags |

Hosted GPTQ weights must match the source model. Source overrides use round-to-nearest unless you supply a matching local checkpoint. See the model guide for supported hosted checkpoints and [Qwen calibration](qwen/int4quant/README.md) for a calibration recipe.

## Graph-edit controls

Exporters supporting graph-edit customization accept the following options:

| Flag | Meaning |
|---|---|
| `--view-graph-edits` | Print the merged graph-edit plan and exit before export or GPTQ downloads |
| `--apply-graph-edit NAME[:ARGS]` | Override an edit or append a registered edit; repeatable, with YAML arguments |
| `--exclude-graph-edit NAME` | Exclude an edit by registered class name; repeatable |
| `--apply-graph-edits-from-file FILE` | Load edits from YAML |
| `--exclude-graph-edits-from-file FILE` | Load exclusions from YAML |
| `--dump-after-edit[=EDITS]` | Save structural ONNX snapshots after all edits or a comma-separated list of edit names |

Precedence is command line > file > defaults. Overrides retain the default edit's position, new edits append once per editor, and exclusions apply last.

## Compiler setup and flags

Use the compiler version validated by the model's `compile_flags.json`. Each version entry has `common` flags and optional component-specific groups such as `transformer` and `lm_head`. The exporter selects the installed version's entry; an unknown version warns and compiles without the model flags. User `--compile-flags` come last.

The compiler Python package provides the compilation API and weight quantization tools. For binary compilation fallback, set `TORQ_COMPILER_PATH=/path/to/torq-compile` (or use `--compiler-path`); otherwise `torq-compile` is resolved on `PATH`.

Each variant records its effective flags in `compile_flags.json`. The standalone driver reuses that snapshot:

```sh
python -m torq.utils.compile /path/to/transformer.onnx -o /path/to/transformer.vmfb
```

For slow compiles, append `--torq-show-pass-progress` to `--compile-flags`.

## Output variants and runtime assets

Model guides show the exact directory trees. Topology selectors such as vocabulary trimming and LM-head splitting get separate paths. Compilation uses the selected base, quantized or converted variant; it does not rebuild other variants' compiled files. Regenerating a variant removes its stale compiled artifacts, while compiled base models can survive a run that uses the base ONNX only as input to quantization. Always deploy the artifacts from the selected variant together.

The shared exporter puts `.vmfb` and imported `.mlir` files in that variant's `compiled/` directory. The `.mlir` files are diagnostic artifacts and are unnecessary on the board. Qwen uses its own Torq output layout, shown in its guide.

Configuration, tokenizer files, chat templates, embeddings and vocabulary tables accompany the variants. When compiler flags include `--torq-convert-io-dtype`, `compiled/` receives runtime-dtype tables (bf16 embeddings and int32 token IDs); variant-root tables retain the host ONNX dtype. Ship the tables beside the VMFB that consumes them.

## Batched prefill

Gemma and Liquid can emit a fixed-N-token prefill companion alongside the single-token decoder. It accepts `[1, N]` token IDs or `[1, N, hidden]` embeddings and a `[1, 1]` start position, updates N cache rows and returns the final position's output. Decode and prefill share cache state. The runner loads a neighboring prefill model automatically, processes full prompt chunks through it, and uses the decode model for the remainder and generation. For VMFB runners that lack shape metadata, supply `prefill_size=` explicitly. Validation compares chunked inference with the unedited dynamic source.

The Gemma prefill model shares the separate LM head. Liquid's prefill stays fused and returns logits directly; its short-conv caches are updated along with attention caches.

## Read-only KV cache

Qwen, Gemma 3, LFM2.5 and SmolLM2 support read-only KV caches. Each combined cache is a time-major `[1, L, 2 * kv, D]` input (K heads, then V heads). The `present.*.key_value` output contains only the step's new rows, which the runtime writes at `position_ids` before the next step. Decode outputs one row; batched prefill outputs N.

`--no-read-only-kv-cache` restores the head-major, full-writeback layout expected by torq-examples demos before v2.4.0. Gemma sliding layers can use bounded ring caches rather than the full context capacity.

## Validation

Validate fp32 ONNX with ONNX Runtime. Its CPU MatMul kernel cannot execute bf16 ONNX, so validate converted models on the target or through the appropriate host casting tools. Check model-specific validation limitations before relying on the export result.

## Standalone weight quantization

Quantization can also be a **separate step** applied to the fp32 static ONNX, using the `torq-quantize-model` tool from the `torq-compiler` dependency (`torq.lab.quantization.onnx`; see the torq-compiler user manual for the full flag reference). It quantizes MatMul weights to int8 / int4 / bf16 and can either emit a DequantizeLinear (DQL) model or a single dequantized-bf16 model ready to compile.

### 1. Export the fp32 static ONNX (quantization input)

Skip dtype conversion and compilation to keep the fp32 static graph and the host tables used by sensitivity analysis. For example, with Gemma:

```sh
torq-export-model gemma3 \
    --instruct-model \
    --trim-vocab \
    --weight-quantization none \
    --skip-torq
# fp32 ONNX: models/<repo>/export/trim/split_lm_head/<dtype>/static/ (transformer.onnx, lm_head.onnx, ...)
```

### 2. Quantize

Quantize each model of the export directory (`transformer.onnx`, `transformer_prefill.onnx`, `lm_head.onnx`) the same way:

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

Output modes: `--bits N` → DQL model; `--bits N --dequantize-weights` → dequantized bf16 model; `--bits 16` → pure fp32→bf16 (no quantization).

### 3. Per-layer sensitivity → mixed int8/int4 config (optional)

`analyze` measures each MatMul's quantization impact (KL divergence vs fp32) and writes a per-layer quantization config. It needs the `token_embeddings.npy` (and `token_id_lut.npy` for trimmed-vocab models) from the export:

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

For a standalone compile (outside the pipeline), both DQL and dequantized-bf16 outputs are directly importable by IREE.

The [compile helper](#compiler-setup-and-flags) loads the validated flags recorded in the export directory:

```sh
python -m torq.utils.compile transformer_mixed_bf16.onnx -o transformer.vmfb
```

In a torq-compiler-only environment (no torq-tools), invoke `torq-compile` with each flag from `compile_flags.json` in the export dir.

When the saved flags include `--torq-convert-io-dtype`, a standalone compile also needs matching runtime tables (see [runtime assets](#output-variants-and-runtime-assets)). It does not stage them as the export pipeline does. To convert tables manually:

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
