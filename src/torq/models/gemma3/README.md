# Gemma-3 (270M-IT) export for Torq

See the [model export guide](../README.md) for setup, common flags, compiler configuration, output variants and cache behavior.

Export pipeline for **Gemma-3 270M** (and 1B), converting the HuggingFace model to a static VMFB for the Torq board. By default the MatMul weights are **INT4** (block size 32, bf16 activations): GPTQ weights for the 270M instruct model, round-to-nearest for the other sources. The model is compiled with the stream-schedule prefetch (`--torq-prefetch-algorithm=stream-schedule`).

## Source model

The source is downloaded automatically from HuggingFace via Optimum on first run:

| `--model-size` / `--instruct-model` | HF repo |
|---|---|
| `270m` | `google/gemma-3-270m` |
| `270m --instruct-model` | `google/gemma-3-270m-it` |
| `1b`   | `google/gemma-3-1b-pt` |
| `1b --instruct-model` | `google/gemma-3-1b-it` |

> Upstream publishes the base 1B checkpoint as `-pt`, unlike 270M which has a bare repo, so plain `-s 1b` resolves to `google/gemma-3-1b-pt`.

> Note: this module exports the **fp32** model and quantizes it itself. It does **not** consume pre-quantized (MatMulNBits) ONNX.

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

The default source is `google/functiongemma-270m-it` (`--instruct-model` is implied). A source override is optional. See the [model card](https://huggingface.co/google/functiongemma-270m-it) for fine-tuning guidance. Local ONNX sources must include `config.json`, `tokenizer.json`, and the source tokenizer metadata and chat template (`tokenizer_config.json` with an embedded template or a separate `chat_template.jinja`).

Export validation checks a tool-call prompt against the source ONNX; it does not measure function-calling task accuracy. Set the context limit large enough for the complete tool definition and prompt. The read-only KV cache and other Gemma defaults also apply.

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

This is the model of the torq-examples `gemma3` demo (`Synaptics/gemma-3-270-GPTQ-int4-it-torq`). The command downloads the source and the GPTQ INT4 weights (the `gptq/` folder of that repository), builds the static model with a trimmed vocabulary and a separate LM head, quantizes the weights to INT4, and compiles `transformer.vmfb`, `transformer_prefill.vmfb` and `lm_head.vmfb.trim` (the trimmed-vocabulary LM head). Equivalent module form: `python -m torq.models.gemma3.export …`.

The `.vmfb` files and the host files (`token_embeddings.npy` in bf16, `token_id_lut.npy` in int32, `tokenizer.json`, `config.json`) are in `models/google/gemma-3-270m-it/export/trim/split_lm_head/quantized/static/compiled/`. Copy this directory to the board and pass its `transformer.vmfb` with `-m` to the torq-examples `gemma3` demo.

Other variants:

```sh
# bf16 weights (no quantization)
torq-export-model gemma3 --instruct-model --trim-vocab --weight-quantization none

# INT8 weights
torq-export-model gemma3 --instruct-model --trim-vocab --weight-quantization int8
```

The selected variant emits separate `transformer.vmfb` and `transformer_prefill.vmfb` files for decoding and prefill. See [output variants](../README.md#output-variants-and-runtime-assets) before reusing compiled files from another run.

## Model-specific export flags

| Flag | Description |
|---|---|
| `-s, --model-size {270m,1b}` | Model size (default: `270m`) |
| `--instruct-model` | Use the instruct-tuned (`-it`) variant |
| `--functiongemma` | Export FunctionGemma 270M using its tool chat template; source overrides remain optional |

`--trim-vocab` and `--split-lm-head` change the output path (see below).

## Output layout

```
models/<repo>/export/<full|trim>/<unified|split_lm_head>/
    <dtype>/static/model.onnx           ← fp32 static ONNX (quantization input)
    quantized/static/model.onnx         ← DQL model (--weight-quantization, the default, or --dynamic-quantize)
    converted/static/model.onnx         ← bf16 model (--weight-quantization none --convert-dtypes)
    …/static/token_embeddings.npy       ← with --extract-embeddings
    …/static/token_id_lut.npy           ← with --trim-vocab
    <variant>/static/compiled/model.vmfb ← compiled VMFB
```

See [output variants and runtime assets](../README.md#output-variants-and-runtime-assets) for compilation and deployment rules.

With `--split-lm-head` the model file is **not** `model.onnx` — each directory above holds `transformer.onnx` (hidden-states output) plus `lm_head.onnx` (hidden states → logits) instead:

```
models/<repo>/export/<full|trim>/split_lm_head/<dtype>/static/
    transformer.onnx                    ← the decoder, outputs last_hidden_states
    transformer_prefill.onnx            ← with --batch-prefill; processes N tokens and outputs the last hidden state
    lm_head.onnx                        ← standalone LM head
```

See [batched prefill](../README.md#batched-prefill) for companion discovery and prompt processing.

## Weight quantization (int4 / int8)

### INT4 with GPTQ (one command, the default)

The exporter quantizes the MatMul weights in the same run. For the 270M instruct source it uses GPTQ INT4 weights (asymmetric, group size 32 along the input features) from the `gptq/` folder of `Synaptics/gemma-3-270-GPTQ-int4-it-torq` instead of round-to-nearest weights:

```sh
torq-export-model gemma3 --instruct-model --trim-vocab
```

- The LM head is split (default). The vocabulary is trimmed and the exporter also quantizes the trimmed LM head rows with the GPTQ weights.
- For each model, the log shows a "Quantization summary" line: KL divergence, cosine similarity and max absolute error of the logits against the fp32 model.
- Output: `models/<repo>/export/trim/split_lm_head/quantized/static/` (`transformer.onnx`, `transformer_prefill.onnx`, `lm_head.onnx`) and `compiled/` (the `.vmfb` files and the host files: `.npy` in the runtime dtype, `tokenizer.json`, `config.json`).
- The prefill component's quantization summary is not a prefill accuracy measurement. Validate complete prompts with the decode and prefill models together.

To use your own GPTQ checkpoint (compressed-tensors `pack-quantized` safetensors, group size 32), pass `--gptq-checkpoint DIR`. Other sources (another `--hf-repo`, the base model, `-s 1b`) and `--no-gptq` use round-to-nearest INT4 weights. `--dynamic-quantize` turns the weight quantization off.

For separate quantization, sensitivity analysis, mixed precision, compilation and benchmarking, see the [quantization recipe](../README.md#standalone-weight-quantization).

## Inference

Run the exported model (ONNX or VMFB) with a prompt:

```sh
torq-infer-model gemma3 -m /path/to/model.vmfb --instruct-model "Capital of France?"
# module form: python -m torq.models.gemma3.infer …
```

See the [validation notes](../README.md#validation) for bf16 limitations.
