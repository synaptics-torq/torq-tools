# Qwen export for Torq

Export pipeline for **Qwen3-0.6B** and **Qwen2.5-0.5B-Instruct**. One command
converts the Hugging Face model to a static, INT4 weight-quantized model and
compiles it to Torq VMFBs.

## Models

| `-s, --model-size` | Source (Hugging Face) | Torq model (Hugging Face) | SL2610 chat demo |
|---|---|---|---|
| `qwen3-0.6b` (default) | `Qwen/Qwen3-0.6B` | `Synaptics/Qwen3-0.6B-GPTQ-int4-torq` | 6.0–6.4 tok/s |
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | `Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq` | 11.3 tok/s |

## Quick start

```sh
torq-export-model qwen -s qwen3-0.6b
torq-export-model qwen -s qwen2.5-0.5b-instruct
```

The command does these steps:

1. It exports the Hugging Face model to ONNX with optimum.
2. It makes a static single-token decoder with a 256-slot KV cache.
3. It moves the token embeddings to the host (`token_embeddings.npy`), trims
   the output vocabulary to Latin text, punctuation and digits
   (`token_id_lut.npy`) and writes the LM head as a separate model.
4. It rewrites attention: grouped-query attention reads the KV heads
   directly, and one Where per layer writes the new token into the combined
   K/V cache.
5. It quantizes all MatMul weights of the transformer and the LM head to INT4
   `DequantizeLinear` (block size 32, bf16 scales). The INT4 weights come from
   GPTQ, downloaded from the `gptq/` folder of the Torq model repository.
6. It converts the model to bf16 and int32 and compiles `transformer.vmfb` and
   `lm_head.vmfb` with the flags in [`compile_flags.json`](compile_flags.json).

The Torq model repositories are private. Log in with `hf auth login` first.

### Compiler

The export needs torq-compiler 2.3.0 or newer. It compiles with input
prefetch, which loads the next weights while the NPU computes. The GPTQ
weight loader is in its Python package (`torq.lab.quantization.onnx.weights`).

## Start from another Hugging Face model

The hosted GPTQ weights belong to the default source of each `--model-size`.
For another model of the same architecture (for example a fine-tune), make
GPTQ weights for that model first:

```sh
# 1. GPTQ calibration in its own virtualenv (see int4quant/README.md), about
#    2 hours on an 8-core CPU
venv-gptq/bin/python src/torq/models/qwen/int4quant/run_gptq.py \
    --model-id <hf-repo> --device cpu --num-samples 256 --max-seq-len 1024 \
    --save-dir gptq-<name>

# 2. Export with these weights
torq-export-model qwen -s qwen2.5-0.5b-instruct --hf-repo <hf-repo> \
    --gptq-checkpoint gptq-<name>
```

Use the `-s` value with the same architecture as `<hf-repo>`. Without
`--gptq-checkpoint`, an export with another `--hf-repo` or with
`--onnx-source-dir` uses round-to-nearest INT4 and shows a warning. It does
not use the hosted weights, because they would replace the weights of your
model.

## Compile flags

[`compile_flags.json`](compile_flags.json) has the flags for all components
(`flags`) and the extra flags per component (`transformer`, `lm_head`). The
flags in `<component>_if_supported` are added when the compiler supports them
(the exporter checks `torq-compile --help-hidden`). Flags after
`--compile-flags` come last and take precedence.

- `--torq-disable-slicing`: the models are compiled without LinalgSlicing.
- `--torq-prefetch-algorithm=stream-schedule`: schedules the DMA transfers
  over the slice compute, so that most of the compute is hidden under the
  weight loads. Decode is DMA-bound, so this sets the token rate.
- `--torq-max-nss-programs-size=12582912` (transformer only): the NSS
  programs need about 11 MB.

## Key export flags

| Flag | Description |
|---|---|
| `-s, --model-size` | Model to export (see [Models](#models)) |
| `-t, --max-gen-tokens N` | Static sequence length / KV-cache size (default: 256) |
| `--split-lm-head` / `--no-split-lm-head` | Separate LM head model (default: on) |
| `--weight-quantization {int4,int8,none}` | DequantizeLinear weight quantization (default: int4) |
| `--gptq` / `--no-gptq` | Use the GPTQ INT4 weights from Hugging Face (default: on) |
| `--gptq-checkpoint DIR` | Local GPTQ checkpoint, made with [`int4quant/run_gptq.py`](int4quant/README.md) |
| `--split-model` | Also split the transformer into Part A and Part B |
| `--keep-individual-kv-io` | Keep separate K and V cache tensors. The SL2610 runtime allows at most 64 dispatch bindings, so this fails for Qwen3-0.6B |
| `--hf-repo`, `--onnx-source-dir` | Override the source model |
| `--skip-torq` | Stop after the ONNX export |
| `--compile-flags …` | Extra `torq-compile` flags (must be last) |

## Output layout

```text
models/<hf-repo>/export/torq/converted/static/
    transformer.vmfb
    lm_head.vmfb
    token_embeddings.npy
    token_id_lut.npy
    tokenizer.json, tokenizer_config.json, config.json, generation_config.json
```

The [torq-examples](https://github.com/synaptics-torq/torq-examples) `qwen3`
and `qwen2_5` demos run this directory.

## Split model

With `--split-model`, the Qwen transformer is divided in two halves. For
Qwen3-0.6B the split is at layer 14:

```text
Part A: layers 0–13
Part B: layers 14–27
LM head: separate
```

The split runtime is intended for memory-constrained Torq targets.

The `--split-vmfb-dir` inference option expects a self-contained directory
containing the split VMFBs and required runtime assets.

## Inference

Run a split Qwen runtime with:

```sh
torq-infer-model qwen \
    --split-vmfb-dir /path/to/static \
    --max-gen-tokens 256 \
    "What is machine learning?"
```

Multiple prompts can be supplied:

```sh
torq-infer-model qwen \
    --split-vmfb-dir /path/to/static \
    --max-gen-tokens 256 \
    "What is machine learning?" \
    "Explain neural networks."
```

Equivalent module form:

```sh
python -m src.torq.models.infer_model qwen \
    --split-vmfb-dir /path/to/static \
    --max-gen-tokens 256 \
    "What is machine learning?"
```

For a single Qwen VMFB, use `--model`:

```sh
torq-infer-model qwen \
    -m /path/to/model.vmfb \
    --max-gen-tokens 256 \
    "Hello"
```

## Notes

- The source ONNX models are large (external data, more than 2 GB of fp32
  weights). The exporter uses path-based ONNX shape inference. It skips the
  generic in-memory ONNX cleanup, which cannot serialize models of this size.
- `--weight-quantization int8` applies block-size-32 INT8 DequantizeLinear
  weights. GPTQ weights are only available for int4.
