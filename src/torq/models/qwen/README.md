# Qwen export for Torq

Export pipeline for **Qwen3-0.6B** and **Qwen2.5-0.5B-Instruct**. One command
converts the Hugging Face model to a static, INT4 weight-quantized model and
compiles it to Torq VMFBs.

## Models

| `-s, --model-size` | Source (Hugging Face) | Torq model (Hugging Face) | SL2610 chat demo |
|---|---|---|---|
| `qwen3-0.6b` (default) | `Qwen/Qwen3-0.6B` | – (no GPTQ weights yet; round-to-nearest INT4) | – |
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | `Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq` | 11.3 tok/s |

## Quick start

```sh
torq-export-model qwen -s qwen2.5-0.5b-instruct
```

The command does these steps:

1. It exports the Hugging Face model to ONNX with optimum.
2. It makes a static single-token decoder with a 256-slot KV cache.
3. It moves the token embeddings to the host (`token_embeddings.npy`), trims
   the output vocabulary to Latin text, punctuation and digits
   (`token_id_lut.npy`) and writes the LM head as a separate model.
4. It rewrites attention: grouped-query attention reads the KV heads
   directly, without a `repeat_kv` copy of the cache.
5. It quantizes all MatMul weights of the transformer and the LM head to INT4
   `DequantizeLinear` (block size 32, bf16 scales). The INT4 weights come from
   GPTQ, downloaded from the `gptq/` folder of the Torq model repository.
6. It converts the model to bf16 and int32 and compiles `transformer.vmfb` and
   `lm_head.vmfb` with the flags in [`compile_flags.json`](compile_flags.json).

The Torq model repositories are private. Log in with `hf auth login` first.

### Compiler

torq-compiler 2.2.1 compiles the model. The measured speeds need a Torq
compiler with input prefetch and `--torq-prefetch-matmul-tile-budget-percent`
(torq-compiler-dev `wip/prefetch-list-scheduler`, a74ffddd or newer). Point
the export at it:

```sh
torq-export-model qwen -s qwen2.5-0.5b-instruct \
    --use-binary --compiler-path /path/to/iree-build/third_party/iree/tools/torq-compile
```

The GPTQ weight loader is in the torq-compiler Python package
(`torq.lab.quantization.onnx.weights`, torq-compiler-dev `wip/weights-quant-gptq`).
torq-compiler 2.2.1 does not have it yet. With 2.2.1, use `--no-gptq`
(round-to-nearest INT4).

## Compile flags

[`compile_flags.json`](compile_flags.json) has the flags for all components
(`flags`) and the extra flags per component (`transformer`, `lm_head`). The
flags in `transformer_if_supported` are added only when the compiler has them
(it checks `torq-compile --help-hidden`). Flags after `--compile-flags` come
last and take precedence.

- `--torq-disable-slicing`: with LinalgSlicing on, the board is 4–8× slower
  than the compile-time estimate.
- `--torq-prefetch-matmul-tile-budget-percent=33` (transformer only): at the
  default budget (50 %) a decode MatMul tile fills LRAM, and the next weight
  tile cannot load during compute. At 33 % prefetch hides 90 % or more of
  the compute.
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
| `--keep-individual-kv-io` | Keep separate K and V cache tensors. The SL2610 runtime allows at most 64 dispatch bindings |
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

The [torq-examples](https://github.com/synaptics-torq/torq-examples) `qwen2_5`
demo runs this directory.

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
