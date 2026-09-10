# Qwen3-0.6B export for Torq

Export pipeline for **Qwen3-0.6B**, converting the ONNX model to a static
Torq-compatible model and compiling it to Torq VMFBs.

For memory-constrained Torq targets such as the Coralboard, the model can be
split into two transformer partitions and a separate LM head. Split exports
also support optional INT8 weight quantization to reduce the compiled model
and runtime memory footprint.

## Source model

The default source model is:

| Model | Hugging Face repo |
|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` |

Use `--hf-repo` / `--hf-repo-subdir` to override the source, or
`--onnx-source-dir DIR` to use a local ONNX model.

## Pipeline overview

```text
Qwen3-0.6B ONNX (fp32)
  -> make static
  -> extract token embeddings
  -> split transformer at layer 14
  -> split LM head
  -> path-based shape inference
  -> optional INT8 MatMul weight quantization
  -> convert fp32 -> bf16
  -> convert int64 -> int32
  -> compile with IREE + Torq backend
  -> Part A / Part B / LM head VMFBs
```

For large external-data Qwen models, path-based shape inference is used before
dtype conversion to preserve intermediate tensor dtype information required by
the BF16 conversion.

For split exports, optional INT8 weight quantization can be enabled with
`--weight-quantization int8`. The quantized path applies block-size-32 INT8
weight quantization to supported MatMul layers while retaining BF16
floating-point computation.

## Quick start — export a split VMFB runtime

From the `torq-tools-dev` repository:

```sh
torq-export-model qwen \
    --split-model \
    --convert-dtypes
```

To reduce the memory footprint of the split model, enable INT8 weight
quantization:

```sh
torq-export-model qwen \
    --split-model \
    --weight-quantization int8 \
    --convert-dtypes
```

`--weight-quantization int8` applies block-size-32 INT8 weight quantization
to supported MatMul weights in Part A, Part B, and the LM head. The remaining
floating-point computation is converted to BF16.

To use a local pre-exported ONNX model:

```sh
torq-export-model qwen \
    --onnx-source-dir models/Qwen/Qwen3-0.6B/source/fp32 \
    --split-model \
    --weight-quantization int8 \
    --convert-dtypes
```

This makes the model static, splits the transformer and LM head, applies INT8
weight quantization, performs the required BF16/INT32 conversion, and compiles
the resulting models to Torq VMFBs.

Equivalent module form:

```sh
python -m src.torq.models.qwen.export \
    --onnx-source-dir models/Qwen/Qwen3-0.6B/source/fp32 \
    --split-model \
    --weight-quantization int8 \
    --convert-dtypes
```

If the Torq compiler Python bindings are not installed in the active
environment, configure the Torq compiler according to the repository's
compiler setup.

## Key export flags

| Flag | Description |
|---|---|
| `-t, --max-gen-tokens N` | Static sequence length / KV-cache size (default: 256) |
| `--hf-repo` | Override the Hugging Face source repository |
| `--hf-repo-subdir DIR` | Subdirectory containing the model files |
| `--onnx-source-dir DIR` | Use a local source ONNX model |
| `--models-dir DIR` | Base directory for source and exported models |
| `--dynamic-models` | Export a dynamic model instead of a static model |
| `--convert-dtypes` | Convert fp32 -> bf16 and int64 -> int32 |
| `--split-model` | Split the transformer into Part A, Part B, and a separate LM head |
| `--weight-quantization int8` | Apply block-size-32 INT8 weight quantization to supported MatMul weights in the split model |
| `--keep-individual-kv-io` | Keep key/value cache tensors separate |
| `--skip-torq` | Stop after ONNX conversion |
| `--skip-validation` | Skip ONNX validation |
| `--compile-flags …` | Extra flags forwarded to `torq-compile` |

## Output layout

With `--split-model`, the compiled VMFB artifacts are placed under:

```text
models/Qwen/Qwen3-0.6B/export/torq/converted/static/
    transformer_part_A.vmfb
    transformer_part_B.vmfb
    lm_head.vmfb
```

The converted runtime tensor assets are generated under:

```text
models/Qwen/Qwen3-0.6B/export/onnx/converted/static/
    token_embeddings.npy
    token_id_lut.npy
```

Tokenizer and model configuration files are copied to the Torq output directory:

```text
models/Qwen/Qwen3-0.6B/export/torq/converted/static/
    tokenizer.json
    config.json
```

A deployable split runtime should package the three VMFBs together with:

```text
token_embeddings.npy
token_id_lut.npy
tokenizer.json
config.json
```

The exact output directories are printed during the export.

## Split model

With `--split-model`, the Qwen transformer is divided at layer 14:

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

Qwen3-0.6B uses a large external-data ONNX model. The exporter uses
path-based ONNX shape inference during dtype conversion to avoid the
in-memory protobuf size limitations of large models.

For split exports, `--weight-quantization int8` applies block-size-32 INT8
weight quantization to supported MatMul weights in Part A, Part B, and the
LM head. The remaining floating-point computation is converted to BF16.

The INT8 weight-quantized configuration reduces the VMFB and runtime memory
footprint, allowing Part A, Part B, and the LM head to remain resident
simultaneously on memory-constrained Torq targets such as the Coralboard.

The currently validated INT8 weight-quantization workflow requires
`--split-model`.
