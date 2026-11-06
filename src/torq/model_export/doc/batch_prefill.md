# Native batch prefill

Native batch prefill processes several prompt tokens per call, then switches to single-token decode. The exporter compiles single-token decode and several prompt chunk sizes into one `transformer.vmfb`, sharing model weights. Batch size here means tokens in one sequence chunk; the sequence batch dimension remains one.

Hardware native VMFB execution is provided by `torq-examples` through `torq.runtime`. This repository supplies model export, compilation and ONNX validation.

## Before you start

Install torq-tools in a virtual environment using the [installation instructions](../../../../README.md#installation). Batch prefill requires torq-compiler **2.3.5 or newer**, including its Python bindings and a compatible `torq-compile` executable in the same environment. These instructions assume native-family support is available in versions 2.3.5 and newer. If selecting another executable, use `--compiler-path /path/to/torq-compile` or set `TORQ_COMPILER_PATH`; pointing at a binary alone does not install the Python bindings.

The exporter checks the importer APIs and that the selected executable advertises `--torq-enable-multi-graph` before exporting. Use a matching board runtime with version-3 entry-table support. Hardware batch-prefill execution is in `torq-examples`; the torq-tools inference runners process prompts one token at a time.

Source models are downloaded from Hugging Face on first use. For gated checkpoints, accept the model's access terms and authenticate with `hf auth login`. Hosted Synaptics GPTQ checkpoints may require private-repository access. The FP32 recipes below avoid the GPTQ download.

## Export and compile

Run one of these commands from the installed torq-tools environment. Each explicitly requests batch prefill, so an unsupported configuration fails instead of falling back to decode-only. All use a 256-token context; this capacity includes the prompt and generated tokens. Omit `--batch-prefill-sizes` for automatic size selection where supported. The explicit sizes below are small examples, not performance recommendations.

| Supported model | Weight format | Size selection | Required cache layout |
| --- | --- | --- | --- |
| Gemma3 270M / FunctionGemma 270M | FP32 or INT4 | Automatic or explicit | Full-cache or read-only |
| LFM2.5 230M / 350M | FP32 | Automatic or explicit | Read-only (default) or full-cache |
| Qwen3-0.6B / Qwen2.5-0.5B-Instruct | FP32 or INT4 | Automatic for FP32; explicit for INT4 | Read-only |
| SmolLM2 135M / 360M | FP32 | Automatic or explicit | Read-only |
| LFM2-VL 450M text decoder | FP32 | Explicit only | Full-cache |

### Gemma3 and FunctionGemma

```sh
# Gemma3 270M instruct, FP32 weights, automatic sizes
torq-export-model gemma3 --instruct-model --batch-prefill \
    --weight-quantization none --max-gen-tokens 256

# FunctionGemma 270M, FP32 weights, automatic sizes
torq-export-model gemma3 --functiongemma --batch-prefill \
    --weight-quantization none --max-gen-tokens 256
```

For INT4, omit `--weight-quantization none`. Embedding extraction and the split LM head are enabled by default.

### LFM2.5

```sh
torq-export-model liquid --model-size 350m --batch-prefill \
    --max-gen-tokens 256
```

Use `--model-size 230m` for the smaller model. Keep FP32 export weights and the default read-only cache; do not add `--convert-dtypes`. `--no-read-only-kv-cache` selects full-cache outputs when needed. Compilation converts floating types as needed.

### Qwen

```sh
# FP32 weights, automatic sizes
torq-export-model qwen --model-size qwen3-0.6b --batch-prefill \
    --weight-quantization none --max-gen-tokens 256

# Default INT4 GPTQ weights, explicit sizes (requires checkpoint access)
torq-export-model qwen --model-size qwen3-0.6b --batch-prefill \
    --batch-prefill-sizes 4,12 --max-gen-tokens 256
```

Use `--model-size qwen2.5-0.5b-instruct` for Qwen2.5. Keep the default split head and read-only cache.

### SmolLM2

```sh
torq-export-model smollm2 --model-size 135M --batch-prefill \
    --split-lm-head --max-gen-tokens 256
```

Use `--model-size 360M` for the larger supported model. The split head is opt-in for this exporter. Keep embedding extraction and the read-only cache enabled.

### LFM2-VL text decoder

```sh
torq-export-model liquid-vl --batch-prefill --split-lm-head \
    --batch-prefill-sizes 4,12 --max-gen-tokens 256
```

This exports the 450M text-decoder family. Keep the full-state cache and FP32 weights; do not add vision-build options or `--convert-dtypes`. Automatic size selection is not supported here.

## Check the result

A successful batch-prefill export produces the following in the selected variant's `static/` directory (the exporter prints the output paths):

- `transformer.onnx`, `transformer_prefill_<B>.onnx` for each selected size, and their shared `transformer.weights.data` file.
- A separate `lm_head.onnx` and the host assets, including embeddings and tokenizer/configuration files.
- Under `compiled/`, `transformer.vmfb`, `transformer.family.json`, `transformer.mlirbc`, the compiled LM head and staged runtime assets. The LM-head filename depends on vocabulary trimming.

Check that `transformer.family.json` records decode and the requested or automatically selected chunk entries. A VMFB alone does not establish that batch prefill was enabled. Automatic selection also writes `prefill_sizes.json` under the model root's `prefill-probes/` directory. Keep the family metadata, compiled head and runtime assets with the decoder VMFB when deploying to `torq-examples`.

If probing fails, retry with explicit ascending sizes such as `--batch-prefill-sizes 4,12` that fit the context. If compiler preflight fails, install compatible Python bindings and select a compatible executable. Do not add `--skip-torq` to a batch-prefill CLI command: that combination is rejected.

## Defaults and advanced options

Batch prefill is on by default for compile runs: `--batch-prefill` / `--no-batch-prefill` selects it explicitly. `--batch-prefill-sizes` defaults to None: when batch prefill is on, the exporter probes the normally exported single-token decoder after graph edits, optimization and any selected weight quantization. The separate logits head is excluded. It uses the selected compiler, model defaults and `--compile-flags`, scans up to 128 tokens bounded by `--max-gen-tokens`, and stores `prefill_sizes.json` plus its reusable cache under the model root's `prefill-probes/` directory. A failed or empty probe stops the export and asks for explicit sizes. Automatic selection uses the final decoder weight representation: INT4 exports are probed after quantization, including their packed weights, scales, zero points and dequantization operations. The exported decoder is retained; only the selected prefill graphs are then built, validated and quantized as needed. Families use the same standard `static/` directory as ordinary static exports, with no chunk-size list in the path.

The resolution rules: an explicit `--batch-prefill` (or `--batch-prefill-sizes`) together with `--skip-torq` fails, because batch prefill always compiles; the default with `--skip-torq` exports decode only; the default on a configuration that does not support batch prefill switches it off with a warning (an explicit `--batch-prefill` on such a configuration fails); otherwise it is on. To pin sizes and bypass probing, pass `--batch-prefill-sizes 4,12,29,51,92,106,124` (sizes require batch prefill to be on). ONNX-only families (no compilation) are built through the Python API: construct the exporter with `batch_prefill=True` and explicit `batch_prefill_sizes`, then call `export_onnx()`.

Every CLI run with batch prefill needs a compatible compiler (`--compiler-path` or `TORQ_COMPILER_PATH`), for the probe and for the family compile.

The unquantized tools export remains FP32; compiler dtype conversion produces BF16 floating I/O and i32 positions. Gemma3 also supports the default INT4 GPTQ weights: omit `--weight-quantization none` to use INT4 weights with the updated time-major, read-only cache. INT4 ONNX uses BF16 activations and i32 positions before compilation. The exporter imports all entries into `compiled/transformer.mlirbc`, compiles `compiled/transformer.vmfb`, compiles the separate language model head, and records the original state ABI and compiler flags in `compiled/transformer.family.json`. Keep this metadata and the runtime assets beside the VMFB.

Native Gemma3 FP32 exports with legacy full-cache I/O retain constant-load prefetch; stream scheduling shrinks their shared weight tiles and regresses throughput. INT4 and read-only-cache exports retain the model’s stream-schedule default. Explicit `--compile-flags` still override these defaults.

Static shapes, extracted embeddings, a split language model head and combined KV state are required. Sizes must be unique, strictly ascending and fit the context. Gemma3 supports FP32 or INT4 weights and either cache layout. LFM2.5 sizes must be at least four and require FP32 weights with either read-only or full-cache outputs. Qwen sizes must be at least four and require the read-only cache with FP32 or INT4 weights; INT4 needs explicit `--batch-prefill-sizes` because automatic probing only covers the FP32 family. SmolLM2 sizes must be at least four and require FP32 weights with the read-only cache; `--split-lm-head` is opt-in there (its exports keep the head in-graph by default). LFM2-VL's text decoder requires `--split-lm-head`, `--batch-prefill-sizes` (no automatic probe) and the full-state cache; its vision build options are not yet validated with native families. Dynamic quantization, explicit dtype conversion, sliding-window execution, vision and arbitrary layer splitting need separate support.

Native compilation requires supplied-family support and a matching runtime with version-3 entry tables. The exporter supplies the compiler options; the hardware runners in `torq-examples` choose the exported functions through `torq.runtime`.

Final decoder ONNX graphs share one `transformer.weights.data` file in their export directory, including after INT4 weight quantization. This applies regardless of `--split-weights`; the separate language model head keeps its existing storage policy. Copy the shared data file alongside any decoder graph you want to load or compile. Integer controls remain inline, and standard ONNX/ONNX Runtime loaders resolve the relative weight references. Optimization intermediates still use per-component data files; sharing reduces final export storage without changing graph construction or avoiding each graph's import and validation work.

## Optional size probe

The optional probe compiles representative projection subgraphs, caches tiling results and prints an exporter option:

```sh
python -m torq.model_export.prefill_sizes transformer.onnx \
    --chip SL2610 --max-batch-size 128 --context-length 256 \
    --tool /path/to/torq-compile -- \
    --torq-convert-dtypes --torq-convert-io-dtype \
    --torq-enable-split-constants-optimization \
    --torq-prefetch-algorithm=stream-schedule
```

Run the probe on the decoder ONNX in the weight format you will compile, with matching compiler flags. FP32, FP16 and BF16 projections retain their source types; INT4 projections retain packed weights, scales, zero points and the dequantization chain. FP32 input cannot predict an INT4 export. Imported ONNX `.mlir` and `.mlirbc` sources are also accepted. Apply the printed `--batch-prefill --batch-prefill-sizes ...` options to a subsequent export. The probe writes `prefill_sizes.json`. Its isolated projections omit surrounding activation producers and attention, so validate the complete family before pinning sizes. Structurally equivalent projections share a representative; different types or quantization structures remain separate even at the same logical shape. The cache fingerprints the extracted graph and real constants. Only direct BF16 constants use the BF16 split grid; other representations use observed strip widths. A failed probe makes no recommendation; provide explicit sizes.

## Runtime contract

| Function | Embeddings | Starting position | State | Outputs |
| --- | --- | --- | --- | --- |
| `main` | `[1,1,D]` | `[1,1]` | Full fixed-shape KV and convolution state | Final hidden token `[1,1,D]`, then updated state or one new KV row |
| `main_B` | `[1,B,D]` | `[1,1]` | Same shapes and order as `main` | Final hidden token `[1,1,D]`, then updated state or B new KV rows |

The updated cache input is `[1,L,2*H,D]` and each output is `[1,B,2*H,D]`. Write the returned contiguous rows into the input cache at the starting position. The legacy layout is `[1,2*H,L,D]` with full-cache outputs. The family metadata records `kv_cache_sequence_axis` as 1 or 2; signatures record every entry’s output shape.

Load the VMFB once. Initialize every state tensor for each new sequence, choose the largest available chunk that fits the remaining prompt and context, pass its starting position and carry every returned state tensor into the next call. Use `main` for short unmatched tails. Apply the separate head to the final prompt hidden token, then use `main` for generated tokens. Every call must satisfy `0 <= position` and `position + B <= context`.

Entries share resident immutable weights with compatible physical layouts, code and initialized data. Scratch and binding addresses are reused, so runtime dispatch is serialized. State persists in caller buffers. The VMFB does not choose chunks, reset caches or implement rolling context. The hardware runners in `torq-examples` handle chunk selection and sequence resets using `torq.runtime` and the VMFB I/O metadata.

Compare chunked and sequential outputs before deployment. BF16 computation may change generated tokens. Check the complete XRAM address span as well as VMFB file size: scratch and binding ranges are uninitialized, and incompatible physical weight layouts can require additional copies.

For model-specific configuration and inference commands, see the [Gemma3/FunctionGemma](../../models/gemma3/README.md#batch-prefill), [LFM2.5](../../models/liquid/README.md#batch-prefill) and [Qwen](../../models/qwen/README.md#batch-prefill) exporter guides. For developer validation commands, see [Contributing](../../../../CONTRIBUTING.md#native-decoder-families).
