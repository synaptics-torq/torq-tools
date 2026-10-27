# Native batch prefill

Native batch prefill processes several prompt tokens per call, then switches to single-token decode. The model exporter supplies static `main` and `main_<B>` functions in one MLIR module. The compiler builds one VMFB containing a shared Torq executable image and a separate schedule for each entry. Batch size here means tokens in one sequence chunk; the sequence batch dimension remains one.

Hardware native VMFB execution is provided by `torq-examples` through `torq.runtime`. This repository supplies model export, compilation and ONNX validation.

## Export and compile

Use the torq-tools native export workflow for Gemma3 270M IT, FunctionGemma 270M or FP32 LFM2.5 230M. For example, from the torq-tools environment:

```sh
torq-export-model gemma3 --instruct-model \
    --weight-quantization none --extract-embeddings --split-lm-head \
    --no-read-only-kv-cache --max-gen-tokens 256 \
    --native-batch-prefill \
    --compiler-path /path/to/torq-compile
```

Native prefill is off by default. `--native-batch-prefill` is mutually exclusive with the older fixed-size `--batch-prefill N`, including `--batch-prefill 0`. `--batch-prefill-sizes` defaults to None: when native mode is enabled, the exporter probes the normally exported single-token decoder after graph edits, optimization and any selected weight quantization. The separate logits head is excluded. It uses the selected compiler, model defaults and `--compile-flags`, scans up to 128 tokens bounded by `--max-gen-tokens`, and stores `prefill_sizes.json` plus its reusable cache under the model root's `prefill-probes/` directory. A failed or empty probe stops the export and asks for explicit sizes. Automatic selection uses the final decoder weight representation: INT4 exports are probed after quantization, including their packed weights, scales, zero points and dequantization operations. The exported decoder is retained; only the selected prefill graphs are then built, validated and quantized as needed. Native families use the same standard `static/` directory as ordinary static exports, with no chunk-size list in the path.

To pin sizes and bypass probing, add `--batch-prefill-sizes 4,12,29,51,92,106,124` alongside `--native-batch-prefill`. Sizes alone do not enable native mode. Automatic probing needs a compatible compiler even with `--skip-torq`; explicit sizes allow ONNX-only exports without running the compiler.

The unquantized tools export remains FP32; compiler dtype conversion produces BF16 floating I/O and i32 positions. Gemma3 also supports the default INT4 GPTQ weights: omit `--weight-quantization none` and `--no-read-only-kv-cache` to use INT4 weights with the updated time-major, read-only cache. INT4 ONNX uses BF16 activations and i32 positions before compilation. The exporter imports all entries into `compiled/transformer.mlirbc`, compiles `compiled/transformer.vmfb`, compiles the separate language model head, and records the original state ABI and compiler flags in `compiled/transformer.family.json`. Keep this metadata and the runtime assets beside the VMFB.

Native Gemma3 FP32 exports with legacy full-cache I/O retain constant-load prefetch; stream scheduling shrinks their shared weight tiles and regresses throughput. INT4 and read-only-cache exports retain the model’s stream-schedule default. Explicit `--compile-flags` still override these defaults.

Static shapes, extracted embeddings, a split language model head and combined KV state are required. Sizes must be unique, strictly ascending and fit the context. Gemma3 supports FP32 or INT4 weights and either cache layout. LFM2.5 sizes must be at least four and require FP32 weights with full-cache outputs. Dynamic quantization, explicit dtype conversion, sliding-window execution, vision and arbitrary layer splitting need separate support.

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

Run the probe on the decoder ONNX in the weight format you will compile, with matching compiler flags. FP32, FP16 and BF16 projections retain their source types; INT4 projections retain packed weights, scales, zero points and the dequantization chain. FP32 input cannot predict an INT4 export. Imported ONNX `.mlir` and `.mlirbc` sources are also accepted. Apply the printed `--native-batch-prefill --batch-prefill-sizes ...` options to a subsequent export. The probe writes `prefill_sizes.json`. Its isolated projections omit surrounding activation producers and attention, so validate the complete family before pinning sizes. Structurally equivalent projections share a representative; different types or quantization structures remain separate even at the same logical shape. The cache fingerprints the extracted graph and real constants. Only direct BF16 constants use the BF16 split grid; other representations use observed strip widths. A failed probe makes no recommendation; provide explicit sizes.

## Runtime contract

| Function | Embeddings | Starting position | State | Outputs |
| --- | --- | --- | --- | --- |
| `main` | `[1,1,D]` | `[1,1]` | Full fixed-shape KV and convolution state | Final hidden token `[1,1,D]`, then updated state or one new KV row |
| `main_B` | `[1,B,D]` | `[1,1]` | Same shapes and order as `main` | Final hidden token `[1,1,D]`, then updated state or B new KV rows |

The updated cache input is `[1,L,2*H,D]` and each output is `[1,B,2*H,D]`. Write the returned contiguous rows into the input cache at the starting position. The legacy layout is `[1,2*H,L,D]` with full-cache outputs. The family metadata records `kv_cache_sequence_axis` as 1 or 2; signatures record every entry’s output shape.

Load the VMFB once. Initialize every state tensor for each new sequence, choose the largest available chunk that fits the remaining prompt and context, pass its starting position and carry every returned state tensor into the next call. Use `main` for short unmatched tails. Apply the separate head to the final prompt hidden token, then use `main` for generated tokens. Every call must satisfy `0 <= position` and `position + B <= context`.

Entries share resident immutable weights with compatible physical layouts, code and initialized data. Scratch and binding addresses are reused, so runtime dispatch is serialized. State persists in caller buffers. The VMFB does not choose chunks, reset caches or implement rolling context. The hardware runners in `torq-examples` handle chunk selection and sequence resets using `torq.runtime` and the VMFB I/O metadata.

Compare chunked and sequential outputs before deployment. BF16 computation may change generated tokens. Check the complete XRAM address span as well as VMFB file size: scratch and binding ranges are uninitialized, and incompatible physical weight layouts can require additional copies.

For model-specific configuration and inference commands, see the [Gemma3/FunctionGemma](../models/gemma3/README.md#experimental-native-batch-prefill) and [LFM2.5](../models/liquid/README.md#experimental-native-batch-prefill) exporter guides. For developer validation commands, see [Contributing](../../../CONTRIBUTING.md#native-decoder-families).
