# Adding native batch prefill to an exporter

Legacy `--batch-prefill N` produces a separate fixed-size prefill model. Native prefill packages single-token decode and several fixed-size prompt chunks into one compiled family. The packaging is reusable; producing a correct chunk graph remains the exporter's responsibility. The compiler lowers the supplied entries and shares compatible resources. It does not derive model-specific prefill semantics from a decode graph.

An exporter that already produces correct fixed-size prefill graphs can usually build a native family by reusing its chunk builder for each size. Starting from decode-only support requires implementing positions, masks and every state transition for a whole chunk. Use the [user guide](batch_prefill.md) for commands and supported configurations; this guide describes the development work.

## Ownership and reusable infrastructure

| Component | Responsibility |
| --- | --- |
| Model exporter | Construct correct decode and chunk graphs, choose supported configurations, and preserve model-specific state and layout semantics. |
| [Shared exporter](../onnx.py) and [native family helpers](../prefill.py) | Check the family ABI and parameter identity, import named entries, share weights, compile the family, record the ABI manifest, and compare chunk outputs against sequential decode. |
| [Size probe](../prefill_sizes.py) | Recommend candidate sizes from isolated projection-matmul tiling using the selected compiler and flags. |
| Compiler | Lower the supplied entries into the target image and share compatible parameter layouts and resources. |
| `torq-examples` and `torq.runtime` | Execute the hardware VMFB, select entries, maintain device caches, and manage sequence resets. |

Reuse the shared helpers and existing chunk-aware graph edits before adding a model-specific rewrite. Native hardware execution belongs in `torq-examples`; adding an exporter does not require adding native VMFB dispatch to the torq-tools inference runners.

The current supported scope is Gemma3/FunctionGemma 270M and LFM2.5 230M. The supported configuration checks are deliberate. Extending them to another model requires validating its graph and state contract, and may require extending the shared ABI or constant classifier. Merely removing an unsupported-model check does not establish support.

## Semantic requirements

For a chunk of B tokens starting at position p, the final hidden output and resulting state must agree with B sequential decode calls starting from the same state. B counts tokens in one sequence; the sequence batch dimension remains one.

For example, p=5 and B=4 means token positions 5, 6, 7 and 8. Query row i can attend to valid earlier cache entries and chunk rows through i. It must not see later chunk rows or unused cache rows, even if those buffers contain nonzero values. The returned state must represent consumption of all four tokens.

### Positions, attention and output selection

Construct all B positions from the starting position, including for nonzero starts. Preserve the model's RoPE variant, rotary dimensions, scaling, tensor layout and query-to-KV head grouping. Replacing a singleton token dimension with B is insufficient when constants, broadcasts or transposes were specialized for decode.

Make the causal mask correct for both the old cache and attention within the chunk. Retain the model's attention scaling and masking rules. Use chunk-aware helpers such as `ReplaceDynamicKVCache`, `MaskFutureAttentionScores` and `ConvertToStaticIndex` where their assumptions match the model.

The decoder body returns only the final token's hidden state, so the separate language model head can run after the final prompt chunk. Slice the correct token and keep its output rank consistent with decode. Short prompt tails use smaller available entries or single-token decode; native execution does not require padding.

### Cache and recurrent state

Advance every state exactly as sequential decode would. That includes convolution history, not just attention KV tensors. Preserve state order, chronological order, dtype and the model's layout conventions. State persists when switching between chunk sizes and decode.

The current attention cache contracts are:

| Mode | Cache input | Cache output | Update |
| --- | --- | --- | --- |
| Legacy full-cache | `[1,2*H,L,d]` | Same full shape | Preserve rows outside the consumed positions and return the complete updated cache. |
| Read-only, time-major | `[1,L,2*H,d]` | `[1,B,2*H,d]` | Return the B new rows; the caller writes them into positions `p:p+B` of the full cache. |

Here H is the KV-head count, L is the exported context capacity and d is the head dimension. Every call must satisfy `0 <= p` and `p+B <= L`. Convolution states use their own fixed shapes and replacement rules. Rolling context, alternative state layouts and additional recurrent state types require explicit support in the exporter, importer, validator and hardware runner.

The current read-only family importer assumes its first state is a KV cache. Hybrid models whose first state is convolution need additional importer and validation work before enabling that cache mode. LFM2.5 currently uses full-cache KV I/O and its existing convolution-state contract.

### Parameters and ONNX representation

Keep parameter names, dtypes, shapes and values identical across family entries after all transformations. Apply compatible optimizer settings to every body. A size-dependent fold into a trained weight or a renamed parameter can break identity checks and resource sharing even when the graphs appear equivalent.

`_constant_roles()` in `prefill.py` distinguishes parameters, fixed position frequencies and controls from their uses. Fixed frequencies are checked for identity but remain literals; shape, position and mask controls can vary with B. Integer controls remain inline so ONNX shape/data propagation can read them.

This classification recognizes specific ONNX forms. Vector biases, different normalization patterns, new quantization operators or unfamiliar mask encodings may be rejected. Extend the classification with focused tests for legitimate new forms, including rejection of mixed parameter/control uses. Do not classify constants by size or treat all floating constants as weights.

Mathematically equivalent expressions can also lower to different layouts or schedules. Check compiler acceptance, parameter sharing and device behavior after changing the representation.

## Current family ABI

The importer checks a narrow contract in addition to model configuration gates:

| Position | Input/output | Contract |
| --- | --- | --- |
| Input 0 | `token_embedding` | Static `[1,B,D]`; FP32 or supported BF16 body. Embeddings are extracted from the decoder. |
| Input 1 | `position_ids` | Starting position `[1,1]`; INT64 for an FP32 body, INT32 for a BF16 body. The graph constructs the B positions. |
| Remaining inputs | Ordered state tensors | Fixed input signatures shared by every entry. State i is named `past_key_values.i.key_value` or `past_conv.i`. |
| Output 0 | Final hidden state | `[1,1,D]`, with the same floating dtype and hidden width as the embedding input. |
| Remaining outputs | Updated states or KV rows | One output per state input, in matching order, named `present.i.key_value` or `present_conv.i`. |

The decoder entries are `main` for B=1 and `main_B` for chunks. The head is compiled separately. A separate `attention_mask` input and arbitrary extra state names are outside this ABI; the graph builds its mask from the position and static dimensions. Supporting another ABI requires updating the checks and the consumers together.

Static dimensions, combined KV I/O, extracted embeddings and a split head are required by this path. Dynamic shapes, vision, arbitrary topology splitting and sliding-window execution beyond the exported context need separate support. Use `validate_native_options()` to reject unsupported combinations before source preparation or output replacement.

## Wiring a new exporter

1. Establish correct single-token decode against the original source ONNX. Then implement one pinned B-token chunk using the model's existing conversion and post-static edits. Extend the [graph-edit tests](../../../../tests/unit/graph_edit) for model-specific rewrites.
2. Register the shared decoder/LLM arguments with native support enabled. Accept `native_batch_prefill=False` and `batch_prefill_sizes=None`, validate the configuration, and keep native mode mutually exclusive with legacy `--batch-prefill`. Explicit sizes require native mode. Use the shared argument helpers, including their handling of an explicitly supplied default fixed size.
3. Track native enablement independently of resolved sizes. `_native_prefill_enabled` must govern graph policy and configuration checks even when `_native_batch_prefill` is still empty because automatic selection is pending. Read-only graph plans must not start probing or export work.
4. Keep native families in the standard static export directories. Implement `_make_native_prefill()` using the chunk builder so the shared `_complete_native_prefill()` can add selected graphs after probing the normal final decoder. Weight-quantizing exporters call completion after decoder quantization and quantize only newly added chunks. Forward `compiler_path` and `torq_compile_args` into `export_onnx()` from the CLI. Explicit sizes bypass probing; automatic selection needs the compiler even with `--skip-torq`.
5. Build `model` for decode and `model_prefill_B` for each resolved size, with the paths expected by the shared export pipeline. Rebuild from the appropriate source for each B so decode-specific constants and layouts cannot leak into chunks. Reuse the existing `_make_model_static(..., chunk_len=B)` pattern where valid, and provide optimizer settings for every component.
6. Extract embeddings and save the independent head once. Apply all graph transforms through the established pipeline, save intermediate components with `_save_component_model()`, and let the base exporter finalize shared weight storage. If a separate weight-quantization stage rewrites the bodies, call `_share_native_weights()` after it finishes, as the Gemma exporter does. Keep graph files beside `transformer.weights.data`; retain the family ABI manifest and runtime assets.
7. Add model-specific equivalence tests, run the shared family checks, and validate compilation and target execution before advertising the new configuration. Extend supported-model checks and their error messages to describe exactly the configurations validated.

The [Gemma exporter](../../models/gemma3/export.py) demonstrates reuse of a static chunk builder, per-component optimization and one shared head. Its `_native_rope_layout()` rewrite preserves a head/token ordering that singleton decode can hide. The [LFM exporter](../../models/liquid/export.py) rebuilds the source conversion per chunk and supplies model-specific GQA and convolution forms. Its native convolution rewrite supports an unbiased depthwise K=3, stride-one, dilation-one, unpadded kernel with `[1,C,3]` channel-major history, oldest first, and B at least four. Those assumptions are specific to that implementation.

## Size selection and performance

Automatic selection probes the normally exported single-token decoder after graph edits, optimization and any selected weight quantization, then builds only the selected prefill graphs in the same standard static directory. INT4 exports are probed with packed weights, scales, zero points and dequantization intact. It excludes the independent logits head and uses the selected compiler plus model/user flags. The existing sweep starts at B=4 and is capped at 128, bounded by the requested context. Manual sizes can cover supported cases outside that probe range. A failed or empty recommendation stops family construction and asks for explicit sizes; the exported single-token decoder remains available.

The probe preserves the exported projection types and constant producers, including INT4 dequantization. It excludes LFM's single-window channel-wise convolution MatMuls (`[C,1,K] @ [C,K,1]`, C > 1), whose channel axis cannot serve as the token axis. It does not account for the full attention graph, convolution/recurrent work or all fused-producer memory. Its results are candidates for full-model validation. Prefetch policy and compiler changes can alter the selected sizes even when projection shapes are unchanged.

For a new final decoder representation, check that the probe extracts the intended decoder projection shapes and excludes the separate head. Pin a known-correct size while establishing correctness. Measure complete-family memory, decode throughput and prompt latency on the target after selecting sizes.

## Validation before enabling support

Keep two independent numerical checks: single-token decode against the original model, and chunked execution against sequential decode. Otherwise a shared decode/chunk bug can pass the family comparison. Continue using `validate_decoder_only_onnx()` for source-model validation where applicable.

Use small synthetic graphs for positions, masks, head grouping, state transitions and layout rewrites. Include nonzero history and future-token perturbations that test causal independence of earlier observable rows. Test unsupported configurations and parameter mutation failures as well as successful import.

Run `validate_native_onnx()` on every exported FP32 chunk. Compare the final hidden output and every state output at empty, populated and final-fit cache positions. For read-only KV, assemble the sequential rows and compare all B returned rows. Verify that full-cache outputs preserve untouched rows. Diagnose mismatches using the existing tolerance policy and per-output comparisons.

The reproducible exporter regressions are:

```sh
tests/.test/bin/python -m pytest \
    tests/unit/model_export/test_native_prefill.py \
    tests/unit/model_export/test_prefill_sizes.py \
    tests/unit/model_export/test_split_weights.py
```

These shared tests complement the new model's own graph and source-equivalence tests. For a full model, record the export command, checkpoint, context, sizes, compiler version and flags. Check that every requested entry is exported and compiles into the intended shared image, inspect parameter sharing and the complete XRAM address span, and validate entry switching, tails and repeated sequence resets on hardware.

Establish FP32 equivalence before applying weight quantization. Verify packed parameters, scales and zero points across entries, then measure BF16/INT4 accuracy and performance through the `torq-examples` hardware runners. ONNX Runtime CPU cannot execute the BF16 MatMul path. State which configurations and hardware checks have actually passed in the model README and PR notes.
