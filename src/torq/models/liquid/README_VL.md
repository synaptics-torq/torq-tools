# LFM2-VL-450M export for Torq

Export the `LiquidAI/LFM2-VL-450M` vision-language model as ONNX components and Torq VMFBs. See the [model export guide](../README.md) for setup, common flags, compiler configuration and runtime assets, and the [LFM2.5 guide](README.md) for the text-decoder options.

## Source model

The exporter downloads ONNX components from the private `Synaptics/liquidAI-LFM2-VLM` mirror, falling back to `onnx-community/LFM2-VL-450M-ONNX`. Configuration and tokenizer files come from `LiquidAI/LFM2-VL-450M` when they are not available locally. See [authentication](../README.md#setup-and-pipeline) for private repositories.

For offline export, use `--onnx-source-dir DIR` with these source files and their external weights:

```text
decoder_model_merged.onnx
decoder_model_merged.onnx_data
vision_encoder.onnx
vision_encoder.onnx_data
embed_tokens.onnx
```

Without a source override, the source directory is `<models-dir>/source/onnx/fp32/`.

| Source component | Exported role |
|---|---|
| `decoder_model_merged.onnx` | Text decoder; split into `transformer.onnx` and `lm_head.onnx` by default |
| `vision_encoder.onnx` | Dynamic fp32 encoder for ONNX Runtime; use `--vision-res` to build a static encoder for Torq |
| `embed_tokens.onnx` | Extracted to the host embedding table `token_embeddings.npy`; not compiled |

## Export and compile

Export and compile the text decoder:

```sh
torq-export-model liquid-vl \
    --models-dir models/liquid-2p5-450M-VL \
    --convert-dtypes
```

To also build a static vision encoder and a two-part image-prefill decoder:

```sh
torq-export-model liquid-vl \
    --models-dir models/liquid-2p5-450M-VL \
    --convert-dtypes \
    --vision-res 256 \
    --image-decoder-parts
```

`--vision-res` and `--image-decoder-parts` require `--convert-dtypes`. Use `--skip-torq` to generate the ONNX components without compiling them.

## Model-specific options

| Flag | Meaning |
|---|---|
| `--models-dir DIR` | Read `DIR/source/onnx/fp32/` and write `DIR/export/` |
| `--vision-res {128,256}` | Build a static vision encoder: 128 produces 16 image tokens; 256 produces 64 |
| `--image-decoder-parts [N]` | Build a 64-image-token prefill decoder split into 2, 3 or 5 layer parts; a bare flag selects 2 |
| `--compile-vision` | Attempt to compile the dynamic source encoder directly; experimental, prefer `--vision-res` |
| `--dynamic-quantize` | Dynamically quantize decoder components; the dynamic vision encoder is skipped unless vision compilation is requested |
| `--keep-conv1d` | Keep the original Conv1D operators for CPU/ONNX Runtime targets |
| `--chunk-lm-head` | Use the legacy chunked LM head for compilers without tile-and-fuse |

## Output and deployment

The conversion command above writes to `models/liquid-2p5-450M-VL/export/split_lm_head/bf16/static/`. Compiled models and runtime assets are in its `compiled/` subdirectory. With `--no-split-lm-head`, the topology directory is `unified/` and the decoder is named `decoder_model_merged`.

| Artifact | Produced by |
|---|---|
| `transformer.vmfb`, `lm_head.vmfb` | Default split decoder |
| `transformer_prefill.vmfb` | Batched text prefill, enabled by default; `--batch-prefill 0` disables it |
| `decoder_model_merged.vmfb` | `--no-split-lm-head --batch-prefill 0` |
| `vision_encoder_128.vmfb` or `vision_encoder_256.vmfb` | `--vision-res 128` or `--vision-res 256` |
| `decoder_image_2part_A.vmfb`, `decoder_image_2part_B.vmfb` | `--image-decoder-parts`; 3- and 5-part exports use the corresponding part count and labels |
| `token_embeddings.npy`, configuration and tokenizer files | Staged alongside compiled models |

Deploy the VMFBs and runtime assets from the same export variant together. See [output variants and runtime assets](../README.md#output-variants-and-runtime-assets) for dtype and file requirements. Existing deployment bundles may name the 128-resolution vision model `vision_encoder.vmfb`; configure the runner to use the generated name or rename it to match the runner's expected filename.

The exporter produces separate components. Image preprocessing, merging image features with text embeddings, and coordinating image-prefill and decode are responsibilities of the deployment runner. A text-only Liquid runner does not provide the complete vision-language pipeline.

## Limitations and validation

- The source vision encoder has dynamic image shapes. Use a static `--vision-res` export for Torq deployment.
- Static vision and image-prefill compilation can take substantial time and memory. The 256-resolution vision VMFB can approach 1.77 GB; check that the generated artifacts fit the target's memory alongside runtime state.
- Automatic end-to-end vision-language validation is not performed by this exporter. Validate image preprocessing and component integration with the deployment runner. See the [validation notes](../README.md#validation) for bf16 ONNX Runtime limitations.
- Set `--max-gen-tokens` large enough for the image tokens, text prompt and generated response.
