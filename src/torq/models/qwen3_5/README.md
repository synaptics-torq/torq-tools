# Qwen3.5 export for Torq

Export pipeline for the text part of **Qwen3.5-0.8B**
([Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache 2.0). One command
converts the Hugging Face model to a static, INT4 weight-quantized model and compiles it to
Torq VMFBs.

| Source (Hugging Face) | Torq model (Hugging Face) | SL2610 chat demo |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | `Synaptics/Qwen3.5-0.8B-int4-torq` | 5.8 tok/s |

## Install

Qwen3.5 needs Hugging Face Transformers 5.x, which conflicts with the optimum pin of the base
package. Use a separate virtualenv:

```sh
python3 -m venv venv-qwen35
. venv-qwen35/bin/activate
pip install -r src/torq/models/qwen3_5/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

The export needs torq-compiler 2.3.0 or newer for compilation (input prefetch). The
`torq-compiler` package in the requirements supplies the ONNX importer and the dtype converter.

## Quick start

```sh
torq-export-model qwen3_5
```

The command does these steps:

1. It builds a static single-token decode step from the Hugging Face weights in PyTorch
   (`_step.py`) and compares its logits with the Hugging Face model on a short chat prompt.
   optimum has no Qwen3.5 export.
2. It exports `transformer.onnx` with `torch.onnx.export`. The inputs are the token embedding
   `[1, 1, 1024]`, `position_ids` `[1, 1]` and three packed states:
   - `past_kv` `[6, 4, 256, 256]`: K and V of the 6 gated-attention layers.
   - `past_conv` `[18, 6144, 3]`: the conv states of the 18 Gated DeltaNet layers.
   - `past_recurrent` `[18, 16, 128, 128]`: the recurrent states of the DeltaNet layers.

   The outputs are the hidden state and the updated states. The states are packed because the
   runtime accepts at most 64 bindings (one tensor per layer state needs 99).
3. It trims the output vocabulary to Latin text, punctuation and digits
   (248,320 → 141,406 tokens, padded to 141,440) and writes `lm_head.onnx` from the tied
   embeddings, plus `token_embeddings.npy` and `token_id_lut.npy` for the host.
4. It converts both models to bf16 and quantizes all MatMul weights to INT4
   `DequantizeLinear` (round-to-nearest, block size 32, bf16 scales, INT4 zero points).
5. It compiles `transformer.vmfb` and `lm_head.vmfb` with the flags in
   [`compile_flags.json`](compile_flags.json).

The output is in `models/Qwen/Qwen3.5-0.8B/export/torq/`.

Options: `--context` sets the static context length (default 256), `--weight-quantization none`
keeps bf16 weights, `--skip-torq` stops after the ONNX export.

## Export rules

The decode step uses only ops that export to static ONNX for Torq:

- No `torch.chunk`: it exports a `Split` with computed sizes, which imports with dynamic
  shapes and stops the compiler.
- No Einsum, OneHot or Expand.
- No MatMul with a constant second input other than the weight layers: the INT4 quantization
  takes every such MatMul as a weight.

## Performance

SL2610 board, NPU at 1 GHz, torq-runtime 2.2.1, [torq-examples](https://github.com/synaptics-torq/torq-examples)
`qwen3_5` demo:

| | Time per call |
|---|---|
| `transformer.vmfb` | 157.6 ms |
| `lm_head.vmfb` | 22.4 ms |
| **Decode (chat demo)** | **5.8 tok/s** |
| Time to first token (short question after the system prompt) | 2.8–3.0 s |

Compile-time profile: the transformer takes 112.4 ms, of which 67 ms is the DMA of the INT4
weights (304 MiB) and about 8 ms is the copy of the packed states. The LM head takes 19.4 ms.

## Limits

- Text only: the vision encoder is not exported.
- The DeltaNet state holds the whole context, so the runner cannot use a KV-cache window.
- The weights use round-to-nearest INT4. GPTQ is not available for Qwen3.5 yet.
