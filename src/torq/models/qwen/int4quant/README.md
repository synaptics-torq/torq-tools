# Qwen GPTQ INT4 (block-32 DQL) export for Torq

Builds a static, weight-only INT4 Qwen model for the Torq NPU. Validated on
`Qwen/Qwen2.5-0.5B-Instruct`. The approach follows the Gemma 3 flow (static
conversion, host embeddings, trimmed vocab, split LM head) with the GPTQ
recipe from `wip/gemma-intfour-quant`, which calibrates on
`HuggingFaceH4/ultrachat_200k`.

```text
fp32 ONNX (optimum export, input_ids + past_key_values)
  -> static (seq=1 decode, fixed KV capacity)       torq-export-model qwen
  -> host token embeddings, combined KV I/O
  -> trimmed vocab (latin/punct/digits), padded to a multiple of 64
  -> separate LM head (lm_head.onnx)
  -> opset 21, MatMul weights -> INT4 DequantizeLinear(axis=0, block_size=32)
     with weights taken from a GPTQ checkpoint (or RTN without one)
  -> fp32 -> bf16, int64 -> int32
  -> model.onnx (transformer) + lm_head.onnx -> torq-compile
```

## 1. GPTQ calibration (separate venv)

`llmcompressor` needs transformers 5.x, which conflicts with the main torq-tools
environment, so use a dedicated virtualenv:

```sh
python3 -m venv venv-gptq
venv-gptq/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r src/torq/models/qwen/int4quant/requirements.txt
venv-gptq/bin/python src/torq/models/qwen/int4quant/run_gptq.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu --grouping in \
    --num-samples 256 --max-seq-len 1024 --save-dir gptq-W4A16-G32-infeat-256
```

On an 8-core CPU without a GPU this takes about 1 h 50 min at 256×1024. The Gemma
recipe default is 512×2048. GPTQ varies a lot from run to run, so check the
result with `compare_static_quant.py` (step 3) and rerun if needed.

## 2. Export

The default export downloads the GPTQ weights of each model from Hugging Face
(see the [Qwen README](../README.md)). To use your own checkpoint, pass it
with `--gptq-checkpoint`:

```sh
torq-export-model qwen -s qwen2.5-0.5b-instruct --gptq-checkpoint gptq-W4A16-G32-infeat-256
```

The `export/onnx/` folders of the model contain:

* `fp32/static/`: fp32 static `transformer` and `lm_head` ONNX,
  `token_embeddings.npy` and `token_id_lut.npy`, checked against the dynamic
  source model
* `converted/static/`: the INT4 DQL bf16 models that are compiled

## 3. Accuracy check

```sh
python src/torq/models/qwen/int4quant/compare_static_quant.py \
    --static-dir models/Qwen/Qwen2.5-0.5B-Instruct/export/onnx/fp32/static \
    --gptq-checkpoint gptq-W4A16-G32-infeat-256 --out results.json
```

This reports KL, cosine, max absolute error and top-1 agreement of the fake-quantized
(RTN and GPTQ) logits against fp32, teacher-forced over chat prompts.

## 4. Compile

`torq-export-model qwen` compiles with the flags in
[`../compile_flags.json`](../compile_flags.json). See the
[Qwen README](../README.md#compile-flags) for the reasons.

Decode MatMuls (M = 1) use the full default tile budget (50 % of LRAM), which
leaves no room to prefetch the next weights. Only 17.6 % of the transformer's
compute was then hidden under DMA. At 33 % it is 87.5 % (at 25 %: 94.9 %, with
the same board time). At 20 % the tiles get too small and weights are re-read.

On an SL2610 board at 1 GHz, with the chat-demo decode speed:

| build | transformer | LM head | chat demo |
|---|---|---|---|
| dev compiler, slicing on | 1085 ms | 224 ms | 0.8 tok/s |
| torq-compiler 2.2.1 + `--torq-disable-slicing` | 161 ms | 28 ms | 6.2 tok/s |
| + grouped-query attention rewrite | 143 ms | 28 ms | 6.8 tok/s |
| prefetch compiler, slicing off, default tile budget | 109 ms | 14 ms | – |
| **+ MatMul tile budget 33 %** | **74 ms** | 14 ms | **11.2 tok/s** |

Slicing is still not usable on the board. With prefetch, the sliced build is
estimated at 57 ms but measures 386–451 ms. Without slicing, the compile-time
estimate is within about 10–30 % of the board.

## Notes

* Qwen2.5 ONNX exports from transformers older than 4.45 slice a per-layer
  `[32768, 64]` rotary cache. The exporter folds this into one static
  `[max_gen_tokens, 64]` table (`fold_legacy_rotary_cache`).
* Tied embeddings: the LM head is `Transpose(embed_tokens)` in the source
  graph. It is materialized as a constant before trimming
  (`materialize_tied_lm_head`). The host embedding table stays full-precision
  and full-vocab.
* The trimmed vocab is padded to a multiple of 64 by repeating the last kept
  token ID. With an odd INT4 width, `torq-compile` fails with
  `unable to create memcpy from memref<...xi4, strided<[odd, 1]>>`.

## Measured results (Qwen2.5-0.5B-Instruct, 2026-10-06)

KV capacity was 256 and the vocab 99520 after trimming and padding. Accuracy is
fake-quant against fp32 over 288 teacher-forced positions.

| variant | KL | cosine | top-1 agree |
|---|---|---|---|
| RTN INT4 (transformer + LM head) | 0.442 | 0.930 | 73.3% |
| GPTQ INT4, 256×1024 calibration (transformer + LM head) | 0.352 | 0.956 | 77.1% |
| GPTQ, transformer only | 0.335 | 0.958 | 80.9% |
| GPTQ, LM head only | 0.011 | 0.998 | 92.0% |

Board performance is in section 4. The dev compiler's compile-time estimate
(87.5 M + 14.7 M cycles per step) was about 12× more optimistic than what that
build measured on the board.
