# Qwen GPTQ INT4 (block-32 DQL) export for Torq

Builds a static, weight-only INT4 Qwen model for the Torq NPU. Validated on
`Qwen/Qwen2.5-0.5B-Instruct`. The approach follows the Gemma 3 flow (static
conversion, host embeddings, trimmed vocab, split LM head) with the GPTQ
recipe of the Gemma 3 INT4 work, which calibrates on
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

Decode MatMuls (M = 1) load one weight tile per compute step. Prefetch hides
the compute only if LRAM has space for the next weight tile next to the
current one. With the default MatMul tile budget, more than 90 % of the
transformer compute is hidden under DMA.

On an SL2610 board at 1 GHz:

| model | transformer | LM head | chat demo |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 74 ms | 14 ms | 11.3 tok/s |
| Qwen3-0.6B | 155.5 ms | 15.7 ms | 6.0–6.4 tok/s |

## Notes

* Qwen2.5 ONNX exports from transformers older than 4.45 slice a per-layer
  `[32768, 64]` rotary cache. The exporter folds this into one static
  `[max_gen_tokens, 64]` table (`fold_legacy_rotary_cache`).
* Tied embeddings: the LM head is `Transpose(embed_tokens)` in the source
  graph. It is materialized as a constant before trimming
  (`materialize_tied_lm_head`). The host embedding table stays full-precision
  and full-vocab.

## Measured results (Qwen2.5-0.5B-Instruct, 2026-10-06)

KV capacity was 256 and the vocab 99520 after trimming and padding. Accuracy is
fake-quant against fp32 over 288 teacher-forced positions.

| variant | KL | cosine | top-1 agree |
|---|---|---|---|
| RTN INT4 (transformer + LM head) | 0.442 | 0.930 | 73.3% |
| GPTQ INT4, 256×1024 calibration (transformer + LM head) | 0.352 | 0.956 | 77.1% |
| GPTQ, transformer only | 0.335 | 0.958 | 80.9% |
| GPTQ, LM head only | 0.011 | 0.998 | 92.0% |

Board performance is in section 4.

## Measured results (Qwen3-0.6B, 2026-10-09)

Same method: 256×1024 GPTQ calibration, KV capacity 256, vocab 99520 after
trimming and padding, 288 teacher-forced positions.

| variant | KL | cosine | top-1 agree |
|---|---|---|---|
| RTN INT4 (transformer + LM head) | 0.152 | 0.948 | 83.3% |
| GPTQ INT4 (transformer + LM head) | 0.107 | 0.968 | 85.4% |

SL2610 board at 1 GHz: transformer 155.5 ms, LM head 15.7 ms, chat demo
6.0–6.4 tok/s. In the transformer, the NPU program takes 112 ms. The runtime
copies the 28 tied KV caches (1 MB each) for about 31 ms per call.
