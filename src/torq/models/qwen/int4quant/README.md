# Qwen GPTQ INT4 (block-32 DQL) export for Torq

Calibrate GPTQ INT4 weights for a Qwen checkpoint, then use them with the [Qwen exporter](../README.md). Calibration uses `HuggingFaceH4/ultrachat_200k`.

## 1. GPTQ calibration (separate venv)

`llmcompressor` needs transformers 5.x, which conflicts with the main torq-tools environment, so use a dedicated virtualenv:

```sh
python3 -m venv venv-gptq
venv-gptq/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r src/torq/models/qwen/int4quant/requirements.txt
venv-gptq/bin/python src/torq/models/qwen/int4quant/run_gptq.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu --grouping in \
    --num-samples 256 --max-seq-len 1024 --save-dir gptq-W4A16-G32-infeat-256
```

Calibration can take several hours on a CPU. Evaluate the checkpoint with the accuracy check below before deploying it.

## 2. Export

The default export downloads the GPTQ weights of each model from Hugging Face (see the [Qwen README](../README.md)). To use your own checkpoint, pass it with `--gptq-checkpoint`:

```sh
torq-export-model qwen -s qwen2.5-0.5b-instruct --gptq-checkpoint gptq-W4A16-G32-infeat-256
```

The `export/onnx/` folders of the model contain:

* `fp32/static/`: fp32 static `transformer` and `lm_head` ONNX, `token_embeddings.npy` and `token_id_lut.npy`, checked against the dynamic source model
* `converted/static/`: the INT4 DQL bf16 models that are compiled

## 3. Accuracy check

```sh
python src/torq/models/qwen/int4quant/compare_static_quant.py \
    --static-dir models/Qwen/Qwen2.5-0.5B-Instruct/export/onnx/fp32/static \
    --gptq-checkpoint gptq-W4A16-G32-infeat-256 --out results.json
```

This reports KL, cosine, max absolute error and top-1 agreement of the fake-quantized (RTN and GPTQ) logits against fp32, teacher-forced over chat prompts.

## 4. Compile

`torq-export-model qwen` compiles with the flags in [`../compile_flags.json`](../compile_flags.json). See the [Qwen README](../README.md#compile-flags) for the reasons.

For quantization sensitivity analysis, mixed precision and benchmarking, see the [quantization recipe](../../README.md#standalone-weight-quantization).
