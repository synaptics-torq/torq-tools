# Weight Quantization Tool

Quantize MatMul weights in fp32 ONNX models to int8, int4, or bf16.

## Usage

```bash
# --- Uniform quantization ---

# Int8 (ORT-matching asymmetric uint8, block_size=32)
torq-quantize-model quantize \
  -i model_fp32.onnx -o model_int8_dql.onnx --bits 8

# Int4 (signed [-8,7], block_size=32)
torq-quantize-model quantize \
  -i model_fp32.onnx -o model_int4_dql.onnx --bits 4

# bf16 only (no quantization, just dtype conversion)
torq-quantize-model quantize \
  -i model_fp32.onnx -o model_bf16.onnx --bits 16

# --- Dequantized bf16 output (ready for Torq compilation) ---

# Quantize to int8, then dequantize back and convert entire model to bf16.
# Scales are truncated to bf16 precision BEFORE dequantization so that the
# output matches what the hardware computes at runtime (bf16 scale × int8 weight).
# The final weights are: bf16( (q - zp) * bf16(scale) )
torq-quantize-model quantize \
  -i model_fp32.onnx -o model_bf16.onnx --bits 8 --dequantize-weights

# --- Mixed quantization from config ---

torq-quantize-model quantize \
  -i model_fp32.onnx -o model_mixed.onnx --config quant_config.json

# With dequantized bf16 output
torq-quantize-model quantize \
  -i model_fp32.onnx -o model_bf16.onnx --config quant_config.json --dequantize-weights

# --- Sensitivity analysis ---

torq-quantize-model analyze \
  -i model_fp32.onnx -o sensitivity_results.json \
  --config-output quant_config.json \
  --embeddings token_embeddings.npy

# Custom thresholds and tokenizer
torq-quantize-model analyze \
  -i model_fp32.onnx -o results.json \
  --config-output config.json \
  --embeddings token_embeddings.npy \
  --tokenizer tokenizer.json \
  --bits 4 8 \
  --bf16-threshold 0.1 \
  --int8-threshold 0.01 \
  --num-tokens 10

# Reduced-vocab models (pass token ID LUT for correct index mapping)
torq-quantize-model analyze \
  -i model_reduced_vocab.onnx -o results.json \
  --config-output config.json \
  --embeddings token_embeddings.npy \
  --tokenizer tokenizer.json \
  --token-lut token_id_lut.npy \
  --bits 4 8
```

## Quantization Config JSON

The config file specifies per-layer quantization settings:

```json
{
  "default": { "bits": 4, "block_size": 32 },
  "layers": {
    "/model/layers.0/self_attn/q_proj/MatMul": { "bits": 8, "block_size": 32 },
    "/model/layers.0/self_attn/v_proj/MatMul": { "bits": 8, "block_size": 32 },
    "/lm_head/MatMul": { "bits": 16, "block_size": 32 }
  }
}
```

- `bits=8` — int8 asymmetric (ORT-matching), DequantizeLinear with uint8 weights + bf16 scales + uint8 zero points
- `bits=4` — int4 signed [-8,7], DequantizeLinear with int8 weights + bf16 scales + int8 zero points
- `bits=16` — keep as fp32/bf16 (no quantization)

Layers not listed in `layers` use the `default` settings.

## Sensitivity Analysis

The `analyze` subcommand measures per-layer quantization impact:

1. For each MatMul layer, quantize its weight independently
2. Run forward pass with calibration prompts
3. Compare output logits against fp32 baseline
4. Classify layers by KL divergence:
   - **CRITICAL** (KL > 1.0) — keep in bf16
   - **HIGH** (KL > 0.1) — keep in bf16
   - **MEDIUM** (KL > 0.01) — use int8
   - **LOW** (KL ≤ 0.01) — use int4

Output: sensitivity results JSON + quantization config JSON.

## Output Formats

| Mode | Output | Description |
|------|--------|-------------|
| `--bits N` | DQL model | DequantizeLinear + MatMul nodes, int8/int4 weights with bf16 scales |
| `--bits N --dequantize-weights` | bf16 model | Single bf16 model with quantization error baked in, ready for compilation |
| `--bits 16` | bf16 model | Pure fp32→bf16 conversion, no quantization |

## Quantization Details

### Int8 (ORT-matching asymmetric)

- Format: uint8 values [0, 255] with per-block zero points
- Block size: 32 (configurable)
- Axis: K dimension (axis=0 in K×N weight layout)
- Formula: `scale = (max - min) / 255`, `zp = round(-min / scale)`, `q = round(w / scale + zp)`
- Produces identical Q/S/ZP values as ORT's `MatMulNBits` quantizer

### Int4 (signed)

- Format: int8 values [-8, 7] (15 quantization levels)
- Block size: 32 (configurable)
- Formula: `scale = (max - min) / 15`, `zp = round(-8 - min / scale)`, `q = round(w / scale + zp)`

## IREE Compatibility

The `quantize` subcommand automatically fixes ONNX graph issues that prevent IREE import:

- **Slice INT64_MAX ends**: ONNX allows `ends=[INT64_MAX]` to mean "to the end," but IREE requires concrete bounds. The tool resolves these via shape inference before int64→int32 conversion.
- **Mixed-type DQL validation**: DQL models with bf16 scales produce `InferenceError` during `check_model`. This is caught and handled gracefully.

Both DQL and dequantized-bf16 outputs are directly importable by `iree.compiler.tools.import_onnx`.

## Benchmarking

Run quantized models through a standard question set and compare results.

### Run benchmark on board

```bash
python -m torq.tools.quantization.weight_quantization.benchmark run \
  -m /path/to/model.vmfb --instruct-model -o results_int8.json

python -m torq.tools.quantization.weight_quantization.benchmark run \
  -m /path/to/model_hybrid.vmfb --instruct-model -o results_hybrid.json
```

Options:
- `-m` — path to model VMFB (or ONNX)
- `--instruct-model` — use Gemma3 instruct chat template
- `--questions-file` — custom JSON list of questions (default: built-in 24 questions)
- `--temperature` — sampling temperature (default: 0.0 = greedy)
- `--runner-path` — path to directory containing `runner.py` (auto-detected if omitted)
- `-j` — number of inference threads

### Compare two benchmark results

```bash
python -m torq.tools.quantization.weight_quantization.benchmark compare \
  -a results_int8.json -b results_hybrid.json \
  --name-a "Pure Int8" --name-b "Hybrid Int8/Int4" \
  -o comparison.md
```

Generates a markdown report with:
- Summary table (TPS, TTFT, total tokens)
- Side-by-side answers for each question
- Per-question performance metrics
