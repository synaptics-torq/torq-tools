# Dynamic Quantization Tool

Dynamically quantize fp32 ONNX models to int8 using `onnxruntime`. Weights are quantized ahead of time; activation scales are computed on the fly at inference, so no calibration dataset is required to quantize.

This is the `dynamic` method of `torq-quantize-model`; the `weights` method (int4/int8/bf16 weight-only quantization with per-layer mixed precision) lives under `torq.tools.quantization.weight_quantization`.

## Usage

```bash
# --- Quantize ---

# Default: per-channel int8 weights
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx

# Unsigned int8 weights, per-tensor
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx --uint8-weights --per-tensor

# Restrict to specific op types or nodes
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx --quantize-only-ops MatMul Gemm
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx --quantize-only-nodes /model/layers.0/MatMul

# Exclude sensitive nodes (e.g. an analysis exclude list) from quantization
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx --exclude-nodes /lm_head/MatMul

# Skip pre-processing, or forward advanced onnxruntime options (must be last)
torq-quantize-model dynamic quantize \
  -i model_fp32.onnx -o model_int8.onnx --skip-preprocess \
  --extra-quant-args reduce_range True

# --- Analyze (per-node sensitivity) ---

# Rank nodes by how much quantizing each one hurts the outputs
torq-quantize-model dynamic analyze \
  -i model_fp32.onnx -o sensitivity_report.json

# Also emit an exclude list of the too-sensitive nodes, ready to feed back in
torq-quantize-model dynamic analyze \
  -i model_fp32.onnx -o sensitivity_report.json \
  --exclude-output exclude_nodes.json --exclude-class HIGH

# Use real calibration inputs and widen the node types tested
torq-quantize-model dynamic analyze \
  -i model_fp32.onnx -o sensitivity_report.json \
  --calibration-data calib.npz \
  --op-types MatMul Gemm --skip-nodes embed_tokens
```

## Analyze → exclude → quantize workflow

The `analyze` subcommand measures per-node quantization impact so you can quantize aggressively while sparing the layers that matter:

1. For each candidate node (`--op-types`, default `MatMul Gemm`), quantize **only that node**.
2. Run the model on calibration inputs and compare its outputs against the fp32 baseline.
3. Score the node by output divergence and classify it by severity.
4. Sort nodes most-sensitive first and (optionally) write the sensitive ones to an exclude list.

Then quantize everything except the sensitive nodes:

```bash
torq-quantize-model dynamic analyze  -i model.onnx -o report.json --exclude-output exclude.json
torq-quantize-model dynamic quantize -i model.onnx -o model_int8.onnx \
  --exclude-nodes $(python -c "import json,sys; print(' '.join(json.load(open('exclude.json'))))")
```

### Metrics

Each node is scored on three metrics, aggregated across all model outputs (shared with the `weights analyze` tool via `torq.utils.metrics`):

- **KL divergence** — softmax-distribution drift on each output's last axis (worst across outputs). Used for classification.
- **Cosine similarity** — direction agreement of the flattened outputs (worst across outputs).
- **Max absolute error** — largest elementwise deviation (worst across outputs).

Nodes are classified by KL divergence:

- **CRITICAL** (KL > 1.0)
- **HIGH** (KL > 0.1)
- **MEDIUM** (KL > 0.01)
- **LOW** (KL ≤ 0.01)

`--exclude-class` (default `HIGH`) sets the threshold at/above which a node joins the exclude list.

### Report format

The report is a JSON list, most-sensitive first:

```json
[
  { "node": "/model/layers.7/mlp/down_proj/MatMul", "op_type": "MatMul", "kl": 0.83, "cosine": 0.94, "max_abs_error": 2.11, "classification": "HIGH" },
  { "node": "/model/layers.0/self_attn/q_proj/MatMul", "op_type": "MatMul", "kl": 0.0004, "cosine": 0.999, "max_abs_error": 0.02, "classification": "LOW" }
]
```

### Calibration inputs

By default, analysis feeds **seeded random inputs** derived from the model's input shapes (unknown/dynamic dimensions default to 1), so it works on any model with zero setup. For representative results, pass real inputs with `--calibration-data path/to/inputs.npz`, where the archive's keys match the model's input names.

## Notes

- Both `quantize` and `analyze` run pre-processing by default (`--skip-preprocess` to opt out); `analyze` pre-processes once and reuses stable node names across the per-node sweep.
- `analyze` runs one quantize + inference pass per candidate node, so runtime scales with the number of `--op-types` nodes — narrow it with `--op-types` / `--skip-nodes` on large models.
