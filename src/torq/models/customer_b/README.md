# Customer B Model Pipeline

End-to-end pipeline for Customer B ONNX models: export to mixed-precision TFLite (int16 activations × int8 weights for FC ops), compile to VMFB for SL2610, run inference, and validate.

## Pipeline

### all_fc / all_conv (generic pipeline)

```
ONNX (.onnx)
  → onnx2tf → SavedModel → fp32 TFLite (.tflite)
    → quantize_fc_ops_in_tflite → mixed int16×int8 TFLite
      → iree-import-tflite → TOSA (.tosa)
        → iree-opt → text MLIR (.mlir)
          → torq-compile → VMFB (.vmfb)
```

Also produces an int8 TFLite + VMFB for comparison (will be removed once mixed-precision is validated).

### all_lstm (specialised pipeline)

```
ONNX (.onnx)
  → manual Keras build → SavedModel → fp32 TFLite (.tflite)
    → quantize_fc_ops_in_tflite → mixed int16×int8 TFLite
      → iree-import-tflite → TOSA (.tosa)
        → iree-opt → text MLIR (.mlir)
          → torq-compile → VMFB (.vmfb)
```

The LSTM model uses a manual Keras reconstruction (4 LSTMs decomposed into per-gate Dense layers) because `onnx2tf` fails on its LSTM state shapes.

## Models

Source ONNX models are in `models/customer_b/`:
- `all_conv.onnx` — convolutional layers model
- `all_fc.onnx` — fully-connected layers model
- `all_lstm.onnx` — LSTM layers model

## Quick Start

### 1. Export (ONNX → TFLite → MLIR → VMFB)

```bash
# Activate the onnx2tf venv (needs specific onnx/tf versions)
source .venv_customer_b/bin/activate

# Export all components
python -m torq.models.customer_b.export --models-dir models/customer_b

# Export only all_fc
python -m torq.models.customer_b.export --models-dir models/customer_b --component all_fc

# Export only all_lstm
python -m torq.models.customer_b.export --models-dir models/customer_b --component all_lstm

# Skip TFLite step (reuse existing .tflite)
python -m torq.models.customer_b.export --skip-tflite --models-dir models/customer_b

# Skip IREE compilation (only do ONNX → TFLite → MLIR)
python -m torq.models.customer_b.export --skip-iree --models-dir models/customer_b
```

### 2. Compile with profiling & IR dumps (in torq-compiler-dev)

```bash
cd ~/synpu_compiler/torq-compiler-dev
source ~/synpu_compiler/venv/bin/activate

# Copy the MLIR to tosa_ops testdata first
cp ~/synpu_compiler/torq-tools-dev/output_customer_b/all_fc/all_fc_fc_int16x8_mixed.mlir \
   tests/testdata/tosa_ops/

# Compile + run on board with profiling and IR dumps
pytest tests/test_tosa_ops.py -k all_fc_fc_int16x8_mixed -v -s \
  --extra-torq-compiler-options="--torq-convert-dtypes --torq-enable-torq-hl-tiling --torq-convert-io-dtype --torq-enable-transpose-optimization" \
  --torq-runtime-hw-type=astra_machina \
  --torq-addr=root@10.3.120.54 \
  --torq-compiler-timeout=5000 \
  --torq-compile-time-profiling-output-dir=profile-compile \
  --torq-runtime-profiling-output-dir=profile-runtime \
  --debug-ir dump-ir \
  --recompute-cache
```

### 3. Inference

```bash
# Run inference (auto-detects TFLite and/or VMFB)
python -m torq.models.customer_b.infer -m output_customer_b/all_fc

# With specific input data
python -m torq.models.customer_b.infer -m output_customer_b/all_fc --input-file input.npy
```

### 4. Validate (compare TFLite vs IREE output)

```bash
# Auto-discover and validate all components
python -m torq.models.customer_b.validate --output-dir output_customer_b

# Validate a specific TFLite/VMFB pair
python -m torq.models.customer_b.validate \
    --tflite output_customer_b/all_fc/all_fc/all_fc_int8.tflite \
    --vmfb output_customer_b/all_fc/all_fc_int8.vmfb

# Custom tolerance
python -m torq.models.customer_b.validate --output-dir output_customer_b --int-tol 2
```

## Output Directory Structure

```
output_customer_b/
  all_fc/
    all_fc/
      all_fc.onnx                       # sanitized copy
      all_fc_float32.tflite             # fp32 TFLite (from onnx2tf)
      all_fc_int8.tflite                # int8 quantized TFLite
      all_fc_fc_int16x8_mixed.tflite    # mixed-precision TFLite (int16×int8 FC)
      saved_model.pb                    # TF SavedModel
    all_fc_int8.mlir                    # int8 MLIR
    all_fc_int8.vmfb                    # int8 VMFB
    all_fc_fc_int16x8_mixed.mlir        # mixed-precision MLIR
    all_fc_fc_int16x8_mixed.vmfb        # mixed-precision VMFB
  all_conv/
    ...
  all_lstm/
    all_lstm/
      saved_model/                      # Keras SavedModel (manual build)
      all_lstm_fp32.tflite              # fp32 TFLite
      all_lstm_fc_int16x8_mixed.tflite  # mixed-precision TFLite
    all_lstm_fc_int16x8_mixed.mlir      # mixed-precision MLIR
    all_lstm_fc_int16x8_mixed.vmfb      # mixed-precision VMFB
```

## Key Files

- `export.py` — main export pipeline (ONNX → TFLite → MLIR → VMFB)
- `quantize_fc_tflite.py` — patches FC ops in fp32 TFLite to int16×int8 mixed-precision
- `__init__.py` — model component registry and CLI arg definitions
- `infer.py` — inference runner
- `validate.py` — TFLite vs VMFB comparison

## Dependencies

See `requirements.txt`. The ONNX → TFLite conversion requires the `.venv_customer_b` virtual environment with specific `onnx2tf` and `tensorflow` versions.
