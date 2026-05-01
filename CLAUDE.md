# Gemma-3 Model Conversion Pipeline

## Overview

The gemma-3-270m-it model is converted from HuggingFace quantized ONNX (int4 or int8) to a static bf16 VMFB for the torq board using `torq-tools-dev`.

## Directory Structure

```
models/google/gemma-3-270m-it/
├── source/
│   ├── fp32/                   # Original fp32 model (from optimum export)
│   │   ├── model.onnx
│   │   ├── config.json
│   │   └── tokenizer.json
│   ├── int4/                   # ORT-quantized int4 source (MatMulNBits bits=4)
│   │   ├── model.onnx          # Full int4 model
│   │   └── model_q4.onnx       # Smaller variant (quantized embeddings)
│   ├── int8/                   # ORT-quantized int8 source (MatMulNBits bits=8)
│   │   ├── model_quantized.onnx
│   │   └── model_quantized.onnx_data
│   └── int4_converted/         # Intermediate: with custom ops replaced
│       └── model.onnx
├── export/onnx/
│   ├── bf16/                   # bf16 original (for weight comparison)
│   │   └── model_gemma3_bf16.onnx
│   ├── int4/
│   │   └── static/             # Static int4 model (dequantized weights)
│   │       ├── model.onnx
│   │       └── token_embeddings.npy
│   ├── int4_bf16_sim/
│   │   └── static/             # bf16-simulated int4 (fp32 I/O, bf16 precision)
│   │       ├── model.onnx
│   │       └── token_embeddings.npy
│   ├── converted/
│   │   └── static/             # bf16 dtype-converted (for IREE/VMFB compile)
│   │       ├── model.onnx
│   │       ├── model.mlir
│   │       ├── model_local.vmfb
│   │       └── token_embeddings.npy
│   ├── int8/
│   │   └── static/             # Static int8 model (dequantized, fp32)
│   │       └── model.onnx
│   ├── int8_converted/
│   │   └── static/             # int8 bf16-converted (for IREE compile)
│   │       └── model.onnx
│   ├── int8_int4_mixed_1/
│   │   └── static/             # Hybrid: int4 all layers + int8 lm_head
│   │       ├── model.onnx
│   │       ├── model.vmfb
│   │       └── token_embeddings.npy
│   └── mixed_bf16/             # Custom: bf16 weights + int4 lm_head only
│       └── model.onnx
└── export/iree/                # Final VMFB for board deployment
```

## Conversion Pipeline

Entry point: `src/torq/models/gemma3/export.py` → `Gemma3ModelExporter`

### Stage 1: Load int4 source (`_load_onnx_int4`)

Source: `source/int4/model.onnx` (or `model_q4.onnx`)

Graph edits performed:
- Replace `SimplifiedLayerNormalization` → standard ONNX ops
- Replace `SkipSimplifiedLayerNormalization` → standard ONNX ops
- `--dequantize-weights`: Replace `MatMulNBits` → fp32 `MatMul` (dequantizes int4 weights inline)
- `--dequantize-weights-linear`: Replace `MatMulNBits` → `DequantizeLinear+Reshape+MatMul` (runtime dequant)
- Replace `GroupQueryAttention` → standard attention ops
- `--extract-embeddings`: Extract `GatherBlockQuantized` embeddings → `token_embeddings.npy`
- Save intermediate to `source/int4_converted/model.onnx`

### Stage 2: Make static (`_make_model_static`)

- Replace dynamic KV cache → fixed-size tensors (`max_gen_tokens=256`)
- Add `position_ids` input, causal attention mask
- Rewire `seqlen_k` → `position_ids` for RoPE indexing
- Output: `export/onnx/int4/static/model.onnx`

### Stage 3: Post-static patches (`_patch_static_model`)

- Eliminate no-op transposes (data-preserving)
- Collapse consecutive Reshape chains
- Collapse Unsqueeze→Expand→Reshape GQA broadcast → single Expand
- Fold scalar MatMul → Mul
- Combine individual KV I/O into merged tensors
- `--extract-embeddings`: Extract/copy `token_embeddings.npy` to export dir

### Stage 4: bf16 simulation (`--simulate-bf16`)

Creates `export/onnx/int4_bf16_sim/static/model.onnx`:
- **Weights**: Round-trip all fp32 constants through bf16 (lossy quantization)
- **Activations**: Insert `Cast(fp32→bf16)→Cast(bf16→fp32)` after every non-Cast node
- I/O stays fp32 so ORT can run normally
- Purpose: Measure bf16 quantization impact without changing graph structure

### Stage 5: bf16 dtype conversion (`--convert-dtypes`)

Uses `torq.tools.convert_dtype.onnx`:
- Convert all fp32 weights and activations → bf16
- Output: `export/onnx/converted/static/model.onnx`
- This is the model compiled to VMFB via IREE

### Stage 6: IREE compile (unless `--skip-iree`)

- Import ONNX → MLIR: `python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir --data-prop`
- Compile MLIR → VMFB with torq backend

## CLI Commands

### Full int4 export pipeline
```bash
cd torq-tools-dev
source .venv/bin/activate
PYTHONPATH=src python -m torq.models.gemma3.export \
    --model-dtype int4 \
    --instruct-model \
    --extract-embeddings \
    --dequantize-weights \
    --simulate-bf16 \
    --convert-dtypes
```

### Int8 export (reuses int4 pipeline with --onnx-source-dir)
```bash
PYTHONPATH=src python -m torq.models.gemma3.export \
    --model-dtype int4 \
    --instruct-model \
    --extract-embeddings \
    --dequantize-weights \
    --convert-dtypes \
    --skip-iree \
    --skip-validation \
    --onnx-source-dir models/google/gemma-3-270m-it/source/int8
```

**Note:** `--model-dtype int4` is reused for int8 because the pipeline handles both
(MatMulNBits with `bits=4` or `bits=8`). `--onnx-source-dir` overrides the source.

### Key flags
| Flag | Description |
|------|-------------|
| `--model-dtype int4` | Use int4 quantized source model |
| `--instruct-model` | Use the `-it` (instruct-tuned) variant |
| `--extract-embeddings` | Extract token embeddings to `.npy` (input becomes embedding vector) |
| `--dequantize-weights` | Dequantize MatMulNBits → fp32 MatMul (bake weights) |
| `--dequantize-weights-linear` | MatMulNBits → DequantizeLinear+MatMul (runtime dequant) |
| `--simulate-bf16` | Create bf16-simulated copy for precision analysis |
| `--convert-dtypes` | Convert fp32 → bf16 dtype for IREE compile |
| `--skip-iree` | Skip VMFB compilation |
| `--skip-validation` | Skip ORT validation against reference |
| `--max-gen-tokens N` | Max sequence length (default: 256) |
| `--model-size 270m\|1b` | Model size variant |
| `--broadcast-ops [OP...]` | Broadcast op inputs to match output shape |
| `--keep-individual-kv-io` | Keep separate K,V I/O instead of merged |

## Compile & Deploy

### Compile ONNX → VMFB
```bash
cd /home/kshanmug/synpu_compiler/torq-compiler-dev
source ../venv/bin/activate

# Compile (outputs model.vmfb next to model.onnx)
./compile_v1.5.sh ../torq-tools-dev/models/google/gemma-3-270m-it/export/onnx/int8_int4_mixed_1/static/model.onnx \
    --torq-enable-transpose-optimization \
    --torq-enable-torq-hl-tiling
```

### SCP to board
```bash
# int8_int4 hybrid model
scp ../torq-tools-dev/models/google/gemma-3-270m-it/export/onnx/int8_int4_mixed_1/static/model.vmfb \
    root@10.3.10.55:/home/root/torq-examples/models/Synaptics/gemma-3-270m-it/model_int8.vmfb

# int4 model (original)
scp ../torq-tools-dev/models/google/gemma-3-270m-it/export/onnx/converted/static/model.vmfb \
    root@10.3.10.55:/home/root/torq-examples/models/Synaptics/gemma-3-270m-it/int4/model.vmfb
```

## Board Inference

### Running on board (10.3.10.55)
```bash
cd ~/torq-examples/gemma3
source ../.venv/bin/activate

# int4 model
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it/int4/model.vmfb --instruct-model

# int8_int4 hybrid model
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it/model_int8.vmfb --instruct-model
```

### Non-interactive (single prompt)
```bash
python src/infer_no_prompt.py \
    -m ../models/Synaptics/gemma-3-270m-it/model_int8.vmfb \
    --instruct-model \
    -p "What is the capital of France?"
```

## Key Model Details

| Property | Value |
|----------|-------|
| Hidden size | 640 |
| Vocab size | 262,144 |
| Num layers | 18 |
| Num attention heads | config-dependent |
| Num KV heads | 1 |
| Head dim | 256 |
| Max seq len (static) | 256 |
| BOS token ID | 2 |
| EOS token ID | 1 |
| End-of-turn ID | 106 |
| System prompt | "You are a helpful AI assistant named Gemma. Answer in 1-2 sentences. No lists, no bullet points, no repetition." |

## Key Files

| File | Purpose |
|------|---------|
| `src/torq/models/gemma3/export.py` | Main export pipeline |
| `src/torq/models/gemma3/_graph.py` | `Gemma3OnnxGraphEditor` - graph surgery |
| `src/torq/models/gemma3/_inference.py` | `Gemma3Static`, `Gemma3Dynamic` inference runners |
| `src/torq/tools/convert_dtype/onnx.py` | fp32→bf16 dtype converter |
| `src/torq/model_export/onnx.py` | Base ONNX exporter, ORT optimizer |
| `src/torq/model_export/hf.py` | HuggingFace optimum ONNX export |

## Weight Comparison Notes

### lm_head weight (640 × 262,144)
- bf16 name: `onnx::MatMul_7445_bf16`
- int4 name: `/lm_head/MatMul/weight_dequantized_bf16`
- Cosine similarity: mean=0.994, min=0.958, all rows > 0.95
- The int4_converted model stores dequantized weights inline (not block-quantized format)
- >16 unique values per block because `int4_val * scale` produces many distinct bf16 values

### int8 vs int4 quality
- int8 produces significantly better answers (correct facts, coherent sentences)
- int4 often hallucinates ("capital of India = United States of Philippines")
- int8 model uses same pipeline via `--onnx-source-dir` pointing to int8 source
- Fixed `_dequantize_matmulnbits_weights()` in `edits.py` to handle `bits=8` (uint8, no nibble unpacking)

### Hybrid int8_int4_mixed_1
- All layer weights: int4 dequantized (from int4 source)
- lm_head weight only: int8 dequantized (from int8 source)
- Script: `create_hybrid_v2.py`

### VMFB repetition issue
- int4 quantization can shift the `end_of_turn` token out of argmax position
- Fix: Check if `eos` or `end_of_turn` is in top-10 logits → force stop
- Applied in board's `runner.py` `_sample()` method
