#!/bin/bash
set -e

# ============================================================================
# Tsuki Part B — Golden ONNX Surgery Pipeline
# ============================================================================
# Produces B1 and B2 matching golden performance (B1≈1,369ms, B2≈1,599ms).
#
# KEY DIFFERENCES FROM PART A PIPELINE:
# - NO bf16 conversion (step 8) — model stays f32, compiler converts
# - NO convert_gemm (step 5c) — keeps MatMul+Add, avoids SDIM crash
# - NO fusion barriers, Gemm splitting, im2col, elementwise chunking
# - NO squeeze_batch_matmul — chunk_attention handles per-head splitting
# - Native Conv1D throughout — im2col not needed for Part B
#
# Source: tests/testdata/onnx_models/tsuki_static_new_fp32_split_stft_final_s50_4s/part_b_post_stft_4s.onnx
# Output: $OUT_DIR/part_b1.onnx and $OUT_DIR/part_b2.onnx
#
# Usage:
#   ./apply_patches_golden_b.sh [output_dir] [--start-from STEP]
# ============================================================================

PARENT="/home/breidy/iree-local-dev"
WD="$PARENT/torq-tools-dev/src/torq/models/tsuki"
DOCKER_TAG="profiler:5000/$(echo $PARENT | sed s/\\//_/g | cut -c2-)"
DOCK="docker run --rm \
    --env PATH=$PARENT/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PARENT/iree-build/third_party/iree/tools \
    --env HOME=$PARENT \
    --env PYTHONPATH=$PARENT/torq-tools-dev/src \
    --mount type=bind,src=$PARENT,dst=$PARENT \
    --mount type=bind,src=/tmp,dst=/tmp \
    --workdir $WD \
    -i $DOCKER_TAG"

SRC="$WD/model/part_b_post_stft_4s.onnx"
STAGE="/tmp/tsuki_pipeline_golden_b"
OUT_DIR=""
START_FROM=0

# Parse arguments
for arg in "$@"; do
    if [ "$prev_arg" = "--start-from" ]; then
        START_FROM="$arg"
        echo "=== Resuming from step $START_FROM (using cached intermediates) ==="
    elif [ "$arg" != "--start-from" ] && [ -z "$OUT_DIR" ]; then
        OUT_DIR="$arg"
    fi
    prev_arg="$arg"
done

OUT_DIR="${OUT_DIR:-/tmp/golden_b_split}"

mkdir -p "$STAGE"
mkdir -p "$OUT_DIR"

run_step() {
    local step_num="$1"
    [ "$step_num" -lt "$START_FROM" ] && return 1
    START_FROM=0
    return 0
}

# ============================================================================
# Phase 1: Base pipeline (docker python3, steps 1-7)
# ============================================================================

if run_step 1; then
echo "=== Step 1: Decompose unsupported ops ==="
$DOCK python3 edit_scripts/decompose_unsupported_ops.py \
    --input="$SRC" --output="$STAGE/step01.onnx"
fi

if run_step 2; then
echo "=== Step 2: Fold constants ==="
$DOCK python3 edit_scripts/fold_onnx_constants.py \
    --input="$STAGE/step01.onnx" --output="$STAGE/step02.onnx"
fi

if run_step 3; then
echo "=== Step 3: Convert normalization ops ==="
$DOCK python3 edit_scripts/convert_normalization_ops.py \
    --input="$STAGE/step02.onnx" --output="$STAGE/step03.onnx"
fi

if run_step 4; then
echo "=== Step 4: Convert conv2d to conv1d ==="
$DOCK python3 edit_scripts/convert_conv2d_to_conv1d.py \
    --input="$STAGE/step03.onnx" --output="$STAGE/step04.onnx"
fi

if run_step 5; then
echo "=== Step 5: Apply reducesum chunked mean mul ==="
$DOCK python3 edit_scripts/apply_reducesum_chunked_mean_mul.py \
    --input="$STAGE/step04.onnx" --output="$STAGE/step05.onnx"

echo "=== Step 5b: Decompose norm (torq-tools-dev) ==="
$DOCK python3 -m torq.tools.decompose_norm \
    -i "$STAGE/step05.onnx" -o "$STAGE/step05b.onnx"

# NOTE: Step 5c (convert_gemm) is SKIPPED for Part B.
# Golden B1 has only 2 Gemm ops (original). convert_gemm creates 42 Gemm
# with Squeeze/Unsqueeze wrappers that change fusion patterns and trigger
# SDIM tile size crashes.
fi

if run_step 6; then
echo "=== Step 6: Apply convtranspose phase matmul ==="
$DOCK python3 edit_scripts/apply_convtranspose_phase_matmul.py \
    --input="$STAGE/step05b.onnx" --output="$STAGE/step06.onnx" \
    --target-output convolution_1
fi

if run_step 7; then
echo "=== Step 7: Compiler patches (stft replacement, ConvTranspose, IsNaN fold) ==="
$DOCK python3 edit_scripts/compiler_patches.py \
    "$STAGE/step06.onnx" "$STAGE/step07.onnx" --skip-json
fi

# Step 7b: Convert any bf16 initializers to f32 (source model has bf16 Conv weights)
# Golden B1 is fully f32 — the compiler handles f32→bf16 via --torq-convert-dtypes.

if run_step 8; then
echo "=== Step 7b: Convert bf16 initializers to f32 ==="
python3 -c "
import onnx, numpy as np
from onnx import numpy_helper
m = onnx.load('$STAGE/step07.onnx')
count = 0
for i, init in enumerate(m.graph.initializer):
    if init.data_type == 16:
        arr = numpy_helper.to_array(init).astype(np.float32)
        m.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, name=init.name))
        count += 1
for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
    if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type == 16:
        vi.type.tensor_type.elem_type = 1
print(f'Converted {count} bf16 initializers to f32')
onnx.save(m, '$STAGE/step07b.onnx')
"
fi

# NOTE: Steps 8-19 (bf16 conversion, bool fixes, etc.) are SKIPPED for Part B.

# ============================================================================
# Phase 2: Remaining transforms (docker python3 for step 20, host for 24+)
# ============================================================================

if run_step 20; then
echo "=== Step 20: Revert tanh→sigmoid decompositions ==="
$DOCK python3 edit_scripts/revert_tanh_to_sigmoid.py \
    --input="$STAGE/step07b.onnx" --output="$STAGE/step20.onnx"
fi

# NOTE: Steps 22-23, 25-27 are SKIPPED for Part B.
# Golden B1 has 0 Pad ops (no fusion barriers), 0 Squeeze (no batch matmul
# squeezing), and no bool elimination or Gemm splitting.

if run_step 24; then
echo "=== Step 24: Chunk attention Q dim (per-head + per-chunk) ==="
python3 edit_scripts/chunk_attention.py \
    -i "$STAGE/step20.onnx" -o "$STAGE/step24.onnx" --num-chunks 8

# Step 24b DISABLED — removing Unsqueeze ops changes fusion patterns and triggers
# TileAndFuse assertion crash ("Unable to fuse a pattern fuse group member")
# echo "=== Step 24b: Merge ReduceMean+Unsqueeze back to keepdims=1 ==="
# python3 edit_scripts/merge_reducemean_unsqueeze.py \
#     -i "$STAGE/step24.onnx" -o "$STAGE/step24b.onnx"
cp "$STAGE/step24.onnx" "$STAGE/step24b.onnx"
fi

# ============================================================================
# Phase 3: Split into B1/B2
# ============================================================================

echo "=== Split into B1/B2 ==="
python3 edit_scripts/split_part_b.py \
    -i "$STAGE/step24b.onnx" --output-dir "$OUT_DIR"

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "=== Verification ==="
for part in part_b1 part_b2; do
python3 -c "
import onnx
from collections import Counter
m = onnx.load('$OUT_DIR/${part}.onnx')
nodes = list(m.graph.node)
ops = Counter(n.op_type for n in nodes)
init_names = {i.name for i in m.graph.initializer}
inputs = [i.name for i in m.graph.input if i.name not in init_names]
no_shape = sum(1 for vi in m.graph.value_info
               if vi.type.HasField('tensor_type') and not vi.type.tensor_type.HasField('shape'))
# Check model dtype
dtype = m.graph.input[0].type.tensor_type.elem_type if m.graph.input else 0
dtype_name = {1:'FLOAT', 10:'FLOAT16', 16:'BFLOAT16'}.get(dtype, f'unknown({dtype})')
print(f'  ${part}:')
print(f'    Nodes: {len(nodes)}, Dtype: {dtype_name}')
print(f'    Softmax: {ops.get(\"Softmax\",0)}, Sigmoid: {ops.get(\"Sigmoid\",0)}, Tanh: {ops.get(\"Tanh\",0)}')
print(f'    Conv: {ops.get(\"Conv\",0)}, MatMul: {ops.get(\"MatMul\",0)}, Gemm: {ops.get(\"Gemm\",0)}')
print(f'    Split: {ops.get(\"Split\",0)}, Concat: {ops.get(\"Concat\",0)}, Pad: {ops.get(\"Pad\",0)}')
print(f'    Squeeze: {ops.get(\"Squeeze\",0)}, Unsqueeze: {ops.get(\"Unsqueeze\",0)}')
print(f'    Transpose: {ops.get(\"Transpose\",0)}, Reshape: {ops.get(\"Reshape\",0)}')
print(f'    Missing shapes: {no_shape}')
print(f'    Inputs ({len(inputs)}): {inputs}')
"
done

echo ""
echo "=== Golden B1 reference ==="
echo "  Nodes: 2393, Dtype: FLOAT, Softmax: 128, Conv: 27, MatMul: 306"
echo "  Gemm: 2, Split: 26, Concat: 49, Pad: 0, Squeeze: 0"
echo ""
echo "=== DONE ==="
echo "Next: MLIR import (NO --data-prop) and compile:"
echo "  source ../easy_docker/env.sh"
echo "  dock python3 -m iree.compiler.tools.import_onnx $OUT_DIR/part_b1.onnx -o $OUT_DIR/part_b1.mlir"
echo "  dock torq-compile $OUT_DIR/part_b1.mlir -o $OUT_DIR/part_b1_board.vmfb \\"
echo "      --torq-hw=SL2610 --torq-target-host-triple=aarch64-linux-gnu \\"
echo "      --torq-convert-dtypes --torq-convert-conv1d-to-matmul=false \\"
echo "      --torq-convert-conv1d-to-generic=false --torq-convert-io-dtype"
