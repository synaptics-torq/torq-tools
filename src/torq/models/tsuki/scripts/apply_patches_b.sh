#!/bin/bash
set -e

# ============================================================================
# Tsuki Part B — Full ONNX Surgery Pipeline  [WIP]
# ============================================================================
# Produces B1 and B2 from the Part B source model.
#
# STATUS: Work-in-progress. We are currently still using the old pre-pipeline
# alpha_v01 B1 and B2 models for board testing. This pipeline has NOT yet been
# validated end-to-end on device. The alpha_v01 models live outside this repo
# (originally at /tmp/alpha_v01_part_b1.onnx and /tmp/alpha_v01_part_b2.onnx).
#
# Source: tests/testdata/onnx_models/tsuki_static_new_fp32_split_stft_final_s50_4s/part_b_post_stft_4s.onnx
# Output: $OUT_DIR/part_b1.onnx and $OUT_DIR/part_b2.onnx
#
# Phase 1 (steps 1-20):  Docker python3 — base ONNX surgery pipeline
# Phase 2 (steps 22-26): Host python3  — TileAndFuse compiler bug workarounds
# Phase 3 (steps 27-31): Host python3  — performance optimization
# Phase 4: Split into B1/B2
#
# IMPORTANT:
# - Steps 22+ use HOST python3, not docker, to avoid protobuf corruption
# - MLIR import must NOT use --data-prop (corrupts broadcast shapes)
# - Always run the FULL pipeline from step 1. Never run partial steps.
#
# Usage:
#   ./apply_patches_b.sh [output_dir] [--start-from STEP]
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
STAGE="/tmp/tsuki_pipeline_b"
OUT_DIR="${1:-/tmp/part_b_split}"
START_FROM=0

# Parse --start-from
for arg in "$@"; do
    if [ "$prev_arg" = "--start-from" ]; then
        START_FROM="$arg"
        echo "=== Resuming from step $START_FROM (using cached intermediates) ==="
    fi
    prev_arg="$arg"
done

mkdir -p "$STAGE"
mkdir -p "$OUT_DIR"

run_step() {
    local step_num="$1"
    [ "$step_num" -lt "$START_FROM" ] && return 1
    START_FROM=0
    return 0
}

# ============================================================================
# Phase 1: Base pipeline (docker python3, steps 1-20)
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

echo "=== Step 5c: Convert Gemm (torq-tools-dev) ==="
$DOCK python3 -m torq.tools.convert_gemm \
    -i "$STAGE/step05b.onnx" -o "$STAGE/step05c.onnx"
fi

if run_step 6; then
echo "=== Step 6: Apply convtranspose phase matmul ==="
# Part B's ConvTranspose output is 'convolution_1' (Part A is 'convolution')
$DOCK python3 edit_scripts/apply_convtranspose_phase_matmul.py \
    --input="$STAGE/step05c.onnx" --output="$STAGE/step06.onnx" \
    --target-output convolution_1
fi

if run_step 7; then
echo "=== Step 7: Compiler patches ==="
$DOCK python3 edit_scripts/compiler_patches.py \
    "$STAGE/step06.onnx" "$STAGE/step07.onnx"
fi

if run_step 8; then
echo "=== Step 8: Make full bf16 (no casts) ==="
$DOCK python3 edit_scripts/make_full_bf16_no_casts.py \
    --input="$STAGE/step07.onnx" --output="$STAGE/step08.onnx"
fi

if run_step 9; then
echo "=== Step 9: Fix cast value_info ==="
$DOCK python3 edit_scripts/fix_cast_value_info.py \
    --input="$STAGE/step08.onnx" --output="$STAGE/step09.onnx"
fi

if run_step 10; then
echo "=== Step 10: Wrap Where/ArgMax in fp32 ==="
$DOCK python3 edit_scripts/wrap_where_argmax_in_fp32.py \
    --input="$STAGE/step09.onnx" --output="$STAGE/step10.onnx"
fi

if run_step 11; then
echo "=== Step 11: Wrap int64 Mul in bf16 ==="
$DOCK python3 edit_scripts/wrap_int64_mul_in_bf16.py \
    --input="$STAGE/step10.onnx" --output="$STAGE/step11.onnx"
fi

if run_step 12; then
echo "=== Step 12: Replace upsample nearest1d ==="
$DOCK python3 edit_scripts/replace_upsample_nearest1d_safe.py \
    --input="$STAGE/step11.onnx" --output="$STAGE/step12.onnx"
fi

if run_step 13; then
echo "=== Step 13: Replace single-index Gather ==="
$DOCK python3 edit_scripts/replace_single_index_gather.py \
    --input="$STAGE/step12.onnx" --output="$STAGE/step13.onnx"
fi

if run_step 14; then
echo "=== Step 14: Replace small ScatterElements ==="
$DOCK python3 edit_scripts/replace_small_scatter_elements.py \
    --input="$STAGE/step13.onnx" --output="$STAGE/step14.onnx"
fi

if run_step 15; then
echo "=== Step 15: Strip redundant BOOL→BF16 casts ==="
$DOCK python3 edit_scripts/strip_redundant_bool_to_bf16_casts.py \
    --input="$STAGE/step14.onnx" --output="$STAGE/step15.onnx"
fi

if run_step 16; then
echo "=== Step 16: Replace dynamic Slice with one-hot ==="
$DOCK python3 edit_scripts/replace_dynamic_slice_with_onehot.py \
    --input="$STAGE/step15.onnx" --output="$STAGE/step16.onnx"
fi

if run_step 17; then
echo "=== Step 17: Replace Equal with INT32 arithmetic ==="
$DOCK python3 edit_scripts/replace_equal_with_int_arithmetic.py \
    --input="$STAGE/step16.onnx" --output="$STAGE/step17.onnx"
fi

if run_step 18; then
echo "=== Step 18: Cast BOOL to INT8 ==="
$DOCK python3 edit_scripts/cast_bool_to_int8.py \
    --input="$STAGE/step17.onnx" --output="$STAGE/step18.onnx"
fi

if run_step 19; then
echo "=== Step 19: Replace bool logic (Not/Or/And) with INT8 ==="
$DOCK python3 edit_scripts/replace_bool_logic_with_int8.py \
    --input="$STAGE/step18.onnx" --output="$STAGE/step19.onnx"
fi

if run_step 20; then
echo "=== Step 20: Revert tanh→sigmoid decompositions ==="
$DOCK python3 edit_scripts/revert_tanh_to_sigmoid.py \
    --input="$STAGE/step19.onnx" --output="$STAGE/step20.onnx"
fi

# Step 21: SKIP — Part A specific (split_part_a_at_mask_boundary)
# No boundary barriers needed for Part B at this stage.
# The B1/B2 split happens at the end (Phase 4).
echo "=== Step 21: SKIPPED (Part A specific) ==="
cp "$STAGE/step20.onnx" "$STAGE/step21b.onnx"

# ============================================================================
# Phase 2: TileAndFuse bug workarounds (HOST python3, steps 22-26)
# ============================================================================

if run_step 22; then
echo "=== Step 22: Eliminate BOOL ops → INT8 arithmetic ==="
python3 edit_scripts/eliminate_bool_ops.py \
    -i "$STAGE/step21b.onnx" -o "$STAGE/step22.onnx"
fi

if run_step 23; then
echo "=== Step 23: Insert fusion barriers before attention ==="
python3 edit_scripts/insert_fusion_barriers.py \
    -i "$STAGE/step22.onnx" -o "$STAGE/step23.onnx"
fi

if run_step 24; then
echo "=== Step 24: Chunk attention Q dim into 8 pieces ==="
python3 edit_scripts/chunk_attention.py \
    -i "$STAGE/step23.onnx" -o "$STAGE/step24.onnx" --num-chunks 8
fi

if run_step 25; then
echo "=== Step 25: Insert Concat barriers on decoder outputs ==="
python3 edit_scripts/insert_concat_barriers.py \
    -i "$STAGE/step24.onnx" -o "$STAGE/step25.onnx" --name-prefix node_cat_
fi

if run_step 26; then
echo "=== Step 26: Split large Gemm ops ≤128KB blocks ==="
python3 edit_scripts/split_large_gemm.py \
    -i "$STAGE/step25.onnx" -o "$STAGE/step26.onnx"
fi

# ============================================================================
# Phase 3: Performance optimization (HOST python3, steps 27-31)
# ============================================================================

if run_step 27; then
echo "=== Step 27: Squeeze batch MatMul to 2D per-head ==="
python3 edit_scripts/squeeze_batch_matmul.py \
    -i "$STAGE/step26.onnx" -o "$STAGE/step27.onnx"
fi

if run_step 30; then
echo "=== Step 30: Conv1D → im2col + Gemm (ALL group=1 Conv1D — native crashes) ==="
python3 edit_scripts/conv1d_im2col.py \
    -i "$STAGE/step27.onnx" -o "$STAGE/step30.onnx" --min-weight-bytes 0 --per-kernel

echo "=== Step 26b: Extra Gemm split (lower threshold, includes im2col Gemms) ==="
python3 edit_scripts/split_large_gemm.py \
    -i "$STAGE/step30.onnx" -o "$STAGE/step26b.onnx" \
    --min-weight-bytes 60000 --max-block-bytes 32768
fi

if run_step 29; then
echo "=== Step 29: Chunk large elementwise ops (1MB threshold) ==="
python3 edit_scripts/chunk_large_elementwise.py \
    -i "$STAGE/step26b.onnx" -o "$STAGE/step29.onnx" \
    --barrier-threshold 32768 --max-io 1048576
fi

if run_step 33; then
echo "=== Step 33: Chunk large reductions (>65536 NDL descriptor limit) ==="
python3 edit_scripts/chunk_large_reduce.py \
    -i "$STAGE/step29.onnx" -o "$STAGE/step33.onnx"
fi

# Step 32: SKIP — Part B has no GatherND
echo "=== Step 32: SKIPPED (no GatherND in Part B) ==="
cp "$STAGE/step33.onnx" "$STAGE/step32.onnx"

if run_step 31; then
echo "=== Step 31: Eliminate ReduceMean axis transposes ==="
python3 edit_scripts/eliminate_reducemean_transposes.py \
    -i "$STAGE/step32.onnx" -o "$STAGE/part_b_processed.onnx"
fi

# ============================================================================
# Phase 4: Split into B1/B2
# ============================================================================

echo "=== Phase 4: Split into B1/B2 ==="
python3 edit_scripts/split_part_b.py \
    -i "$STAGE/part_b_processed.onnx" -o "$OUT_DIR"

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "=== Verification ==="
for part in part_b1 part_b2; do
python3 -c "
import onnx
m = onnx.load('$OUT_DIR/${part}.onnx')
nodes = len(m.graph.node)
transposes = sum(1 for n in m.graph.node if n.op_type == 'Transpose')
convs = sum(1 for n in m.graph.node if n.op_type == 'Conv')
inits = len(m.graph.initializer)
no_shape = sum(1 for vi in m.graph.value_info
               if vi.type.HasField('tensor_type') and not vi.type.tensor_type.HasField('shape'))
print(f'  ${part}: {nodes} nodes, {transposes} transposes, {convs} Conv, {inits} inits, {no_shape} missing shapes')
"
done
echo ""
echo "=== DONE ==="
echo "Next: MLIR import (NO --data-prop) and compile each part:"
echo "  docker ... python3 -m iree.compiler.tools.import_onnx $OUT_DIR/part_b1.onnx -o part_b1.mlir"
echo "  docker ... torq-compile part_b1.mlir -o part_b1_board.vmfb \\"
echo "      --torq-hw=SL2610 --torq-target-host-triple=aarch64-linux-gnu \\"
echo "      --torq-convert-dtypes --torq-convert-conv1d-to-matmul=false \\"
echo "      --torq-convert-conv1d-to-generic=false --torq-convert-io-dtype"
