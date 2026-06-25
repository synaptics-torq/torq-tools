#!/bin/bash
set -e

# ============================================================================
# Tsuki Part A — Full ONNX Surgery Pipeline
# ============================================================================
# Produces the best unsplit part_a model from the original fp32 source.
#
# Source: tests/testdata/onnx_models/tsuki_static_new_fp32_split_stft_final_s50_4s/part_a_pre_stft_4s.onnx
# Output: $OUT (default: /tmp/part_a_best.onnx)
#
# Phase 1 (steps 1-20):  Docker python3 — base ONNX surgery pipeline
# Phase 2 (steps 22-26): Host python3  — TileAndFuse compiler bug workarounds
# Phase 3 (steps 27-31): Host python3  — performance optimization
#
# IMPORTANT:
# - Steps 22+ use HOST python3, not docker, to avoid protobuf corruption
# - MLIR import must NOT use --data-prop (corrupts broadcast shapes)
# - Always run the FULL pipeline from step 1. Never run partial steps.
#
# Usage:
#   ./apply_patches.sh [output.onnx] [--start-from STEP]
#
# --start-from STEP: Skip steps before STEP, resume from cached intermediates.
#   Requires /tmp/tsuki_pipeline/step*.onnx from a previous full run.
#   Example: ./apply_patches.sh /tmp/out.onnx --start-from 26
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

SRC="$WD/model/part_a_pre_stft_4s.onnx"
STAGE="/tmp/tsuki_pipeline"
OUT="${1:-/tmp/part_a_best.onnx}"
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

run_step() {
    local step_num="$1"
    [ "$step_num" -lt "$START_FROM" ] && return 1
    START_FROM=0  # once we start, run all subsequent steps
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
$DOCK python3 edit_scripts/apply_convtranspose_phase_matmul.py \
    --input="$STAGE/step05c.onnx" --output="$STAGE/step06.onnx" \
    --target-output convolution
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

echo "=== Step 8b: Fix view_175 precision (INT32 floor + fp32 frac) ==="
python3 edit_scripts/fix_view175_precision.py \
    -i "$STAGE/step08.onnx" -o "$STAGE/step08.onnx" \
    --fp32-ref "$STAGE/step07.onnx"

echo "=== Step 8c: Fix add_18087 precision (bucket decomposition) ==="
python3 edit_scripts/fix_add18087_precision.py \
    -i "$STAGE/step08.onnx" -o "$STAGE/step08.onnx"
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
echo "=== Step 12: Replace upsample nearest1d Gather ops (exact) ==="
python3 -c "
import onnx; from onnx import shape_inference
m = shape_inference.infer_shapes(onnx.load('$STAGE/step11.onnx'))
onnx.save(m, '$STAGE/step11.onnx')
print('Shape inference done')
"
$DOCK python3 edit_scripts/replace_upsample_nearest1d_exact.py \
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

# Step 21 (temp): Split to get boundary tensor names, then insert barriers
# in the unsplit model. This prevents TileAndFuse from fusing across the
# encoder-decoder boundary.
if run_step 21; then
echo "=== Step 21 (temp): Split to extract boundary tensor names ==="
TMPDIR_SPLIT="$STAGE/tmp_split"
mkdir -p "$TMPDIR_SPLIT"
$DOCK python3 edit_scripts/split_part_a_at_mask_boundary.py \
    --input="$STAGE/step20.onnx" --output-dir="$TMPDIR_SPLIT"

echo "=== Step 21b: Insert boundary barriers (unsplit model) ==="
python3 edit_scripts/insert_boundary_barriers.py \
    -i "$STAGE/step20.onnx" -o "$STAGE/step21b.onnx" \
    --from-split-model "$TMPDIR_SPLIT/part_a2_post_mask.onnx"

rm -rf "$TMPDIR_SPLIT"
fi

# ============================================================================
# Phase 2: TileAndFuse bug workarounds (HOST python3, steps 22-26)
# IMPORTANT: Use host python3 to avoid docker protobuf corruption
# ============================================================================

#if run_step 22; then
#echo "=== Step 22: Eliminate BOOL ops → INT8 arithmetic (Bugs A+B) ==="
#python3 edit_scripts/eliminate_bool_ops.py \
#    -i "$STAGE/step21b.onnx" -o "$STAGE/step22.onnx"
#fi
if run_step 22; then
echo "=== Step 22: Eliminate BOOL ops → INT8 arithmetic (Bugs A+B) ==="
python3 edit_scripts/eliminate_bool_ops_safe.py \
    -i "$STAGE/step21b.onnx" -o "$STAGE/step22.onnx"
fi

if run_step 23; then
echo "=== Step 23: Insert fusion barriers before attention (Bug C) ==="
python3 edit_scripts/insert_fusion_barriers.py \
    -i "$STAGE/step22.onnx" -o "$STAGE/step23.onnx"
fi

if run_step 24; then
echo "=== Step 24: Chunk attention Q dim into 8 pieces (Bug C) ==="
python3 edit_scripts/chunk_attention.py \
    -i "$STAGE/step23.onnx" -o "$STAGE/step24.onnx" --num-chunks 8
fi

if run_step 25; then
echo "=== Step 25: Insert Concat barriers on decoder outputs (Bug D) ==="
python3 edit_scripts/insert_concat_barriers.py \
    -i "$STAGE/step24.onnx" -o "$STAGE/step25.onnx" --name-prefix node_cat_
fi

if run_step 26; then
echo "=== Step 26: Split large Gemm ops ≤128KB blocks (Bug E / LRAM) ==="
python3 edit_scripts/split_large_gemm.py \
    -i "$STAGE/step25.onnx" -o "$STAGE/step26.onnx"
fi

# ============================================================================
# Phase 3: Performance optimization (HOST python3, steps 27-31)
# ============================================================================

# Step 27 (squeeze_batch_matmul) disabled — adds too many ops for split compilation
# echo "=== Step 27: Squeeze batch MatMul to 2D per-head ==="
# python3 edit_scripts/squeeze_batch_matmul.py \
#     -i "$STAGE/step26.onnx" -o "$STAGE/step27.onnx"

if run_step 30; then
echo "=== Step 30: Conv1D → im2col + Gemm (large Conv1D only, weight >= 512KB) ==="
python3 edit_scripts/conv1d_im2col.py \
    -i "$STAGE/step26.onnx" -o "$STAGE/step30.onnx" --min-weight-bytes 512000 --per-kernel

echo "=== Step 30b: Conv1D → im2col + Gemm (K3 128↔512 L=50 convs) ==="
python3 edit_scripts/conv1d_im2col.py \
    -i "$STAGE/step30.onnx" -o "$STAGE/step30.onnx" --min-weight-bytes 393000 --max-input-length 50 --per-kernel

echo "=== Step 30c: Conv1D → im2col + Gemm (K3 256x256/290x256 L=320 convs) ==="
python3 edit_scripts/conv1d_im2col.py \
    -i "$STAGE/step30.onnx" -o "$STAGE/step30.onnx" --min-weight-bytes 393000 --max-input-length 320 --per-kernel

# Steps 30c2/30d DISABLED — any additional im2col beyond 30c triggers TileAndFuse
# assertion crash (Operation.cpp:509 "not already in an operation block").
# 30c2 alone crashes on batch_matmul_1x320x128x256; 30d alone crashes on
# conv_1d_ncw_fcw_1x256x320x130x3. Both are topology-dependent compiler bugs.

echo "=== Step 26b: Extra Gemm split (lower threshold, includes im2col Gemms) ==="
python3 edit_scripts/split_large_gemm.py \
    -i "$STAGE/step30.onnx" -o "$STAGE/step26b.onnx" \
    --min-weight-bytes 60000 --max-block-bytes 32768
fi

# Step 29 (chunk_large_elementwise) REMOVED — AddOpPattern compiler fix marks
# bf16 add/sub for fuse-group, so TileAndFuse tiles them on NSS natively.
# Chunking was adding 878 nodes of overhead and hurting performance (1,814ms → 1,454ms without).

if run_step 33; then
echo "=== Step 33: Chunk large reductions (>65536 NDL descriptor limit) ==="
python3 edit_scripts/chunk_large_reduce.py \
    -i "$STAGE/step26b.onnx" -o "$STAGE/step33.onnx"
fi

if run_step 32; then
echo "=== Step 32: Split large GatherND ops into per-channel slices (LRAM overflow) ==="
python3 edit_scripts/split_gathernd_channels.py \
    -i "$STAGE/step33.onnx" -o "$STAGE/step32.onnx" --max-bytes 509952
fi


if run_step 31; then
echo "=== Step 31: Eliminate ReduceMean axis transposes ==="
python3 edit_scripts/eliminate_reducemean_transposes.py \
    -i "$STAGE/step32.onnx" -o "$STAGE/step31.onnx"
fi

if run_step 99; then
echo "=== Step 99: Deduplicate node names ==="
python3 edit_scripts/dedup_node_names.py \
    -i "$STAGE/step31.onnx" -o "$OUT"
fi

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "=== Verification ==="
python3 -c "
import onnx
m = onnx.load('$OUT')
nodes = len(m.graph.node)
transposes = sum(1 for n in m.graph.node if n.op_type == 'Transpose')
inits = len(m.graph.initializer)
no_shape = sum(1 for vi in m.graph.value_info
               if vi.type.HasField('tensor_type') and not vi.type.tensor_type.HasField('shape'))
print(f'Output: $OUT')
print(f'  Nodes: {nodes}')
print(f'  Transposes: {transposes}')
print(f'  Initializers: {inits}')
print(f'  Value_info missing shape: {no_shape}')
"
echo ""
echo "=== DONE ==="
echo "Next: MLIR import (NO --data-prop) and compile:"
echo "  docker ... python3 -m iree.compiler.tools.import_onnx $OUT -o model.mlir"
echo "  docker ... torq-compile model.mlir -o model_board.vmfb \\"
echo "      --torq-hw=SL2610 --torq-target-host-triple=aarch64-linux-gnu \\"
echo "      --torq-convert-dtypes --torq-convert-conv1d-to-matmul=false \\"
echo "      --torq-convert-conv1d-to-generic=false --torq-convert-io-dtype"
