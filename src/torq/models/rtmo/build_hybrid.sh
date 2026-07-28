#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the RTMO hybrid: post-stripped fp32 ONNX -> three TFLite parts
# (int8 backbone / bf16 transformer / int8 head, via _hybrid.py) -> three
# NSS-only vmfbs (via tosa-converter-for-tflite + torq-compile). The int8 parts
# compile unsliced; the bf16 transformer compiles sliced (faster for bf16).
#
# Config is via environment variables (required ones error if unset):
#   PY      python with onnx2tf + tf + onnx-graphsurgeon (default: python)
#   TOSA    tosa-converter-for-tflite (default: on PATH)
#   TORQC   torq-compile built from main            (required)
#   ONNX    post-stripped fp32 rtmo onnx (rtmo_nopost_fp32.onnx, from export.py) (required)
#   IMAGES  directory of representative calibration images  (required)
#   N_CALIB calibration image count (default: 100)
#
# Usage:  TORQC=... ONNX=... IMAGES=... ./build_hybrid.sh [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PY=${PY:-python}
TOSA=${TOSA:-tosa-converter-for-tflite}
TORQC=${TORQC:?set TORQC to a torq-compile built from main}
ONNX=${ONNX:?set ONNX to the post-stripped fp32 rtmo onnx (rtmo_nopost_fp32.onnx)}
IMAGES=${IMAGES:?set IMAGES to a directory of calibration images}
N_CALIB=${N_CALIB:-100}
OUT=${1:-./hybrid_vmfb}
TFL_DIR="$OUT/tflite"
mkdir -p "$OUT" "$TFL_DIR"

echo "[1/2] ONNX -> TFLite parts (int8 backbone / bf16 transformer / int8 head)"
"$PY" "$HERE/_hybrid.py" \
    --onnx "$ONNX" \
    --out-dir "$TFL_DIR" \
    --images-dir "$IMAGES" \
    --n-calib "$N_CALIB" \
    --transformer-scheme bf16

echo "[2/2] TFLite -> NSS-only vmfb"
COMMON="--torq-hw=SL2610 --torq-disable-css --torq-disable-host --torq-convert-dtypes --torq-convert-io-dtype"
compile_part () {   # <tflite> <out.vmfb> <extra-flags>
  "$TOSA" "$1" --text -o "$OUT/${2%.vmfb}.tosa.mlir"
  "$TORQC" $COMMON ${3:-} -o "$OUT/$2" "$OUT/${2%.vmfb}.tosa.mlir"
  echo "    -> $OUT/$2 ($(stat -c%s "$OUT/$2") bytes)"
}
compile_part "$TFL_DIR/rtmo_hybrid_backbone_int8.tflite"    rtmo_hyb_backbone_int8.vmfb    --torq-disable-slicing
compile_part "$TFL_DIR/rtmo_hybrid_transformer_bf16.tflite" rtmo_hyb_transformer_bf16.vmfb
compile_part "$TFL_DIR/rtmo_hybrid_head_int8.tflite"        rtmo_hyb_head_int8.vmfb        --torq-disable-slicing

echo
echo "Done. Three vmfbs in $OUT — run with:"
echo "  python -m torq.models.infer_model rtmo IMAGE.jpg -m $OUT"
