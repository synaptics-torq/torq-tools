#!/bin/bash
# Stage — compile ONNX/TOSA pieces to torq .vmfb (NSS-only).
#
# bf16 ONNX path:    iree import_onnx -> torq-compile
# int8 TFLite path:  tosa-converter-for-tflite -> torq-compile
#
# Flags (NSS only, no host/CSS fallback):
#   --torq-hw=SL2610 --torq-disable-css --torq-disable-host
#   --torq-tile-and-fuse-distance-limit=1   (the full ACT model needs this to compile;
#                                            without it, fusion pressure fails compilation)
#   --torq-enable-split-constants-optimization
# IMPORTANT: pass flags INLINE (shell var expansion can mangle --torq-hw=...).
#
# Usage:  TC=/path/to/torq-compile ./compile.sh
set -e
TC=${TC:-torq-compile}

compile_onnx () {  # $1 = name (without .onnx)
  python3 -m iree.compiler.tools.import_onnx "$1.onnx" -o "$1.mlir" --data-prop
  "$TC" "$1.mlir" -o "$1.vmfb" \
    --torq-hw=SL2610 --torq-disable-css --torq-disable-host \
    --torq-tile-and-fuse-distance-limit=1 --torq-enable-split-constants-optimization
  echo "compiled $1.vmfb ($(stat -c%s "$1.vmfb") bytes)"
}

compile_tflite () {  # $1 = name (without .tflite)
  tosa-converter-for-tflite "$1.tflite" --bytecode -o "$1.mlirbc"
  "$TC" "$1.mlirbc" -o "$1.vmfb" \
    --torq-hw=SL2610 --torq-disable-css --torq-disable-host \
    --torq-tile-and-fuse-distance-limit=1
  echo "compiled $1.vmfb ($(stat -c%s "$1.vmfb") bytes)"
}

# --- the verified pipeline ---
compile_onnx piece_A_folded                   # transformer encoder L1+L2  (bf16, const-folded)
compile_onnx piece_B_folded                   # transformer encoder L3+L4 + decoder (bf16, const-folded)
# backbone: pick ONE
# compile_onnx backbone_folded                # bf16 backbone, BatchNorm-folded (faithful weights, ~482 ms)
compile_tflite resnet18_backbone_int8         # int8 backbone (~118 ms)
