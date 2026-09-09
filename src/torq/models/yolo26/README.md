# YOLO26 (yolo26n) export for Torq

NMS-free Ultralytics YOLO26 object detection, quantized to full-integer **int8
TFLite** (int8 NHWC IO, `1x320x320x3 -> 1x84x2100`) and compiled to a Torq NPU
vmfb.

## Pipeline

```
Ultralytics yolo26n.pt
  → ONNX export (static 320, one2one head) + strip TopK/decode tail   [yolo26]
  → onnx2tf NCHW→NHWC + full-integer int8 PTQ (COCO128 calib)          [yolo26]
  → hosted on HF: Synaptics/yolov26n_od / yolo26n_full_integer_quant_320_od.tflite
  → tosa-converter-for-tflite → torq-compile → yolo26n_npu.vmfb  [yolo26-compile]
```

## Usage

```sh
# export + quantize from the Ultralytics weights (downloads .pt + COCO128)
torq-export-model yolo26 --download

# compile the HF-hosted int8 TFLite to a vmfb (writes yolo26n_npu.vmfb)
torq-export-model yolo26-compile -o models/yolo26/compile
```

`yolo26-compile` needs `tosa-converter-for-tflite` on PATH and a `torq-compile`
that includes the fixes in review as torq-compiler PRs **#2280 / #2285** — point
`--compiler-path` (or `TORQ_COMPILER_PATH`) at such a build. Flags used:
`--torq-hw=SL2610 --torq-disable-slicing`.

The stripped model emits the six raw head tensors (box/cls per stride 8/16/32);
the fixed-k TopK / xyxy decode runs host-side.
