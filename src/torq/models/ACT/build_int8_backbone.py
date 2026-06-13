#!/usr/bin/env python3
"""Optimization stage — int8 ResNet-18 backbone via the TFLite -> TOSA -> torq path.

WHY a separate path (not ONNX QDQ): torq has no NSS lowering for ONNX
Quantize/DequantizeLinear (see gh-issues/qdq-int8-nss-lowering/), and
--torq-convert-dtypes crashes (gh-issues/02-convert-dtypes-const-outline/). The
SUPPORTED int8 route is TFLite full-int8 PTQ -> tosa-converter-for-tflite -> torq,
which works because IREE folds the quantized pattern into a quantized Conv.
int8 backbone measured ~118 ms vs bf16 ~483 ms (~4x).

LAYOUT MATTERS: build the model NHWC with batch=1 and the spatial dims as H/W.
(An int8 fully-connected/Dense flattens to batch=N and torq's conv codegen rejects
that layout -- the same lesson that makes the transformer weight-matmuls compile
only as batch=1 seq-on-spatial 1x1 convs.)

This builds a STRUCTURALLY-EQUIVALENT ResNet-18 (image[1,480,640,3] ->
[1,15,20,512]) for runtime/plumbing. For a numerically-correct backbone, port the
real exported conv weights into this Keras graph before PTQ (TODO).

Output: resnet18_backbone_int8.tflite  (then: tosa-converter-for-tflite ... | torq-compile)

Usage:  python build_int8_backbone.py -o resnet18_backbone_int8.tflite
"""
import argparse, numpy as np, tensorflow as tf, keras
from keras import layers as L


def resnet_block(x, ch, stride=1):
    s = x
    x = L.Conv2D(ch, 3, stride, padding="same", use_bias=False)(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.Conv2D(ch, 3, 1, padding="same", use_bias=False)(x); x = L.BatchNormalization()(x)
    if stride != 1 or s.shape[-1] != ch:
        s = L.Conv2D(ch, 1, stride, use_bias=False)(s); s = L.BatchNormalization()(s)
    return L.ReLU()(L.Add()([x, s]))


def build_resnet18():
    inp = keras.Input(shape=(480, 640, 3), batch_size=1)          # NHWC, batch=1
    x = L.Conv2D(64, 7, 2, padding="same", use_bias=False)(inp); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPool2D(3, 2, padding="same")(x)
    for ch, n, st in [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]:
        for b in range(n):
            x = resnet_block(x, ch, st if b == 0 else 1)
    return keras.Model(inp, x)                                    # -> [1,15,20,512]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="resnet18_backbone_int8.tflite")
    a = ap.parse_args()
    m = build_resnet18()

    def rep():
        for _ in range(50):
            yield [np.random.randn(1, 480, 640, 3).astype(np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    open(a.out, "wb").write(c.convert())
    print(f"wrote {a.out}")
    print("next: tosa-converter-for-tflite", a.out, "--bytecode -o bb_int8.mlirbc")
    print("      torq-compile bb_int8.mlirbc -o backbone_int8.vmfb "
          "--torq-hw=SL2610 --torq-disable-css --torq-disable-host --torq-tile-and-fuse-distance-limit=1")


if __name__ == "__main__":
    main()
