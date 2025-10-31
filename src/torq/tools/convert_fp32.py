import argparse

from .convert_dtype.onnx import add_onnx_fp32_convert_args


def main():
    parser = argparse.ArgumentParser(description="Convert FP32 models to BF16 and FP16")
    model_type = parser.add_subparsers(dest="model_type", required=True)

    onnx_type = model_type.add_parser("onnx", help="Convert ONNX FP32 models")
    add_onnx_fp32_convert_args(onnx_type)

    args = parser.parse_args()

    if args.model_type == "onnx":
        from .convert_dtype.onnx import onnx_fp32_convert_from_args
        onnx_fp32_convert_from_args(args)
