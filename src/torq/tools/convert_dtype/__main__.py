# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .onnx import add_onnx_dtype_convert_args


def main():
    parser = argparse.ArgumentParser(description="Convert FP32 models to BF16 and FP16")
    model_type = parser.add_subparsers(dest="model_type", required=True)

    onnx_type = model_type.add_parser("onnx", help="Convert ONNX FP32 models")
    add_onnx_dtype_convert_args(onnx_type)

    args = parser.parse_args()

    if args.model_type == "onnx":
        from .onnx import onnx_dtype_convert_from_args
        onnx_dtype_convert_from_args(args)


if __name__ == "__main__":
    main()
