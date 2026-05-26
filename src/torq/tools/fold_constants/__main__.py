# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import onnx
from ..common import maybe_load_onnx_model
from .onnx import fold_constants


def fold_constants_from_file(input_path: str) -> onnx.ModelProto:
    model = maybe_load_onnx_model(input_path)
    folded_model, _ = fold_constants(model)
    return folded_model


def main():
    parser = argparse.ArgumentParser(
        description="Constant-fold an ONNX model into materialized initializers"
    )
    parser.add_argument("-i", "--input", help="Input ONNX model path", required=True)
    parser.add_argument("-o", "--output", help="Output ONNX model path", required=True)
    args = parser.parse_args()
    model = fold_constants_from_file(args.input)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
