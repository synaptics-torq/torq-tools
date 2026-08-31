# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging

from .onnx import add_onnx_cleanup_args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Clean up common exporter artifacts in a model"
    )
    model_type = parser.add_subparsers(dest="model_type", required=True)

    onnx_type = model_type.add_parser(
        "onnx",
        help="Clean up ONNX exporter artifacts (unrolled Concats, constant "
             "islands, unfused Conv+BatchNorm)",
    )
    add_onnx_cleanup_args(onnx_type)

    args = parser.parse_args()

    if args.model_type == "onnx":
        from .onnx import onnx_cleanup_from_args
        onnx_cleanup_from_args(args)


if __name__ == "__main__":
    main()
