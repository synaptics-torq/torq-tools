# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse

from .tflite import add_tflite_static_convert_args


def main():
    parser = argparse.ArgumentParser(description="Convert dynamic models to static, using the default shapes")
    model_type = parser.add_subparsers(dest="model_type", required=True)

    tflite_type = model_type.add_parser("tflite", help="Convert TFLite dynamic models")
    add_tflite_static_convert_args(tflite_type)

    args = parser.parse_args()

    if args.model_type == "tflite":
        from .tflite import tflite_static_convert_from_args
        tflite_static_convert_from_args(args)


if __name__ == "__main__":
    main()
