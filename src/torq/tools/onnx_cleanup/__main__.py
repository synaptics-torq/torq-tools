# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging

from .onnx import add_onnx_cleanup_args, onnx_cleanup_from_args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Clean up common exporter artifacts in an ONNX model "
                    "(unrolled Concats, constant islands, unfused Conv+BatchNorm)"
    )
    add_onnx_cleanup_args(parser)
    onnx_cleanup_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
