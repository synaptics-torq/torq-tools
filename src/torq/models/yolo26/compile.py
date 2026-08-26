# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Compile the HuggingFace-hosted YOLO26 int8 TFLite (``Synaptics/yolov26n_od``, int8 NHWC IO,
1x320x320x3 -> 1x84x2100 — the artifact this module's export+quantize pipeline produced)
to a Torq NPU vmfb: download, convert to TOSA MLIR via ``tosa-converter-for-tflite``
(must be on PATH), then ``torq-compile --torq-hw=SL2610 --torq-disable-slicing``.

Requires torq-compiler fixes in review (torq-compiler PRs #2280 / #2285); point
``--compiler-path`` or ``TORQ_COMPILER_PATH`` at a build that includes them.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from ...model_export.hf import hf_download_source_model
from ...utils.compile import compile_mlir_for_vm
from ...utils.logging import add_logging_args, configure_logging

logger = logging.getLogger("yolo26-compile")

DEFAULT_HF_REPO = "Synaptics/yolov26n_od"
DEFAULT_TFLITE = "yolo26n_full_integer_quant_320_od.tflite"
OUTPUT_VMFB = "yolo26n_npu.vmfb"
COMPILER_ARGS = ["--torq-hw=SL2610", "--torq-disable-slicing"]


def convert_to_tosa(tflite_model: Path, mlir_model: Path) -> None:
    """TFLite flatbuffer -> textual TOSA MLIR via ``tosa-converter-for-tflite``."""
    try:
        subprocess.check_output(
            ["tosa-converter-for-tflite", str(tflite_model), "--text", "-o", str(mlir_model)],
            text=True, stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        raise RuntimeError("tosa-converter-for-tflite binary not found in PATH") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert TFLite model via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def compile_yolo26(output_dir="models/yolo26/compile", hf_repo=DEFAULT_HF_REPO,
                   tflite_filename=DEFAULT_TFLITE, compiler_path=None) -> Path:
    """Download -> TOSA-convert -> torq-compile; return the written vmfb path."""
    output_dir = Path(output_dir)
    tflite = output_dir / "source" / tflite_filename
    if not tflite.exists():
        logger.info("downloading %s from %s", tflite_filename, hf_repo)
        hf_download_source_model(hf_repo, tflite_filename, tflite.parent)
    mlir = output_dir / (Path(tflite_filename).stem + ".tosa.mlir")
    logger.info("converting %s to TOSA MLIR", tflite.name)
    convert_to_tosa(tflite, mlir)
    vmfb = output_dir / OUTPUT_VMFB
    logger.info("compiling %s for the Torq NPU", mlir.name)
    compile_mlir_for_vm(mlir, vmfb, compiler_args=COMPILER_ARGS, use_binary=True, compiler_path=compiler_path)
    return vmfb


def add_yolo26_compile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output-dir", type=str, default="models/yolo26/compile", help="Directory for the TFLite/MLIR/vmfb artifacts (default: %(default)s)")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO, help="HuggingFace repo hosting the quantized TFLite (default: %(default)s)")
    parser.add_argument("--tflite", type=str, default=DEFAULT_TFLITE, help="TFLite filename inside the repo (default: %(default)s)")
    parser.add_argument("--compiler-path", type=str, default=None, help="torq-compile binary (default: TORQ_COMPILER_PATH env var, or torq-compile from PATH)")
    add_logging_args(parser)


def compile_yolo26_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    print(f"vmfb: {compile_yolo26(args.output_dir, args.hf_repo, args.tflite, args.compiler_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the HF-hosted YOLO26 int8 TFLite to a Torq vmfb")
    add_yolo26_compile_args(parser)
    compile_yolo26_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
