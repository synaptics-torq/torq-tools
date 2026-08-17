# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export PP-OCRv6-tiny (paddle-paddle) for Torq: freeze the dynamic det/rec sources to
static shapes, apply the exact detector surgeries (issue torq-compiler-dev#2236), convert
to bf16, then (optionally) compile each model to a vmfb."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx

from ...tools.convert_dtype.onnx import convert_model
from ...utils.logging import add_logging_args, configure_logging
from ._surgery import cascade_resizes, decompose_avgpool, fold_bn, freeze, gemm_to_matmul, split_anisotropic_dw, split_convtranspose

logger = logging.getLogger("ppocr-export")

DEFAULT_DET_HW = (800, 608)
DEFAULT_BUCKETS = (320, 640, 1280, 2432)
COMPILER_ARGS = ["--torq-hw=SL2610", "--torq-disable-slicing"]


def export_ppocr(output_dir, det_hw=DEFAULT_DET_HW, buckets=DEFAULT_BUCKETS, budget=150_000, compile_vmfb=True, compiler_path=None) -> dict[str, Path]:
    """Download, freeze, surger, bf16-convert and (optionally) compile all five models."""
    from ...utils.compile import export_torq
    from .download_source import download_source

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = download_source(output_dir / "source")
    written = {}

    det = freeze(onnx.load(src["det"]), [1, 3, *det_hw])
    logger.info("det surgery: %d resize(s) cascaded, %d convtranspose(s) split", cascade_resizes(det), split_convtranspose(det, budget))
    written["det"] = _finish(det, output_dir / f"ppocr_det_{det_hw[0]}x{det_hw[1]}.onnx")

    for w in buckets:
        rec = freeze(onnx.load(src["rec"]), [1, 3, 48, w])
        logger.info("rec_w%d surgery: %d bn folded, %d avgpool decomposed, %d dw split, %d gemm converted", w, fold_bn(rec), decompose_avgpool(rec), split_anisotropic_dw(rec), gemm_to_matmul(rec))
        written[f"rec_w{w}"] = _finish(rec, output_dir / f"rec_w{w}.onnx")

    if compile_vmfb:
        for tag, path in written.items():
            logger.info("compiling %s", path.name)
            export_torq(path, output_dir, compiler_args=COMPILER_ARGS, use_binary=bool(compiler_path), compiler_path=compiler_path, opset=22)
    return written


def _finish(model: onnx.ModelProto, dest: Path) -> Path:
    onnx.checker.check_model(model)
    fp32 = dest.with_name(dest.stem + "_fp32.onnx")
    onnx.save(model, fp32)
    convert_model(fp32, dest, convert_dtype="bf16", convert_io=True, torq_onnx_finalize=True)
    logger.info("wrote %s", dest)
    return dest


def add_ppocr_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output-dir", type=str, default="models/ppocr/export", help="Output directory (default: %(default)s)")
    parser.add_argument("--det-hw", type=int, nargs=2, metavar=("H", "W"), default=list(DEFAULT_DET_HW), help="Static detector input, multiples of 32 (default: %(default)s)")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS), help="Recognizer width buckets (default: %(default)s)")
    parser.add_argument("--budget", type=int, default=150_000, help="Max bytes per ConvTranspose slice (default: %(default)s)")
    parser.add_argument("--no-compile", action="store_true", help="Stop after the bf16 ONNX exports")
    parser.add_argument("--compiler-path", type=str, default=None, help="torq-compile binary (default: TORQ_COMPILER_PATH or the wheel)")
    add_logging_args(parser)


def export_ppocr_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    written = export_ppocr(args.output_dir, tuple(args.det_hw), tuple(args.buckets), args.budget, compile_vmfb=not args.no_compile, compiler_path=args.compiler_path)
    for tag, path in written.items():
        print(f"{tag}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PP-OCRv6-tiny (paddle-paddle) to Torq")
    add_ppocr_export_args(parser)
    export_ppocr_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
