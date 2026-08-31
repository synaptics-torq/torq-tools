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
from ._surgery import cascade_resizes, decompose_avgpool, decompose_hardsigmoid, fold_bn, freeze, gemm_to_matmul, global_reduce_to_gap, split_anisotropic_dw, split_convtranspose

logger = logging.getLogger("ppocr-export")

# DBNet's stride-32 backbone needs H and W that are multiples of 32. 640x384 is the
# 16:9-ish bucket: 640x360 rounded UP to the next multiple of 32, so letterboxed 16:9
# content is padded rather than cropped.
DEFAULT_DET_SIZES = ((800, 608), (640, 384))
DEFAULT_BUCKETS = (320, 640, 1280, 2432)
COMPILER_ARGS = ["--torq-hw=SL2610", "--torq-disable-slicing"]


def _round32(hw: tuple[int, int]) -> tuple[int, int]:
    rounded = tuple(-(-d // 32) * 32 for d in hw)
    if rounded != tuple(hw):
        logger.warning("det input %s rounded up to %s (dims must be multiples of 32)", tuple(hw), rounded)
    return rounded


def export_ppocr(output_dir, det_sizes=DEFAULT_DET_SIZES, buckets=DEFAULT_BUCKETS, budget=150_000, compile_vmfb=True, compiler_path=None) -> dict[str, Path]:
    """Download, freeze, surger, bf16-convert and (optionally) compile all the models."""
    from ...utils.compile import export_torq
    from .download_source import download_source

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = download_source(output_dir / "source")
    written = {}

    for hw in det_sizes:
        h, w = _round32(hw)
        det = freeze(onnx.load(src["det"]), [1, 3, h, w])
        logger.info("det_%dx%d surgery: %d resize(s) cascaded, %d convtranspose(s) split, %d hardsigmoid decomposed, %d gap converted", h, w, cascade_resizes(det), split_convtranspose(det, budget), decompose_hardsigmoid(det), global_reduce_to_gap(det))
        written[f"det_{h}x{w}"] = _finish(det, output_dir / f"ppocr_det_{h}x{w}.onnx")

    for w in buckets:
        rec = freeze(onnx.load(src["rec"]), [1, 3, 48, w])
        logger.info("rec_w%d surgery: %d bn folded, %d avgpool decomposed, %d dw split, %d gemm converted, %d hardsigmoid decomposed, %d gap converted", w, fold_bn(rec), decompose_avgpool(rec), split_anisotropic_dw(rec), gemm_to_matmul(rec), decompose_hardsigmoid(rec), global_reduce_to_gap(rec))
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
    parser.add_argument("--det-hw", type=int, nargs="+", metavar="D", default=[d for hw in DEFAULT_DET_SIZES for d in hw], help="Static detector inputs as H W pairs; dims are rounded up to multiples of 32 (default: %(default)s)")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS), help="Recognizer width buckets (default: %(default)s)")
    parser.add_argument("--budget", type=int, default=150_000, help="Max bytes per ConvTranspose slice (default: %(default)s)")
    parser.add_argument("--no-compile", action="store_true", help="Stop after the bf16 ONNX exports")
    parser.add_argument("--compiler-path", type=str, default=None, help="torq-compile binary (default: TORQ_COMPILER_PATH or the wheel)")
    add_logging_args(parser)


def export_ppocr_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    if len(args.det_hw) % 2:
        raise SystemExit("--det-hw takes H W pairs; got an odd number of values")
    det_sizes = tuple(zip(args.det_hw[::2], args.det_hw[1::2]))
    written = export_ppocr(args.output_dir, det_sizes, tuple(args.buckets), args.budget, compile_vmfb=not args.no_compile, compiler_path=args.compiler_path)
    for tag, path in written.items():
        print(f"{tag}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PP-OCRv6-tiny (paddle-paddle) to Torq")
    add_ppocr_export_args(parser)
    export_ppocr_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
