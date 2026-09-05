# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export YOLO26 for Torq: run Ultralytics' own ONNX export (static ``imgsz``, the
default NMS-free/DFL-free one2one head), strip the fixed-k TopK/decode tail (runs
host-side instead), then convert the fp32 ONNX to bf16 via ``torq.tools.convert_dtype``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs

from ...tools.convert_dtype.onnx import convert_model
from ...utils.logging import add_logging_args, configure_logging
from ._surgery import strip_postprocess

logger = logging.getLogger("yolo26-export")

DEFAULT_INPUT_SIZE = 320
DEFAULT_VARIANT = "yolo26n"
DEFAULT_OPSET = 17


def export_yolo26(
    source, output_dir, input_size=DEFAULT_INPUT_SIZE, opset=DEFAULT_OPSET,
    convert_bf16=True, bf16_convert_io=False, validate=True,
) -> dict[str, Path]:
    """Run Ultralytics ONNX export + strip + (bf16) pipeline; return the written paths."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_size % 32 != 0:
        raise ValueError(f"input_size must be divisible by 32 (stride-32 level), got {input_size}")

    logger.info("Loading YOLO26 weights: %s", source)
    model = YOLO(str(source))
    raw_path = Path(model.export(format="onnx", imgsz=input_size, dynamic=False, simplify=True, opset=opset))
    logger.info("Ultralytics raw ONNX export: %s", raw_path)

    graph = gs.import_onnx(onnx.load(str(raw_path)))
    strip_postprocess(graph)
    stripped = gs.export_onnx(graph)
    onnx.checker.check_model(stripped)

    fp32_path = output_dir / "model_nopost_fp32.onnx"
    onnx.save(stripped, str(fp32_path))
    logger.info("Wrote stripped fp32 model: %s", fp32_path)
    written = {"fp32": fp32_path}

    if validate:
        _validate_onnxruntime(stripped, input_size)

    if convert_bf16:
        io_suffix = "_io" if bf16_convert_io else ""
        bf16_path = output_dir / f"model_nopost_bf16{io_suffix}.onnx"
        convert_model(fp32_path, bf16_path, convert_dtype="bf16", convert_io=bf16_convert_io, torq_onnx_finalize=True)
        written["bf16_io" if bf16_convert_io else "bf16"] = bf16_path
    return written


def _validate_onnxruntime(model: onnx.ModelProto, input_size: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime/numpy unavailable; skipping validation run")
        return
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, input_size, input_size), dtype=np.float32)})
    logger.info("Validation run OK: %s", ", ".join(f"{o.name}{list(v.shape)}" for o, v in zip(sess.get_outputs(), outs)))


def add_yolo26_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--source", type=str, default="models/yolo26/source/yolo26n.pt", help="Source .pt weights or model scale, e.g. 'yolo26n' (default: %(default)s)")
    parser.add_argument("--download", action="store_true", help="Download weights (+calib images) from Ultralytics assets first; also automatic if --source is missing")
    parser.add_argument("-o", "--output-dir", type=str, default="models/yolo26/export", help="Directory for exported models (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Square input size, divisible by 32 (default: %(default)s)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset (default: %(default)s)")
    parser.add_argument("--no-bf16", action="store_true", help="Stop after the fp32 stripped model")
    parser.add_argument("--bf16-convert-io", action="store_true", help="Also cast model I/O to bf16 (default: fp32 I/O, matching --torq-convert-io-dtype)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the onnxruntime validation run")
    add_logging_args(parser)


def export_yolo26_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    source = args.source
    if getattr(args, "download", False) or not Path(source).exists():
        from .download_source import download_source

        models_dir = Path(args.output_dir).parent if Path(source).suffix == ".pt" else "models/yolo26"
        variant = Path(source).stem if Path(source).suffix == ".pt" else source or DEFAULT_VARIANT
        source = str(download_source(models_dir, variant))
    written = export_yolo26(source, args.output_dir, args.input_size, args.opset, convert_bf16=not args.no_bf16, bf16_convert_io=args.bf16_convert_io, validate=not args.skip_validation)
    for tag, path in written.items():
        print(f"{tag}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26 detection model to Torq")
    add_yolo26_export_args(parser)
    export_yolo26_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
