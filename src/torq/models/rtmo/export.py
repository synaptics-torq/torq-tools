# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Export RTMO tiny for Torq: strip the mmdeploy post-processing (runs
host-side instead), re-target to a square ``--input-size``, then convert the
fp32 ONNX to bf16 via ``torq.tools.convert_dtype``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx

from ...tools.convert_dtype.onnx import convert_model
from ...utils.logging import add_logging_args, configure_logging
from ._surgery import build_stripped_model

logger = logging.getLogger("rtmo-export")

DEFAULT_INPUT_SIZE = 320
DEFAULT_BATCH = 1


def export_rtmo(source, output_dir, input_size=DEFAULT_INPUT_SIZE, batch=DEFAULT_BATCH, convert_bf16=True, bf16_convert_io=False, validate=True) -> dict[str, Path]:
    """Run the strip+resize (+bf16) pipeline; return the written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading source RTMO model: %s", source)
    model = onnx.load(str(source), load_external_data=True)
    stripped = build_stripped_model(model, input_size=input_size, batch=batch)
    onnx.checker.check_model(stripped)

    fp32_path = output_dir / "model_nopost_fp32.onnx"
    onnx.save(stripped, str(fp32_path))
    logger.info("Wrote stripped fp32 model: %s", fp32_path)
    written = {"fp32": fp32_path}

    if validate:
        _validate_onnxruntime(stripped, input_size, batch)

    if convert_bf16:
        io_suffix = "_io" if bf16_convert_io else ""  # bf16-I/O variant coexists with fp32-I/O
        bf16_path = output_dir / f"model_nopost_bf16{io_suffix}.onnx"
        convert_model(fp32_path, bf16_path, convert_dtype="bf16", convert_io=bf16_convert_io, torq_onnx_finalize=True)
        written["bf16_io" if bf16_convert_io else "bf16"] = bf16_path
    return written


def _validate_onnxruntime(model: onnx.ModelProto, input_size: int, batch: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime/numpy unavailable; skipping validation run")
        return
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: np.zeros((batch, 3, input_size, input_size), dtype=np.float32)})
    logger.info("Validation run OK: %s", ", ".join(f"{o.name}{list(v.shape)}" for o, v in zip(sess.get_outputs(), outs)))


def add_rtmo_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--source", type=str, default="models/rtmo/model.onnx", help="Source RTMO ONNX (with post-processing) (default: %(default)s)")
    parser.add_argument("--download", action="store_true", help="Download source + calib from HF; also automatic if --source is missing")
    parser.add_argument("-o", "--output-dir", type=str, default="models/rtmo/export", help="Directory for exported models (default: %(default)s)")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Square input size, divisible by 32 (default: %(default)s)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Static batch size (default: %(default)s)")
    parser.add_argument("--no-bf16", action="store_true", help="Stop after the fp32 stripped model")
    parser.add_argument("--bf16-convert-io", action="store_true", help="Also cast model I/O to bf16 (default: fp32 I/O, matching --torq-convert-io-dtype)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the onnxruntime validation run")
    add_logging_args(parser)


def export_rtmo_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    source = args.source
    if getattr(args, "download", False) or not Path(source).exists():
        from .download_source import download_source
        source = str(download_source(Path(source).parent))
    written = export_rtmo(source, args.output_dir, args.input_size, args.batch, convert_bf16=not args.no_bf16, bf16_convert_io=args.bf16_convert_io, validate=not args.skip_validation)
    for tag, path in written.items():
        print(f"{tag}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RTMO tiny pose model to Torq")
    add_rtmo_export_args(parser)
    export_rtmo_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
