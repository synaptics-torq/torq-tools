# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export a Piper (VITS) voice for Torq: split into partA (text encoder + duration,
CPU/onnxruntime, fp32) and partB (HiFi-GAN vocoder, NPU), pin partB to static windows,
convert those to bf16 with bf16 IO, then (optionally) compile each to a vmfb.

partA yields the exact output frame count before partB runs, so a runtime pads each
utterance into the smallest window that fits and crops the audio back. Single-speaker
voices have no speaker embedding: their partB takes the latent alone (the boundary is
discovered from the graph, not assumed). Compiling NSS-only needs the vocoder fixes
from torq-compiler branch wip/partb-vocoder-nss-fixes."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import onnx

from ...tools.convert_dtype.onnx import convert_model
from ...utils.logging import add_logging_args, configure_logging
from ._graph import CUTS, freeze, split

logger = logging.getLogger("piper-export")

DEFAULT_VOICE = "en_US-libritts_r-medium"
DEFAULT_SECONDS = (1, 2, 4, 6, 8)
HOP = 256
COMPILER_ARGS = ["--torq-hw=SL2610", "--torq-disable-css", "--torq-disable-host",
                 "--torq-max-nss-programs-size=25165824"]


def export_piper(voice=DEFAULT_VOICE, output_dir="models/piper/export", seconds=DEFAULT_SECONDS,
                 compile_vmfb=True, compiler_path=None) -> dict[str, Path]:
    """Download, split, pin the vocoder windows, bf16-convert and (optionally) compile."""
    from ...utils.compile import export_torq
    from .download_source import download_source

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = download_source(voice, output_dir / "source")
    sample_rate = json.load(open(src["config"]))["audio"]["sample_rate"]
    written = {"config": Path(shutil.copy(src["config"], output_dir))}
    written["partA"] = output_dir / "partA.onnx"
    part_b = output_dir / "partB.onnx"
    boundary = split(src["model"], written["partA"], part_b)
    logger.info("split at %s; partB consumes: %s", CUTS[0], ", ".join(boundary))

    for s in seconds:
        frames = round(s * sample_rate / HOP)
        fp32 = output_dir / f"partB_static_{s:g}s_fp32.onnx"
        onnx.save(freeze(part_b, CUTS[1], frames), fp32)
        dest = output_dir / f"partB_static_{s:g}s.onnx"
        convert_model(fp32, dest, convert_dtype="bf16", convert_io=True, torq_onnx_finalize=True)
        logger.info("wrote %s (%d frames)", dest.name, frames)
        written[f"{s:g}s"] = dest
        if compile_vmfb:
            export_torq(dest, output_dir, compiler_args=COMPILER_ARGS,
                        use_binary=bool(compiler_path), compiler_path=compiler_path, opset=22)
    return written


def add_piper_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--voice", type=str, default=DEFAULT_VOICE,
                        help="Voice key on rhasspy/piper-voices (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Output directory (default: models/piper/<voice>/export)")
    parser.add_argument("--seconds", type=float, nargs="+", default=list(DEFAULT_SECONDS),
                        help="Vocoder window sizes in seconds (default: %(default)s)")
    parser.add_argument("--no-compile", action="store_true", help="Stop after the bf16 ONNX exports")
    parser.add_argument("--compiler-path", type=str, default=None,
                        help="torq-compile binary (default: TORQ_COMPILER_PATH or the wheel)")
    add_logging_args(parser)


def export_piper_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    output_dir = args.output_dir or f"models/piper/{args.voice}/export"
    written = export_piper(args.voice, output_dir, tuple(args.seconds),
                           compile_vmfb=not args.no_compile, compiler_path=args.compiler_path)
    for tag, path in written.items():
        print(f"{tag}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Piper TTS voice to Torq")
    add_piper_export_args(parser)
    export_piper_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
