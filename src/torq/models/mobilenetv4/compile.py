# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Minimal end-to-end MobileNetV4 tflite -> vmfb pipeline.

    1. Download the source .tflite (if not already on disk).
    2. Strip its dynamic batch shape signature so it's fully static
       (`torq.tools.convert_static`, invoked as a subprocess -- torq-compile
       rejects dynamic TOSA operand shapes with a level-check error).
    3. Legalize the static tflite flatbuffer to TOSA MLIR via Arm's
       ``tosa-converter-for-tflite`` (the actual tflite frontend for this
       toolchain -- IREE's own tflite importer is not part of this build).
    4. Compile the TOSA MLIR to .vmfb via the `torq-compile` binary.

    Kept self-contained to this package: steps 2-4 shell out to external
    tools/binaries (subprocess) rather than importing torq.utils.compile /
    torq.tools.convert_static as Python modules, so this new tflite feature
    stays isolated while it's still being proven out.

    python src/torq/models/mobilenetv4/compile.py --models-dir models
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .download_source import (
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_SIZE,
    MODEL_DTYPES,
    MODEL_SIZES,
    _SOURCE_FILES,
    download_source,
)

logger = logging.getLogger("MobileNetV4.compile")


def make_static_tflite(tflite_model: str | Path, static_model: str | Path):
    tflite_model = Path(tflite_model)
    if not tflite_model.exists():
        raise FileNotFoundError(f"TFLite model '{tflite_model}' not found")
    cmd = [
        sys.executable, "-m", "torq.tools.convert_static", "tflite",
        "-i", str(tflite_model),
        "-o", str(static_model),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        logger.info(out.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to make TFLite model static via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def tflite_to_tosa_mlir(
    tflite_model: str | Path,
    mlir_model: str | Path,
    emit_debug_info: bool = False,
):
    tflite_model = Path(tflite_model)
    if not tflite_model.exists():
        raise FileNotFoundError(f"TFLite model '{tflite_model}' not found")
    cmd = ["tosa-converter-for-tflite", str(tflite_model), "--text", "-o", str(mlir_model)]
    if emit_debug_info:
        cmd.append("--emit-debug-info")
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        logger.info(out.strip())
    except FileNotFoundError:
        raise RuntimeError(
            "tosa-converter-for-tflite binary not found in PATH "
            "(pip install tosa-converter-for-tflite)"
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert TFLite model via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def compile_mlir_to_vmfb(
    mlir_model: str | Path,
    vmfb_model: str | Path,
    compiler_args: list[str] | None = None,
):
    mlir_model = Path(mlir_model)
    if not mlir_model.exists():
        raise FileNotFoundError(f"MLIR model '{mlir_model}' not found")
    cmd = ["torq-compile", str(mlir_model), "-o", str(vmfb_model), *(compiler_args or [])]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        logger.info(out.strip())
    except FileNotFoundError:
        raise RuntimeError("torq-compile binary not found in PATH") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to compile MLIR model via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def compile_vmfb(
    models_dir: str | Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dtype: str = DEFAULT_MODEL_DTYPE,
    compiler_args: list[str] | None = None,
) -> Path:
    models_dir = Path(models_dir)
    model_root = models_dir / "mobilenetv4"
    filename = _SOURCE_FILES[model_size][model_dtype]
    stem = Path(filename).stem

    source_dir = model_root / "source" / "tflite" / model_size / model_dtype
    source_model = source_dir / filename
    if not source_model.exists():
        logger.info("Source tflite not found, downloading...")
        download_source(models_dir, model_size, model_dtype)

    static_dir = model_root / "export" / "tflite" / model_size / model_dtype / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    static_model = static_dir / filename
    logger.info("Fixing dynamic batch shape: '%s' -> '%s'", source_model, static_model)
    make_static_tflite(source_model, static_model)

    torq_dir = model_root / "export" / "torq" / model_size / model_dtype / "static"
    torq_dir.mkdir(parents=True, exist_ok=True)
    mlir_model = torq_dir / f"{stem}.mlir"
    logger.info("Converting '%s' to TOSA MLIR: '%s'", static_model, mlir_model)
    tflite_to_tosa_mlir(static_model, mlir_model)

    vmfb_model = torq_dir / f"{stem}.vmfb"
    logger.info("Compiling '%s' to '%s'...", mlir_model, vmfb_model)
    compile_mlir_to_vmfb(mlir_model, vmfb_model, compiler_args=compiler_args)
    logger.info("Done: '%s'", vmfb_model)
    return vmfb_model


def main():
    parser = argparse.ArgumentParser(
        description="Compile MobileNetV4 tflite to a Torq .vmfb.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source/export models (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="Model size (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--model-dtype",
        type=str,
        choices=MODEL_DTYPES,
        default=DEFAULT_MODEL_DTYPE,
        help="Model dtype (default: %(default)s)",
    )
    parser.add_argument(
        "--compile-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for torq-compile. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.logging, format="%(message)s")
    compile_vmfb(
        args.models_dir,
        args.model_size,
        args.model_dtype,
        compiler_args=args.compile_flags or [],
    )


if __name__ == "__main__":
    main()
