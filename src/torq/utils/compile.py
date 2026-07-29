# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from shutil import copy2

try:
    import torq.compiler.tools as torq_c
    import iree.compiler.tflite as iree_tflite_compile
    TORQ_C_PYAPI = True
except ImportError:
    TORQ_C_PYAPI = False

logger = logging.getLogger("Torq-compile")


def _resolve_compiler(compiler_path: str | Path | None = None) -> str:
    if compiler_path:
        return str(compiler_path)
    return os.environ.get("TORQ_COMPILER_PATH", "torq-compile")


def get_iree_version(compiler: str | None = None) -> str | None:
    import re

    compiler = compiler or _resolve_compiler()
    try:
        out = subprocess.check_output([compiler, "--version"], text=True)
    except FileNotFoundError:
        logger.warning("Failed to check torq-compile version; Ensure '%s' is installed and accessible from PATH", compiler)
        return None
    m = re.search(r"IREE compiler version ([\w.\-+]+)", out)
    if not m:
        return None
    ver_str = m.group(1)
    if ver_str.lower() == "unknown":
        return None
    return ver_str


def export_onnx_to_mlir(
    onnx_model: str | Path,
    mlir_model: str | Path,
    opset: int | None = None,
):
    if not Path(onnx_model).exists():
        raise FileNotFoundError(f"ONNX model '{onnx_model}' not found")

    if TORQ_C_PYAPI:
        try:
            import sys
            import_onnx_args = [
                sys.executable, "-m", "iree.compiler.tools.import_onnx",
                str(onnx_model),
                "-o", str(mlir_model),
                "--data-prop",
            ]
            if opset and opset > 0:
                import_onnx_args += ["--opset-version", str(opset)]
            subprocess.check_output(
                import_onnx_args,
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None
    else:
        logger.warning("Torq compiler python API not found, will attempt fallback to `iree-import-onnx` binary")
        try:
            import_onnx_args = [
                "iree-import-onnx",
                str(onnx_model),
                "-o", str(mlir_model),
                "--data-prop",
            ]
            if opset and opset > 0:
                import_onnx_args += ["--opset-version", str(opset)]
            subprocess.check_output(
                import_onnx_args,
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "iree-import-onnx binary not found in PATH"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None


def export_tflite_to_mlir(
    tflite_model: str | Path,
    mlir_model: str | Path | None = None,
):
    if not Path(tflite_model).exists():
        raise FileNotFoundError(f"TFLite model '{tflite_model}' not found")

    # Preferred: the self-contained tosa-converter-for-tflite (its own compiled
    # converter, in-process, no subprocess). The iree.compiler.tflite path below
    # routes through TensorFlow's pywrap_mlir, which is broken on recent TF (the
    # ExperimentalTFLiteToTosaBytecode symbol was renamed to snake_case).
    try:
        from tosa_converter_for_tflite import (
            TosaConverterDebugInfo,
            TosaConverterOutputFormat,
            tflite_flatbuffer_to_tosa_mlir,
        )
    except ImportError:
        tflite_flatbuffer_to_tosa_mlir = None
    if tflite_flatbuffer_to_tosa_mlir is not None:
        tflite_flatbuffer_to_tosa_mlir(
            str(tflite_model), str(mlir_model),
            TosaConverterOutputFormat.Text, TosaConverterDebugInfo.Disabled,
        )
        return

    if TORQ_C_PYAPI:
        iree_tflite_compile.compile_file(
            str(tflite_model),
            import_only=True,
            output_file=str(mlir_model)
        )
    else:
        logger.warning("Torq compiler python API not found, will attempt fallback to `iree-import-tflite` binary")
        try:
            subprocess.check_output(
                [
                    "iree-import-tflite",
                    str(tflite_model),
                    "-o", str(mlir_model)
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "iree-import-tflite binary not found in PATH"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to export TFLite model via '{' '.join(e.cmd)}':\n    "
                + "\n    ".join(e.output.strip().splitlines())
            ) from None


def compile_mlir_for_vm(
    mlir_model: str | Path,
    output_model: str | Path,
    target: str = "torq",
    compiler_args: list[str] | None = None,
    local_compile: bool = False,
    use_binary: bool = False,
    compiler_path: str | Path | None = None,
):
    compiler_args = compiler_args or []
    if target == "llvm-cpu":
        compiler_args += [
            "--iree-hal-target-backends=llvm-cpu"
        ]
        if not local_compile:
            compiler_args += [
                "--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu",
                "--iree-llvmcpu-target-cpu-features=+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse",
                "--iree-llvmcpu-target-cpu=generic"
            ]
        else:
            compiler_args += [
                "--iree-llvmcpu-target-cpu=host"
            ]

            from packaging import version
            if (iree_version := get_iree_version(_resolve_compiler(compiler_path))) and version.parse(iree_version) >= version.parse("3.7.1"):
                compiler_args.append("--iree-hal-target-device=local")
    elif target == "torq":
        if local_compile:
            compiler_args += [
                "--torq-css-qemu",
                "--torq-target-host-triple=native"
            ]

    if TORQ_C_PYAPI and not use_binary:
        compiled_bytes = torq_c.compile_file(
            str(mlir_model),
            target_backends=[target],
            extra_args=compiler_args,
        )
        with open(output_model, "wb") as f:
            f.write(compiled_bytes)
    else:
        if not TORQ_C_PYAPI:
            logger.warning("Torq compiler python API not found, will attempt fallback to `torq-compile` binary")
        compiler = _resolve_compiler(compiler_path)
        try:
            compile_cmd = [
                compiler,
                str(mlir_model),
                "-o", str(output_model)
            ] + [str(arg) for arg in compiler_args]
            logger.debug("Full compile command: '%s'", " ".join(compile_cmd))
            subprocess.check_output(
                compile_cmd,
                text=True,
                stderr=subprocess.STDOUT
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"torq-compile binary not found: '{compiler}'"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to compile MLIR model with: '{' '.join(e.cmd)}:\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def export_torq(
    input_model: str | Path,
    output_dir: str | Path,
    compile_vmfb: bool = True,
    compiler_args: list[str] | None = None,
    local_compile: bool = False,
    use_binary: bool = False,
    compiler_path: str | Path | None = None,
    opset: int | None = None,
):
    input_model = Path(input_model)
    output_dir = Path(output_dir)
    model_name = input_model.stem
    model_type = input_model.suffix
    if model_type not in (".onnx", ".tflite"):
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported extensions are: .onnx, .tflite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        mlir_model = Path(temp_dir) / f"{model_name}.mlir"
        if model_type == ".onnx":
            export_onnx_to_mlir(input_model, mlir_model, opset=opset)
        else:
            export_tflite_to_mlir(input_model, mlir_model)
        copy2(mlir_model, output_dir / f"{model_name}.mlir")
        if compile_vmfb:
            vmfb_model = output_dir / f"{model_name}.vmfb"
            compile_mlir_for_vm(
                mlir_model,
                vmfb_model,
                compiler_args=compiler_args,
                local_compile=local_compile,
                use_binary=use_binary,
                compiler_path=compiler_path,
            )


def add_torq_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Torq args")
    group.add_argument(
        "--opset",
        type=int,
        default=22,
        help="ONNX opset to use, older models will be updated to this opset (default: %(default)s)"
    )
    group.add_argument(
        "--local-compile",
        action="store_true",
        default=False,
        help="Compile for the local host instead of cross-compiling for aarch64"
    )
    group.add_argument(
        "--use-binary",
        action="store_true",
        default=False,
        help="Enforce using the `torq-compile` binary instead of Python API"
    )
    group.add_argument(
        "--compiler-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to torq-compile binary (overrides TORQ_COMPILER_PATH env var, default: torq-compile)"
    )
    group.add_argument(
        "--compile-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq compiler. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )


def main():
    import argparse
    import sys
    from shutil import rmtree
    from .logging import configure_logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=Path,
        metavar=".onnx | .tflite | .mlir",
        help="Path to MLIR or ONNX/TFLite model"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        choices=["torq", "llvm-cpu"],
        default="torq",
        help="Torq compile target (choices: %(choices)s, default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        metavar="FILE",
        default=None,
        help="Output .vmfb file path (default: <model stem>.vmfb in model's directory)"
    )
    parser.add_argument(
        "-d", "--dump-debug",
        action="store_true",
        default=False,
        help="Dump debug symbols"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Root directory for dumping debug symbols (default: <output_dir>/debug)"
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    add_torq_args(parser)
    args = parser.parse_args()

    configure_logging(args.logging)

    if not args.model.exists():
        print(f"Invalid model file '{args.model}'")
        sys.exit(1)

    model_file: Path = args.model
    output_model: Path = args.output or model_file.parent / (model_file.stem + ".vmfb")
    output_dir: Path = output_model.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory set to '%s'", str(output_dir))

    debug_dir = None
    if args.dump_debug:
        debug_dir = args.debug_dir or (output_dir / model_file.stem)
        logger.debug("Debug directory set to '%s'", str(debug_dir))
        if debug_dir.exists():
            rmtree(debug_dir)
            logger.debug("Deleted existing debug files at '%s'", str(debug_dir))
        debug_dir.mkdir(exist_ok=True)

    torq_compile_args: list[str] = args.compile_flags or []
    if debug_dir:
        torq_compile_args += [
            "--mlir-print-ir-after-all",
            f"--mlir-print-ir-tree-dir={debug_dir}/ir",
            f"--dump-compilation-phases-to={debug_dir}/compile",
        ]
    logger.debug("Added torq-compile debug args, current args: %s", str(torq_compile_args))

    model_type: str = model_file.suffix.lower()
    if model_type != ".mlir":
        mlir_model: Path = output_dir / (model_file.stem + ".mlir")
        if model_type == ".onnx":
            logger.info("Exporting ONNX model '%s' to MLIR...", str(model_file))
            export_onnx_to_mlir(model_file, mlir_model, opset=args.opset)
        elif model_type == ".tflite":
            logger.info("Exporting TFLite model '%s' to MLIR...", str(model_file))
            export_tflite_to_mlir(model_file, mlir_model)
        else:
            print(f"Unsupported model type '{model_type}'")
            sys.exit(1)
        logger.info("Successfully exported '%s'", str(mlir_model))
    else:
        mlir_model: Path = model_file

    logger.info("Compiling MLIR model '%s' for %s...", str(mlir_model), args.target)
    try:
        compile_mlir_for_vm(
            mlir_model,
            output_model,
            args.target,
            torq_compile_args,
            args.local_compile,
            args.use_binary,
            args.compiler_path,
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error("Compilation failed for target %s: %s", args.target, str(e))
        if debug_dir:
            logger.info("Debug symbols dumped to '%s'", str(debug_dir))
        else:
            logger.info("Run with '-d' to dump debug symbols")
        sys.exit(1)

    logger.info("Successfully compiled '%s'", str(output_model))


if __name__ == "__main__":
    main()
