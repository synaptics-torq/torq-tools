# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from subprocess import check_output, CalledProcessError, STDOUT

__all__ = [
    "hf_download_models",
    "optimum_export_onnx",
]


def hf_download_models(
    repo: str,
    models: list[str],
    subfolder: str,
    local_dir: str | os.PathLike,
):
    for model_name in models:
        hf_hub_download(
            repo,
            model_name,
            subfolder=subfolder,
            local_dir=local_dir,
        )


def optimum_export_onnx(
    onnx_dir: str | os.PathLike,
    hf_repo: str,
    dtype: str,
    models: list[str],
    *,
    opset: int = 22,
    opt_level: str | None = "O1",
):
    if all(
        (Path(onnx_dir) / name).exists()
        for name in models
    ):
        return
    cmd = [
        "optimum-cli", "export", "onnx",
        str(onnx_dir),
        "--model", hf_repo,
        "--dtype", dtype,
        "--opset", str(opset),
    ]
    if opt_level:
        cmd += ["--optimize", str(opt_level)]
    try:
        check_output(
            cmd,
            text=True,
            stderr=STDOUT,
        )
    except CalledProcessError as e:
        raise RuntimeError(
            f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None