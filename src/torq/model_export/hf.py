# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi
from subprocess import check_output, CalledProcessError, STDOUT

_logger = logging.getLogger(__name__)

__all__ = [
    "hf_download_models",
    "hf_download_source_model",
    "optimum_export_onnx",
]


def hf_download_models(
    repo: str,
    models: list[str],
    subfolder: str,
    local_dir: str | os.PathLike,
):
    for model_name in models:
        _logger.debug("Downloading model '%s' from %s", model_name, repo)
        hf_hub_download(
            repo,
            model_name,
            subfolder=subfolder,
            local_dir=local_dir,
        )


def hf_download_source_model(
    repo: str,
    model_filename: str,
    local_dir: str | os.PathLike,
    subfolder: str | None = None,
    peripheral_files: list[str] | None = None,
) -> Path:
    """Download an ONNX model and peripheral files from a HF repo.

    Returns the path to the downloaded model file.

    Raises:
        FileNotFoundError: If the model file is not found in the repo.
    """
    api = HfApi()
    repo_files = set(api.list_repo_files(repo))

    def _repo_path(filename: str) -> str:
        return f"{subfolder}/{filename}" if subfolder else filename

    model_filepath = _repo_path(model_filename)
    if model_filepath not in repo_files:
        raise FileNotFoundError(
            f"'{model_filepath}' not found in repo '{repo}'"
        )

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(repo, model_filename, subfolder=subfolder, local_dir=local_dir)

    for filename in (peripheral_files or []):
        full_filepath = _repo_path(filename)
        if full_filepath not in repo_files:
            _logger.warning("Peripheral file '%s' not found in repo %s", full_filepath, repo)
            continue
        _logger.debug("Downloading file '%s' from repo %s", full_filepath, repo)
        hf_hub_download(repo, filename, subfolder=subfolder, local_dir=local_dir)

    return local_dir / model_filename


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
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "onnx",
        str(onnx_dir),
        "--model", hf_repo,
        "--dtype", dtype,
        "--opset", str(opset),
    ]
    if opt_level:
        cmd += ["--optimize", str(opt_level)]
    try:
        _logger.info("Exporting %s to ONNX via optimum-cli @ '%s'", hf_repo, str(onnx_dir))
        _logger.debug("optimum-cli full cmd: %s", " ".join(cmd))
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