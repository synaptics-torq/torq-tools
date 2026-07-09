# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
import os
import shutil
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

    Each file is located independently: the requested ``subfolder`` is preferred
    but the repo root is used as a fallback, so repos that keep the model under
    ``onnx/`` while leaving config/tokenizer at the root (e.g. the onnx-community
    mirrors) work. Every file — including the model's external-data sidecar
    (``<model_filename>_data``) — is flattened into ``local_dir`` by basename so
    the model and its metadata sit together.

    Returns the path to the downloaded model file.

    Raises:
        FileNotFoundError: If the model file is not found in the repo.
    """
    api = HfApi()
    repo_files = set(api.list_repo_files(repo))
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    def _locate(filename: str) -> str | None:
        """Repo subfolder holding ``filename`` (preferred subfolder, else root),
        or None if the file is absent from the repo."""
        if subfolder and f"{subfolder}/{filename}" in repo_files:
            return subfolder
        if filename in repo_files:
            return ""
        return None

    def _fetch(filename: str, *, required: bool) -> Path | None:
        found_in = _locate(filename)
        if found_in is None:
            if required:
                raise FileNotFoundError(
                    f"'{filename}' not found in repo '{repo}' (subfolder={subfolder!r})"
                )
            _logger.warning("File '%s' not found in repo %s", filename, repo)
            return None
        _logger.debug("Downloading file '%s' from repo %s", filename, repo)
        cached = Path(hf_hub_download(repo, filename, subfolder=found_in or None))
        dest = local_dir / Path(filename).name
        if cached.resolve() != dest.resolve():
            shutil.copy2(cached, dest)
        return dest

    model_dest = _fetch(model_filename, required=True)
    # ONNX external-data sidecar (e.g. model.onnx_data), present for larger models.
    _fetch(f"{model_filename}_data", required=False)
    for filename in (peripheral_files or []):
        _fetch(filename, required=False)

    return model_dest


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