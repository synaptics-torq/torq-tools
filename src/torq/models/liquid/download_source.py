# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the LFM2.5 (Liquid) source ONNX from HuggingFace into the
canonical source location the exporter reads from.

The exporter (`torq-export-model liquid`) auto-downloads on first run, so this
is only needed when you want the source on disk ahead of time — for offline
export, inspection, or to avoid a re-download.

This script is self-contained (no compiler-toolchain imports), so run it as a
file:

    python src/torq/models/liquid/download_source.py \\
        --models-dir /home/kshanmug/torq/torq-tools-dev/models
"""

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("Liquid.download")

# Kept self-contained (no package imports) so downloading the source never
# requires the compiler toolchain. Keep in sync with the maps in `export.py`.
DEFAULT_MODEL_SIZE = "350m"
MODEL_SIZES = ["350m", "230m"]
# Tried in order: Synaptics-hosted mirror first (so this keeps working if the
# upstream LiquidAI repo is renamed/removed), then upstream as fallback. The
# Synaptics repo is private — set HF_TOKEN (or `huggingface-cli login`) to use
# it; without a token it 401s and we fall through to the public upstream repo.
HF_REPO_ONNX = {
    "350m": (
        "Synaptics/LiquidAI-LFM2p5-350M-LLM",
        "LiquidAI/LFM2.5-350M-ONNX",
    ),
    "230m": (
        "Synaptics/LiquidAI-LFM2p5-230M-LLM",
        "LiquidAI/LFM2.5-230M-ONNX",
    ),
}

# Files to pull from the HF repo: (repo path, required?)
_SOURCE_FILES: tuple[tuple[str, bool], ...] = (
    ("onnx/model.onnx", True),
    ("onnx/model.onnx_data", True),
    ("config.json", False),
    ("tokenizer.json", False),
    ("tokenizer_config.json", False),
)


def download_source(
    models_dir: str | Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dtype: str = "fp32",
) -> Path:
    """Download the source ONNX into
    ``<models-dir>/liquid-2p5-<size>/source/onnx/<dtype>/`` and return the dir.
    """
    from huggingface_hub import hf_hub_download

    repos = HF_REPO_ONNX[model_size]
    if isinstance(repos, str):
        repos = (repos,)
    target = (
        Path(models_dir)
        / f"liquid-2p5-{model_size}"
        / "source"
        / "onnx"
        / model_dtype
    )
    target.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for repo in repos:
        logger.info("Downloading source ONNX from '%s' into %s", repo, target)
        ok = True
        for filename, required in _SOURCE_FILES:
            dest = target / Path(filename).name
            if dest.exists():
                logger.info("  exists, skip: %s", dest.name)
                continue
            try:
                cached = hf_hub_download(repo, filename)
                shutil.copy(cached, dest)
                size_mb = dest.stat().st_size / 1e6
                logger.info("  ok: %s (%.1f MB)", dest.name, size_mb)
            except Exception as e:
                last_err = e
                if required:
                    logger.warning("  %s missing from %s: %s", filename, repo, e)
                    ok = False
                    break
                logger.debug("  optional missing %s: %s", filename, e)
        if ok:
            logger.info("Source ready at %s", target)
            return target
        logger.warning("  repo '%s' unavailable; trying next", repo)

    raise RuntimeError(
        f"Failed to download source ONNX from any of {list(repos)}: {last_err}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download the LFM2.5 (Liquid) source ONNX from HuggingFace.",
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
        help="Model size to download (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_source(args.models_dir, args.model_size)


if __name__ == "__main__":
    main()
