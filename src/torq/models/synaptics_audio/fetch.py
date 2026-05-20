# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Fetch a recipe's source FP32 ONNX from the HuggingFace Hub.

Thin wrapper around :func:`huggingface_hub.hf_hub_download`. Uses the standard
HF cache (``~/.cache/huggingface/hub/`` or ``$HF_HOME``) by default; downloads
are deduplicated and only re-fetched when the upstream file changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .recipes import Recipe

logger = logging.getLogger("synaptics-audio.fetch")


def fetch_source(
    recipe: Recipe,
    *,
    source_filename: str | None = None,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download a recipe source ONNX from ``recipe.repo_id`` via HF Hub.

    Args:
        recipe: the recipe to fetch a source ONNX for.
        source_filename: optional specific source filename from the recipe.
            ``None`` fetches the first declared source.
        cache_dir: optional override for the HF cache directory. ``None``
            uses the default (``$HF_HOME`` or ``~/.cache/huggingface/hub/``).

    Returns:
        Local filesystem path to the downloaded (or cached) ONNX file.

    Raises:
        ValueError: if no source has been declared.
    """
    source_filenames = recipe.source_filenames()
    if not source_filenames:
        raise ValueError(
            f"recipe {recipe.key!r} has no source_filename; cannot auto-fetch from HF. "
            f"Either set source_filename in the recipe or provide an explicit src path."
        )
    source_filename = source_filename or source_filenames[0]

    from huggingface_hub import hf_hub_download

    logger.info(
        "fetching %s/%s from HuggingFace Hub",
        recipe.repo_id,
        source_filename,
    )
    local = hf_hub_download(
        repo_id=recipe.repo_id,
        filename=source_filename,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    logger.info("source ONNX cached at %s", local)
    return Path(local)


def fetch_sources(
    recipe: Recipe,
    *,
    cache_dir: Path | str | None = None,
) -> list[Path]:
    """Download every source ONNX declared by ``recipe``."""
    return [
        fetch_source(recipe, source_filename=source, cache_dir=cache_dir)
        for source in recipe.source_filenames()
    ]
