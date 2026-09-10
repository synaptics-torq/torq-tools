# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the dynamic-shape PP-OCRv6-tiny det/rec ONNX sources from HF."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("ppocr-download")

HF_REPO = "Synaptics/paddle-paddle-tiny"
FILES = {"det": "ppocr_det_dynamic.onnx", "rec": "ppocr_rec_dynamic.onnx", "dict": "ppocr_rec.yml"}


def download_source(models_dir: str | Path = "models/ppocr") -> dict[str, Path]:
    """Idempotently fetch the det/rec sources (+ char dict); return their paths."""
    from huggingface_hub import hf_hub_download

    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, name in FILES.items():
        dest = target / name
        if not dest.exists():
            logger.info("downloading %s from %s", name, HF_REPO)
            shutil.copy(hf_hub_download(HF_REPO, name), dest)
        out[tag] = dest
    return out
