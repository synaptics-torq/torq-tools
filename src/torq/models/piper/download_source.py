# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download a Piper voice (ONNX + config) from rhasspy/piper-voices."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("piper-download")

HF_REPO = "rhasspy/piper-voices"


def download_source(voice: str, models_dir: str | Path = "models/piper") -> dict[str, Path]:
    """Idempotently fetch ``<voice>.onnx(.json)``; voice keys look like ``es_MX-ald-medium``."""
    from huggingface_hub import hf_hub_download

    locale, name, quality = voice.split("-")
    prefix = f"{locale.split('_')[0]}/{locale}/{name}/{quality}"
    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, suffix in (("model", ".onnx"), ("config", ".onnx.json")):
        dest = target / f"{voice}{suffix}"
        if not dest.exists():
            logger.info("downloading %s from %s", dest.name, HF_REPO)
            shutil.copy(hf_hub_download(HF_REPO, f"{prefix}/{voice}{suffix}"), dest)
        out[tag] = dest
    return out
