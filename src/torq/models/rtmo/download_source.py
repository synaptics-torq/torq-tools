# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Download the RTMO source model + calibration images from Hugging Face.

The upstream RTMO artifacts live at https://huggingface.co/Synaptics/RTMO_pose
(``model.onnx`` — the original with post-processing baked in — plus a small
``calib/`` set of COCO images for PTQ). Kept self-contained (no package/compiler
imports) so fetching the source never needs the toolchain. The Synaptics repo
may be private — set ``HF_TOKEN`` or run ``huggingface-cli login`` if so.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("rtmo-download")

HF_REPO = "Synaptics/RTMO_pose"
MODEL_FILE = "model.onnx"


def download_source(models_dir: str | Path = "models/rtmo", with_calib: bool = True) -> Path:
    """Download ``model.onnx`` (and, by default, the ``calib/`` images) from
    :data:`HF_REPO` into ``models_dir``; return the local ``model.onnx`` path.

    Existing files are left untouched (idempotent).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)

    model_dest = target / MODEL_FILE
    if model_dest.exists():
        logger.info("source model exists, skip: %s", model_dest)
    else:
        logger.info("downloading %s from %s", MODEL_FILE, HF_REPO)
        shutil.copy(hf_hub_download(HF_REPO, MODEL_FILE), model_dest)
        logger.info("  ok: %s (%.1f MB)", model_dest, model_dest.stat().st_size / 1e6)

    if with_calib:
        calib = target / "calib"
        calib.mkdir(exist_ok=True)
        new = 0
        for f in list_repo_files(HF_REPO):
            if f.startswith("calib/") and f.lower().endswith((".jpg", ".jpeg", ".png")):
                dest = calib / Path(f).name
                if not dest.exists():
                    shutil.copy(hf_hub_download(HF_REPO, f), dest)
                    new += 1
        logger.info("calibration images: %d new in %s", new, calib)

    return model_dest


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Download the RTMO source ONNX + calib from HF.")
    ap.add_argument("-o", "--models-dir", default="models/rtmo",
                    help="Destination directory (default: %(default)s)")
    ap.add_argument("--no-calib", action="store_true", help="Skip the calibration images")
    args = ap.parse_args()
    print("source model:", download_source(args.models_dir, with_calib=not args.no_calib))


if __name__ == "__main__":
    main()
