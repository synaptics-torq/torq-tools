# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download YOLO26 pretrained weights (via Ultralytics' own asset releases) plus a
calibration image set (COCO128) for PTQ.

Self-contained (no torq package/compiler imports).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("yolo26-download")

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DEFAULT_VARIANT = "yolo26n"


def download_weights(models_dir: str | Path = "models/yolo26", variant: str = DEFAULT_VARIANT) -> Path:
    """Idempotently fetch ``<variant>.pt`` into ``models_dir/source`` (Ultralytics auto-downloads)."""
    from ultralytics import YOLO

    source_dir = Path(models_dir) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    dest = source_dir / f"{variant}.pt"
    if dest.exists():
        logger.info("weights exist, skip: %s", dest)
        return dest
    logger.info("downloading %s.pt via Ultralytics assets", variant)
    YOLO(f"{variant}.pt")  # downloads into cwd
    cwd_file = Path(f"{variant}.pt")
    if cwd_file.exists() and cwd_file.resolve() != dest.resolve():
        cwd_file.replace(dest)
    return dest


def download_calib(models_dir: str | Path = "models/yolo26", n_images: int | None = None) -> Path:
    """Idempotently fetch the COCO128 image set into ``models_dir/calib`` for PTQ calibration."""
    from ultralytics.utils.downloads import download as ul_download

    calib_dir = Path(models_dir) / "calib"
    if calib_dir.exists() and any(calib_dir.glob("*.jpg")):
        logger.info("calibration images exist, skip: %s", calib_dir)
        return calib_dir
    calib_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(models_dir) / "_coco128_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ul_download([COCO128_URL], dir=tmp_dir, unzip=True, delete=True, threads=1)
    images = sorted((tmp_dir / "coco128" / "images" / "train2017").glob("*.jpg"))
    if n_images:
        images = images[:n_images]
    for img in images:
        img.replace(calib_dir / img.name)
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("fetched %d calibration images into %s", len(images), calib_dir)
    return calib_dir


def download_source(models_dir: str | Path = "models/yolo26", variant: str = DEFAULT_VARIANT, with_calib: bool = True) -> Path:
    weights = download_weights(models_dir, variant)
    if with_calib:
        download_calib(models_dir)
    return weights


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Download YOLO26 pretrained weights + COCO128 calib images.")
    ap.add_argument("-o", "--models-dir", default="models/yolo26", help="Destination directory (default: %(default)s)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="Model scale, e.g. yolo26n/s/m/l/x (default: %(default)s)")
    ap.add_argument("--no-calib", action="store_true", help="Skip the calibration images")
    args = ap.parse_args()
    print("weights:", download_source(args.models_dir, args.variant, with_calib=not args.no_calib))


if __name__ == "__main__":
    main()
