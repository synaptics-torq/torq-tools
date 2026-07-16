# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the MobileNetV4 source .tflite from the MLCommons ``mobile_open``
GitHub release into the canonical source location the exporter will read from.

Kept self-contained (no package imports) so downloading the source never
requires the compiler toolchain — run it as a file:

    python src/torq/models/mobilenetv4/download_source.py \\
        --models-dir models
"""

import argparse
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("MobileNetV4.download")

DEFAULT_MODEL_SIZE = "large"
MODEL_SIZES = ["large"]

# MLCommons only publishes a single fp32 tflite asset per size today (no
# pre-quantized variants) — see https://github.com/mlcommons/mobile_open/releases
# (tag "model_upload"). Quantized variants, once needed, are produced locally
# from this fp32 baseline rather than downloaded, so DTYPES has one entry for now.
DEFAULT_MODEL_DTYPE = "fp32"
MODEL_DTYPES = ["fp32"]

_RELEASE_URL = "https://github.com/mlcommons/mobile_open/releases/download/model_upload"
_SOURCE_FILES = {
    "large": {
        "fp32": "MobileNetV4-Conv-Large-fp32.tflite",
    },
}


def download_source(
    models_dir: str | Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dtype: str = DEFAULT_MODEL_DTYPE,
) -> Path:
    """Download the source tflite into
    ``<models-dir>/mobilenetv4/source/tflite/<size>/<dtype>/`` and return the dir.
    """
    filename = _SOURCE_FILES[model_size][model_dtype]
    target = (
        Path(models_dir)
        / "mobilenetv4"
        / "source"
        / "tflite"
        / model_size
        / model_dtype
    )
    target.mkdir(parents=True, exist_ok=True)

    dest = target / filename
    if dest.exists():
        logger.info("exists, skip: %s", dest)
        return target

    url = f"{_RELEASE_URL}/{filename}"
    logger.info("Downloading source tflite from '%s' into %s", url, target)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        tmp_dest.rename(dest)
    except Exception:
        tmp_dest.unlink(missing_ok=True)
        raise
    size_mb = dest.stat().st_size / 1e6
    logger.info("ok: %s (%.1f MB)", dest.name, size_mb)
    logger.info("Source ready at %s", target)
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Download the MobileNetV4 source tflite from MLCommons mobile_open.",
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
    parser.add_argument(
        "-d", "--model-dtype",
        type=str,
        choices=MODEL_DTYPES,
        default=DEFAULT_MODEL_DTYPE,
        help="Model dtype to download (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_source(args.models_dir, args.model_size, args.model_dtype)


if __name__ == "__main__":
    main()
