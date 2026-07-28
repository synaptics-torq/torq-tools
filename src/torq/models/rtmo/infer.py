# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""RTMO hybrid pose demo: run the three chained vmfbs on an image and draw the
detected people (boxes + 17-keypoint skeletons)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from . import add_rtmo_infer_args
from ._inference import load_rtmo
from ...utils.logging import configure_logging

logger = logging.getLogger("RTMO")

# COCO 17-keypoint skeleton (nose, eyes, ears, shoulders, elbows, wrists, hips,
# knees, ankles).
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_predictions(image_path, dets, keypoints, meta, output_path,
                     score_threshold=0.30, keypoint_threshold=0.30):
    """Draw detections + poses on the original image; return (path, n_drawn)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read '{image_path}'.")
    scale = float(meta["scale"])
    pad_x, pad_y = float(meta.get("pad_x", 0)), float(meta.get("pad_y", 0))
    h, w = image.shape[:2]

    image_dets = np.asarray(dets[0], dtype=np.float32)
    image_kpts = np.asarray(keypoints[0], dtype=np.float32)

    drawn = 0
    for detection, pose in zip(image_dets, image_kpts):
        score = float(detection[4])
        if not np.isfinite(score) or score < score_threshold:
            continue
        x1, y1, x2, y2 = detection[:4]
        x1 = int(np.clip(round((x1 - pad_x) / scale), 0, w - 1))
        y1 = int(np.clip(round((y1 - pad_y) / scale), 0, h - 1))
        x2 = int(np.clip(round((x2 - pad_x) / scale), 0, w - 1))
        y2 = int(np.clip(round((y2 - pad_y) / scale), 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        mp = pose.copy()
        mp[:, 0] = np.clip((mp[:, 0] - pad_x) / scale, 0, w - 1)
        mp[:, 1] = np.clip((mp[:, 1] - pad_y) / scale, 0, h - 1)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {score:.2f}"
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - lh - base - 6)
        cv2.rectangle(image, (x1, top), (min(w - 1, x1 + lw + 8), y1), (0, 220, 0), -1)
        cv2.putText(image, label, (x1 + 4, max(lh + 1, y1 - base - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        for a, b in COCO_SKELETON:
            if mp[a, 2] < keypoint_threshold or mp[b, 2] < keypoint_threshold:
                continue
            p1 = tuple(np.rint(mp[a, :2]).astype(int))
            p2 = tuple(np.rint(mp[b, :2]).astype(int))
            cv2.line(image, p1, p2, (255, 180, 0), 2, cv2.LINE_AA)
        for x, y, vis in mp:
            if vis < keypoint_threshold:
                continue
            cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        drawn += 1

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write output image '{output_path}'.")
    return str(output_path), drawn


def infer_rtmo(args: argparse.Namespace):
    configure_logging(args.logging)
    if cv2 is None:
        raise ImportError("opencv-python is required for the RTMO demo")
    runner = load_rtmo(args.model_dir, device_uri=args.device, n_threads=args.threads)
    for image in args.inputs:
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read '{image}'.")
        dets, keypoints, meta = runner.infer(img)
        output = args.output or f"{Path(image).stem}_rtmo.jpg"
        path, n = draw_predictions(image, dets, keypoints, meta, output)
        logger.info("%s -> %d people (%.1f ms) -> %s",
                    image, n, runner.infer_time_ms, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RTMO hybrid pose inference.")
    add_rtmo_infer_args(parser)
    infer_rtmo(parser.parse_args())
