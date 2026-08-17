# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""2D sin-cos positional embedding for the RTMO neck (AIFI transformer).

The export bakes ``neck.pos_enc_0`` as a constant sized for the export-time
input; re-targeting to a new input size regenerates it for the new stride-32
grid. Reproduces the mmpose formula (validated against the baked constant).
"""

from __future__ import annotations

import numpy as np


def build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0) -> np.ndarray:
    """Return the ``[1, w*h, embed_dim]`` fp32 embedding: ``[cos_w, sin_w, cos_h, sin_h]``."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    grid_w, grid_h = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing="ij")
    pos_dim = embed_dim // 4
    omega = 1.0 / (temperature ** (np.arange(pos_dim, dtype=np.float32) / pos_dim))
    out_w = grid_w.reshape(-1)[:, None] * omega[None]
    out_h = grid_h.reshape(-1)[:, None] * omega[None]
    pos_emb = np.concatenate([np.cos(out_w), np.sin(out_w), np.cos(out_h), np.sin(out_h)], axis=1).astype(np.float32)
    return pos_emb[None, :, :]
