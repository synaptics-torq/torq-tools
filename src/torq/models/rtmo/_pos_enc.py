# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""2D sin-cos positional embedding used by the RTMO neck (HybridEncoder AIFI).

The exported RTMO graph bakes the transformer positional encoding
(``neck.pos_enc_0``) as a constant sized for the input the model was exported
with (416x416 -> a 13x13 stride-32 feature map -> ``[1, 169, 256]``). When we
re-target the model to a different input size we must regenerate this constant
for the new stride-32 grid, otherwise the encoder's ``Add`` shape-mismatches.

``build_2d_sincos_position_embedding`` reproduces the mmpose formula. It has
been validated against the baked ``neck.pos_enc_0`` (max abs error ~1e-3, i.e.
below bf16 rounding) so the same call regenerates a correct constant for any
square grid.
"""

from __future__ import annotations

import numpy as np


def build_2d_sincos_position_embedding(
    w: int,
    h: int,
    embed_dim: int = 256,
    temperature: float = 10000.0,
) -> np.ndarray:
    """Return the ``[1, w*h, embed_dim]`` fp32 sin-cos position embedding.

    Matches mmpose ``HybridEncoder.build_2d_sincos_position_embedding``: the
    grid is built with ``meshgrid(arange(w), arange(h), indexing="ij")`` and the
    four quarters are concatenated as ``[cos_w, sin_w, cos_h, sin_h]``.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    grid_w, grid_h = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing="ij",
    )
    pos_dim = embed_dim // 4
    omega = np.arange(pos_dim, dtype=np.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    out_w = grid_w.reshape(-1)[:, None] * omega[None]
    out_h = grid_h.reshape(-1)[:, None] * omega[None]

    pos_emb = np.concatenate(
        [np.cos(out_w), np.sin(out_w), np.cos(out_h), np.sin(out_h)],
        axis=1,
    ).astype(np.float32)
    return pos_emb[None, :, :]
