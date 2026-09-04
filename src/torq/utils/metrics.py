# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Numeric metrics for quantization sensitivity analysis.

Shared by the weight-quantization (``weights analyze``) and dynamic-quantization
(``dynamic analyze``) tools to score how far a quantized model's outputs drift
from the fp32 baseline.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kl_divergence",
    "cosine_similarity",
    "classify_severity",
]


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL divergence ``D(P || Q)`` between the softmax distributions of two logit vectors."""
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    p -= p.max()
    q -= q.max()
    p_exp = np.exp(p)
    q_exp = np.exp(q)
    p_sum = p_exp.sum()
    q_sum = q_exp.sum()
    log_p = p - np.log(p_sum)
    log_q = q - np.log(q_sum)
    p_prob = p_exp / p_sum
    kl = float(np.sum(p_prob * (log_p - log_q)))
    return max(0.0, kl)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays (0.0 if either is all-zero)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify_severity(kl_divergence: float) -> str:
    """Bucket a KL divergence into a severity label: LOW / MEDIUM / HIGH / CRITICAL."""
    if kl_divergence > 1.0:
        return "CRITICAL"
    if kl_divergence > 0.1:
        return "HIGH"
    if kl_divergence > 0.01:
        return "MEDIUM"
    return "LOW"
