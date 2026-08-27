# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import numpy as np

from torq.utils.metrics import classify_severity, cosine_similarity, kl_divergence


def test_kl_divergence_zero_for_identical_logits():
    logits = np.array([2.0, 1.0, 0.1, -3.0])
    assert kl_divergence(logits, logits) == 0.0


def test_kl_divergence_positive_and_shift_invariant():
    p = np.array([3.0, 1.0, 0.0])
    q = np.array([0.0, 1.0, 3.0])
    kl = kl_divergence(p, q)
    assert kl > 0.0
    # softmax is shift-invariant, so adding a constant to either logit vector is a no-op
    assert kl == kl_divergence(p + 5.0, q)


def test_cosine_similarity_bounds():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, -a) == -1.0
    assert cosine_similarity(a, np.zeros_like(a)) == 0.0


def test_classify_severity_buckets():
    assert classify_severity(2.0) == "CRITICAL"
    assert classify_severity(0.5) == "HIGH"
    assert classify_severity(0.05) == "MEDIUM"
    assert classify_severity(0.001) == "LOW"
    # boundaries are exclusive (> threshold), so an exact threshold falls in the lower bucket
    assert classify_severity(0.01) == "LOW"
