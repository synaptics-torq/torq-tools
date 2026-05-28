# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""ONNX model verification helpers."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import onnx

from .ort import make_cpu_session

logger = logging.getLogger(__name__)


def _run(
    model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    sess = make_cpu_session(model.SerializeToString())
    outputs = sess.run(None, dict(feeds))
    return {o.name: a for o, a in zip(sess.get_outputs(), outputs)}


def verify_equivalence(
    source: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_shapes: Mapping[str, Sequence[int]],
    *,
    seed: int = 42,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Assert that two FP32 ONNX models produce equivalent random-input outputs."""
    rng = np.random.default_rng(seed)
    feeds = {
        name: rng.standard_normal(tuple(shape)).astype(np.float32)
        for name, shape in input_shapes.items()
    }

    src_out = _run(source, feeds)
    simp_out = _run(simplified, feeds)

    if set(src_out) != set(simp_out):
        raise AssertionError(
            "output name set differs: "
            f"source={sorted(src_out)} simplified={sorted(simp_out)}"
        )

    for name in sorted(src_out):
        np.testing.assert_allclose(
            simp_out[name],
            src_out[name],
            atol=atol,
            rtol=rtol,
            err_msg=f"FP32 output {name!r} drifted after simplification pipeline",
        )
        logger.info("output %s OK (shape=%s)", name, src_out[name].shape)
