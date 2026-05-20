# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Internal CPU-only ONNX Runtime helper.

This package's verification and shape-probing flows run **strictly on CPU**:

* ``providers`` is locked to ``CPUExecutionProvider``;
* per-session ``log_severity_level`` is raised to ERROR so warnings stay quiet;
* the process-wide default logger severity is also raised to ERROR;
* during the very first ``import onnxruntime`` in the process, ORT 1.20+'s
  native autoEP discovery prints a ``GPU device discovery failed`` warning
  *to file descriptor 2* before any Python-level logging hook runs. We
  bracket the import in an FD-level stderr redirect so that probe stays
  silent. Once ORT is loaded the redirect is unwound and stderr is restored,
  so any subsequent error is still visible.

All ORT session creation in :mod:`torq.models.synaptics_audio` should go
through :func:`make_cpu_session` so the no-GPU policy is enforced in one
place.
"""

from __future__ import annotations

import os
import sys
from typing import Union


def _import_ort_silently():
    """Import ``onnxruntime`` with the native GPU-discovery warning silenced."""
    if "onnxruntime" in sys.modules:
        return sys.modules["onnxruntime"]

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Stderr is not a real fd (e.g. captured by a test runner);
        # fall back to a plain import. Any warning will be captured anyway.
        import onnxruntime  # noqa: PLC0415

        return onnxruntime

    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        import onnxruntime  # noqa: PLC0415
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)
    return onnxruntime


ort = _import_ort_silently()

# 0 = verbose, 1 = info, 2 = warning, 3 = error, 4 = fatal.
ort.set_default_logger_severity(3)

CPU_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)

ModelInput = Union[str, bytes, "bytearray"]


def make_cpu_session(
    model: ModelInput,
    *,
    sess_options: "ort.SessionOptions | None" = None,
) -> "ort.InferenceSession":
    """Return a CPU-only :class:`onnxruntime.InferenceSession` with quiet logging.

    Args:
        model: filesystem path (``str`` / ``Path``-like) or serialized ONNX
            bytes.
        sess_options: optional pre-configured ``SessionOptions``. When
            ``None`` a fresh one is allocated and its ``log_severity_level``
            is set to ERROR.

    The returned session is guaranteed to use *only* the CPU EP regardless
    of what ORT discovers on the host.
    """
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model,
        sess_options=opts,
        providers=list(CPU_PROVIDERS),
    )
