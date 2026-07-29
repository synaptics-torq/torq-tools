# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Subprocess worker: compile already-quantized RTMO TFLite parts to vmfbs in a
**TensorFlow-free** interpreter.

:func:`torq.models.rtmo._hybrid.quantize_hybrid` imports TensorFlow, which
statically links one copy of LLVM; the ``torq.compiler`` wheel links another.
Both live in one process and the second registration aborts with::

    Option 'remarks-section' registered more than once!
    LLVM ERROR: inconsistency in registered CommandLine options

:func:`torq.models.rtmo._hybrid.compile_hybrid` spawns this worker (which never
imports TF) so the compile runs cleanly. This module must not import TensorFlow,
directly or transitively.

Usage::

    python -m torq.models.rtmo._compile_worker <job.json>

``job.json`` carries the ``compile_hybrid`` arguments plus a ``result`` path the
worker writes the ``{name: vmfb}`` map to.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ._hybrid import _compile_parts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    job = json.loads(Path(sys.argv[1]).read_text())
    vmfbs = _compile_parts(
        job["parts"], job["out_dir"],
        extra_flags=job.get("extra_flags"),
        local_compile=job.get("local_compile", False),
        use_binary=job.get("use_binary", False),
        compiler_path=job.get("compiler_path"),
    )
    Path(job["result"]).write_text(
        json.dumps({k: str(v) for k, v in vmfbs.items()})
    )


if __name__ == "__main__":
    main()
