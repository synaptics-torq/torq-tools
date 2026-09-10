# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""TF-free subprocess worker: compile quantized RTMO TFLite parts to vmfbs.

TensorFlow (imported by quantize_hybrid) and the torq.compiler wheel each link
their own LLVM; both in one process abort with "Option 'remarks-section'
registered more than once". compile_hybrid spawns this worker, which must never
import TensorFlow. Usage: ``python -m torq.models.rtmo._compile_worker <job.json>``.
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
    vmfbs = _compile_parts(job["parts"], job["out_dir"], extra_flags=job.get("extra_flags"), local_compile=job.get("local_compile", False), use_binary=job.get("use_binary", False), compiler_path=job.get("compiler_path"))
    Path(job["result"]).write_text(json.dumps({k: str(v) for k, v in vmfbs.items()}))


if __name__ == "__main__":
    main()
