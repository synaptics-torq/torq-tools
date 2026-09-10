# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.


from . import (
    moonshine,
    smollm2,
    gemma3,
    liquid,
)
"""Model preparation entry points (LLM and Synaptics audio).

LLM subpackages (``gemma3``, ``moonshine``, ``smollm2``) are lazy-loaded so
``python -m torq.models.synaptics_audio`` does not require ``torq.compile``.
"""


import importlib
from typing import Any

__all__ = [
    "gemma3",
    "moonshine",
    "smollm2",
    "gemma3",
    "liquid",
    "rtmo",
    "synaptics_audio",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
