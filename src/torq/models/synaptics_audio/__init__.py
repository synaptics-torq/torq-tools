# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics audio model preparation: FP32 ONNX -> simplified FP32 -> BF16 ONNX.

Public surface is intentionally tiny:

* :data:`recipes.ALL` / :data:`recipes.BY_KEY` -- the registry of audio recipes.
* :func:`prepare.prepare` -- the one and only entry point.
* :func:`fetch.fetch_sources` -- download a recipe's source ONNX files from HF Hub.
* :data:`passes.PIPELINE` -- the fixed list of value-preserving rewrites
  applied in order.

CLI: ``python -m torq.models.synaptics_audio <recipe-key> <dst> [--src <src.onnx>]``
"""

from .fetch import fetch_source, fetch_sources
from .recipes import ALL, BY_KEY, Recipe

__all__ = ["ALL", "BY_KEY", "Recipe", "fetch_source", "fetch_sources"]
