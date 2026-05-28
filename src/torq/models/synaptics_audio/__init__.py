# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Synaptics audio model preparation: FP32 ONNX -> simplified FP32 -> BF16 ONNX.

Public surface is intentionally tiny:

* :data:`recipes.ALL` / :data:`recipes.BY_KEY` -- the registry of audio recipes.
* :func:`prepare.prepare` -- the one and only entry point.
* :func:`fetch.fetch_sources` -- download a recipe's source ONNX files from HF Hub.

The simplification pipeline runs through :class:`torq.graph_edit.OnnxGraphEditor`
plus the per-node ``OnnxGraphEdit`` rewrites in :mod:`torq.graph_edit.edits`;
the audio-specific ordering lives in :func:`prepare._SynapticsAudioGraphEditor.run_audio_pipeline`.

CLI: ``python -m torq.models.synaptics_audio <recipe-key> <dst> [--src <src.onnx>]``
"""

from .fetch import fetch_source, fetch_sources
from .recipes import ALL, BY_KEY, Recipe

__all__ = ["ALL", "BY_KEY", "Recipe", "fetch_source", "fetch_sources"]
