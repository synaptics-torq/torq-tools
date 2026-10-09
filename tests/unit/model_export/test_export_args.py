# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared export-argument defaults.

The chip-optimal export options (``--extract-embeddings`` and its siblings)
are defined once in the shared ``add_onnx_args`` / ``add_decoder_args`` /
``add_llm_args`` / ``add_torq_args`` helpers rather than in each model's
``__init__.py``, so a model family cannot silently drift from the optimal
defaults (moonshine was exactly that case: its local
``--extract-embeddings`` copy stayed off while every LLM moved on).
These tests pin the optimal defaults across every exporter."""

import argparse

import pytest

from torq.models.gemma3 import add_gemma3_export_args
from torq.models.liquid import add_liquid_export_args, add_liquid_vl_export_args
from torq.models.moonshine import add_moonshine_export_args
from torq.models.moonshine_streaming import add_moonshine_streaming_export_args
from torq.models.smollm2 import add_smollm2_export_args

pytestmark = pytest.mark.unit

# liquid-vl hardcodes _extract_embeddings=False (no flag, by design).
EMBEDDINGS_EXPORTERS = [
    ("gemma3", add_gemma3_export_args),
    ("liquid", add_liquid_export_args),
    ("smollm2", add_smollm2_export_args),
    ("moonshine", add_moonshine_export_args),
    ("moonshine_streaming", add_moonshine_streaming_export_args),
]

# --split-lm-head / --batch-prefill: the LLM decoder exporters only.
SPLIT_PREFILL_EXPORTERS = [
    ("gemma3", add_gemma3_export_args),
    ("liquid", add_liquid_export_args),
    ("liquid-vl", add_liquid_vl_export_args),
]


def _parse(add_args, *argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args(list(argv))


@pytest.mark.parametrize("name,add_args", EMBEDDINGS_EXPORTERS)
def test_extract_embeddings_is_on_by_default(name, add_args):
    assert _parse(add_args).extract_embeddings is True
    assert _parse(add_args, "--no-extract-embeddings").extract_embeddings is False


@pytest.mark.parametrize("name,add_args", SPLIT_PREFILL_EXPORTERS)
def test_split_lm_head_on_and_batch_prefill_default(name, add_args):
    assert _parse(add_args).split_lm_head is True
    assert _parse(add_args).batch_prefill == 64
    off = _parse(add_args, "--no-split-lm-head", "--batch-prefill", "0")
    assert off.split_lm_head is False
    assert off.batch_prefill == 0
