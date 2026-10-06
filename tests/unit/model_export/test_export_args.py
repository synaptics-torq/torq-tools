# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared export-argument defaults.

The chip-optimal export options (``--extract-embeddings`` and its siblings)
are defined once in the shared ``add_onnx_args`` / ``add_decoder_args`` /
``add_llm_args`` / ``add_torq_args`` helpers rather than in each model's
``__init__.py``, so a model family cannot silently drift from the optimal
defaults (moonshine was exactly that case: its local
``--extract-embeddings`` copy stayed off while every LLM moved on).
These tests pin the per-exporter CLI surface the shared helpers provide."""

import argparse

import pytest

from torq.models.gemma3 import add_gemma3_export_args
from torq.models.liquid import add_liquid_export_args, add_liquid_vl_export_args
from torq.models.moonshine import add_moonshine_export_args
from torq.models.moonshine_streaming import add_moonshine_streaming_export_args
from torq.models.smollm2 import add_smollm2_export_args

pytestmark = pytest.mark.unit

ALL_EXPORTERS = [
    ("gemma3", add_gemma3_export_args),
    ("liquid", add_liquid_export_args),
    ("liquid-vl", add_liquid_vl_export_args),
    ("smollm2", add_smollm2_export_args),
    ("moonshine", add_moonshine_export_args),
    ("moonshine_streaming", add_moonshine_streaming_export_args),
]

# liquid-vl hardcodes _extract_embeddings=False (no flag, by design).
EMBEDDINGS_EXPORTERS = [(n, f) for n, f in ALL_EXPORTERS if n != "liquid-vl"]

# moonshine / moonshine_streaming use their own component-selectable
# --skip-torq (a list) instead of the bool one.
BOOL_SKIP_TORQ_EXPORTERS = [
    (n, f) for n, f in ALL_EXPORTERS if not n.startswith("moonshine")
]

# -s/--model-size: (choices, default); None = no selector (VL size is fixed).
MODEL_SIZE_CHOICES = {
    "gemma3": (["270m", "1b"], "270m"),
    "liquid": (["350m", "230m"], "350m"),
    "liquid-vl": (None, None),
    "smollm2": (["135M", "360M", "1.7B"], "135M"),
    "moonshine": (["base", "tiny"], "tiny"),
    "moonshine_streaming": (["tiny", "small", "medium"], "tiny"),
}

# --split-lm-head / --batch-prefill: the LLM decoder exporters only.
SPLIT_PREFILL_EXPORTERS = [
    (n, f) for n, f in ALL_EXPORTERS if n in ("gemma3", "liquid", "liquid-vl")
]

# --instruct-model: gemma3 and smollm2 only.
INSTRUCT_EXPORTERS = [
    (n, f) for n, f in ALL_EXPORTERS if n in ("gemma3", "smollm2")
]

MAX_GEN_TOKENS_DEFAULTS = {"gemma3": 256, "liquid": 256, "liquid-vl": 256, "smollm2": 64}


def _parser(add_args) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser


def _parse(add_args, *argv) -> argparse.Namespace:
    return _parser(add_args).parse_args(list(argv))


def _action(add_args, option):
    for action in _parser(add_args)._actions:
        if option in action.option_strings:
            return action
    return None


@pytest.mark.parametrize("name,add_args", ALL_EXPORTERS)
def test_shared_export_args_are_present(name, add_args):
    args = _parse(add_args)
    assert args.models_dir == "models"
    assert args.broadcast_ops is None


@pytest.mark.parametrize("name,add_args", EMBEDDINGS_EXPORTERS)
def test_extract_embeddings_is_on_by_default(name, add_args):
    assert _parse(add_args).extract_embeddings is True
    assert _parse(add_args, "--no-extract-embeddings").extract_embeddings is False
    assert _parse(add_args, "--extract-embeddings").extract_embeddings is True


@pytest.mark.parametrize("name,add_args", BOOL_SKIP_TORQ_EXPORTERS)
def test_skip_torq_is_a_bool_opt_out(name, add_args):
    assert _parse(add_args).skip_torq is False
    assert _parse(add_args, "--skip-torq").skip_torq is True


def test_moonshine_skip_torq_stays_component_selectable():
    assert _parse(add_moonshine_export_args, "--skip-torq", "decoder").skip_torq == ["decoder"]


def test_dynamic_models_absent_from_streaming_exporter():
    # moonshine_streaming is always static; the shared flag must not leak in.
    assert _action(add_moonshine_streaming_export_args, "--dynamic-models") is None


@pytest.mark.parametrize("name,add_args", ALL_EXPORTERS)
def test_model_size_selector(name, add_args):
    choices, default = MODEL_SIZE_CHOICES[name]
    action = _action(add_args, "--model-size")
    if choices is None:
        assert action is None
        return
    assert action is not None
    assert list(action.choices) == choices
    assert action.default == default
    assert _parse(add_args, "-s", choices[0]).model_size == choices[0]


@pytest.mark.parametrize("name,add_args", SPLIT_PREFILL_EXPORTERS)
def test_split_lm_head_on_and_batch_prefill_default(name, add_args):
    assert _parse(add_args).split_lm_head is True
    assert _parse(add_args).batch_prefill == 64
    off = _parse(add_args, "--no-split-lm-head", "--batch-prefill", "0")
    assert off.split_lm_head is False
    assert off.batch_prefill == 0


@pytest.mark.parametrize("name,add_args", INSTRUCT_EXPORTERS)
def test_instruct_model_is_opt_in(name, add_args):
    assert _parse(add_args).instruct_model is False
    assert _parse(add_args, "--instruct-model").instruct_model is True


def test_instruct_model_absent_from_liquid():
    assert _action(add_liquid_export_args, "--instruct-model") is None
    assert _action(add_liquid_vl_export_args, "--instruct-model") is None


@pytest.mark.parametrize("name,add_args", ALL_EXPORTERS[:4])
def test_max_gen_tokens_defaults(name, add_args):
    assert _parse(add_args).max_gen_tokens == MAX_GEN_TOKENS_DEFAULTS[name]
