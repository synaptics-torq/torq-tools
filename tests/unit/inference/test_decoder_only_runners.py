# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

from pathlib import Path

import numpy as np

from torq.inference.transformers import (
    DecoderOnlyConfig,
    DynamicDecoderOnlyRunner,
    StaticDecoderOnlyRunner,
)
from torq.models.gemma3._inference import Gemma3Dynamic
from torq.models.smollm2._inference import SmolLM2Dynamic


class _Encoded:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeTokenizer:
    def encode(self, text: str) -> _Encoded:
        if text == "\n":
            return _Encoded([10])
        if text == "\n\n":
            return _Encoded([11])
        return _Encoded([ord(c) % 50 for c in text] or [0])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return ",".join(str(i) for i in ids)

    def token_to_id(self, token: str) -> int | None:
        if token == "<end_of_turn>":
            return 99
        return None


class _FakeRunner:
    def __init__(self, tokens: list[int] | None = None):
        self.model_path = Path("model.onnx")
        self.tokens = tokens or [3]
        self.calls = []

    def infer(self, inputs):
        self.calls.append(inputs)
        token = self.tokens[min(len(self.calls) - 1, len(self.tokens) - 1)]
        logits = np.zeros((1, 1, max(token + 1, 4)), dtype=np.float32)
        logits[0, 0, token] = 1.0
        cache = [
            np.asarray(v).copy()
            for k, v in inputs.items()
            if k.startswith("past_key_values.")
        ]
        return [logits, *cache]


_CONFIG = DecoderOnlyConfig(
    n_layers=2,
    n_kv_heads=3,
    head_dim=5,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)


class _DemoMixin:
    def _init_model_metadata(self):
        self._nl_token_id = 10

    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]:
        return [int(c) for c in input] if input else []

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        return next_token == self._eos_token_id


class _DemoDynamic(_DemoMixin, DynamicDecoderOnlyRunner):
    def __init__(self, model: _FakeRunner, *, include_position_ids: bool):
        DynamicDecoderOnlyRunner.__init__(
            self,
            model,
            _CONFIG,
            max_prompt_tokens=4,
            max_gen_tokens=3,
            tokenizer=_FakeTokenizer(),
            sys_prompt=None,
            include_position_ids=include_position_ids,
        )


class _DemoStatic(_DemoMixin, StaticDecoderOnlyRunner):
    def __init__(
        self,
        model: _FakeRunner,
        *,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
        token_id_lut: np.ndarray | None = None,
    ):
        StaticDecoderOnlyRunner.__init__(
            self,
            model,
            _CONFIG,
            max_prompt_tokens=4,
            max_gen_tokens=7,
            tokenizer=_FakeTokenizer(),
            sys_prompt=None,
            combined_kv_io=combined_kv_io,
            token_embeddings=token_embeddings,
            token_id_lut=token_id_lut,
        )


def test_dynamic_runner_can_omit_or_include_position_ids():
    no_pos_model = _FakeRunner()
    no_pos = _DemoDynamic(no_pos_model, include_position_ids=False)
    no_pos._llm_step(5, 2)

    with_pos_model = _FakeRunner()
    with_pos = _DemoDynamic(with_pos_model, include_position_ids=True)
    with_pos._llm_step(5, 2)

    assert "position_ids" not in no_pos_model.calls[0]
    assert np.array_equal(with_pos_model.calls[0]["position_ids"], np.array([[2]], dtype=np.int64))


def test_decoder_only_prompt_tokens_are_padded_and_truncated():
    runner = _DemoDynamic(_FakeRunner(), include_position_ids=True)

    assert runner._format_input_tokens([1, 2]) == [1, 2, 0, 0]
    assert runner._format_input_tokens([1, 2, 3, 4, 5]) == [1, 2, 3, 4]


def test_static_runner_builds_combined_and_separate_cache_shapes():
    combined = _DemoStatic(_FakeRunner(), combined_kv_io=True)
    separate = _DemoStatic(_FakeRunner(), combined_kv_io=False)

    assert list(combined._kv_cache) == [
        "past_key_values.0.key_value",
        "past_key_values.1.key_value",
    ]
    assert combined._kv_cache["past_key_values.0.key_value"].shape == (1, 6, 7, 5)
    assert len(separate._kv_cache) == 4
    assert separate._kv_cache["past_key_values.0.key"].shape == (1, 3, 7, 5)


def test_static_runner_uses_token_embeddings_and_token_lut():
    model = _FakeRunner(tokens=[1])
    embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)
    runner = _DemoStatic(
        model,
        token_embeddings=embeddings,
        token_id_lut=np.array([8, 42], dtype=np.int64),
    )

    next_token, _ = runner._llm_step(2, 0)

    assert next_token == 42
    assert model.calls[0]["token_embedding"].shape == (1, 1, 3)
    assert np.array_equal(model.calls[0]["token_embedding"][0, 0], embeddings[2])


def test_gemma3_stop_rules_are_preserved():
    runner = Gemma3Dynamic.__new__(Gemma3Dynamic)
    runner._eos_token_id = 2
    runner._end_of_turn_id = 99
    runner._instruct_model = False
    runner._double_nl_token_id = 11
    runner._nl_token_id = 10

    assert runner._stop_decoding(2, [2])
    assert runner._stop_decoding(99, [99])
    assert runner._stop_decoding(11, [1, 2, 11])
    assert runner._stop_decoding(7, [1, 10, 10])
    assert not runner._stop_decoding(7, [1, 2, 7])


def test_smollm2_stop_rules_are_preserved():
    runner = SmolLM2Dynamic.__new__(SmolLM2Dynamic)
    runner._eos_token_id = 2
    runner._instruct_model = False
    runner._nl_token_id = 10

    assert runner._stop_decoding(2, [2])
    assert runner._stop_decoding(7, [1, 10, 10])
    assert not runner._stop_decoding(7, [1, 2, 7])

    runner._instruct_model = True
    assert runner._stop_decoding(7, [7]) is None
