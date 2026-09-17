# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from torq.models.liquid._inference import LiquidBase, LiquidDynamic, LiquidStatic, ModelConfig


class _Encoded:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeTokenizer:
    def encode(self, text: str) -> _Encoded:
        if text == "\n":
            return _Encoded([10])
        return _Encoded([ord(c) % 50 for c in text] or [0])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return ",".join(str(i) for i in ids)


class _FakeSession:
    def __init__(self, names: list[str]):
        self._names = names

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._names]


class _FakeRunner:
    """Stands in for an `InferenceRunner`.

    `input_names=None` mimics a runner that cannot report its signature
    (e.g. VMFB), which exposes no `_sess`.
    """

    def __init__(self, input_names: list[str] | None = None):
        self.model_path = Path("model.onnx")
        self.calls = []
        if input_names is not None:
            self._sess = _FakeSession(input_names)

    def infer(self, inputs):
        self.calls.append(inputs)
        logits = np.zeros((1, 1, 4), dtype=np.float32)
        logits[0, 0, 3] = 1.0
        return [logits, *(np.asarray(v) for k, v in inputs.items() if k.startswith("past_"))]


_CONFIG = ModelConfig(
    n_layers=2,
    n_kv_heads=3,
    head_dim=5,
    conv_dim=6,
    conv_L_cache=3,
    layer_types=("conv", "full_attention"),
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

# Graph input order of the two published source exports, minus the caches.
UPSTREAM_INPUTS = ["input_ids", "attention_mask", "num_logits_to_keep"]
MIRROR_INPUTS = ["input_ids", "attention_mask", "position_ids"]


class _DemoDynamic(LiquidDynamic):
    def __init__(self, model: _FakeRunner):
        LiquidBase.__init__(
            self,
            model,
            _CONFIG,
            max_prompt_tokens=4,
            max_gen_tokens=3,
            tokenizer=_FakeTokenizer(),
            sys_prompt=None,
        )


class _DemoStatic(LiquidStatic):
    def __init__(
        self,
        model: _FakeRunner,
        prefill_model: _FakeRunner | None = None,
        prefill_size: int | None = None,
    ):
        self._combined_kv_io = True
        self._token_embeddings = None
        self._kv_cache_len = 7
        self._prefill_model = prefill_model
        self._prefill_size = prefill_size
        LiquidBase.__init__(
            self,
            model,
            _CONFIG,
            max_prompt_tokens=4,
            max_gen_tokens=7,
            tokenizer=_FakeTokenizer(),
            sys_prompt=None,
        )


def test_dynamic_feeds_position_ids_for_mirror_source():
    model = _FakeRunner(MIRROR_INPUTS)

    _DemoDynamic(model)._llm_step(5, 2)

    fed = model.calls[0]
    assert "num_logits_to_keep" not in fed
    assert np.array_equal(fed["position_ids"], np.array([[2]], dtype=np.int64))


def test_dynamic_feeds_num_logits_to_keep_for_upstream_source():
    model = _FakeRunner(UPSTREAM_INPUTS)

    _DemoDynamic(model)._llm_step(5, 2)

    fed = model.calls[0]
    assert "position_ids" not in fed
    assert np.array_equal(fed["num_logits_to_keep"], np.array(1, dtype=np.int64))


def test_dynamic_feed_order_matches_declared_graph_inputs():
    """VMFB runners feed positionally, so insertion order has to line up."""
    for names in (UPSTREAM_INPUTS, MIRROR_INPUTS):
        model = _FakeRunner(names)

        _DemoDynamic(model)._llm_step(5, 2)

        assert [k for k in model.calls[0] if not k.startswith("past_")] == names


def test_dynamic_keeps_num_logits_to_keep_when_signature_unknown():
    model = _FakeRunner(None)

    runner = _DemoDynamic(model)
    runner._llm_step(5, 2)

    assert runner._input_names is None
    assert "num_logits_to_keep" in model.calls[0]
    assert "position_ids" not in model.calls[0]


def test_static_feeds_attention_mask_only_when_declared():
    with_mask = _FakeRunner(["input_ids", "position_ids", "attention_mask"])
    without_mask = _FakeRunner(["input_ids", "position_ids"])
    unknown = _FakeRunner(None)

    _DemoStatic(with_mask)._llm_step(5, 2)
    _DemoStatic(without_mask)._llm_step(5, 2)
    _DemoStatic(unknown)._llm_step(5, 2)

    assert with_mask.calls[0]["attention_mask"].shape == (1, 7)
    assert "attention_mask" not in without_mask.calls[0]
    assert "attention_mask" not in unknown.calls[0]


def test_static_uses_prefill_model_for_full_chunks_then_decode_for_remainder():
    decode_model = _FakeRunner(input_names=["input_ids", "position_ids"])
    prefill_model = _FakeRunner(input_names=["input_ids", "position_ids"])
    runner = _DemoStatic(decode_model, prefill_model=prefill_model, prefill_size=2)

    next_token, curr_seq_len = runner._prefill_prompt([3, 4, 5], start_seq_len=1)

    assert next_token == 3
    assert curr_seq_len == 4
    # Full 2-token chunk goes to the prefill model at the chunk start position.
    assert np.array_equal(prefill_model.calls[0]["input_ids"], np.array([[3, 4]]))
    assert np.array_equal(prefill_model.calls[0]["position_ids"], np.array([[1]]))
    # The single remainder token falls back to the decode model.
    assert np.array_equal(decode_model.calls[0]["input_ids"], np.array([[5]]))
    assert np.array_equal(decode_model.calls[0]["position_ids"], np.array([[3]]))


def test_static_prefill_chunks_repeatedly_advance_cache_and_position():
    decode_model = _FakeRunner(input_names=["input_ids", "position_ids"])
    prefill_model = _FakeRunner(input_names=["input_ids", "position_ids"])
    runner = _DemoStatic(decode_model, prefill_model=prefill_model, prefill_size=2)

    runner._prefill_prompt([1, 2, 3, 4, 5, 6], start_seq_len=0)

    assert len(prefill_model.calls) == 3
    assert decode_model.calls == []
    assert [c["position_ids"].item() for c in prefill_model.calls] == [0, 2, 4]
    assert np.array_equal(prefill_model.calls[-1]["input_ids"], np.array([[5, 6]]))


def test_static_prefill_feeds_token_embeddings_for_chunks():
    decode_model = _FakeRunner(input_names=["token_embedding", "position_ids"])
    prefill_model = _FakeRunner(input_names=["token_embedding", "position_ids"])
    runner = _DemoStatic(decode_model, prefill_model=prefill_model, prefill_size=2)
    runner._token_embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)

    runner._prefill_prompt([3, 4], start_seq_len=0)

    fed = prefill_model.calls[0]["token_embedding"]
    assert fed.shape == (1, 2, 3)
    assert np.array_equal(fed[0, 0], runner._token_embeddings[3])
    assert np.array_equal(fed[0, 1], runner._token_embeddings[4])


def test_static_requires_prefill_model_and_size_together(tmp_path):

    def _init(**overrides):
        kwargs = dict(
            model=_FakeRunner(input_names=["input_ids", "position_ids"]),
            max_prompt_tokens=4,
            max_gen_tokens=7,
            config_path=tmp_path / "config.json",
            tokenizer_path=tmp_path / "tokenizer.json",
        )
        kwargs.update(overrides)
        LiquidStatic.__init__(LiquidStatic.__new__(LiquidStatic), **kwargs)

    with pytest.raises(ValueError, match="must be provided together"):
        _init(prefill_model=_FakeRunner(input_names=["input_ids", "position_ids"]))
    with pytest.raises(ValueError, match="must be positive"):
        _init(prefill_model=_FakeRunner(input_names=["input_ids", "position_ids"]),
              prefill_size=0)


def test_static_infer_prefill_size_from_input_shapes():
    class _ShapedRunner:
        model_path = Path("model_prefill.onnx")
        input_shapes = {"input_ids": [1, 8]}

    assert LiquidStatic._infer_prefill_size(_ShapedRunner()) == 8

    class _EmbShapedRunner:
        model_path = Path("model_prefill.onnx")
        input_shapes = {"token_embedding": [1, 16, 4]}

    assert LiquidStatic._infer_prefill_size(_EmbShapedRunner()) == 16

    class _OpaqueRunner:
        model_path = Path("model_prefill.onnx")
        input_shapes = {}

    with pytest.raises(ValueError, match="fixed prefill size"):
        LiquidStatic._infer_prefill_size(_OpaqueRunner())
