# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
    def __init__(self, model: _FakeRunner):
        self._combined_kv_io = True
        self._token_embeddings = None
        self._kv_cache_len = 7
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
