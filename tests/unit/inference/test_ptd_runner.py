# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Bookkeeping tests for StaticPTDRunner with a deterministic fake model.

The fake predicts ``succ(t) = (t + 1) % 3`` for every input row t (a fixed
token cycle that never reaches EOS=3), so ordinary greedy and PTD greedy both
produce the same cycle and the candidate pool warms up to real hits. The fake
PTD body mimics the exported contract: per-row logits, the anchor blended into
the present cache at row L, and per-layer K/V scratch outputs carrying the
slot tokens (so cache promotion is checkable).
"""

from pathlib import Path

import numpy as np
import pytest

from torq.inference.transformers import DecoderOnlyConfig, StaticDecoderOnlyRunner, StaticPTDRunner

VOCAB = 4
EOS = 3
ROWS = 8
MAX_TOKENS = 14
N_LAYERS, N_KV, HEAD_DIM = 2, 3, 5
PROMPT_LEN = 1  # the prompt "0"


def _succ(t: int) -> int:
    return (t + 1) % 3


def _expected_gen() -> list[int]:
    gen = [1]
    while len(gen) < MAX_TOKENS:
        gen.append(_succ(gen[-1]))
    return gen


class _FakePTDModel:
    """1-row (decode body) or N-row (PTD body) fake with the static contract."""

    def __init__(self, rows: int, vocab: int = VOCAB, succ=None, combined_scratch=True):
        self.model_path = Path(f"fake_{rows}.onnx")
        self.rows = rows
        self.vocab = vocab
        self.succ = succ or _succ
        self.combined_scratch = combined_scratch
        self.input_calls = []

    @property
    def input_shapes(self):
        shapes = {
            "input_ids": [1, self.rows],
            "position_ids": [1, self.rows],
        }
        if self.rows > 1:
            shapes["tree_vis"] = [1, 1, self.rows, self.rows - 1]
        for i in range(N_LAYERS):
            shapes[f"past_key_values.{i}.key_value"] = [1, 2 * N_KV, MAX_TOKENS, HEAD_DIM]
        return shapes

    @property
    def output_names(self):
        names = ["logits"]
        for i in range(N_LAYERS):
            names.append(f"present.{i}.key_value")
            if self.rows > 1:
                names += ([f"present.{i}.key_value_scratch"] if self.combined_scratch else
                          [f"present.{i}.key_scratch", f"present.{i}.value_scratch"])
        return names

    def infer(self, inputs):
        toks = np.asarray(inputs["input_ids"]).reshape(-1)
        self.input_calls.append((toks.copy(), np.asarray(inputs["position_ids"]).reshape(-1).copy()))
        logits = np.zeros((1, self.rows, self.vocab), dtype=np.float32)
        for i in range(self.rows):
            logits[0, i, self.succ(int(toks[i]))] = 1.0
        out = [logits]
        for l in range(N_LAYERS):
            cache = np.asarray(inputs[f"past_key_values.{l}.key_value"]).copy()
            L = int(np.asarray(inputs["position_ids"]).reshape(-1)[0])
            anchor = int(toks[0])
            cache[0, :N_KV, L, 0] = anchor
            cache[0, N_KV:, L, 0] = anchor + 100
            out.append(cache)
            if self.rows > 1:
                scratch_outputs = []
                for role in ("key", "value"):
                    scratch = np.zeros((1, N_KV, self.rows - 1, HEAD_DIM), dtype=np.float32)
                    scratch[0, :, np.arange(self.rows - 1), 0] = toks[1:][:, None] + (100 if role == "value" else 0)
                    scratch_outputs.append(scratch)
                out.extend([np.concatenate(scratch_outputs, axis=1)] if self.combined_scratch else scratch_outputs)
        return out


class _FakeTokenizer:
    def encode(self, text: str):
        class _E:
            ids = [int(c) for c in text] if text else []
        return _E()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(i) for i in ids)

    def token_to_id(self, token):
        return None


class _PTDRunner(StaticPTDRunner):
    def __init__(self, model, ptd_model, eos: int = EOS, **kwargs):
        config = DecoderOnlyConfig(
            n_layers=N_LAYERS, n_kv_heads=N_KV, head_dim=HEAD_DIM,
            bos_token_id=1, eos_token_id=eos, pad_token_id=0,
        )
        StaticPTDRunner.__init__(
            self, model, config, max_prompt_tokens=4, max_gen_tokens=MAX_TOKENS,
            tokenizer=_FakeTokenizer(), sys_prompt=None,
            ptd_model=ptd_model, ptd_rows=ROWS,
            **kwargs,
        )

    def _init_model_metadata(self):
        self._nl_token_id = 99
        self._double_nl_token_id = 99
        self._end_of_turn_id = None

    def _tokenize_input(self, input, role=None):
        return [int(c) for c in input] if input else []

    def _stop_decoding(self, next_token, gen_tokens):
        return next_token == self._eos_token_id


def _run_prompt(prompt: str) -> _PTDRunner:
    runner = _PTDRunner(_FakePTDModel(1), _FakePTDModel(ROWS))
    runner.run(prompt)
    return runner


def _gen_from_cache(runner) -> list[int]:
    """Generated tokens from the committed cache (row i >= 1 holds gen[i-1])."""
    return [int(runner._kv_cache["past_key_values.0.key_value"][0, 0, i, 0])
            for i in range(1, MAX_TOKENS)
            if runner._kv_cache["past_key_values.0.key_value"][0, 0, i, 0] != 0]


@pytest.mark.parametrize("combined_scratch", [True, False])
def test_ptd_runner_cache_holds_full_greedy_cycle(combined_scratch):
    runner = _PTDRunner(_FakePTDModel(1), _FakePTDModel(ROWS, combined_scratch=combined_scratch))
    runner.run("0")
    gen = _expected_gen()
    assert runner._n_tokens_gen == len(gen)
    # Cache row i (i >= 1) holds the KV of gen[i-1]; the final token has none.
    for l in range(N_LAYERS):
        cache = runner._kv_cache[f"past_key_values.{l}.key_value"]
        assert cache[0, 0, 0, 0] == 0  # prompt token
        for i in range(1, len(gen)):
            assert cache[0, 0, i, 0] == gen[i - 1], f"layer {l} key row {i}"
            assert cache[0, N_KV, i, 0] == gen[i - 1] + 100, f"layer {l} value row {i}"


def test_ptd_runner_hits_and_emission_bookkeeping():
    runner = _run_prompt("0")
    log = runner.ptd_round_log
    assert len(log) >= 5, log
    hits = [e for e in log if e["hit"]]
    assert hits, "expected at least one candidate hit with a warmed pool"
    for e in hits:
        assert e["accepted"] >= 1
        assert e["emitted"] == e["accepted"] + 1  # accepted path + terminal bonus
    # Emitted tokens across rounds + fallback steps fill the cache to M.
    ptd_emitted = sum(e["emitted"] for e in log)
    fallback_steps = (MAX_TOKENS - PROMPT_LEN) - ptd_emitted
    assert fallback_steps > 0, "cache fullness should engage the 1-token fallback"
    # The last PTD round started within the row budget; after it, the
    # remaining steps are 1-token fallbacks up to the full cache.
    last = log[-1]
    assert last["L"] + ROWS <= MAX_TOKENS
    assert last["L"] + last["emitted"] + fallback_steps == MAX_TOKENS


def test_ptd_runner_row_positions_track_committed_length():
    runner = _run_prompt("0")
    ptd_model = runner._ptd_model
    # Each PTD call: position_ids[0] == L (committed length), rows 1.. follow
    # the packed nodes' depths (>= L + 1); invalid pad rows are at L.
    for toks, positions in ptd_model.input_calls:
        L = int(positions[0])
        assert 1 <= L <= MAX_TOKENS - ROWS, (L, "PTD round started without row budget")
        for p in positions[1:]:
            assert p >= L, (positions, "row position before the anchor position")


def test_ptd_runner_stop_check_in_fallback_region():
    """A stop token (EOS) generated by the 1-token fallback steps must stop
    generation, exactly like the base runner — the fallback covers the last
    ROWS-1 cache rows, so without the check PTD would emit past the EOS."""
    vocab, eos = 16, 10

    def succ(t: int) -> int:
        return t + 1  # 1,2,...,9 -> 10 (EOS); EOS is never an input

    runner = _PTDRunner(_FakePTDModel(1, vocab=vocab, succ=succ),
                        _FakePTDModel(ROWS, vocab=vocab, succ=succ),
                        eos=eos)
    runner.run("1")  # prefill: 1 -> 2, then the cycle walks up to EOS=10
    gen = _gen_from_cache(runner)
    # The cache holds every generated token except the final one (it has no
    # KV); the full generation is 2..10 (nine tokens), ending at EOS.
    assert gen == [2, 3, 4, 5, 6, 7, 8, 9], f"committed cache diverged: {gen}"
    assert runner._n_tokens_gen == 9, (
        f"fallback region missed the stop check (generated {runner._n_tokens_gen} tokens, "
        f"expected 9 ending at EOS)"
    )
    # The final model call predicted EOS=10 from input 9 (a fallback step).
    last_tokens, _ = runner._model.input_calls[-1]
    assert int(last_tokens[0]) == 9
    # PTD rounds ran before the fallback, and the final token (10) came from a
    # fallback step: the PTD-emitted count is short of the full generation.
    ptd_emitted = sum(e["emitted"] for e in runner.ptd_round_log)
    assert 1 <= ptd_emitted < runner._n_tokens_gen - 1


def test_ptd_trimmed_vocabulary_matches_sequential_token_ids():
    lut = np.array([0, 5, 10], dtype=np.int32)

    class CompactModel(_FakePTDModel):
        def __init__(self, rows):
            super().__init__(rows, vocab=11, succ=lambda t: int(lut[(list(lut).index(t) + 1) % 3]))

        def infer(self, inputs):
            out = super().infer(inputs)
            out[0] = out[0][:, :, lut]
            return out

    class SequentialRunner(_PTDRunner):
        _run = StaticDecoderOnlyRunner._run

    ptd = _PTDRunner(CompactModel(1), CompactModel(ROWS), token_id_lut=lut)
    sequential = SequentialRunner(CompactModel(1), CompactModel(ROWS), token_id_lut=lut)
    expected = sequential.run_tokens("0")
    actual = ptd.run_tokens("0")
    assert actual == expected
    assert set(actual) == set(lut)
    assert any(e["accepted"] > 0 for e in ptd.ptd_round_log)
    for i, token in enumerate(actual[:-1], start=1):
        assert ptd._kv_cache["past_key_values.0.key_value"][0, 0, i, 0] == token


def test_first_prediction_eos_resets_generation_count():
    runner = _run_prompt("0")
    assert runner._n_tokens_gen == MAX_TOKENS
    runner._eos_token_id = 1
    assert runner.run_tokens("0") == [1]
    assert runner._n_tokens_gen == 1
    assert runner.ptd_round_log == []


@pytest.mark.parametrize("kwargs", [{"temperature": 0.5}, {"top_p": 0.9}])
def test_ptd_rejects_sampling_settings(kwargs):
    with pytest.raises(ValueError, match="greedy"):
        _PTDRunner(_FakePTDModel(1), _FakePTDModel(ROWS), **kwargs)
