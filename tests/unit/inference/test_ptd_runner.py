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
        self.visibility_calls = []

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
        assert np.asarray(inputs["input_ids"]).shape == (1, self.rows)
        assert np.asarray(inputs["position_ids"]).shape == (1, self.rows)
        self.visibility_calls.append(np.asarray(inputs["tree_vis"]).copy())
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
    def __init__(self, model, eos: int = EOS, **kwargs):
        instruct_model = kwargs.pop("instruct_model", False)
        config = DecoderOnlyConfig(
            n_layers=N_LAYERS, n_kv_heads=N_KV, head_dim=HEAD_DIM,
            bos_token_id=1, eos_token_id=eos, pad_token_id=0, instruct_model=instruct_model,
        )
        StaticPTDRunner.__init__(
            self, model, config, max_prompt_tokens=4, max_gen_tokens=MAX_TOKENS,
            tokenizer=_FakeTokenizer(), sys_prompt="0" if instruct_model else None,
            ptd_rows=ROWS,
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
    runner = _PTDRunner(_FakePTDModel(ROWS))
    runner.run(prompt)
    return runner


def _gen_from_cache(runner) -> list[int]:
    """Generated tokens from the committed cache (row i >= 1 holds gen[i-1])."""
    return [int(runner._kv_cache["past_key_values.0.key_value"][0, 0, i, 0])
            for i in range(1, MAX_TOKENS)
            if runner._kv_cache["past_key_values.0.key_value"][0, 0, i, 0] != 0]


@pytest.mark.parametrize("combined_scratch", [True, False])
def test_ptd_runner_cache_holds_full_greedy_cycle(combined_scratch):
    runner = _PTDRunner(_FakePTDModel(ROWS, combined_scratch=combined_scratch))
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
    assert sum(e["emitted"] for e in log) == MAX_TOKENS - PROMPT_LEN
    assert any(e["L"] + ROWS > MAX_TOKENS for e in log)
    for e in log:
        assert e["L"] + 1 + e["n_candidates"] + e["n_drafts"] <= MAX_TOKENS
    assert log[-1]["L"] + log[-1]["emitted"] == MAX_TOKENS


def test_ptd_runner_row_positions_track_committed_length():
    runner = _run_prompt("0")
    ptd_model = runner._model
    # Each PTD call: position_ids[0] == L (committed length), rows 1.. follow
    # the packed nodes' depths (>= L + 1); invalid pad rows are at L.
    for toks, positions in ptd_model.input_calls[1:]:
        L = int(positions[0])
        assert 1 <= L < MAX_TOKENS
        for p in positions[1:]:
            assert L <= p < MAX_TOKENS, (positions, "row position before the anchor position")


def test_ptd_runner_stop_check_near_cache_capacity():
    """Stop rules still apply when fewer than N real rows fit in the cache."""
    vocab, eos = 16, 10

    def succ(t: int) -> int:
        return t + 1  # 1,2,...,9 -> 10 (EOS); EOS is never an input

    runner = _PTDRunner(_FakePTDModel(ROWS, vocab=vocab, succ=succ),
                        eos=eos)
    runner.run("1")  # prefill: 1 -> 2, then the cycle walks up to EOS=10
    gen = _gen_from_cache(runner)
    # The cache holds every generated token except the final one (it has no
    # KV); the full generation is 2..10 (nine tokens), ending at EOS.
    assert gen == [2, 3, 4, 5, 6, 7, 8, 9], f"committed cache diverged: {gen}"
    assert runner._n_tokens_gen == 9, (
        f"padded region missed the stop check (generated {runner._n_tokens_gen} tokens, "
        f"expected 9 ending at EOS)"
    )
    # The final model call predicted EOS=10 from input 9 (a padded PTD step).
    last_tokens, _ = runner._model.input_calls[-1]
    assert int(last_tokens[0]) == 9
    assert sum(e["emitted"] for e in runner.ptd_round_log) == runner._n_tokens_gen - 1
    assert runner.ptd_round_log[-1]["L"] + ROWS > MAX_TOKENS


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

    ptd = _PTDRunner(CompactModel(ROWS), token_id_lut=lut)
    sequential = SequentialRunner(CompactModel(ROWS), token_id_lut=lut)
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
        _PTDRunner(_FakePTDModel(ROWS), **kwargs)


@pytest.mark.parametrize("prompt_size", [1, ROWS, ROWS + 2, MAX_TOKENS])
def test_ptd_prefill_chunks_commit_only_real_rows_and_run_head_once(prompt_size):
    class Body(_FakePTDModel):
        @property
        def output_names(self):
            return ["last_hidden_states"] + super().output_names[1:]

    class Head:
        def __init__(self):
            self.calls = 0

        def infer(self, inputs):
            self.calls += 1
            assert inputs["last_hidden_states"].shape == (1, ROWS, VOCAB)
            return [inputs["last_hidden_states"]]

    body, head = Body(ROWS), Head()
    runner = _PTDRunner(body, lm_head=head)
    prompt = [i % 3 for i in range(prompt_size)]
    prediction, length = runner._prefill_prompt(prompt)
    assert (prediction, length) == (_succ(prompt[-1]), prompt_size)
    assert len(body.input_calls) == (prompt_size + ROWS - 1) // ROWS
    assert head.calls == 1
    for cache in runner._kv_cache.values():
        np.testing.assert_array_equal(cache[0, 0, :length, 0], prompt)
        np.testing.assert_array_equal(cache[0, N_KV, :length, 0], np.array(prompt) + 100)
        assert not cache[:, :, length:, :].any()
    for i, ((_, positions), vis) in enumerate(zip(body.input_calls, body.visibility_calls)):
        size = min(ROWS, prompt_size - i * ROWS)
        np.testing.assert_array_equal(positions[:size], np.arange(i * ROWS, i * ROWS + size))
        for row in range(size):
            np.testing.assert_array_equal(vis[0, 0, row], np.arange(ROWS - 1) < row)
        assert not vis[0, 0, size:, :].any()
        assert not vis[0, 0, :, max(size - 1, 0):].any()


def test_ptd_warmup_and_repeat_requests_use_only_ptd_body():
    body = _FakePTDModel(ROWS)
    runner = _PTDRunner(body, instruct_model=True)
    assert runner._warmup_len == 1
    expected = runner.run_tokens("1")
    assert runner.run_tokens("1") == expected
    assert all(tokens.shape == (ROWS,) for tokens, _ in body.input_calls)
    assert any(log["L"] == MAX_TOKENS - 1 for log in runner.ptd_round_log)


def test_ptd_single_token_at_last_cache_slot_uses_masked_padding():
    body = _FakePTDModel(ROWS)
    runner = _PTDRunner(body)
    next_token, cache = runner._llm_step(1, MAX_TOKENS - 1)
    assert next_token == 2
    assert not body.visibility_calls[-1].any()
    np.testing.assert_array_equal(body.input_calls[-1][1], MAX_TOKENS - 1)
    assert cache[0][0, 0, -1, 0] == 1
    assert not cache[0][:, :, :-1, :].any()


def test_ptd_prefill_rejects_prompt_larger_than_cache():
    runner = _PTDRunner(_FakePTDModel(ROWS))
    with pytest.raises(ValueError, match="cache capacity"):
        runner._prefill_prompt([1] * (MAX_TOKENS + 1))


def test_standalone_ptd_prefill_and_generation_with_individual_kv_io():
    class SeparateBody(_FakePTDModel):
        @property
        def input_shapes(self):
            shapes = super().input_shapes
            for layer in range(N_LAYERS):
                shape = shapes.pop(f"past_key_values.{layer}.key_value")
                for role in ("key", "value"):
                    shapes[f"past_key_values.{layer}.{role}"] = [1, N_KV, *shape[2:]]
            return shapes

        @property
        def output_names(self):
            return ["logits"] + [
                f"present.{layer}.{role}{suffix}"
                for layer in range(N_LAYERS)
                for suffix in ("", "_scratch")
                for role in ("key", "value")
            ]

        def infer(self, inputs):
            combined = {name: value for name, value in inputs.items() if not name.startswith("past_key_values.")}
            for layer in range(N_LAYERS):
                combined[f"past_key_values.{layer}.key_value"] = np.concatenate(
                    [inputs[f"past_key_values.{layer}.{role}"] for role in ("key", "value")], axis=1,
                )
            raw = super().infer(combined)
            out = [raw[0]]
            for layer in range(N_LAYERS):
                committed, key_scratch, value_scratch = raw[1 + 3 * layer:4 + 3 * layer]
                out.extend([committed[:, :N_KV], committed[:, N_KV:], key_scratch, value_scratch])
            return out

    runner = _PTDRunner(SeparateBody(ROWS, combined_scratch=False), combined_kv_io=False)
    generated = runner.run_tokens("0")
    assert generated == _expected_gen()
    assert any(row["accepted"] > 0 for row in runner.ptd_round_log)
    for layer in range(N_LAYERS):
        for role, offset in (("key", 0), ("value", 100)):
            cache = runner._kv_cache[f"past_key_values.{layer}.{role}"]
            np.testing.assert_array_equal(cache[0, 0, :, 0], np.array([0] + generated[:-1]) + offset)
