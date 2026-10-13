# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from types import SimpleNamespace

from torq.tools import benchmark_ptd


def test_benchmark_compares_ids_even_when_decoded_text_matches(monkeypatch):
    static = SimpleNamespace(run_tokens=lambda *args, **kwargs: [1, 2],
                             run=lambda *args, **kwargs: "same text",
                             _tokenizer=SimpleNamespace(decode=lambda ids: "same text"))
    ptd = SimpleNamespace(ptd_round_log=[], _tokenizer=static._tokenizer)
    monkeypatch.setattr(benchmark_ptd.Gemma3Static, "from_onnx", lambda *args, **kwargs: static)
    monkeypatch.setattr(benchmark_ptd.Gemma3PTD, "from_onnx", lambda *args, **kwargs: ptd)
    monkeypatch.setattr(benchmark_ptd, "_decode_step_time", lambda *args: (0.0, 0.1, 2))
    monkeypatch.setattr(benchmark_ptd, "_ptd_timings", lambda *args: (0.0, 0.1, 2, [3, 4]))
    args = SimpleNamespace(max_gen_tokens=8, model_max_tokens=16, prompts=["prompt"], model="model",
                           ptd_model="ptd", max_inp_len=None, n_threads=1, instruct_model=False,
                           repo_id=None, lm_head=None, ptd_lm_head=None, ptd_rows=4, ptd_width=4, ptd_depth=6)
    report = benchmark_ptd.run_benchmark(args)
    assert report["prompts"]["prompt"]["static"]["output"] == "same text"
    assert report["prompts"]["prompt"]["ptd"]["output"] == "same text"
    assert not report["totals"]["outputs_token_identical"]


def test_benchmark_prints_no_rounds_without_formatting_error(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_ptd, "run_benchmark", lambda args: {
        "totals": {"outputs_token_identical": True, "mean_step_s": 0.1,
                   "speedup_est": None, "peak_rss_mb": 1.0},
    })
    monkeypatch.setattr("sys.argv", ["benchmark_ptd", "--model", "model", "--ptd-model", "ptd"])
    benchmark_ptd.main()
    assert "speedup est n/a" in capsys.readouterr().out


def test_benchmark_counts_padded_rows_and_measures_whole_round():
    runner = SimpleNamespace(ptd_round_log=[{
        "round": 1, "rows": 8, "n_candidates": 0, "n_drafts": 1,
        "emitted": 1, "accepted": 0, "hit": False, "body_ms": 2.0, "round_ms": 3.0,
    }])
    summary = benchmark_ptd._ptd_summary(runner)
    assert summary["rows_processed"] == 8
    assert summary["active_rows"] == 2
    assert summary["mean_round_ms"] == 3.0
