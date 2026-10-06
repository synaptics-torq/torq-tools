# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""E-7: smollm2 host inference must honor -s (repo id) and prefer the
config/tokenizer staged next to the export over an HF download."""

import json

import numpy as np
import pytest

from torq.models.smollm2._inference import SmolLM2Static, _repo_id_for_size
from torq.models.smollm2.infer import _find_local_assets

pytestmark = pytest.mark.unit


class _Encoded:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """Stands in for tokenizers.Tokenizer (not installed in the test venv)."""

    def encode(self, text: str) -> _Encoded:
        if text == "\n":
            return _Encoded([3])
        return _Encoded([1])  # any word -> token 1

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return ",".join(str(i) for i in ids)


class _FakeModel:
    def __init__(self, model_path):
        self.model_path = model_path


_CONFIG = {
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}


def _write_assets(directory):
    """Stage config.json + tokenizer.json (a hand-written WordLevel file)."""
    from pathlib import Path

    directory = Path(directory)
    cfg = directory / "config.json"
    cfg.write_text(json.dumps(_CONFIG))
    tok = directory / "tokenizer.json"
    tok.write_text(json.dumps({
        "model": {
            "type": "WordLevel",
            "vocab": {"Hello": 1, "world": 2, "<unk>": 3},
            "unk_token": "<unk>",
        },
        "version": "1.0",
    }))
    return cfg, tok


def test_repo_id_follows_model_size_and_instruct():
    assert _repo_id_for_size("135M", False) == "HuggingFaceTB/SmolLM2-135M"
    assert _repo_id_for_size("135M", True) == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert _repo_id_for_size("360M", True) == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert _repo_id_for_size("1.7B", False) == "HuggingFaceTB/SmolLM2-1.7B"


def test_find_local_assets_walks_up_to_the_variant_dir(tmp_path):
    variant = tmp_path / "export" / "split_lm_head" / "fp32" / "static"
    compiled = variant / "compiled"
    compiled.mkdir(parents=True)
    cfg, tok = _write_assets(variant)
    model = compiled / "model.vmfb"
    model.write_bytes(b"vmfb")

    found_cfg, found_tok = _find_local_assets(model)

    assert found_cfg == cfg
    assert found_tok == tok


def test_find_local_assets_stops_at_the_variant_dir(tmp_path):
    """Assets above the variant dir (a sibling model's tree, an HF snapshot
    root) must not be picked up — the walk is bounded, not unbounded."""
    variant = tmp_path / "export" / "static"
    compiled = variant / "compiled"
    compiled.mkdir(parents=True)
    _write_assets(tmp_path)  # two levels above the variant: out of bounds
    model = compiled / "model.vmfb"
    model.write_bytes(b"vmfb")

    assert _find_local_assets(model) == (None, None)


def test_static_runner_prefers_staged_config_and_tokenizer(tmp_path, monkeypatch):
    """No repo_id, no network: the assets staged next to the export must be
    enough to build the runner (this venv has no huggingface_hub)."""
    import torq.models.smollm2._inference as si

    cfg, tok = _write_assets(tmp_path)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(si, "_load_tokenizer", lambda path: _FakeTokenizer())

    runner = SmolLM2Static(
        _FakeModel(model), 2, 8, instruct_model=False,
        config_path=cfg, tokenizer_path=tok,
    )

    assert runner._n_layers == _CONFIG["num_hidden_layers"]


def test_static_runner_falls_back_to_the_repo_id_download(tmp_path, monkeypatch):
    import torq.models.smollm2._inference as si

    cfg, tok = _write_assets(tmp_path)
    seen = []

    def fake_download(repo_id, filename):
        seen.append((repo_id, filename))
        return str(cfg if filename == "config.json" else tok)

    monkeypatch.setattr(si, "_hf_hub_download", fake_download)
    monkeypatch.setattr(si, "_load_tokenizer", lambda path: _FakeTokenizer())
    SmolLM2Static(
        _FakeModel(tmp_path / "model.onnx"), 2, 8, instruct_model=False,
        repo_id="HuggingFaceTB/SmolLM2-360M",
    )

    assert seen == [
        ("HuggingFaceTB/SmolLM2-360M", "config.json"),
        ("HuggingFaceTB/SmolLM2-360M", "tokenizer.json"),
    ]


def test_infer_smollm2_selects_the_vmfb_loader_for_vmfb_inputs(tmp_path, monkeypatch):
    """A .vmfb path must select from_vmfb (the board runner), as the -m
    metavar always advertised — not the ONNX/ORT runner."""
    import argparse

    import torq.models.smollm2._inference as si
    from torq.models.smollm2.infer import infer_smollm2

    _write_assets(tmp_path)
    model = tmp_path / "model.vmfb"
    model.write_bytes(b"vmfb")

    runners = []

    class _FakeVMFB:
        def __init__(self, model_path, n_threads=None):
            runners.append(str(model_path))
            self.model_path = model_path

        def infer(self, inputs):
            logits = np.zeros((1, 1, 4), dtype=np.float32)
            logits[0, 0, 2] = 1.0  # eos -> the run stops after one step
            cache = [
                np.asarray(v)
                for k, v in inputs.items()
                if k.startswith("past_key_values.")
            ]
            return [logits, *cache]

    def _no_ort(*args, **kwargs):
        raise AssertionError("ORT runner selected for a .vmfb input")

    monkeypatch.setattr(si, "VMFBInferenceRunner", _FakeVMFB)
    monkeypatch.setattr(si, "ORTInferenceRunner", _no_ort)
    monkeypatch.setattr(si, "_load_tokenizer", lambda path: _FakeTokenizer())

    args = argparse.Namespace(
        inputs=["Hello"],
        model=str(model),
        model_size="135M",
        max_inp_len=None,
        threads=None,
        instruct_model=False,
        dynamic_model=False,
        max_gen_tokens=8,
    )
    infer_smollm2(args)

    assert runners == [str(model)]


def test_infer_smollm2_forwards_size_and_local_assets(tmp_path, monkeypatch):
    """The CLI used to drop -s entirely (dead arg): the runner must be built
    against the 360M repo and the staged assets, not the 135M default."""
    import argparse

    import torq.models.smollm2._inference as si
    from torq.models.smollm2.infer import infer_smollm2

    cfg, tok = _write_assets(tmp_path)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")

    class _FakeORT:
        def __init__(self, model_path, n_threads=None):
            self.model_path = model_path

        def infer(self, inputs):
            logits = np.zeros((1, 1, 4), dtype=np.float32)
            logits[0, 0, 2] = 1.0  # eos -> the run stops after one step
            cache = [
                np.asarray(v)
                for k, v in inputs.items()
                if k.startswith("past_key_values.")
            ]
            return [logits, *cache]

    built = {}
    original_init = SmolLM2Static.__init__

    def spy_init(self, model_runner, max_prompt_tokens, max_gen_tokens, **kwargs):
        built.update(kwargs)
        original_init(self, model_runner, max_prompt_tokens, max_gen_tokens, **kwargs)

    monkeypatch.setattr(si, "ORTInferenceRunner", _FakeORT)
    monkeypatch.setattr(si, "_load_tokenizer", lambda path: _FakeTokenizer())
    monkeypatch.setattr(si.SmolLM2Static, "__init__", spy_init)

    args = argparse.Namespace(
        inputs=["Hello"],
        model=str(model),
        model_size="360M",
        max_inp_len=None,
        threads=None,
        instruct_model=False,
        dynamic_model=False,
        max_gen_tokens=8,
    )
    infer_smollm2(args)

    assert built["repo_id"] == "HuggingFaceTB/SmolLM2-360M"
    assert built["config_path"] == str(cfg)
    assert built["tokenizer_path"] == str(tok)
