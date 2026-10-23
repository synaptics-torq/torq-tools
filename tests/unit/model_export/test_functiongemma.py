# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torq.models.gemma3 import FUNCTIONGEMMA_HF_REPO, add_gemma3_export_args
from torq.models.gemma3 import export as gemma3_export


def _args(*argv):
    parser = argparse.ArgumentParser()
    add_gemma3_export_args(parser)
    return parser.parse_args(list(argv))


@pytest.mark.parametrize("source", [None, "org/my-finetune"])
def test_functiongemma_source_and_gptq(tmp_path, monkeypatch, source):
    monkeypatch.setattr(gemma3_export.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace(
        hidden_size=8, vocab_size=16, num_attention_heads=2,
    )))
    exporter_cls = gemma3_export.Gemma3ModelExporter
    captured = []

    def make_exporter(*args, **kwargs):
        exporter = exporter_cls(*args, **kwargs)
        captured.append(exporter)
        return exporter

    monkeypatch.setattr(gemma3_export, "Gemma3ModelExporter", make_exporter)
    monkeypatch.setattr(exporter_cls, "export_onnx", Mock())
    monkeypatch.setattr(exporter_cls, "weight_quantize_models", Mock())
    argv = ["--functiongemma", "--skip-torq", "--models-dir", str(tmp_path)]
    if source:
        argv.extend(["--hf-repo", source])
    gemma3_export.export_gemma3_from_args(_args(*argv))
    assert captured[0]._hf_repo == (source or FUNCTIONGEMMA_HF_REPO)
    assert captured[0]._instruct_model is True
    assert captured[0]._read_only_kv_cache is True


def test_functiongemma_does_not_use_hosted_gemma_gptq(monkeypatch):
    import huggingface_hub

    download = Mock(side_effect=AssertionError("Wrong checkpoint"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = _args("--functiongemma", "--instruct-model")
    assert gemma3_export._resolve_gptq_checkpoint(args, "int4") is None
    download.assert_not_called()
    args.gptq_checkpoint = "/my/functiongemma-gptq"
    assert str(gemma3_export._resolve_gptq_checkpoint(args, "int4")) == args.gptq_checkpoint


def test_functiongemma_rejects_1b_before_loading_config(monkeypatch):
    load = Mock(side_effect=AssertionError("Must validate before loading config"))
    monkeypatch.setattr(gemma3_export.AutoConfig, "from_pretrained", load)
    with pytest.raises(ValueError, match="270m"):
        gemma3_export.export_gemma3_from_args(_args("--functiongemma", "--model-size", "1b"))
    with pytest.raises(ValueError, match="270m"):
        gemma3_export.Gemma3ModelExporter(functiongemma=True, model_size="1b")
    load.assert_not_called()


def test_functiongemma_local_source_stays_offline(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    load = Mock(return_value=SimpleNamespace(hidden_size=8, vocab_size=16, num_attention_heads=2))
    monkeypatch.setattr(gemma3_export.AutoConfig, "from_pretrained", load)
    exporter = gemma3_export.Gemma3ModelExporter(functiongemma=True, onnx_source_dir=tmp_path)
    load.assert_called_once_with(tmp_path, local_files_only=True)
    assert exporter._source_asset_dirs == [tmp_path]


@pytest.mark.parametrize(("size", "instruct", "repo"), [
    ("270m", False, "google/gemma-3-270m"),
    ("270m", True, "google/gemma-3-270m-it"),
    ("1b", False, "google/gemma-3-1b-pt"),
    ("1b", True, "google/gemma-3-1b-it"),
])
def test_existing_gemma_sources(size, instruct, repo):
    assert gemma3_export._resolve_hf_repo(size, instruct, None, False) == repo
