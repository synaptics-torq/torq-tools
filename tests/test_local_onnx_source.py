# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from pathlib import Path
from types import SimpleNamespace

import pytest

from torq.model_export.onnx import OnnxModelExporterBase
from torq.models.gemma3 import export as gemma3_export
from torq.models.moonshine import export as moonshine_export
from torq.models.smollm2 import export as smollm2_export


def _gemma3_config():
    return SimpleNamespace(hidden_size=8, vocab_size=16, num_attention_heads=2)


def _moonshine_config():
    return SimpleNamespace(
        hidden_size=8,
        vocab_size=16,
        encoder_num_attention_heads=2,
        decoder_num_attention_heads=2,
    )


@pytest.mark.parametrize(
    ("module", "exporter_cls"),
    [
        (gemma3_export, gemma3_export.Gemma3ModelExporter),
        (moonshine_export, moonshine_export.MoonshineModelExporter),
        (smollm2_export, smollm2_export.SmolLM2ModelExporter),
    ],
)
@pytest.mark.parametrize(
    ("create_source_dir", "error"),
    [
        (False, "ONNX source directory does not exist"),
        (True, "Expected config.json in ONNX source directory"),
    ],
)
def test_invalid_local_onnx_source_never_loads_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    exporter_cls,
    create_source_dir: bool,
    error: str,
):
    def fail_config_load(*args, **kwargs):
        pytest.fail("local source validation must happen before loading model configuration")

    monkeypatch.setattr(module.AutoConfig, "from_pretrained", fail_config_load)
    source_dir = tmp_path / "source"
    if create_source_dir:
        source_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=error):
        exporter_cls(onnx_source_dir=source_dir)


@pytest.mark.parametrize(
    ("module", "exporter_cls"),
    [
        (gemma3_export, gemma3_export.Gemma3ModelExporter),
        (smollm2_export, smollm2_export.SmolLM2ModelExporter),
    ],
)
def test_decoder_local_onnx_source_requires_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    exporter_cls,
):
    def fail_config_load(*args, **kwargs):
        pytest.fail("local source validation must happen before loading model configuration")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}")
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", fail_config_load)

    with pytest.raises(FileNotFoundError, match="Expected tokenizer.json in ONNX source directory"):
        exporter_cls(onnx_source_dir=source_dir)


@pytest.mark.parametrize(
    ("module", "exporter_cls", "config"),
    [
        (gemma3_export, gemma3_export.Gemma3ModelExporter, _gemma3_config),
        (moonshine_export, moonshine_export.MoonshineModelExporter, _moonshine_config),
        (smollm2_export, smollm2_export.SmolLM2ModelExporter, _gemma3_config),
    ],
)
def test_local_onnx_source_loads_config_without_hub_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    exporter_cls,
    config,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}")
    (source_dir / "tokenizer.json").write_text("{}")
    calls = []

    def load_config(path, *, local_files_only=False):
        calls.append((path, local_files_only))
        return config()

    monkeypatch.setattr(module.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(OnnxModelExporterBase, "__init__", lambda *args, **kwargs: None)

    exporter = exporter_cls(onnx_source_dir=source_dir)

    assert exporter._onnx_source_dir == source_dir
    assert calls == [(source_dir, True)]
