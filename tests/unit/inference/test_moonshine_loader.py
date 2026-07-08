# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import pytest

from torq.models.moonshine import _inference as moonshine


def _touch(path):
    path.write_bytes(b"")
    return path


def test_find_models_rejects_missing_and_mixed_formats(tmp_path):
    with pytest.raises(FileNotFoundError):
        moonshine.find_models(tmp_path)

    _touch(tmp_path / "encoder.onnx")
    _touch(tmp_path / "decoder.vmfb")

    with pytest.raises(ValueError, match="multiple formats"):
        moonshine.find_models(tmp_path)


def test_load_moonshine_dispatches_dynamic_onnx(tmp_path, monkeypatch):
    encoder = _touch(tmp_path / "encoder.onnx")
    decoder = _touch(tmp_path / "decoder_merged.onnx")
    called = {}

    def fake_from_onnx(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return "dynamic"

    monkeypatch.setattr(moonshine.MoonshineDynamic, "from_onnx", staticmethod(fake_from_onnx))

    result = moonshine.load_moonshine(tmp_path, "tiny", None, None, n_threads=2)

    assert result == "dynamic"
    assert called["args"] == (encoder, decoder, "tiny", None)
    assert called["kwargs"] == {"n_threads": 2}


def test_load_moonshine_dispatches_static_onnx(tmp_path, monkeypatch):
    encoder = _touch(tmp_path / "encoder.onnx")
    gen_encoder_cache = _touch(tmp_path / "gen_encoder_cache.onnx")
    decoder = _touch(tmp_path / "decoder.onnx")
    called = {}

    def fake_from_onnx(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return "static"

    monkeypatch.setattr(moonshine.MoonshineStatic, "from_onnx", staticmethod(fake_from_onnx))

    result = moonshine.load_moonshine(tmp_path, "base", None, None, n_threads=3)

    assert result == "static"
    assert called["args"] == (encoder, gen_encoder_cache, decoder, "base")
    assert called["kwargs"] == {"n_threads": 3}


def test_load_moonshine_requires_static_vmfb_lengths(tmp_path):
    _touch(tmp_path / "encoder.vmfb")
    _touch(tmp_path / "decoder.vmfb")

    with pytest.raises(ValueError, match="maximum input length"):
        moonshine.load_moonshine(tmp_path, "tiny", None, 12)
