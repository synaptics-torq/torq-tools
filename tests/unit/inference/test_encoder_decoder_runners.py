# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

from pathlib import Path

import numpy as np

from torq.inference.transformers import (
    DynamicEncoderDecoderRunner,
    EncoderDecoderConfig,
    StaticEncoderDecoderRunner,
)


_CONFIG = EncoderDecoderConfig(
    n_layers=1,
    n_kv_heads=1,
    head_dim=1,
    start_token_id=1,
    end_token_id=2,
    encoder_pad_id=2,
)


def _cache(value: float, shape=(1, 1, 1, 1)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _logits(token: int) -> np.ndarray:
    logits = np.zeros((1, 1, max(token + 1, 4)), dtype=np.float32)
    logits[0, 0, token] = 1.0
    return logits


class _FakeRunner:
    def __init__(self, outputs=None, *, model_path: str = "model.onnx"):
        self.model_path = Path(model_path)
        self.outputs = outputs
        self.calls = []

    def infer(self, inputs):
        self.calls.append(inputs)
        if callable(self.outputs):
            return self.outputs(inputs)
        return self.outputs


def _decoder_outputs(token: int):
    def infer(inputs):
        dec_cache = [
            np.asarray(v).copy()
            for k, v in inputs.items()
            if "decoder" in k and k.startswith("past_key_values.")
        ]
        return [_logits(token), *dec_cache]

    return infer


def test_dynamic_encoder_decoder_cache_update_preserves_cross_cache_after_first_step():
    runner = DynamicEncoderDecoderRunner(
        _FakeRunner([np.zeros((1, 2, 3), dtype=np.float32)]),
        _FakeRunner(_decoder_outputs(3)),
        _CONFIG,
    )

    first_values = [_cache(i) for i, _ in enumerate(runner._all_cache_names)]
    runner._update_cache(first_values, update_all=True)
    second_values = [_cache(10 + i) for i, _ in enumerate(runner._all_cache_names)]
    runner._update_cache(second_values, update_all=False)

    assert np.all(runner._kv_cache["past_key_values.0.decoder.key"] == 10)
    assert np.all(runner._kv_cache["past_key_values.0.decoder.value"] == 11)
    assert np.all(runner._kv_cache["past_key_values.0.encoder.key"] == 2)
    assert np.all(runner._kv_cache["past_key_values.0.encoder.value"] == 3)


def test_static_encoder_decoder_pads_self_attention_cache():
    runner = StaticEncoderDecoderRunner(
        _FakeRunner([_cache(1), _cache(2)]),
        None,
        _FakeRunner(_decoder_outputs(3)),
        _CONFIG,
        max_inp_len=8,
        max_dec_len=4,
        combined_kv_io=False,
    )

    padded = runner._pad_cache_tensor("past_key_values.0.decoder.key", _cache(9))

    assert padded.shape == (1, 1, 4, 1)
    assert np.all(padded[:, :, 0:1, :] == 9)
    assert np.all(padded[:, :, 1:, :] == 0)


def test_static_encoder_decoder_uses_folded_encoder_cache_and_preprocessor():
    preprocessor = _FakeRunner([np.full((1, 8), 5, dtype=np.float32)])
    encoder = _FakeRunner([_cache(7), _cache(8)])
    decoder = _FakeRunner(_decoder_outputs(3))
    runner = StaticEncoderDecoderRunner(
        encoder,
        None,
        decoder,
        _CONFIG,
        max_inp_len=8,
        max_dec_len=4,
        preprocessor=preprocessor,
        combined_kv_io=False,
    )

    tokens = runner.run(np.arange(4, dtype=np.float32), max_tokens=0)

    assert np.array_equal(tokens, np.array([[1, 3]]))
    assert "input_values" in preprocessor.calls[0]
    assert "input_features" in encoder.calls[0]
    assert np.all(decoder.calls[0]["past_key_values.0.encoder.key"] == 7)
    assert np.all(decoder.calls[0]["past_key_values.0.encoder.value"] == 8)


def test_static_encoder_decoder_uses_unfolded_encoder_cache_model():
    encoder_hidden = np.ones((1, 2, 3), dtype=np.float32)
    encoder = _FakeRunner([encoder_hidden])
    gen_encoder_cache = _FakeRunner([_cache(4), _cache(5)])
    decoder = _FakeRunner(_decoder_outputs(3))
    runner = StaticEncoderDecoderRunner(
        encoder,
        gen_encoder_cache,
        decoder,
        _CONFIG,
        max_inp_len=8,
        max_dec_len=4,
        combined_kv_io=False,
    )

    runner.run(np.arange(4, dtype=np.float32), max_tokens=0)

    assert np.array_equal(
        gen_encoder_cache.calls[0]["encoder_hidden_states"],
        encoder_hidden,
    )
    assert np.all(decoder.calls[0]["past_key_values.0.encoder.key"] == 4)
    assert np.all(decoder.calls[0]["past_key_values.0.encoder.value"] == 5)


def test_static_encoder_decoder_uses_token_embedding_input():
    embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
    decoder = _FakeRunner(_decoder_outputs(3))
    runner = StaticEncoderDecoderRunner(
        _FakeRunner([_cache(1), _cache(2)]),
        None,
        decoder,
        _CONFIG,
        max_inp_len=8,
        max_dec_len=4,
        combined_kv_io=False,
        token_embeddings=embeddings,
    )

    runner._run_decoder([2], seq_len=0)

    assert "token_embedding" in decoder.calls[0]
    assert np.array_equal(decoder.calls[0]["token_embedding"][0, 0], embeddings[2])
