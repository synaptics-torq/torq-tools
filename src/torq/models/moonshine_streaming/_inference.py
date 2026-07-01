# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import ml_dtypes

from ...inference.runners import InferenceRunner, ORTInferenceRunner


_ORT_TYPE_TO_NP: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(bfloat16)": np.dtype(ml_dtypes.bfloat16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
}


def _np_dtype(ort_type: str) -> np.dtype:
    return _ORT_TYPE_TO_NP.get(ort_type, np.dtype(np.float32))


class MoonshineStreaming:
    """Static two-model streaming inference: a fused encoder + a KV-cache decoder.

    The fused encoder runs frontend / encoder / adapter / cross-KV generation in one
    dispatch per audio chunk; the decoder is the pre-allocated static-KV-cache graph
    (``current_len`` input). Warmup (discarding outputs while the right-context
    ``features_buffer`` fills) and KV accumulation are managed here on the host.

    Model dimensions and I/O dtypes are derived from the ONNX graphs, so the same
    runner drives both the float and the dtype-converted (bf16) exports — the feed
    buffers and the host-side LUTs simply track the model's I/O dtype.
    """

    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        model_size: Literal["tiny", "small", "medium"] | None = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_size = model_size
        self._encoder = encoder
        self._decoder = decoder

        # Fused-encoder configuration (probed from the ONNX inputs)
        enc_inputs = {inp.name: inp for inp in self._encoder._sess.get_inputs()}
        self._chunk_len: int = int(enc_inputs["audio_chunk"].shape[1])
        pos_emb_inp = enc_inputs["position_embeddings"]
        self._feature_stride: int = int(pos_emb_inp.shape[1])   # F: output frames / chunk
        self._hidden_size: int = int(pos_emb_inp.shape[2])
        self._total_lookahead: int = int(enc_inputs["features_buffer"].shape[1])
        self._c1_channels: int = int(enc_inputs["conv2_buffer"].shape[1])
        self._enc_left_ctx: list[int] = [
            int(enc_inputs[f"buf_{i}"].shape[1])
            for i in range(len(enc_inputs))
            if f"buf_{i}" in enc_inputs
        ]
        self._n_enc_layers: int = len(self._enc_left_ctx)
        self._enc_dtype: np.dtype = _np_dtype(pos_emb_inp.type)
        # ceil(total_la / F) warmup chunks before the encoder buffers are meaningful
        self._warmup_chunks: int = math.ceil(self._total_lookahead / self._feature_stride)

        # Static-decoder configuration (probed from the ONNX inputs)
        dec_inputs = {inp.name: inp for inp in self._decoder._sess.get_inputs()}
        if "current_len" not in dec_inputs:
            raise ValueError(
                "Decoder is not a static KV-cache model (missing 'current_len' input)"
            )
        self._extract_embeddings: bool = "inputs_embeds" in dec_inputs
        self._n_layers: int = sum(1 for n in dec_inputs if n.startswith("k_self_"))
        k_self_0 = dec_inputs["k_self_0"]
        self._n_kv_heads: int = int(k_self_0.shape[1])
        self._max_tokens: int = int(k_self_0.shape[2])
        self._head_dim: int = int(k_self_0.shape[3])
        self._max_memory_len: int = int(dec_inputs["k_cross_0"].shape[2])
        self._dec_dtype: np.dtype = _np_dtype(k_self_0.type)

        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._infer_times: deque[float] = deque(maxlen=100)

        self._token_embeddings = self._find_token_embeddings()
        self._pos_emb = self._find_pos_emb()

        self._logger.info(
            "Encoder: chunk_len=%d, F=%d, hidden=%d, total_la=%d, warmup=%d, enc_layers=%d, dtype=%s",
            self._chunk_len, self._feature_stride, self._hidden_size, self._total_lookahead,
            self._warmup_chunks, self._n_enc_layers, self._enc_dtype,
        )
        self._logger.info(
            "Decoder: layers=%d, kv_heads=%d, head_dim=%d, max_tokens=%d, max_memory_len=%d, "
            "extract_embeddings=%s, dtype=%s",
            self._n_layers, self._n_kv_heads, self._head_dim, self._max_tokens,
            self._max_memory_len, self._extract_embeddings, self._dec_dtype,
        )

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small", "medium"] | None = None,
        n_threads: int | None = None,
    ) -> "MoonshineStreaming":
        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
        )

    def _find_token_embeddings(self) -> np.ndarray:
        paths = list(self._decoder.model_path.parent.glob("decoder*token_embeddings.npy"))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")
        return np.load(paths[0])

    def _find_pos_emb(self) -> np.ndarray:
        paths = list(self._encoder.model_path.parent.glob("adapter_pos_emb.npy"))
        if not paths:
            raise FileNotFoundError("Missing position embedding file 'adapter_pos_emb.npy'")
        return np.load(paths[0])

    def _encoder_state(self) -> dict[str, np.ndarray]:
        dt = self._enc_dtype
        return {
            "conv1_buffer": np.zeros((1, self._hidden_size, 4), dtype=dt),
            "conv2_buffer": np.zeros((1, self._c1_channels, 4), dtype=dt),
            "features_buffer": np.zeros((1, self._total_lookahead, self._hidden_size), dtype=dt),
            **{f"buf_{i}": np.zeros((1, lc, self._hidden_size), dtype=dt)
               for i, lc in enumerate(self._enc_left_ctx)},
        }

    def run(
        self,
        input_audio: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        st = time.time()

        audio_len = input_audio.shape[-1]
        F = self._feature_stride

        # Fused-encoder state (encoder I/O dtype)
        state = self._encoder_state()
        conv1_buf = state["conv1_buffer"]
        conv2_buf = state["conv2_buffer"]
        feat_buf = state["features_buffer"]
        enc_bufs = {f"buf_{i}": state[f"buf_{i}"] for i in range(self._n_enc_layers)}

        k_cross_buf = np.zeros(
            (self._n_layers, 1, self._n_kv_heads, self._max_memory_len, self._head_dim),
            dtype=self._dec_dtype,
        )
        v_cross_buf = np.zeros_like(k_cross_buf)
        cross_kv_fill = 0

        pos_offset = 0
        # Chunk loop
        for chunk_idx, offset in enumerate(range(0, audio_len, self._chunk_len)):
            audio_chunk = input_audio[:, offset:offset + self._chunk_len]
            if audio_chunk.shape[-1] < self._chunk_len:
                audio_chunk = np.pad(
                    audio_chunk, ((0, 0), (0, self._chunk_len - audio_chunk.shape[-1]))
                )
            in_warmup = chunk_idx < self._warmup_chunks

            # Position embeddings are looked up on the host (avoids a large Gather in ONNX).
            pos_emb = self._pos_emb[pos_offset:pos_offset + F].reshape(1, F, -1)

            fe_feed = {
                "audio_chunk": audio_chunk.astype(self._enc_dtype),
                "conv1_buffer": conv1_buf,
                "conv2_buffer": conv2_buf,
                "features_buffer": feat_buf,
                "position_embeddings": pos_emb.astype(self._enc_dtype),
                # During warmup the encoder buffers are zeroed and outputs discarded.
                **(enc_bufs if not in_warmup else
                   {f"buf_{i}": np.zeros((1, lc, self._hidden_size), dtype=self._enc_dtype)
                    for i, lc in enumerate(self._enc_left_ctx)}),
            }

            new_k, new_v, conv1_buf, conv2_buf, feat_buf, *buf_outs = \
                self._encoder.infer(fe_feed)

            if in_warmup:
                continue  # keep conv/feature state only; discard encoder buf + cross-KV

            enc_bufs = {f"buf_{i}": buf_outs[i] for i in range(self._n_enc_layers)}
            pos_offset += F

            n_new = new_k.shape[3]
            end = min(cross_kv_fill + n_new, self._max_memory_len)
            actual = end - cross_kv_fill
            k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
            v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
            cross_kv_fill = end

        valid_len = cross_kv_fill
        if max_tokens is None:
            max_tokens = int((audio_len / 16000) * 6)
        max_tokens = min(max_tokens, self._max_tokens)

        # Decode (static pre-allocated self-KV cache)
        k_self = np.zeros(
            (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
            dtype=self._dec_dtype,
        )
        v_self = np.zeros_like(k_self)
        cross_attn_bias = np.zeros((1, self._n_kv_heads, 1, self._max_memory_len), dtype=self._dec_dtype)
        cross_attn_bias[:, :, :, valid_len:] = -1e9

        tokens = [self._start_token_id]
        for step in range(max_tokens):
            current_len = np.array([[step]], dtype=np.int64)
            if self._extract_embeddings:
                dec_feed = {"inputs_embeds": self._token_embeddings[tokens[-1]].reshape(1, 1, -1)}
            else:
                dec_feed = {"token": np.array([[tokens[-1]]], dtype=np.int64)}
            dec_feed.update({
                "current_len": current_len,
                "cross_attn_bias": cross_attn_bias,
                "position_ids": current_len,
            })
            for i in range(self._n_layers):
                dec_feed[f"k_self_{i}"] = k_self[i]
                dec_feed[f"v_self_{i}"] = v_self[i]
                dec_feed[f"k_cross_{i}"] = k_cross_buf[i]
                dec_feed[f"v_cross_{i}"] = v_cross_buf[i]

            logits, *cache = self._decoder.infer(dec_feed)
            for i in range(self._n_layers):
                k_self[i] = cache[i * 2]
                v_self[i] = cache[i * 2 + 1]

            next_token = int(np.asarray(logits, dtype=np.float32)[0, -1, :].argmax())
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


def load_moonshine_streaming(
    model_dir: str | os.PathLike,
    model_size: Literal["tiny", "small", "medium"] | None = None,
    n_threads: int | None = None,
) -> MoonshineStreaming:
    """Load a streaming runner from a directory containing ``encoder`` + ``decoder`` models."""
    d = Path(model_dir)
    encoder_onnx, decoder_onnx = d / "encoder.onnx", d / "decoder.onnx"
    if encoder_onnx.exists() and decoder_onnx.exists():
        return MoonshineStreaming.from_onnx(encoder_onnx, decoder_onnx, model_size, n_threads=n_threads)
    if (d / "encoder.vmfb").exists() and (d / "decoder.vmfb").exists():
        raise NotImplementedError(
            "VMFB streaming inference is not yet wired up; point at the ONNX export directory."
        )
    raise FileNotFoundError(f"Expected 'encoder.onnx' and 'decoder.onnx' in '{d}'")
