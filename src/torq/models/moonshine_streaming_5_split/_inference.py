# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from ...inference.runners import (
    InferenceRunner,
    ORTInferenceRunner,
)

class MoonshineStreaming5Split:

    def __init__(
        self,
        frontend: InferenceRunner,
        encoder: InferenceRunner,
        adapter: InferenceRunner,
        cross_kv: InferenceRunner,
        decoder: InferenceRunner,
        model_size: Literal["tiny", "small"],
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_size = model_size

        self._frontend = frontend
        self._encoder = encoder
        self._adapter = adapter
        self._cross_kv = cross_kv
        self._decoder = decoder

        if self._model_size == "tiny":
            self._n_layers: int = 6
            self._n_kv_heads: int = 8
            self._head_dim: int = 40
            self._hidden_size: int = 320
        else: # small
            self._n_layers: int = 14
            self._n_kv_heads: int = 10
            self._head_dim: int = 64
            self._hidden_size: int = 640

        # Detect streaming encoder by presence of 'right_ctx' input in encoder.onnx
        enc_inputs = self._encoder._sess.get_inputs()
        enc_input_names = {inp.name for inp in enc_inputs}
        if "right_ctx" in enc_input_names:
            self._enc_left_ctx: list[int] = [
                inp.shape[1] for inp in enc_inputs if inp.name.startswith("buf_")
            ]
            self._total_lookahead: int = next(
                inp.shape[1] for inp in enc_inputs if inp.name == "right_ctx"
            )
            self._is_streaming_encoder = True
            self._logger.info(
                "Streaming encoder detected: %d layers, left_ctx=%s, total_lookahead=%d",
                len(self._enc_left_ctx), self._enc_left_ctx, self._total_lookahead,
            )
        else:
            self._enc_left_ctx = []
            self._total_lookahead = 0
            self._is_streaming_encoder = False

        self._start_token_id: int = 1
        self._end_token_id: int = 2

        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)

        # Load token embeddings
        self._token_embeddings = self._find_token_embeddings()

    @classmethod
    def from_onnx(
        cls,
        frontend_model: str | os.PathLike,
        encoder_model: str | os.PathLike,
        adapter_model: str | os.PathLike,
        cross_kv_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
    ) -> "MoonshineStreaming5Split":
        return cls(
            ORTInferenceRunner(frontend_model, n_threads=n_threads),
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(adapter_model, n_threads=n_threads),
            ORTInferenceRunner(cross_kv_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
        )

    def _find_token_embeddings(self) -> np.ndarray:
        paths = list(self._decoder.model_path.parent.glob("decoder*token_embeddings.npy"))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")
        return np.load(paths[0])

    def run(
        self,
        input_audio: np.ndarray,
        max_tokens: int | None = None,
        chunk_len: int = 640,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        st = time.time()

        # Initialize frontend state
        sample_buffer = np.zeros((1, 79), dtype=np.float32)
        sample_len = np.zeros(1, dtype=np.int64)
        c1 = self._hidden_size * 2 if self._model_size == "tiny" else 1536
        conv1_buffer = np.zeros((1, self._hidden_size, 4), dtype=np.float32)
        conv2_buffer = np.zeros((1, c1, 4), dtype=np.float32)
        frame_count = np.zeros(1, dtype=np.int64)

        audio_len = input_audio.shape[-1]

        if self._is_streaming_encoder:
            # True streaming: frontend → encoder (with buffer state) → adapter, per audio chunk.
            # Cross KV and decoder run once after all memory chunks are accumulated.
            total_la = self._total_lookahead
            n_enc_layers = len(self._enc_left_ctx)
            enc_bufs = {
                f"buf_{i}": np.zeros((1, lc, self._hidden_size), dtype=np.float32)
                for i, lc in enumerate(self._enc_left_ctx)
            }
            pending = np.zeros((1, 0, self._hidden_size), dtype=np.float32)
            memory_chunks: list[np.ndarray] = []
            pos_offset = np.zeros(1, dtype=np.int64)

            for offset in range(0, audio_len, chunk_len):
                audio_chunk = input_audio[:, offset:offset + chunk_len]
                if audio_chunk.shape[-1] < chunk_len:
                    audio_chunk = np.pad(audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1])))

                res = self._frontend.infer({
                    "audio_chunk": audio_chunk,
                    "sample_buffer": sample_buffer,
                    "sample_len": sample_len,
                    "conv1_buffer": conv1_buffer,
                    "conv2_buffer": conv2_buffer,
                    "frame_count": frame_count,
                })
                features, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count = res

                pending = np.concatenate([pending, features], axis=1)
                stable_count = max(0, pending.shape[1] - total_la)

                if stable_count > 0:
                    stable_feats = pending[:, :stable_count, :]
                    right_ctx = pending[:, stable_count:stable_count + total_la, :]
                    if right_ctx.shape[1] < total_la:
                        pad = np.zeros((1, total_la - right_ctx.shape[1], self._hidden_size), dtype=np.float32)
                        right_ctx = np.concatenate([right_ctx, pad], axis=1)

                    enc_res = self._encoder.infer({
                        "stable_features": stable_feats,
                        "right_ctx": right_ctx,
                        **enc_bufs,
                    })
                    encoded_stable = enc_res[0]
                    enc_bufs = {f"buf_{i}": enc_res[i + 1] for i in range(n_enc_layers)}

                    mem = self._adapter.infer({"encoded": encoded_stable, "pos_offset": pos_offset})[0]
                    memory_chunks.append(mem)
                    pos_offset = np.array([int(pos_offset[0]) + encoded_stable.shape[1]], dtype=np.int64)
                    pending = pending[:, stable_count:, :]

            # Flush remaining pending frames with zero right context
            if pending.shape[1] > 0:
                right_ctx = np.zeros((1, total_la, self._hidden_size), dtype=np.float32)
                enc_res = self._encoder.infer({
                    "stable_features": pending,
                    "right_ctx": right_ctx,
                    **enc_bufs,
                })
                encoded_stable = enc_res[0]
                mem = self._adapter.infer({"encoded": encoded_stable, "pos_offset": pos_offset})[0]
                memory_chunks.append(mem)

            if not memory_chunks:
                raise ValueError("No features were processed from the audio input.")
            memory = np.concatenate(memory_chunks, axis=1)

        else:
            # Batch path: accumulate all features, encode once, then adapt once.
            accum_features: list[np.ndarray] = []
            for offset in range(0, audio_len, chunk_len):
                audio_chunk = input_audio[:, offset:offset + chunk_len]
                if audio_chunk.shape[-1] < chunk_len:
                    audio_chunk = np.pad(audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1])))

                res = self._frontend.infer({
                    "audio_chunk": audio_chunk,
                    "sample_buffer": sample_buffer,
                    "sample_len": sample_len,
                    "conv1_buffer": conv1_buffer,
                    "conv2_buffer": conv2_buffer,
                    "frame_count": frame_count,
                })
                features, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count = res
                accum_features.append(features)

            if not accum_features:
                raise ValueError("No features were processed from the audio input.")
            all_features = np.concatenate(accum_features, axis=1)
            encoded = self._encoder.infer({"features": all_features})[0]
            memory = self._adapter.infer({"encoded": encoded, "pos_offset": np.zeros(1, dtype=np.int64)})[0]

        # Cross KV once, then autoregressive decode
        out_k_cross, out_v_cross = self._cross_kv.infer({"memory": memory})

        tokens = [self._start_token_id]
        k_self = np.zeros((self._n_layers, 1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)
        v_self = np.zeros((self._n_layers, 1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)

        if max_tokens is None:
            max_tokens = int((audio_len / 16000) * 6)

        for _ in range(max_tokens):
            token_input = np.array([[tokens[-1]]], dtype=np.int64)
            res = self._decoder.infer({
                "token": token_input,
                "k_self": k_self,
                "v_self": v_self,
                "out_k_cross": out_k_cross,
                "out_v_cross": out_v_cross,
            })
            logits, k_self, v_self, _, _ = res
            next_token = int(logits[0, -1, :].argmax())
            tokens.append(next_token)
            self._n_tokens_gen += 1
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])
