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

        # Detect simplified static frontend (StaticStreamingFrontendWrapper):
        # no sample_buffer / sample_len — takes (audio_chunk, conv1_buffer, conv2_buffer).
        fe_inputs = self._frontend._sess.get_inputs()
        fe_input_names = {inp.name for inp in fe_inputs}
        self._is_static_frontend = "sample_buffer" not in fe_input_names
        # Read the fixed chunk_len from the static frontend's audio_chunk input shape.
        # Falls back to None for the dynamic frontend (chunk_len passed by the caller).
        if self._is_static_frontend:
            audio_inp = next(inp for inp in fe_inputs if inp.name == "audio_chunk")
            self._fixed_chunk_len: int | None = int(audio_inp.shape[1])
            self._logger.info("Static frontend detected: fixed chunk_len=%d", self._fixed_chunk_len)
        else:
            self._fixed_chunk_len = None

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

        # Static streaming: static frontend + streaming encoder → encoder expects exactly
        # F stable frames per call.  Read F from the encoder's stable_features input shape.
        self._is_static_streaming = self._is_static_frontend and self._is_streaming_encoder
        if self._is_static_streaming:
            stable_inp = next(inp for inp in enc_inputs if inp.name == "stable_features")
            self._feature_stride: int = int(stable_inp.shape[1])
            self._logger.info("Static streaming detected: F=%d", self._feature_stride)
        else:
            self._feature_stride = None

        # Detect static decoder: presence of 'current_len' input means pre-allocated KV buffers.
        dec_inputs = self._decoder._sess.get_inputs()
        dec_input_names = {inp.name for inp in dec_inputs}
        self._is_static_decoder = "current_len" in dec_input_names
        # Detect embedding extraction: decoder takes float inputs_embeds instead of int token.
        self._extract_embeddings = "inputs_embeds" in dec_input_names
        if self._is_static_decoder:
            # Derive max_tokens and max_memory_len from the pre-allocated buffer shapes.
            # k_self shape: [n_layers, 1, n_kv_heads, max_tokens, head_dim]
            k_self_inp = next(inp for inp in dec_inputs if inp.name == "k_self_0")
            self._max_tokens: int = k_self_inp.shape[2]
            # k_cross (out_k_cross) shape: [n_layers, 1, n_kv_heads, max_memory_len, head_dim]
            k_cross_inp = next(inp for inp in dec_inputs if inp.name == "k_cross_0")
            self._max_memory_len: int = k_cross_inp.shape[2]
            self._logger.info(
                "Static decoder detected: max_tokens=%d, max_memory_len=%d",
                self._max_tokens, self._max_memory_len,
            )
        else:
            self._max_tokens = None
            self._max_memory_len = None

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
        # Static frontend has a fixed chunk_len baked in; override the caller's default.
        if self._fixed_chunk_len is not None:
            chunk_len = self._fixed_chunk_len

        # Initialize frontend state
        c1 = self._hidden_size * 2 if self._model_size == "tiny" else 1536
        conv1_buffer = np.zeros((1, self._hidden_size, 4), dtype=np.float32)
        conv2_buffer = np.zeros((1, c1, 4), dtype=np.float32)
        if not self._is_static_frontend:
            sample_buffer = np.zeros((1, 79), dtype=np.float32)
            sample_len = np.zeros(1, dtype=np.int64)
            frame_count = np.zeros(1, dtype=np.int64)

        audio_len = input_audio.shape[-1]

        if self._is_streaming_encoder:
            total_la = self._total_lookahead
            n_enc_layers = len(self._enc_left_ctx)
            enc_bufs = {
                f"buf_{i}": np.zeros((1, lc, self._hidden_size), dtype=np.float32)
                for i, lc in enumerate(self._enc_left_ctx)
            }
            pending = np.zeros((1, 0, self._hidden_size), dtype=np.float32)
            pos_offset = np.zeros(1, dtype=np.int64)

            if self._is_static_decoder:
                # Pre-allocate fixed-size cross-KV buffers for incremental accumulation.
                k_cross_buf = np.zeros(
                    (self._n_layers, 1, self._n_kv_heads, self._max_memory_len, self._head_dim),
                    dtype=np.float32,
                )
                v_cross_buf = np.zeros(
                    (self._n_layers, 1, self._n_kv_heads, self._max_memory_len, self._head_dim),
                    dtype=np.float32,
                )
                cross_kv_fill = 0  # number of memory slots written so far
            else:
                memory_chunks: list[np.ndarray] = []
                kv_k_chunks: list[np.ndarray] = []
                kv_v_chunks: list[np.ndarray] = []

            for offset in range(0, audio_len, chunk_len):
                audio_chunk = input_audio[:, offset:offset + chunk_len]
                if audio_chunk.shape[-1] < chunk_len:
                    audio_chunk = np.pad(audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1])))

                if self._is_static_frontend:
                    res = self._frontend.infer({
                        "audio_chunk": audio_chunk,
                        "conv1_buffer": conv1_buffer,
                        "conv2_buffer": conv2_buffer,
                    })
                    features, conv1_buffer, conv2_buffer = res
                else:
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

                if self._is_static_streaming:
                    # Static path: encoder accepts exactly F frames per call.
                    # Drain pending in F-sized batches whenever we have enough right context.
                    F = self._feature_stride
                    while pending.shape[1] >= F + total_la:
                        stable_feats = pending[:, :F, :]
                        right_ctx    = pending[:, F:F + total_la, :]
                        enc_res = self._encoder.infer({"stable_features": stable_feats, "right_ctx": right_ctx, **enc_bufs})
                        encoded_stable = enc_res[0]
                        enc_bufs = {f"buf_{i}": enc_res[i + 1] for i in range(n_enc_layers)}
                        mem = self._adapter.infer({"encoded": encoded_stable, "pos_offset": pos_offset})[0]
                        pos_offset = np.array([int(pos_offset[0]) + F], dtype=np.int64)
                        kv_outs = self._cross_kv.infer({"memory": mem})
                        new_k, new_v = kv_outs[0], kv_outs[1]
                        if self._is_static_decoder:
                            n_new = new_k.shape[3]
                            end = min(cross_kv_fill + n_new, self._max_memory_len)
                            actual = end - cross_kv_fill
                            k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
                            v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
                            cross_kv_fill = end
                        else:
                            kv_k_chunks.append(new_k)
                            kv_v_chunks.append(new_v)
                        pending = pending[:, F:, :]
                else:
                    # Dynamic path: pass all stable frames at once.
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
                        pos_offset = np.array([int(pos_offset[0]) + encoded_stable.shape[1]], dtype=np.int64)

                        kv_outs = self._cross_kv.infer({"memory": mem})
                        new_k, new_v = kv_outs[0], kv_outs[1]

                        if self._is_static_decoder:
                            n_new = new_k.shape[3]
                            end = min(cross_kv_fill + n_new, self._max_memory_len)
                            actual = end - cross_kv_fill
                            k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
                            v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
                            cross_kv_fill = end
                        else:
                            kv_k_chunks.append(new_k)
                            kv_v_chunks.append(new_v)

                        pending = pending[:, stable_count:, :]

            # Flush remaining pending frames with zero right context
            if pending.shape[1] > 0:
                zero_right = np.zeros((1, total_la, self._hidden_size), dtype=np.float32)
                if self._is_static_streaming:
                    # Flush in F-sized batches, padding the last partial batch with zeros.
                    F = self._feature_stride
                    while pending.shape[1] > 0:
                        batch = pending[:, :F, :]
                        if batch.shape[1] < F:
                            pad = np.zeros((1, F - batch.shape[1], self._hidden_size), dtype=np.float32)
                            batch = np.concatenate([batch, pad], axis=1)
                        enc_res = self._encoder.infer({"stable_features": batch, "right_ctx": zero_right, **enc_bufs})
                        encoded_stable = enc_res[0]
                        enc_bufs = {f"buf_{i}": enc_res[i + 1] for i in range(n_enc_layers)}
                        mem = self._adapter.infer({"encoded": encoded_stable, "pos_offset": pos_offset})[0]
                        pos_offset = np.array([int(pos_offset[0]) + F], dtype=np.int64)
                        kv_outs = self._cross_kv.infer({"memory": mem})
                        new_k, new_v = kv_outs[0], kv_outs[1]
                        if self._is_static_decoder:
                            n_new = new_k.shape[3]
                            end = min(cross_kv_fill + n_new, self._max_memory_len)
                            actual = end - cross_kv_fill
                            k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
                            v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
                            cross_kv_fill = end
                        else:
                            kv_k_chunks.append(new_k)
                            kv_v_chunks.append(new_v)
                        pending = pending[:, F:, :]
                else:
                    enc_res = self._encoder.infer({
                        "stable_features": pending,
                        "right_ctx": zero_right,
                        **enc_bufs,
                    })
                    encoded_stable = enc_res[0]
                    mem = self._adapter.infer({"encoded": encoded_stable, "pos_offset": pos_offset})[0]
                    kv_outs = self._cross_kv.infer({"memory": mem})
                    new_k, new_v = kv_outs[0], kv_outs[1]

                if not self._is_static_streaming:
                    if self._is_static_decoder:
                        n_new = new_k.shape[3]
                        end = min(cross_kv_fill + n_new, self._max_memory_len)
                        actual = end - cross_kv_fill
                        k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
                        v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
                        cross_kv_fill = end
                    else:
                        kv_k_chunks.append(new_k)
                        kv_v_chunks.append(new_v)

            if self._is_static_decoder:
                out_k_cross = k_cross_buf
                out_v_cross = v_cross_buf
            else:
                if not kv_k_chunks:
                    raise ValueError("No features were processed from the audio input.")
                out_k_cross = np.concatenate(kv_k_chunks, axis=3)
                out_v_cross = np.concatenate(kv_v_chunks, axis=3)

        else:
            # Batch path: accumulate all features, encode once, then adapt once.
            accum_features: list[np.ndarray] = []
            for offset in range(0, audio_len, chunk_len):
                audio_chunk = input_audio[:, offset:offset + chunk_len]
                if audio_chunk.shape[-1] < chunk_len:
                    audio_chunk = np.pad(audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1])))

                if self._is_static_frontend:
                    res = self._frontend.infer({
                        "audio_chunk": audio_chunk,
                        "conv1_buffer": conv1_buffer,
                        "conv2_buffer": conv2_buffer,
                    })
                    features, conv1_buffer, conv2_buffer = res
                else:
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
            out_k_cross, out_v_cross = self._cross_kv.infer({"memory": memory})
            cross_kv_fill = None  # not used in batch path

        if max_tokens is None:
            max_tokens = int((audio_len / 16000) * 6)

        tokens = [self._start_token_id]

        if self._is_static_decoder:
            # Pre-allocated self-KV buffers: [n_layers, 1, n_kv_heads, max_tokens, head_dim]
            k_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )
            v_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )
            valid_len = cross_kv_fill if self._is_streaming_encoder else out_k_cross.shape[3]

            for step in range(max_tokens):
                current_len = np.array([[step]], dtype=np.int64)  # [1, 1]
                cross_kv_valid = np.array([valid_len], dtype=np.int64)  # [1]

                if self._extract_embeddings:
                    dec_feed = {"inputs_embeds": self._token_embeddings[tokens[-1]].reshape(1, 1, -1)}
                else:
                    dec_feed = {"token": np.array([[tokens[-1]]], dtype=np.int64)}

                dec_feed_kv = {
                    **dec_feed,
                    "current_len": current_len,
                    "cross_kv_valid_len": cross_kv_valid,
                    "position_ids": current_len,
                }
                for i in range(self._n_layers):
                    dec_feed_kv[f"k_self_{i}"] = k_self[i]
                    dec_feed_kv[f"v_self_{i}"] = v_self[i]
                    dec_feed_kv[f"k_cross_{i}"] = out_k_cross[i]
                    dec_feed_kv[f"v_cross_{i}"] = out_v_cross[i]
                res = self._decoder.infer(dec_feed_kv)
                logits = res[0]
                for i in range(self._n_layers):
                    k_self[i] = res[1 + i * 2]
                    v_self[i] = res[2 + i * 2]
                next_token = int(logits[0, -1, :].argmax())
                tokens.append(next_token)
                self._n_tokens_gen += 1
                if next_token == self._end_token_id:
                    break
        else:
            # Per-layer lists: seq dim grows each step, can't use a fixed stacked array.
            k_self_layers = [
                np.zeros((1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)
                for _ in range(self._n_layers)
            ]
            v_self_layers = [
                np.zeros((1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)
                for _ in range(self._n_layers)
            ]

            for _ in range(max_tokens):
                if self._extract_embeddings:
                    dec_feed = {"inputs_embeds": self._token_embeddings[tokens[-1]].reshape(1, 1, -1)}
                else:
                    dec_feed = {"token": np.array([[tokens[-1]]], dtype=np.int64)}

                dec_feed_kv = {**dec_feed}
                for i in range(self._n_layers):
                    dec_feed_kv[f"k_self_{i}"] = k_self_layers[i]
                    dec_feed_kv[f"v_self_{i}"] = v_self_layers[i]
                    dec_feed_kv[f"k_cross_{i}"] = out_k_cross[i]
                    dec_feed_kv[f"v_cross_{i}"] = out_v_cross[i]
                res = self._decoder.infer(dec_feed_kv)
                logits = res[0]
                for i in range(self._n_layers):
                    k_self_layers[i] = res[1 + i * 2]
                    v_self_layers[i] = res[2 + i * 2]
                next_token = int(logits[0, -1, :].argmax())
                tokens.append(next_token)
                self._n_tokens_gen += 1
                if next_token == self._end_token_id:
                    break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])
