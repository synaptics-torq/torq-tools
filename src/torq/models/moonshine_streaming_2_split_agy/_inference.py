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

class MoonshineStreaming2Split:

    def __init__(
        self,
        fused_encoder: InferenceRunner,
        decoder: InferenceRunner,
        model_size: Literal["tiny", "small"],
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_size = model_size

        self._fused_encoder = fused_encoder
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

        # Inspect fused_encoder inputs to derive left-contexts, total_lookahead, F, and chunk_len.
        fe_inputs = self._fused_encoder._sess.get_inputs()
        buf_inputs = sorted(
            [inp for inp in fe_inputs if inp.name.startswith("buf_")],
            key=lambda inp: int(inp.name.split("_")[1])
        )
        self._enc_left_ctx = [inp.shape[1] for inp in buf_inputs]

        feats_buf_inp = next(inp for inp in fe_inputs if inp.name == "features_buffer")
        self._total_lookahead = feats_buf_inp.shape[1]

        pos_emb_inp = next(inp for inp in fe_inputs if inp.name == "position_embeddings")
        self._feature_stride = pos_emb_inp.shape[1]

        audio_chunk_inp = next(inp for inp in fe_inputs if inp.name == "audio_chunk")
        self._fixed_chunk_len = audio_chunk_inp.shape[1]

        self._logger.info(
            "Fused encoder inputs derived: chunk_len=%d, F=%d, total_lookahead=%d, left_contexts=%s",
            self._fixed_chunk_len, self._feature_stride, self._total_lookahead, self._enc_left_ctx
        )

        # Inspect decoder inputs to see if static KV caches are used.
        dec_inputs = self._decoder._sess.get_inputs()
        dec_input_names = {inp.name for inp in dec_inputs}
        self._is_static_decoder = "current_len" in dec_input_names
        self._extract_embeddings = "inputs_embeds" in dec_input_names
        if self._is_static_decoder:
            k_self_inp = next(inp for inp in dec_inputs if inp.name == "k_self_0")
            self._max_tokens: int = k_self_inp.shape[2]
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

        # Load token embeddings and position embedding table
        self._token_embeddings = self._find_token_embeddings()
        self._pos_emb = self._find_pos_emb()

    @classmethod
    def from_onnx(
        cls,
        fused_encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
    ) -> "MoonshineStreaming2Split":
        return cls(
            ORTInferenceRunner(fused_encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
        )

    def _find_token_embeddings(self) -> np.ndarray:
        parent = self._decoder.model_path.parent
        paths = list(parent.glob("*token_embeddings.npy"))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")
        return np.load(paths[0])

    def _find_pos_emb(self) -> np.ndarray:
        parent = self._fused_encoder.model_path.parent
        paths = list(parent.glob("adapter_pos_emb.npy"))
        if not paths:
            parent = self._decoder.model_path.parent
            paths = list(parent.glob("adapter_pos_emb.npy"))
        if not paths:
            raise FileNotFoundError("Missing position embedding file 'adapter_pos_emb.npy'")
        return np.load(paths[0])

    def run(
        self,
        input_audio: np.ndarray,
        max_tokens: int | None = None,
        chunk_len: int = 1280,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        st = time.time()

        chunk_len = self._fixed_chunk_len
        audio_len = input_audio.shape[-1]
        F = self._feature_stride
        warmup_chunks = (self._total_lookahead + F - 1) // F

        # Initialize fused encoder states
        c1 = self._hidden_size * 2 if self._model_size == "tiny" else 1536
        conv1_buffer = np.zeros((1, self._hidden_size, 4), dtype=np.float32)
        conv2_buffer = np.zeros((1, c1, 4), dtype=np.float32)
        features_buffer = np.zeros((1, self._total_lookahead, self._hidden_size), dtype=np.float32)
        enc_bufs = {
            f"buf_{i}": np.zeros((1, lc, self._hidden_size), dtype=np.float32)
            for i, lc in enumerate(self._enc_left_ctx)
        }

        # Initialize cross-KV accumulation buffer
        if self._is_static_decoder:
            k_cross_buf = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_memory_len, self._head_dim),
                dtype=np.float32,
            )
            v_cross_buf = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_memory_len, self._head_dim),
                dtype=np.float32,
            )
            cross_kv_fill = 0
        else:
            kv_k_chunks: list[np.ndarray] = []
            kv_v_chunks: list[np.ndarray] = []

        num_audio_chunks = (audio_len + chunk_len - 1) // chunk_len
        total_chunks = num_audio_chunks + warmup_chunks
        pos_offset = 0

        for chunk_idx in range(total_chunks):
            offset = chunk_idx * chunk_len
            if offset < audio_len:
                audio_chunk = input_audio[:, offset:offset + chunk_len]
                if audio_chunk.shape[-1] < chunk_len:
                    audio_chunk = np.pad(audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1])))
            else:
                audio_chunk = np.zeros((1, chunk_len), dtype=np.float32)

            # Look up position embeddings from host-side table
            pos_emb = self._pos_emb[pos_offset : pos_offset + F].reshape(1, F, -1)

            # Build inputs dictionary
            feed = {
                "audio_chunk": audio_chunk,
                "conv1_buffer": conv1_buffer,
                "conv2_buffer": conv2_buffer,
                "features_buffer": features_buffer,
                "position_embeddings": pos_emb,
                **enc_bufs,
            }

            # Run inference
            res = self._fused_encoder.infer(feed)
            
            # Unpack results:
            # Outputs: k_cross, v_cross, conv1_buffer_out, conv2_buffer_out, features_buffer_out, *buf_out
            new_k, new_v = res[0], res[1]
            conv1_buffer = res[2]
            conv2_buffer = res[3]
            features_buffer = res[4]

            # Warmup logic:
            if chunk_idx < warmup_chunks:
                # Discard cross-KV outputs and encoder buffer updates.
                pass
            else:
                # Active step:
                # Update encoder buffers
                for i in range(len(self._enc_left_ctx)):
                    enc_bufs[f"buf_{i}"] = res[5 + i]

                # Update position offset
                pos_offset += F

                # Save cross-KV
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

        if max_tokens is None:
            max_tokens = int((audio_len / 16000) * 6)

        tokens = [self._start_token_id]

        if self._is_static_decoder:
            # Pre-allocated self-KV buffers
            k_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )
            v_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )
            valid_len = cross_kv_fill

            # Precompute cross-attention bias
            cross_attn_bias = np.zeros(
                (1, self._n_kv_heads, 1, self._max_memory_len), dtype=np.float32
            )
            cross_attn_bias[:, :, :, valid_len:] = -1e9

            for step in range(max_tokens):
                current_len = np.array([[step]], dtype=np.int64)

                if self._extract_embeddings:
                    dec_feed = {"inputs_embeds": self._token_embeddings[tokens[-1]].reshape(1, 1, -1)}
                else:
                    dec_feed = {"token": np.array([[tokens[-1]]], dtype=np.int64)}

                dec_feed_kv = {
                    **dec_feed,
                    "current_len": current_len,
                    "cross_attn_bias": cross_attn_bias,
                    "position_ids": current_len,
                }
                for i in range(self._n_layers):
                    dec_feed_kv[f"k_self_{i}"] = k_self[i]
                    dec_feed_kv[f"v_self_{i}"] = v_self[i]
                    dec_feed_kv[f"k_cross_{i}"] = k_cross_buf[i]
                    dec_feed_kv[f"v_cross_{i}"] = v_cross_buf[i]
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
            k_self_layers = [
                np.zeros((1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)
                for _ in range(self._n_layers)
            ]
            v_self_layers = [
                np.zeros((1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32)
                for _ in range(self._n_layers)
            ]

            out_k_cross = np.concatenate(kv_k_chunks, axis=3)
            out_v_cross = np.concatenate(kv_v_chunks, axis=3)

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
