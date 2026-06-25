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

from ...inference.runners import InferenceRunner, ORTInferenceRunner


class MoonshineStreaming:
    """Two-model streaming inference: fused_encoder + decoder_kv.

    The fused encoder combines frontend / encoder / adapter / cross-KV generation
    into one ONNX dispatch per chunk.  Warmup logic (discarding outputs while the
    right-context features_buffer fills up) is managed entirely on the host.

    Compared to the 5-split, this reduces per-chunk encoder-side dispatches from 4
    down to 1, cutting host-device launch overhead significantly.
    """

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
        else:  # small
            self._n_layers: int = 14
            self._n_kv_heads: int = 10
            self._head_dim: int = 64
            self._hidden_size: int = 640

        # ── Detect fused-encoder configuration from ONNX inputs ─────────────
        fe_inputs = {inp.name: inp for inp in self._fused_encoder._sess.get_inputs()}

        audio_inp = fe_inputs["audio_chunk"]
        self._chunk_len: int = int(audio_inp.shape[1])

        pos_emb_inp = fe_inputs["position_embeddings"]
        self._feature_stride: int = int(pos_emb_inp.shape[1])  # F (actual output frames/chunk)

        feat_buf_inp = fe_inputs["features_buffer"]
        self._total_lookahead: int = int(feat_buf_inp.shape[1])  # sum of per-layer right-ctx

        # conv2_buffer shape is [1, c1_channels, 4] — read directly to avoid hardcoding
        self._c1_channels: int = int(fe_inputs["conv2_buffer"].shape[1])

        # Per-layer left-context sizes from buf_* inputs
        self._enc_left_ctx: list[int] = [
            int(fe_inputs[f"buf_{i}"].shape[1])
            for i in range(self._n_layers)
            if f"buf_{i}" in fe_inputs
        ]
        self._n_enc_layers: int = len(self._enc_left_ctx)

        # ceil(total_la / F) warmup chunks before encoder buffers become meaningful
        self._warmup_chunks: int = math.ceil(self._total_lookahead / self._feature_stride)

        self._logger.info(
            "Fused encoder: chunk_len=%d, F=%d, total_la=%d, warmup=%d, enc_layers=%d",
            self._chunk_len, self._feature_stride, self._total_lookahead,
            self._warmup_chunks, self._n_enc_layers,
        )

        # ── Detect static decoder configuration ──────────────────────────────
        dec_inputs = {inp.name: inp for inp in self._decoder._sess.get_inputs()}
        self._is_static_decoder: bool = "current_len" in dec_inputs
        self._extract_embeddings: bool = "inputs_embeds" in dec_inputs

        if self._is_static_decoder:
            k_self_0 = dec_inputs["k_self_0"]
            self._max_tokens: int = int(k_self_0.shape[2])
            k_cross_0 = dec_inputs["k_cross_0"]
            self._max_memory_len: int = int(k_cross_0.shape[2])
            self._logger.info(
                "Static decoder: max_tokens=%d, max_memory_len=%d",
                self._max_tokens, self._max_memory_len,
            )
        else:
            self._max_tokens = None
            self._max_memory_len = None

        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)

        self._token_embeddings = self._find_token_embeddings()
        self._pos_emb = self._find_pos_emb()

    @classmethod
    def from_onnx(
        cls,
        fused_encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
    ) -> "MoonshineStreaming":
        return cls(
            ORTInferenceRunner(fused_encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
        )

    def _find_token_embeddings(self) -> np.ndarray:
        paths = list(self._decoder.model_path.parent.glob("decoder*token_embeddings.npy"))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")
        return np.load(paths[0])

    def _find_pos_emb(self) -> np.ndarray:
        paths = list(self._fused_encoder.model_path.parent.glob("adapter_pos_emb.npy"))
        if not paths:
            raise FileNotFoundError("Missing position embedding file 'adapter_pos_emb.npy'")
        return np.load(paths[0])

    def run(
        self,
        input_audio: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        st = time.time()

        audio_len = input_audio.shape[-1]
        chunk_len = self._chunk_len
        F = self._feature_stride
        total_la = self._total_lookahead
        enc_hidden = self._hidden_size
        c1 = self._c1_channels  # conv1 output channels, read from ONNX input shape

        # ── Initialise fused-encoder state ───────────────────────────────────
        conv1_buf = np.zeros((1, enc_hidden, 4), dtype=np.float32)
        conv2_buf = np.zeros((1, c1, 4), dtype=np.float32)
        feat_buf  = np.zeros((1, total_la, enc_hidden), dtype=np.float32)
        enc_bufs  = {
            f"buf_{i}": np.zeros((1, lc, enc_hidden), dtype=np.float32)
            for i, lc in enumerate(self._enc_left_ctx)
        }

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

        pos_offset = 0
        chunk_idx = 0

        # ── Chunk loop ───────────────────────────────────────────────────────
        for offset in range(0, audio_len, chunk_len):
            audio_chunk = input_audio[:, offset:offset + chunk_len]
            if audio_chunk.shape[-1] < chunk_len:
                audio_chunk = np.pad(
                    audio_chunk, ((0, 0), (0, chunk_len - audio_chunk.shape[-1]))
                )

            in_warmup = chunk_idx < self._warmup_chunks

            # Position embeddings: looked up on host to avoid large embedding Gather in ONNX
            pos_emb = self._pos_emb[pos_offset:pos_offset + F].reshape(1, F, -1)

            fe_feed = {
                "audio_chunk":         audio_chunk,
                "conv1_buffer":        conv1_buf,
                "conv2_buffer":        conv2_buf,
                "features_buffer":     feat_buf,
                "position_embeddings": pos_emb,
                # During warmup pass zero encoder buffers; outputs are discarded.
                **(enc_bufs if not in_warmup else
                   {f"buf_{i}": np.zeros((1, lc, enc_hidden), dtype=np.float32)
                    for i, lc in enumerate(self._enc_left_ctx)}),
            }

            fe_res = self._fused_encoder.infer(fe_feed)

            # Unpack outputs:
            # [0] k_cross [n_layers,1,heads,F,head_dim]
            # [1] v_cross
            # [2] conv1_buffer_out
            # [3] conv2_buffer_out
            # [4] features_buffer_out
            # [5..5+n_enc_layers-1] buf_i_out
            new_k, new_v       = fe_res[0], fe_res[1]
            conv1_buf          = fe_res[2]
            conv2_buf          = fe_res[3]
            feat_buf           = fe_res[4]
            buf_outs           = fe_res[5:]

            if in_warmup:
                # Warmup: keep only conv/feature state; discard encoder buf and cross-KV.
                chunk_idx += 1
                continue

            # Active step: update encoder buffers and accumulate cross-KV
            enc_bufs = {f"buf_{i}": buf_outs[i] for i in range(self._n_enc_layers)}
            pos_offset += F

            if self._is_static_decoder:
                n_new  = new_k.shape[3]
                end    = min(cross_kv_fill + n_new, self._max_memory_len)
                actual = end - cross_kv_fill
                k_cross_buf[:, :, :, cross_kv_fill:end, :] = new_k[:, :, :, :actual, :]
                v_cross_buf[:, :, :, cross_kv_fill:end, :] = new_v[:, :, :, :actual, :]
                cross_kv_fill = end
            else:
                kv_k_chunks.append(new_k)
                kv_v_chunks.append(new_v)

            chunk_idx += 1

        # ── Assemble cross-KV for decoder ────────────────────────────────────
        if self._is_static_decoder:
            out_k_cross  = k_cross_buf
            out_v_cross  = v_cross_buf
            valid_len    = cross_kv_fill
        else:
            if not kv_k_chunks:
                raise ValueError("No active (post-warmup) chunks were processed.")
            out_k_cross = np.concatenate(kv_k_chunks, axis=3)
            out_v_cross = np.concatenate(kv_v_chunks, axis=3)
            valid_len   = out_k_cross.shape[3]

        if max_tokens is None:
            max_tokens = int((audio_len / 16000) * 6)

        tokens = [self._start_token_id]

        # ── Decode ───────────────────────────────────────────────────────────
        if self._is_static_decoder:
            k_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )
            v_self = np.zeros(
                (self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim),
                dtype=np.float32,
            )

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

                dec_feed.update({
                    "current_len":    current_len,
                    "cross_attn_bias": cross_attn_bias,
                    "position_ids":    current_len,
                })
                for i in range(self._n_layers):
                    dec_feed[f"k_self_{i}"]  = k_self[i]
                    dec_feed[f"v_self_{i}"]  = v_self[i]
                    dec_feed[f"k_cross_{i}"] = out_k_cross[i]
                    dec_feed[f"v_cross_{i}"] = out_v_cross[i]

                res = self._decoder.infer(dec_feed)
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

            for _ in range(max_tokens):
                if self._extract_embeddings:
                    dec_feed = {"inputs_embeds": self._token_embeddings[tokens[-1]].reshape(1, 1, -1)}
                else:
                    dec_feed = {"token": np.array([[tokens[-1]]], dtype=np.int64)}

                for i in range(self._n_layers):
                    dec_feed[f"k_self_{i}"]  = k_self_layers[i]
                    dec_feed[f"v_self_{i}"]  = v_self_layers[i]
                    dec_feed[f"k_cross_{i}"] = out_k_cross[i]
                    dec_feed[f"v_cross_{i}"] = out_v_cross[i]

                res = self._decoder.infer(dec_feed)
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
