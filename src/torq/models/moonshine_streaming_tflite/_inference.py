# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""TFLite streaming inference: fused_encoder + static-cache decoder_kv.

Mirrors the 2-split orchestration (host-side warmup, cross-KV ring, static decode)
but runs the LiteRT `.tflite` graphs produced by `tflite_export.py`. The decoder
uses a fixed-size self-KV cache (see `_StaticSelfCache` in `export.py`), so a single
graph is reused across decode steps. Embeddings are looked up on the host.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np

import ai_edge_litert.interpreter as lite_rt


class _LiteRTRunner:
    """Feeds inputs by export arg-order (`serving_default_args_<N>`) and returns
    outputs in the model's output order (= wrapper return-tuple order)."""

    def __init__(self, model_path: str | os.PathLike, n_threads: int | None = None):
        self.model_path = Path(model_path)
        self._it = lite_rt.Interpreter(str(self.model_path), num_threads=n_threads)
        self._it.allocate_tensors()
        self._in = sorted(self._it.get_input_details(), key=self._arg_index)
        self._out = self._it.get_output_details()

    @staticmethod
    def _arg_index(detail) -> int:
        # 'serving_default_args_0', 'serving_default_args_12:0', ...
        name = detail["name"]
        try:
            return int(name.split("args_")[1].split("_")[0].split(":")[0])
        except (IndexError, ValueError):
            return detail["index"]

    @property
    def input_details(self):
        return self._in

    def infer(self, feeds: list[np.ndarray]) -> list[np.ndarray]:
        for d, a in zip(self._in, feeds):
            self._it.set_tensor(d["index"], a.astype(d["dtype"]))
        self._it.invoke()
        return [self._it.get_tensor(o["index"]) for o in self._out]


class MoonshineStreamingTFLite:
    """Two-model streaming inference over LiteRT graphs."""

    def __init__(
        self,
        fused_encoder: _LiteRTRunner,
        decoder: _LiteRTRunner,
        model_size: Literal["tiny", "small"],
        config: dict,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._fe = fused_encoder
        self._dec = decoder

        if model_size == "tiny":
            self._n_layers, self._n_kv_heads, self._head_dim, self._hidden = 6, 8, 40, 320
        else:
            self._n_layers, self._n_kv_heads, self._head_dim, self._hidden = 14, 10, 64, 640

        self._chunk_len = int(config["chunk_len"])
        self._F = int(config["feature_stride"])
        self._total_la = int(config["total_lookahead"])
        self._max_tokens = int(config["max_tokens"])
        self._max_memory_len = int(config["max_memory_len"])
        self._extract_embeddings = bool(config["extract_embeddings"])
        self._warmup_chunks = math.ceil(self._total_la / self._F)

        # Encoder buffer geometry, read from the fused_encoder input shapes (arg order:
        # audio, conv1, conv2, features_buffer, position_embeddings, buf_0 .. buf_5).
        fe_in = self._fe.input_details
        self._enc_hidden = int(fe_in[1]["shape"][1])
        self._c1 = int(fe_in[2]["shape"][1])
        self._enc_left_ctx = [int(d["shape"][1]) for d in fe_in[5:]]
        self._n_enc_layers = len(self._enc_left_ctx)

        self._start_token_id = 1
        self._end_token_id = 2

        self._token_embeddings = self._load(self._dec.model_path, "decoder*token_embeddings.npy")
        self._pos_emb = self._load(self._fe.model_path, "adapter_pos_emb.npy")

        self._logger.info(
            "TFLite streaming: chunk_len=%d F=%d total_la=%d warmup=%d enc_layers=%d "
            "max_tokens=%d max_memory_len=%d",
            self._chunk_len, self._F, self._total_la, self._warmup_chunks,
            self._n_enc_layers, self._max_tokens, self._max_memory_len,
        )

    @staticmethod
    def _load(model_path: Path, pattern: str) -> np.ndarray:
        hits = list(model_path.parent.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"Missing '{pattern}' next to {model_path}")
        return np.load(hits[0])

    @classmethod
    def from_tflite(
        cls,
        fused_encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
    ) -> "MoonshineStreamingTFLite":
        cfg_path = Path(fused_encoder_model).parent / "streaming_config.json"
        with open(cfg_path) as f:
            config = json.load(f)
        return cls(
            _LiteRTRunner(fused_encoder_model, n_threads),
            _LiteRTRunner(decoder_model, n_threads),
            model_size,
            config,
        )

    def run(self, input_audio: np.ndarray, max_tokens: int | None = None) -> np.ndarray:
        f32 = np.float32
        L = input_audio.shape[-1]
        chunk_len, F, total_la, eh = self._chunk_len, self._F, self._total_la, self._enc_hidden

        # ── Encoder streaming state ──────────────────────────────────────────
        conv1 = np.zeros((1, eh, 4), f32)
        conv2 = np.zeros((1, self._c1, 4), f32)
        feat = np.zeros((1, total_la, eh), f32)
        bufs = [np.zeros((1, lc, eh), f32) for lc in self._enc_left_ctx]

        kc, vc = [], []
        pos_off = 0
        for ci, off in enumerate(range(0, L, chunk_len)):
            chunk = input_audio[:, off:off + chunk_len]
            if chunk.shape[-1] < chunk_len:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_len - chunk.shape[-1])))
            warm = ci < self._warmup_chunks
            pos_emb = self._pos_emb[pos_off:pos_off + F].reshape(1, F, -1).astype(f32)
            feed_bufs = bufs if not warm else [np.zeros((1, lc, eh), f32) for lc in self._enc_left_ctx]

            out = self._fe.infer([chunk.astype(f32), conv1, conv2, feat, pos_emb, *feed_bufs])
            new_k, new_v, conv1, conv2, feat = out[0], out[1], out[2], out[3], out[4]
            buf_outs = out[5:]
            if warm:
                continue
            bufs = list(buf_outs)
            pos_off += F
            kc.append(new_k)
            vc.append(new_v)

        if not kc:
            raise ValueError("No active (post-warmup) chunks were processed.")
        K = np.concatenate(kc, axis=3)  # [n_layers, 1, heads, T, head_dim]
        V = np.concatenate(vc, axis=3)
        T = K.shape[3]

        # ── Cross-KV window (cap at max_memory_len, keep earliest frames) ────
        ml = self._max_memory_len
        valid = min(T, ml)
        Kc = np.zeros((self._n_layers, 1, self._n_kv_heads, ml, self._head_dim), f32)
        Vc = np.zeros_like(Kc)
        Kc[:, :, :, :valid] = K[:, :, :, :valid]
        Vc[:, :, :, :valid] = V[:, :, :, :valid]
        cross_attn_bias = np.zeros((1, self._n_kv_heads, 1, ml), f32)
        cross_attn_bias[:, :, :, valid:] = -1e9

        # ── Static-cache decode ──────────────────────────────────────────────
        if max_tokens is None:
            max_tokens = min(self._max_tokens, int((L / 16000) * 6))
        max_tokens = min(max_tokens, self._max_tokens)

        k_self = np.zeros((self._n_layers, 1, self._n_kv_heads, self._max_tokens, self._head_dim), f32)
        v_self = np.zeros_like(k_self)
        tokens = [self._start_token_id]

        for step in range(max_tokens):
            position_ids = np.array([[step]], dtype=np.int64)
            self_attn_bias = np.full((1, 1, 1, self._max_tokens), -1e9, f32)
            self_attn_bias[:, :, :, :step + 1] = 0.0

            if self._extract_embeddings:
                first = self._token_embeddings[tokens[-1]].reshape(1, 1, -1).astype(f32)
            else:
                first = np.array([[tokens[-1]]], dtype=np.int64)

            feeds = [first]
            for i in range(self._n_layers):
                feeds += [k_self[i], v_self[i]]
            for i in range(self._n_layers):
                feeds += [Kc[i], Vc[i]]
            feeds += [cross_attn_bias, position_ids, self_attn_bias]

            res = self._dec.infer(feeds)
            logits = res[0]
            for i in range(self._n_layers):
                k_self[i] = res[1 + i * 2]
                v_self[i] = res[2 + i * 2]

            next_token = int(logits[0, -1, :].argmax())
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        return np.array([tokens])
