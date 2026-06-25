# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path
from typing import Literal, Final

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import ml_dtypes
import torch
from datasets import load_dataset, Audio
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

from ...utils.logging import configure_logging

from . import (
    ONNX_DTYPES,
    OPTIMUM_DTYPES,
    STATIC_MODEL_COMPONENTS,
    add_moonshine_streaming_export_args,
)

from ._graph import MoonshineStreamingOnnxGraphEditor
from ._inference import MoonshineStreaming
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig


def _layer_from_kv(k, v):
    from transformers.cache_utils import DynamicLayer
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


# Patch asinh with polynomial approximation to avoid Log+Sqrt in exported graph
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingAsinhCompression

_ASINH_R = 6.0
_ASINH_C = [9.701152e-01, -7.976821e-02, 5.723432e-03, -1.902770e-04, 2.265939e-06]


def _patched_asinh_forward(self, x):
    val = torch.exp(self.log_k) * x
    val = val.clamp(-_ASINH_R, _ASINH_R)
    u = val * val
    p = val.new_full((), _ASINH_C[-1])
    for c in reversed(_ASINH_C[:-1]):
        p = p * u + c
    return val * p


MoonshineStreamingAsinhCompression.forward = _patched_asinh_forward


# ── Wrapper modules ──────────────────────────────────────────────────────────

class StaticStreamingFrontendWrapper(torch.nn.Module):
    """Fixed-chunk frontend: audio_chunk + conv buffers → features + updated buffers."""

    def __init__(self, model, chunk_len: int):
        super().__init__()
        embedder = model.model.encoder.embedder
        self.cmvn = embedder.cmvn
        self.comp = embedder.comp
        self.linear = embedder.linear
        self.conv1 = embedder.conv1
        self.conv2 = embedder.conv2
        self.frame_len: int = int(embedder.frame_len)
        self.n_frames: int = chunk_len // self.frame_len

    def forward(self, audio_chunk, conv1_buffer, conv2_buffer):
        x = audio_chunk.reshape(1, self.n_frames, self.frame_len)
        x = self.cmvn(x)
        x = self.comp(x)
        x = torch.nn.functional.silu(self.linear(x))

        x = x.transpose(1, 2)
        x1_padded = torch.cat([conv1_buffer, x], dim=2)
        x1_conv = torch.nn.functional.conv1d(
            x1_padded, self.conv1.weight, self.conv1.bias,
            stride=self.conv1.stride, dilation=self.conv1.dilation,
        )
        conv1_buffer_out = x1_padded[:, :, -4:]
        x1_silu = torch.nn.functional.silu(x1_conv)

        x2_padded = torch.cat([conv2_buffer, x1_silu], dim=2)
        x2_conv = torch.nn.functional.conv1d(
            x2_padded, self.conv2.weight, self.conv2.bias,
            stride=self.conv2.stride, dilation=self.conv2.dilation,
        )
        conv2_buffer_out = x2_padded[:, :, -4:]

        features = x2_conv.transpose(1, 2)
        return features, conv1_buffer_out, conv2_buffer_out


class StatefulEncoderWrapper(torch.nn.Module):
    """Stateful streaming encoder with per-layer left-context sliding-window buffers."""

    def __init__(self, model):
        super().__init__()
        enc = model.model.encoder
        self.layers = enc.layers
        self.final_norm = enc.final_norm
        self.config = enc.config
        self._left_ctx = [int(w[0]) for w in self.config.sliding_windows]
        self._right_ctx = [int(w[1]) for w in self.config.sliding_windows]
        self._total_lookahead = sum(self._right_ctx)

    def forward(
        self,
        stable_features: torch.Tensor,
        right_ctx: torch.Tensor,
        buf_0: torch.Tensor,
        buf_1: torch.Tensor,
        buf_2: torch.Tensor,
        buf_3: torch.Tensor,
        buf_4: torch.Tensor,
        buf_5: torch.Tensor,
    ) -> tuple:
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )
        bufs = [buf_0, buf_1, buf_2, buf_3, buf_4, buf_5]
        bufs_out = []
        rc = self._total_lookahead

        stable_in = stable_features
        right_ctx_h = right_ctx

        for layer_idx, (layer, buf) in enumerate(zip(self.layers, bufs)):
            lc = self._left_ctx[layer_idx]
            layer_rc = self._right_ctx[layer_idx]

            window = torch.cat([buf, stable_in, right_ctx_h], dim=1)
            attn_mask = torch.ones(
                window.shape[0], window.shape[1], dtype=torch.bool, device=window.device
            )
            layer_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=window,
                attention_mask=attn_mask,
                and_mask_function=sliding_window_mask_function((lc, layer_rc)),
            )

            out = layer(window, attention_mask=layer_mask)
            out = out[0] if isinstance(out, tuple) else out

            bufs_out.append(torch.cat([buf, stable_in], dim=1)[:, -lc:, :])

            out_trimmed = out[:, lc:, :]
            stable_in = out_trimmed[:, :-rc, :]
            right_ctx_h = out_trimmed[:, -rc:, :]

        return (self.final_norm(stable_in), *bufs_out)


class AdapterWrapper(torch.nn.Module):
    """Encoder-to-decoder projection with host-side positional embeddings."""

    def __init__(self, decoder):
        super().__init__()
        self.pos_emb = decoder.pos_emb
        self.proj = decoder.proj

    def forward(self, encoded, position_embeddings):
        memory = encoded + position_embeddings
        memory = self.proj(memory)
        return memory


class CrossKVGeneratorWrapper(torch.nn.Module):
    """Generates stacked per-layer cross key-values from adapter memory."""

    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers
        self.depth = len(decoder.layers)
        self.num_heads = decoder.config.num_key_value_heads
        self.head_dim = getattr(
            decoder.config, "head_dim",
            decoder.config.hidden_size // decoder.config.num_attention_heads
        )

    def forward(self, memory):
        bsz, seq_len = memory.shape[:-1]
        k_list, v_list = [], []
        for layer in self.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(memory).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(memory).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_list.append(k)
            v_list.append(v)
        return torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)


class ZeroEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, indices):
        return torch.zeros(indices.shape[0], self.embedding_dim, device=indices.device)


class DecoderKVWrapper(torch.nn.Module):
    """Decoder with flat per-layer self/cross KV caches.

    When extract_embeddings=True the first argument is float inputs_embeds
    [1, 1, hidden_size] instead of an integer token id [1, 1].
    """

    def __init__(self, model, extract_embeddings: bool = False, output_attention: bool = False):
        super().__init__()
        self.base_model = model.model
        self.proj_out = model.proj_out
        self.n_layers = len(self.base_model.decoder.layers)
        self.decoder = self.base_model.decoder
        self._extract_embeddings = extract_embeddings
        self._output_attention = output_attention
        assert self.n_layers == 6, f"DecoderKVWrapper expects 6 decoder layers, got {self.n_layers}"

        self.decoder.pos_emb = ZeroEmbedding(self.decoder.pos_emb.embedding_dim)
        self.decoder.proj = torch.nn.Identity()

    def forward(self, token_or_embed,
                k_self_0, v_self_0,
                k_self_1, v_self_1,
                k_self_2, v_self_2,
                k_self_3, v_self_3,
                k_self_4, v_self_4,
                k_self_5, v_self_5,
                k_cross_0, v_cross_0,
                k_cross_1, v_cross_1,
                k_cross_2, v_cross_2,
                k_cross_3, v_cross_3,
                k_cross_4, v_cross_4,
                k_cross_5, v_cross_5,
                cross_attn_bias: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None):
        k_selves  = [k_self_0,  k_self_1,  k_self_2,  k_self_3,  k_self_4,  k_self_5]
        v_selves  = [v_self_0,  v_self_1,  v_self_2,  v_self_3,  v_self_4,  v_self_5]
        k_crosses = [k_cross_0, k_cross_1, k_cross_2, k_cross_3, k_cross_4, k_cross_5]
        v_crosses = [v_cross_0, v_cross_1, v_cross_2, v_cross_3, v_cross_4, v_cross_5]

        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        for i in range(self.n_layers):
            self_cache.layers.append(_layer_from_kv(k_selves[i], v_selves[i]))
            cross_cache.layers.append(_layer_from_kv(k_crosses[i], v_crosses[i]))

        pkv = EncoderDecoderCache(self_cache, cross_cache)
        for i in range(self.n_layers):
            pkv.is_updated[i] = True

        enc_seq_len = k_cross_0.shape[-2]
        dummy_encoder_hidden = torch.zeros(
            1, enc_seq_len, self.decoder.config.hidden_size,
            dtype=k_self_0.dtype, device=k_self_0.device,
        )

        encoder_attention_mask = cross_attn_bias

        if self._extract_embeddings:
            dec_out = self.decoder(
                inputs_embeds=token_or_embed,
                past_key_values=pkv,
                use_cache=True,
                position_ids=position_ids,
                encoder_hidden_states=dummy_encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=self._output_attention,
            )
        else:
            dec_out = self.decoder(
                input_ids=token_or_embed,
                past_key_values=pkv,
                use_cache=True,
                position_ids=position_ids,
                encoder_hidden_states=dummy_encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=self._output_attention,
            )

        logits = self.proj_out(dec_out.last_hidden_state)

        out_k_selves = [layer.keys   for layer in pkv.self_attention_cache.layers]
        out_v_selves = [layer.values for layer in pkv.self_attention_cache.layers]

        # Cross-KV is input-only and unchanged by the decoder, so it is NOT
        # re-emitted as outputs — doing so was a wasteful per-token D2H copy that
        # every orchestrator discards. Outputs are logits + updated self-KV only
        # (plus the optional cross_attn weights below). NOTE: any consumer that
        # detected the cross_attn output by output *count* must use
        # `len(out) > 1 + 2*depth` (was `1 + 4*depth`) — see full_streaming_demo.
        outputs = (logits,
                   out_k_selves[0], out_v_selves[0],
                   out_k_selves[1], out_v_selves[1],
                   out_k_selves[2], out_v_selves[2],
                   out_k_selves[3], out_v_selves[3],
                   out_k_selves[4], out_v_selves[4],
                   out_k_selves[5], out_v_selves[5])

        if self._output_attention:
            # Per-layer decoder→memory cross-attention probabilities, already
            # computed inside the eager attention; stack into one output tensor
            # [depth, 1, heads, 1, enc_seq_len]. argmax over the last axis gives
            # the audio frame each token attends to (host-side token→frame
            # alignment for window left-masking / word timestamps).
            cross_attn = torch.stack(dec_out.cross_attentions, dim=0)
            outputs = outputs + (cross_attn,)

        return outputs


class StatefulFusedEncoderWrapper(torch.nn.Module):
    """Single ONNX graph: frontend + stateful encoder + adapter + cross-KV generation.

    Fuses the four 5-split encoder-side components into one model dispatch per chunk,
    reducing host-device launch overhead from 4 calls/chunk down to 1.

    The features_buffer accumulates `total_lookahead` right-context frames across
    chunks so the encoder always sees a full right-context window.  Warmup logic
    (discarding early outputs while the buffer fills) is managed by the host.

    Inputs
    ------
    audio_chunk         [1, chunk_len]
    conv1_buffer        [1, hidden_size, 4]
    conv2_buffer        [1, c1, 4]
    features_buffer     [1, total_lookahead, hidden_size]  right-context accumulator
    position_embeddings [1, F, hidden_size]                host-side lookup
    buf_0 .. buf_5      [1, left_ctx_i, hidden_size]       encoder sliding-window caches

    Outputs
    -------
    k_cross             [depth, 1, heads, F, head_dim]
    v_cross             [depth, 1, heads, F, head_dim]
    conv1_buffer_out    [1, hidden_size, 4]
    conv2_buffer_out    [1, c1, 4]
    features_buffer_out [1, total_lookahead, hidden_size]
    buf_0_out .. buf_5_out  [1, left_ctx_i, hidden_size]
    """

    def __init__(self, model, chunk_len: int):
        super().__init__()
        self.frontend = StaticStreamingFrontendWrapper(model, chunk_len)
        self.encoder = StatefulEncoderWrapper(model)
        self.adapter = AdapterWrapper(model.model.decoder)
        self.cross_kv = CrossKVGeneratorWrapper(model.model.decoder)
        self.total_la = self.encoder._total_lookahead

        # Measure actual output frame count by running the frontend once.
        # n_frames is the number of INPUT frames (before strided convs); the two
        # stride-2 convs reduce it further, so we cannot use n_frames directly.
        embedder = model.model.encoder.embedder
        enc_hidden = int(embedder.linear.out_features)
        c1_channels = int(embedder.conv1.out_channels)
        with torch.no_grad():
            _feats, _, _ = self.frontend(
                torch.zeros(1, chunk_len),
                torch.zeros(1, enc_hidden, 4),
                torch.zeros(1, c1_channels, 4),
            )
        self.F: int = int(_feats.shape[1])

    def forward(
        self,
        audio_chunk,
        conv1_buffer,
        conv2_buffer,
        features_buffer,
        position_embeddings,
        buf_0, buf_1, buf_2, buf_3, buf_4, buf_5,
    ):
        # 1. Frontend feature extraction
        new_feats, conv1_buf_out, conv2_buf_out = self.frontend(
            audio_chunk, conv1_buffer, conv2_buffer
        )

        # 2. Shift right-context accumulator: drop oldest F frames, append new_feats
        combined = torch.cat([features_buffer, new_feats], dim=1)  # [1, total_la + F, hidden]
        stable_features = combined[:, :self.F, :]                   # [1, F, hidden]
        features_buffer_out = combined[:, self.F:, :]               # [1, total_la, hidden]
        right_ctx = features_buffer_out

        # 3. Sliding-window stateful encoder
        encoded, buf0_out, buf1_out, buf2_out, buf3_out, buf4_out, buf5_out = self.encoder(
            stable_features, right_ctx, buf_0, buf_1, buf_2, buf_3, buf_4, buf_5
        )

        # 4. Adapter: position projection
        memory = self.adapter(encoded, position_embeddings)

        # 5. Cross-KV generation
        k_cross, v_cross = self.cross_kv(memory)

        return (
            k_cross, v_cross,
            conv1_buf_out, conv2_buf_out,
            features_buffer_out,
            buf0_out, buf1_out, buf2_out, buf3_out, buf4_out, buf5_out,
        )


# ── Model Exporter ───────────────────────────────────────────────────────────

class MoonshineStreamingExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["tiny", "small"] = "tiny",
        model_dtype: str = "float",
        *,
        extract_embeddings: bool = False,
        export_attention: bool = False,
        hf_repo: str | None = None,
        max_audio_s: int = 5,
        max_tok_per_s: int = 6,
        chunk_len: int = 1280,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
        **edit_args,
    ):
        self._model_size = model_size
        self._extract_embeddings = extract_embeddings
        self._export_attention = export_attention
        self._onnx_source_dir = onnx_source_dir
        self._hf_repo = hf_repo or f"UsefulSensors/moonshine-streaming-{self._model_size}"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._num_samples = max_audio_s * 16_000
        self._max_tokens = max_audio_s * max_tok_per_s
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._chunk_len = chunk_len

        self._enc_seq_len = self._num_samples // 320
        self._feature_stride: int | None = None
        self._max_memory_len: int | None = None
        self._total_lookahead: int | None = None

        self._n_layers = getattr(self._config, "decoder_num_hidden_layers",
                                 getattr(self._config, "num_hidden_layers", 6))
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        dec_heads = getattr(self._config, "num_attention_heads", 8)
        self._num_kv_heads = getattr(self._config, "num_key_value_heads", 8)
        self._head_dim = self._hidden_size // dec_heads

        opt_configs = {
            comp: ORTOptimizerConfig(
                num_heads=dec_heads,
                hidden_size=self._config.hidden_size,
            )
            for comp in STATIC_MODEL_COMPONENTS
        }

        super().__init__(
            model_dtype,
            True,  # always static
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs=opt_configs,
            skip_export=skip_export,
        )

    def _setup_dirs(self) -> list[Path]:
        # Source ONNX is parameterised by (chunk_len, max_tokens) — the torch export
        # bakes those into the graph shapes — so it keeps a config-specific subdir to
        # avoid reusing a stale source across configs. The export/convert/torq dirs
        # follow the shared convention used by the other model exporters.
        onnx_dir = (
            self._models_dir / "source" / "onnx" / "merged" / self._model_size
            / self._model_dtype / f"c{self._chunk_len}_t{self._max_tokens}"
        )
        export_dir = (
            self._models_dir / "export" / "onnx" / self._model_dtype / "static"
        )
        convert_dir = (
            self._models_dir / "export" / "onnx" / "converted" / "static"
        )
        torq_dir = (
            self._models_dir / "export" / "torq"
            / ("converted" if self._convert_dtypes else self._model_dtype)
            / "static"
        )
        return onnx_dir, export_dir, convert_dir, torq_dir

    def _generate_source_onnx(self):
        import json
        from huggingface_hub import snapshot_download
        from transformers import MoonshineStreamingForConditionalGeneration

        local_dir = self._models_dir / "weights" / self._model_size
        if not (local_dir / "model.safetensors").exists():
            self._logger.info("Downloading %s to %s ...", self._hf_repo, str(local_dir))
            snapshot_download(
                repo_id=self._hf_repo,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )

        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            str(local_dir),
            torch_dtype=torch.float32,
            local_files_only=True,
            attn_implementation="eager",
        ).eval()

        self._onnx_dir.mkdir(parents=True, exist_ok=True)

        # Save token embeddings and position table for host-side lookup
        embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
        np.save(self._onnx_dir / "decoder_token_embeddings.npy", embeddings)
        pos_emb = model.model.decoder.pos_emb.weight.detach().cpu().numpy()
        np.save(self._onnx_dir / "adapter_pos_emb.npy", pos_emb)

        shutil.copy2(local_dir / "tokenizer.json", self._onnx_dir / "tokenizer.json")
        shutil.copy2(local_dir / "config.json", self._onnx_dir / "config.json")

        # ── Build fused wrapper (measures F internally) ──────────────────────
        fused_dummy = StatefulFusedEncoderWrapper(model, self._chunk_len).eval()
        F        = fused_dummy.F         # actual output frames per chunk (post-conv)
        total_la = fused_dummy.total_la  # sum of per-layer right-context sizes
        self._feature_stride  = F
        self._total_lookahead = total_la
        self._max_memory_len  = self._enc_seq_len
        enc_hidden   = self._hidden_size
        embedder     = model.model.encoder.embedder
        c1           = int(embedder.conv1.out_channels)  # actual conv1 output channels
        left_ctxs    = fused_dummy.encoder._left_ctx
        n_enc_layers = len(fused_dummy.encoder.layers)
        self._logger.info(
            "Fused encoder: chunk_len=%d, F=%d (actual output frames), "
            "total_la=%d, warmup_chunks=%d",
            self._chunk_len, F, total_la, (total_la + F - 1) // F,
        )

        # ── Export encoder.onnx ────────────────────────────────────────
        self._logger.info("Exporting StatefulFusedEncoderWrapper to ONNX ...")
        dummy_audio     = torch.zeros(1, self._chunk_len)
        dummy_conv1     = torch.zeros(1, enc_hidden, 4)
        dummy_conv2     = torch.zeros(1, c1, 4)
        dummy_feats_buf = torch.zeros(1, total_la, enc_hidden)
        dummy_pos_emb   = torch.zeros(1, F, enc_hidden)
        dummy_bufs      = [torch.zeros(1, lc, enc_hidden) for lc in left_ctxs]
        buf_in_names    = [f"buf_{i}" for i in range(n_enc_layers)]
        buf_out_names   = [f"buf_{i}_out" for i in range(n_enc_layers)]

        torch.onnx.export(
            fused_dummy,
            (dummy_audio, dummy_conv1, dummy_conv2, dummy_feats_buf,
             dummy_pos_emb, *dummy_bufs),
            str(self._onnx_dir / "encoder.onnx"),
            dynamo=True,
            input_names=[
                "audio_chunk", "conv1_buffer", "conv2_buffer",
                "features_buffer", "position_embeddings",
                *buf_in_names,
            ],
            output_names=[
                "k_cross", "v_cross",
                "conv1_buffer_out", "conv2_buffer_out",
                "features_buffer_out",
                *buf_out_names,
            ],
        )

        # ── Export decoder.onnx (identical to 5-split streaming decoder) ──
        self._logger.info(
            "Exporting DecoderKVWrapper to ONNX (max_tokens=%d, max_memory_len=%d)...",
            self._max_tokens, self._max_memory_len,
        )
        decoder_kv = DecoderKVWrapper(
            model, extract_embeddings=self._extract_embeddings,
            output_attention=self._export_attention,
        ).eval()
        self_kv_dummies = [
            torch.zeros(1, self._num_kv_heads, self._max_tokens, self._head_dim)
            for _ in range(self._n_layers * 2)
        ]
        self_kv_in_names  = [name for i in range(self._n_layers) for name in (f"k_self_{i}", f"v_self_{i}")]
        self_kv_out_names = [f"out_{name}" for name in self_kv_in_names]
        cross_kv_dummies = [
            torch.zeros(1, self._num_kv_heads, self._max_memory_len, self._head_dim)
            for _ in range(self._n_layers * 2)
        ]
        cross_kv_in_names  = [name for i in range(self._n_layers) for name in (f"k_cross_{i}", f"v_cross_{i}")]
        # Cross-KV is input-only — not re-emitted as decoder outputs (see DecoderKVWrapper.forward).
        dummy_cross_attn_bias = torch.zeros(1, self._num_kv_heads, 1, self._max_memory_len)
        dummy_position_ids    = torch.tensor([[self._max_tokens]], dtype=torch.long)

        if self._extract_embeddings:
            dummy_first_input = torch.zeros(1, 1, self._hidden_size)
            first_input_name  = "inputs_embeds"
        else:
            dummy_first_input = torch.ones(1, 1, dtype=torch.long)
            first_input_name  = "token"

        decoder_out_names = ["logits", *self_kv_out_names]
        if self._export_attention:
            # one stacked output [depth, 1, heads, 1, max_memory_len]
            decoder_out_names.append("cross_attn")
            self._logger.info("  + exporting cross-attention weights output 'cross_attn'")

        torch.onnx.export(
            decoder_kv,
            (dummy_first_input, *self_kv_dummies, *cross_kv_dummies,
             dummy_cross_attn_bias, dummy_position_ids),
            str(self._onnx_dir / "decoder.onnx"),
            dynamo=True,
            input_names=[first_input_name, *self_kv_in_names,
                         *cross_kv_in_names, "cross_attn_bias", "position_ids"],
            output_names=decoder_out_names,
        )

        # ── streaming_config.json ────────────────────────────────────────────
        warmup_chunks = (total_la + F - 1) // F
        config = {
            "chunk_len": self._chunk_len,
            "feature_stride": F,
            "total_lookahead": total_la,
            "warmup_chunks": warmup_chunks,
            "max_tokens": self._max_tokens,
            "max_memory_len": self._max_memory_len,
            "extract_embeddings": self._extract_embeddings,
            "export_attention": self._export_attention,
        }
        with open(self._onnx_dir / "streaming_config.json", "w") as f:
            json.dump(config, f, indent=2)
        self._logger.info("Streaming config: %s", config)

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        import json as _json

        source_files = {
            "encoder": self._onnx_dir / "encoder.onnx",
            "decoder":    self._onnx_dir / "decoder.onnx",
        }

        any_missing = any(not path.exists() for path in source_files.values())
        if any_missing:
            self._logger.info("Source ONNX models not found. Exporting from PyTorch...")
            self._generate_source_onnx()

        if self._feature_stride is None:
            config_path = self._onnx_dir / "streaming_config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"streaming_config.json not found at {config_path}. "
                    "Delete cached ONNX files and re-export to regenerate."
                )
            with open(config_path) as _f:
                _cfg = _json.load(_f)
            self._feature_stride  = _cfg["feature_stride"]
            self._total_lookahead = _cfg["total_lookahead"]
            self._max_memory_len  = _cfg["max_memory_len"]
            self._logger.info(
                "Loaded streaming config: F=%d, total_la=%d, max_memory_len=%d",
                self._feature_stride, self._total_lookahead, self._max_memory_len,
            )

        return {comp: onnx.load(path) for comp, path in source_files.items()}

    # ── Graph-edit helpers ───────────────────────────────────────────────────

    def _make_fused_encoder_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
            model, "encoder", self._onnx_export_dtype
        )
        # Dynamo export produces concrete shapes; fix_io_dims handles any residual
        # symbolic dims on batch or seq axes.
        editor.fix_fused_encoder_io(
            chunk_len=self._chunk_len,
            feat_len=self._feature_stride,
        )
        editor.decompose_layer_normalization()
        # Ensure kernel_shape is present on all 1-D Conv nodes (dynamo may omit it).
        for node in list(editor._graph.nodes):
            if node.op == "Conv":
                weight = node.inputs[1]
                if (
                    ("kernel_shape" not in node.attrs or not node.attrs["kernel_shape"])
                    and weight.shape is not None
                    and len(weight.shape) == 3
                ):
                    node.attrs["kernel_shape"] = [weight.shape[2]]
        editor._graph.cleanup().toposort()
        # StaticStreamingFrontendWrapper uses explicit torch.cat for causal padding,
        # so no Pad nodes exist in the graph — decompose_strided_conv1d is still
        # needed since the hardware cannot execute strided Conv1D directly.
        editor.decompose_strided_conv1d()
        editor.decompose_gelu()
        editor.decompose_boolean_and()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model, skip_data_prop=True)

    def _make_decoder_kv_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
            model, "decoder", self._onnx_export_dtype
        )
        editor.make_decoder_static(self._max_tokens)
        editor.decompose_layer_normalization()
        editor.decompose_gelu()
        editor.decompose_boolean_and()
        editor.clear_intermediate_shapes()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def make_static(self):
        F = self._feature_stride
        assert F is not None, "feature_stride is None — _load_onnx must run first"
        self._logger.info(
            "Applying streaming static graph edits (F=%d, max_tokens=%d)...",
            F, self._max_tokens,
        )
        self._components["encoder"] = self._make_fused_encoder_static(
            self._components["encoder"]
        )
        self._components["decoder"] = self._make_decoder_kv_static(
            self._components["decoder"]
        )

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        # Copy embeddings / tokenizer alongside the exported ONNX
        for fname in ("decoder_token_embeddings.npy", "adapter_pos_emb.npy"):
            src = self._onnx_dir / fname
            dst = Path(model_path).parent / fname
            if src.exists():
                shutil.copy2(src, dst)

        if "encoder" in component:
            editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
                model_path, component, self._onnx_export_dtype
            )
            editor.decompose_asinh()
            editor.remove_identity_gather_nd()
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        if "decoder" in component:
            editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
                model_path, component, self._onnx_export_dtype
            )
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        for filename in ("tokenizer.json", "config.json"):
            src = self._onnx_dir / filename
            dst = Path(model_path).parent / filename
            if src.exists():
                shutil.copy2(src, dst)

    # Host-side runtime LUTs fed directly into the models: the token-embedding table
    # (-> decoder ``inputs_embeds``) and the position table (-> encoder
    # ``position_embeddings``). Their dtype must track the model's I/O dtype.
    EMBEDDING_SIDECARS: Final[tuple[str, ...]] = (
        "decoder_token_embeddings.npy", "adapter_pos_emb.npy",
    )
    # Metadata that rides alongside the models, copied verbatim.
    METADATA_SIDECARS: Final[tuple[str, ...]] = (
        "tokenizer.json", "config.json", "streaming_config.json",
    )
    SIDECAR_FILES: Final[tuple[str, ...]] = EMBEDDING_SIDECARS + METADATA_SIDECARS

    def export_onnx(self, validate: bool = True):
        super().export_onnx(validate=False)
        for fname in self.SIDECAR_FILES:
            src = self._onnx_dir / fname
            if src.exists():
                shutil.copy2(src, self._export_dir / fname)
        if validate:
            self.validate_onnx()

    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
        skip: list[str] | None = None,
    ):
        # The converted models take bf16 ``position_embeddings`` / ``inputs_embeds``,
        # so the host-side LUTs must be bf16 too (the golden export left them float32,
        # which would dtype-mismatch the converted graph at runtime). The base
        # converter loads each .npy, casts to the target dtype, and writes it into the
        # converted dir. Unless ``--preserve-io-dtypes`` keeps the I/O float.
        external_data = []
        if not preserve_io:
            external_data = [
                (self._export_dir / fname, np.dtype(ml_dtypes.bfloat16))
                for fname in self.EMBEDDING_SIDECARS
                if (self._export_dir / fname).exists()
            ]
        super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            skip=skip,
            external_data=external_data,
        )
        # If I/O dtypes were preserved, copy the LUTs verbatim instead.
        embed_to_copy = self.EMBEDDING_SIDECARS if preserve_io else ()
        for fname in self.METADATA_SIDECARS + embed_to_copy:
            src = self._export_dir / fname
            if src.exists():
                shutil.copy2(src, self._convert_dir / fname)

    def validate_onnx(self, n_iters: int = 5):
        self._logger.info("Validating exported streaming ONNX models...")
        runner = MoonshineStreaming.from_onnx(
            encoder_model=self._export_dir / "encoder.onnx",
            decoder_model=self._export_dir / "decoder.onnx",
            model_size=self._model_size,
        )

        tokenizer_path = self._export_dir / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists() else None

        try:
            dataset = load_dataset(
                path="hf-internal-testing/librispeech_asr_dummy",
                name="clean",
                split="validation",
            )
            dataset = dataset.cast_column("audio", Audio(16_000))
        except Exception as exc:  # offline / dataset unavailable
            self._logger.warning(
                "Could not load validation dataset (%s); using dummy audio", exc
            )
            dataset = None

        for i in range(n_iters):
            if dataset is not None and i < len(dataset):
                audio = dataset[i]["audio"]["array"].astype(np.float32)[np.newaxis, :]
            else:
                if dataset is not None:
                    break
                audio = np.random.randn(1, 80_000).astype(np.float32)

            tokens = runner.run(audio)
            if tokenizer is not None:
                text = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
                self._logger.info(
                    "(ONNX-validation) [iter %d, %.1f ms] '%s'",
                    i, runner.last_infer_time * 1000, text,
                )
            else:
                self._logger.info(
                    "(ONNX-validation) [iter %d, %.1f ms] tokens=%s",
                    i, runner.last_infer_time * 1000, tokens.tolist(),
                )
            if dataset is None:
                break


def export_moonshine_streaming_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    if getattr(args, "chunk_len", None) is None:
        raise ValueError("--chunk-len is required for the streaming architecture")
    exporter = MoonshineStreamingExporter(
        args.model_size,
        args.dtype,
        extract_embeddings=args.extract_embeddings,
        export_attention=args.export_attention,
        hf_repo=args.hf_repo,
        max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec,
        chunk_len=args.chunk_len,
        models_dir=args.models_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        skip_export=args.skip_export,
        broadcast_ops=args.broadcast_ops,
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if args.skip_torq is None or "all" not in args.skip_torq:
        exporter.export_torq(
            torq_compile_args=args.compile_flags or [],
            use_binary=args.use_binary,
            skip=args.skip_torq or [],
            local_compile=args.local_compile,
            compiler_path=args.compiler_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Export Moonshine Streaming to Torq")
    add_moonshine_streaming_export_args(parser)
    export_moonshine_streaming_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
