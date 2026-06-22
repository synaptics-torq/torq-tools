# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import torch
from transformers.cache_utils import EncoderDecoderCache, DynamicCache


def _layer_from_kv(k, v):
    from transformers.cache_utils import DynamicLayer
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


class _StaticSelfCache(DynamicCache):
    """Fixed-size self-attention cache: writes the new token's K/V in-place at a
    stored position via a mask/select instead of concatenating.

    This keeps the per-layer self-KV shape constant ([1, heads, max_tokens, dim])
    across decode steps, so the exported graph is a reusable fixed-point — the
    torch-level equivalent of the 5/2-split ``make_decoder_static`` graph surgery.
    The caller sets ``self.pos`` (a 1-D index tensor) before each forward.

    Note: we deliberately do NOT use ``index_copy``/``index_put``. Those lower to
    ``SCATTER_ND`` (→ ``stablehlo.scatter`` into a *middle* axis) which the Torq/TOSA
    path cannot legalize (``iree_linalg_ext.scatter`` requires indexing the leading
    dim). The mask form below lowers to supported elementwise ops (iota/equal/select/
    broadcast). All index math is int32 — the NPU supports int32/bf16 only.
    """

    pos: torch.Tensor

    def _write_at(self, cache: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        # cache: [1, H, L, D]; new: [1, H, 1, D]; write `new` at sequence position self.pos.
        # Keep everything rank-4 (no 0-D scalars): rank-0 operands make the TorqHW elementwise
        # lowering choke. Streaming decode positions are always >= 0, so no negative-index
        # normalization is needed (unlike index_copy, which added it defensively).
        L = cache.shape[2]
        pos = self.pos.to(torch.int32).view(1, 1, 1, 1)
        idx = torch.arange(L, dtype=torch.int32, device=cache.device).view(1, 1, L, 1)
        onehot = idx == pos  # [1, 1, L, 1] bool, true only at the write position
        return torch.where(onehot, new.expand(-1, -1, L, -1), cache)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        layer.keys = self._write_at(layer.keys, key_states)
        layer.values = self._write_at(layer.values, value_states)
        return layer.keys, layer.values


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
        self._has_precomputed_masks = False

    def precompute_masks(self, stable_len: int):
        """Bake the per-layer sliding-window masks as constant buffers.

        The window length per layer (``lc + stable_len + total_lookahead``) and the
        sliding-window pattern are fully static, and the padding mask is all-ones, so
        each layer's attention mask is a compile-time constant. Precomputing it here
        (eagerly) keeps ``create_bidirectional_mask``'s internal ``torch.vmap`` out of
        the traced graph — vmap does not survive PT2E export→convert→re-export.
        """
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )
        hidden = int(self.config.hidden_size)
        rc = self._total_lookahead
        with torch.no_grad():
            for layer_idx in range(len(self.layers)):
                lc = self._left_ctx[layer_idx]
                layer_rc = self._right_ctx[layer_idx]
                window = torch.zeros(1, lc + stable_len + rc, hidden)
                attn_mask = torch.ones(1, window.shape[1], dtype=torch.bool)
                mask = create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=window,
                    attention_mask=attn_mask,
                    and_mask_function=sliding_window_mask_function((lc, layer_rc)),
                )
                self.register_buffer(f"_mask_{layer_idx}", mask, persistent=False)
        self._has_precomputed_masks = True

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
            if self._has_precomputed_masks:
                layer_mask = getattr(self, f"_mask_{layer_idx}")
            else:
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

    def __init__(self, model, extract_embeddings: bool = False, static_self_cache: bool = False):
        super().__init__()
        self.base_model = model.model
        self.proj_out = model.proj_out
        self.n_layers = len(self.base_model.decoder.layers)
        self.decoder = self.base_model.decoder
        self._extract_embeddings = extract_embeddings
        self._static_self_cache = static_self_cache
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
                position_ids: torch.Tensor | None = None,
                self_attn_bias: torch.Tensor | None = None):
        k_selves  = [k_self_0,  k_self_1,  k_self_2,  k_self_3,  k_self_4,  k_self_5]
        v_selves  = [v_self_0,  v_self_1,  v_self_2,  v_self_3,  v_self_4,  v_self_5]
        k_crosses = [k_cross_0, k_cross_1, k_cross_2, k_cross_3, k_cross_4, k_cross_5]
        v_crosses = [v_cross_0, v_cross_1, v_cross_2, v_cross_3, v_cross_4, v_cross_5]

        if self._static_self_cache:
            self_cache = _StaticSelfCache()
            self_cache.pos = position_ids[0]  # [1] index into the max_tokens axis
        else:
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
        # In static mode the self-KV cache is a fixed max_tokens buffer; a 4D
        # additive mask gates which positions [0..current] are valid. A prepared 4D
        # attention_mask is passed through create_causal_mask unchanged.
        self_attention_mask = self_attn_bias if self._static_self_cache else None

        dec_kwargs = dict(
            past_key_values=pkv,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=self_attention_mask,
            encoder_hidden_states=dummy_encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
        )
        if self._extract_embeddings:
            dec_out = self.decoder(inputs_embeds=token_or_embed, **dec_kwargs)
        else:
            dec_out = self.decoder(input_ids=token_or_embed, **dec_kwargs)

        logits = self.proj_out(dec_out.last_hidden_state)

        out_k_selves = [layer.keys   for layer in pkv.self_attention_cache.layers]
        out_v_selves = [layer.values for layer in pkv.self_attention_cache.layers]

        return (logits,
                out_k_selves[0], out_v_selves[0],
                out_k_selves[1], out_v_selves[1],
                out_k_selves[2], out_v_selves[2],
                out_k_selves[3], out_v_selves[3],
                out_k_selves[4], out_v_selves[4],
                out_k_selves[5], out_v_selves[5],
                k_cross_0, v_cross_0,
                k_cross_1, v_cross_1,
                k_cross_2, v_cross_2,
                k_cross_3, v_cross_3,
                k_cross_4, v_cross_4,
                k_cross_5, v_cross_5)


class StatefulFusedEncoderWrapper(torch.nn.Module):
    """Single graph: frontend + stateful encoder + adapter + cross-KV generation.

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

        # Bake static sliding-window masks (removes torch.vmap from the traced graph).
        self.encoder.precompute_masks(self.F)

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
