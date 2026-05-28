# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""PyTorch wrapper modules for Moonshine Streaming ONNX export.

These wrappers isolate model subcomponents for individual ONNX export:
  - PreprocessorWrapper: CNN embedder (audio → features)
  - TransformerEncoderWrapper: transformer layers (features → hidden states)
  - DecoderWrapper: first decode step (no KV cache)
  - DecoderWithPastWrapper: subsequent decode steps (with KV cache)
  - FullEncoderWrapper: combined encoder (for validation reference)
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers.cache_utils import (
    EncoderDecoderCache,
    DynamicCache,
    DynamicLayer,
)


# ── DynamicLayer patch ───────────────────────────────────────────────────────
# DynamicLayer.lazy_initialization creates torch.tensor([]) (1D) then update()
# does cat([1D_empty, 4D_key_states]) which confuses both TorchScript and dynamo.
# Fix: on first call (not initialized), skip the cat and assign directly.

def _patched_dynamic_update(self, key_states, value_states, cache_kwargs=None):
    if not self.is_initialized:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True
        return self.keys, self.values
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values

DynamicLayer.update = _patched_dynamic_update


# ── Helpers ──────────────────────────────────────────────────────────────────

def _layer_from_kv(k, v):
    """Create a DynamicLayer pre-filled with key/value tensors."""
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


# ── KV cache naming helpers ──────────────────────────────────────────────────

def kv_output_names(n_layers, self_prefix="present", cross_prefix="present"):
    names = []
    for i in range(n_layers):
        names.append(f"{self_prefix}_self_key_{i}")
        names.append(f"{self_prefix}_self_value_{i}")
    for i in range(n_layers):
        names.append(f"{cross_prefix}_cross_key_{i}")
        names.append(f"{cross_prefix}_cross_value_{i}")
    return names


def kv_input_names(n_layers):
    names = []
    for i in range(n_layers):
        names.append(f"past_self_key_{i}")
        names.append(f"past_self_value_{i}")
    for i in range(n_layers):
        names.append(f"past_cross_key_{i}")
        names.append(f"past_cross_value_{i}")
    return names


# ── Wrapper modules ──────────────────────────────────────────────────────────

class FullEncoderWrapper(nn.Module):
    """Wraps full encoder (embedder + transformer). Used for validation reference."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        encoder_outputs = self.encoder(
            input_values, attention_mask=attention_mask, return_dict=True,
        )
        return encoder_outputs.last_hidden_state


class PreprocessorWrapper(nn.Module):
    """CNN embedder: raw audio + attention_mask → feature sequence + frame-level mask.

    Wraps model.model.encoder.embedder which contains:
      CMVN → Asinh compression → Linear+SiLU → CausalConv1d(stride=2) → CausalConv1d(stride=2)
    """

    def __init__(self, model):
        super().__init__()
        self.embedder = model.model.encoder.embedder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor):
        hidden_states, padding_mask = self.embedder(input_values, padding_mask=attention_mask)
        return hidden_states, padding_mask


class TransformerEncoderWrapper(nn.Module):
    """Transformer encoder layers + final norm with sliding-window attention.

    Input: embedder features (B, seq_len, hidden_size) + frame-level mask (B, seq_len)
    Output: encoder hidden states (B, seq_len, hidden_size)
    """

    def __init__(self, model):
        super().__init__()
        self.layers = model.model.encoder.layers
        self.final_norm = model.model.encoder.final_norm
        self.config = model.model.encoder.config

    def forward(self, input_features: torch.FloatTensor, attention_mask: torch.Tensor) -> torch.FloatTensor:
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )

        hidden_states = input_features

        for layer_idx, encoder_layer in enumerate(self.layers):
            layer_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                and_mask_function=sliding_window_mask_function(
                    self.config.sliding_windows[layer_idx]
                ),
            )
            layer_out = encoder_layer(hidden_states, attention_mask=layer_mask)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class DecoderWrapper(nn.Module):
    """First decode step — no past KV cache.

    Returns last_hidden_state + flat past KV tensors (self + cross).
    Logits are computed externally via decoder_token_embeddings.npy.
    """

    def __init__(self, model, num_decoder_layers: int):
        super().__init__()
        self.base_model = model.model
        self.n_layers = num_decoder_layers

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.Tensor,
    ):
        out = self.base_model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=encoder_attention_mask.bool(),
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden_state = out.last_hidden_state
        pkv: EncoderDecoderCache = out.past_key_values

        self_cache = pkv.self_attention_cache
        cross_cache = pkv.cross_attention_cache

        flat = [hidden_state]
        for layer in self_cache.layers:
            flat.append(layer.keys)
            flat.append(layer.values)
        for layer in cross_cache.layers:
            flat.append(layer.keys)
            flat.append(layer.values)

        return tuple(flat)


class DecoderWithPastWrapper(nn.Module):
    """Subsequent decode steps — accepts and returns flat KV tensors.

    Returns last_hidden_state (not logits). Logits computed externally.
    """

    def __init__(self, model, num_decoder_layers: int):
        super().__init__()
        self.base_model = model.model
        self.n_layers = num_decoder_layers

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.Tensor,
        *flat_past,
    ):
        n = self.n_layers
        self_cache = DynamicCache()
        cross_cache = DynamicCache()

        for i in range(n):
            self_cache.layers.append(_layer_from_kv(flat_past[2 * i], flat_past[2 * i + 1]))
        for i in range(n):
            cross_cache.layers.append(_layer_from_kv(flat_past[2 * n + 2 * i], flat_past[2 * n + 2 * i + 1]))

        pkv = EncoderDecoderCache(self_cache, cross_cache)

        out = self.base_model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=encoder_attention_mask.bool(),
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        hidden_state = out.last_hidden_state
        new_pkv: EncoderDecoderCache = out.past_key_values

        new_self = new_pkv.self_attention_cache
        new_cross = new_pkv.cross_attention_cache

        flat_out = [hidden_state]
        for layer in new_self.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)
        for layer in new_cross.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)

        return tuple(flat_out)
