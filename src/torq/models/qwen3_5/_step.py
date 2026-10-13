# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Static single-token decode step for the Qwen3.5 text model.

optimum cannot export Qwen3.5 (Gated DeltaNet layers), so the exporter builds the
decode step in PyTorch from the Hugging Face weights and exports it with
``torch.onnx.export``. All state is explicit:

- full-attention layers: K and V caches ``[1, kv_heads, ctx, head_dim]``. The new
  token is written in place with one Where.
- Gated DeltaNet layers: conv state ``[1, conv_dim, kernel - 1]`` and recurrent
  state ``[1, v_heads, k_dim, v_dim]``.

:class:`PackedStep` packs the states per kind into three tensors, because the
runtime accepts at most 64 bindings.

The step only uses ops that export to static ONNX for Torq: no Einsum, OneHot,
Expand or ``torch.chunk`` (chunk exports a Split with computed sizes, which
imports with dynamic shapes), and no MatMul with a constant B other than the
weight layers (the weight quantizer takes every such MatMul as a weight).
"""

import torch
import torch.nn.functional as F
from torch import nn


def _rms_norm_1p(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Qwen3.5 RMSNorm: the weight is zero-centered, the scale is (1 + w).
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(x)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class StaticStep(nn.Module):
    """One decode step of the Qwen3.5 text model with explicit per-layer state."""

    def __init__(self, text_model: nn.Module, cfg, ctx: int):
        super().__init__()
        self.m = text_model
        self.cfg = cfg
        self.ctx = ctx
        self.layer_types = list(cfg.layer_types)
        self.eps = cfg.rms_norm_eps
        # Text-only M-RoPE has equal T/H/W positions, so it is plain RoPE on the
        # first head_dim * partial_rotary_factor dims.
        rope = cfg.rope_parameters
        rot = int(cfg.head_dim * rope.get("partial_rotary_factor", 1.0))
        inv = 1.0 / (rope["rope_theta"] ** (torch.arange(0, rot, 2).float() / rot))
        freqs = torch.arange(ctx).float()[:, None] * inv[None, :]
        emb = torch.cat([freqs, freqs], -1)
        self.register_buffer("cos_tab", emb.cos(), persistent=False)  # [ctx, rot]
        self.register_buffer("sin_tab", emb.sin(), persistent=False)
        self.register_buffer("slot_ids", torch.arange(ctx).view(1, -1), persistent=False)
        self.rot = rot

    def _attention(self, layer, x, pos_onehot, mask, k_cache, v_cache, cos, sin):
        a = layer.self_attn
        hd = a.head_dim
        # Static slices, not torch.chunk.
        qp = a.q_proj(x).view(1, 1, -1, hd * 2)
        q, gate = qp[..., :hd], qp[..., hd:]
        gate = gate.reshape(1, 1, -1)
        q = _rms_norm_1p(q, a.q_norm.weight, self.eps).transpose(1, 2)
        k = _rms_norm_1p(a.k_proj(x).view(1, 1, -1, hd), a.k_norm.weight, self.eps).transpose(1, 2)
        v = a.v_proj(x).view(1, 1, -1, hd).transpose(1, 2)

        def rope(t):
            tr, tp = t[..., : self.rot], t[..., self.rot :]
            h = self.rot // 2
            rh = torch.cat([-tr[..., h:], tr[..., :h]], -1)
            return torch.cat([tr * cos + rh * sin, tp], -1)

        q, k = rope(q), rope(k)
        sel = pos_onehot.view(1, 1, self.ctx, 1) > 0.5
        k_cache = torch.where(sel, k, k_cache)
        v_cache = torch.where(sel, v, v_cache)
        # Grouped-query attention without repeat_kv.
        nkv = k_cache.shape[1]
        rep = q.shape[1] // nkv
        qg = q.view(1, nkv, rep, hd)
        scores = torch.matmul(qg, k_cache.transpose(-1, -2)) * a.scaling + mask
        p = torch.softmax(scores.float(), -1).type_as(q)
        o = torch.matmul(p, v_cache).reshape(1, 1, -1)
        o = o * torch.sigmoid(gate)
        return a.o_proj(o), k_cache, v_cache

    def _deltanet(self, layer, x, conv_state, rec_state):
        d = layer.linear_attn
        qkv = d.in_proj_qkv(x).transpose(1, 2)  # [1, conv_dim, 1]
        z = d.in_proj_z(x)
        b = d.in_proj_b(x)
        a = d.in_proj_a(x)
        # Causal depthwise conv over (state, new token); keep the last kernel - 1 inputs.
        full = torch.cat([conv_state, qkv], -1)
        new_conv = full[:, :, 1:]
        w = d.conv1d.weight.squeeze(1)  # [conv_dim, kernel]
        y = F.silu((full * w.unsqueeze(0)).sum(-1, keepdim=True)).transpose(1, 2)
        q, k, v = torch.split(y, [d.key_dim, d.key_dim, d.value_dim], dim=-1)
        q = q.reshape(1, -1, d.head_k_dim).float()
        k = k.reshape(1, -1, d.head_k_dim).float()
        v = v.reshape(1, -1, d.head_v_dim).float()
        beta = b.sigmoid().float().reshape(1, -1)
        g = (-d.A_log.float().exp() * F.softplus(a.float() + d.dt_bias)).reshape(1, -1)
        q = _l2norm(q) / (d.head_k_dim ** 0.5)
        k = _l2norm(k)
        # Gated delta rule, one token.
        s = rec_state * g.exp()[..., None, None]
        kv_mem = (s * k.unsqueeze(-1)).sum(-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        s = s + k.unsqueeze(-1) * delta.unsqueeze(-2)
        o = (s * q.unsqueeze(-1)).sum(-2).to(x.dtype).reshape(-1, d.head_v_dim)
        # Gated RMSNorm: norm, weight, then silu(z) gate.
        of = o.float()
        o = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + d.norm.variance_epsilon)).to(x.dtype)
        o = (d.norm.weight * o) * F.silu(z.reshape(-1, d.head_v_dim).float()).to(x.dtype)
        return d.out_proj(o.reshape(1, 1, -1)), new_conv, s

    def forward(self, token_embedding, position_ids, *states):
        dtype = token_embedding.dtype
        pos_onehot = (self.slot_ids == position_ids.view(1, 1)).to(dtype)  # [1, ctx]
        # Mul + ReduceSum, not MatMul (see the module docstring).
        cos = (pos_onehot.view(-1, 1) * self.cos_tab).sum(0).view(1, 1, 1, -1).to(dtype)
        sin = (pos_onehot.view(-1, 1) * self.sin_tab).sum(0).view(1, 1, 1, -1).to(dtype)
        mask = torch.where(self.slot_ids.view(1, 1, 1, -1) <= position_ids.view(1, 1, 1, 1), 0.0, -1e9).to(dtype)
        x = token_embedding
        it = iter(states)
        outs = []
        for i, lt in enumerate(self.layer_types):
            layer = self.m.layers[i]
            h = _rms_norm_1p(x, layer.input_layernorm.weight, self.eps)
            if lt == "full_attention":
                h, kc, vc = self._attention(layer, h, pos_onehot, mask, next(it), next(it), cos, sin)
                outs += [kc, vc]
            else:
                h, cs, rs = self._deltanet(layer, h, next(it), next(it))
                outs += [cs, rs]
            x = x + h
            x = x + layer.mlp(_rms_norm_1p(x, layer.post_attention_layernorm.weight, self.eps))
        return (_rms_norm_1p(x, self.m.norm.weight, self.eps), *outs)

    def init_states(self) -> list[torch.Tensor]:
        cfg = self.cfg
        conv_dim = 2 * cfg.linear_key_head_dim * cfg.linear_num_key_heads + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        st = []
        for lt in self.layer_types:
            if lt == "full_attention":
                st += [torch.zeros(1, cfg.num_key_value_heads, self.ctx, cfg.head_dim)] * 2
            else:
                st += [torch.zeros(1, conv_dim, cfg.linear_conv_kernel_dim - 1),
                       torch.zeros(1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)]
        return st


class PackedStep(nn.Module):
    """:class:`StaticStep` with the states packed per kind.

    - ``kv``: ``[n_attention, 2 * kv_heads, ctx, head_dim]``, K heads then V heads
    - ``conv``: ``[n_deltanet, conv_dim, kernel - 1]``
    - ``rec``: ``[n_deltanet, v_heads, k_dim, v_dim]``
    """

    INPUT_NAMES = ["token_embedding", "position_ids", "past_kv", "past_conv", "past_recurrent"]
    OUTPUT_NAMES = ["hidden_states", "present_kv", "present_conv", "present_recurrent"]

    def __init__(self, step: StaticStep):
        super().__init__()
        self.step = step
        self.types = step.layer_types

    def unpack(self, kv, conv, rec) -> list[torch.Tensor]:
        states, ia, il = [], 0, 0
        nkv = kv.shape[1] // 2
        for lt in self.types:
            if lt == "full_attention":
                states += [kv[ia : ia + 1, :nkv], kv[ia : ia + 1, nkv:]]
                ia += 1
            else:
                states += [conv[il : il + 1], rec[il : il + 1]]
                il += 1
        return states

    def pack(self, states) -> list[torch.Tensor]:
        kvs, convs, recs = [], [], []
        it = iter(states)
        for lt in self.types:
            a, b = next(it), next(it)
            if lt == "full_attention":
                kvs.append(torch.cat([a, b], 1))
            else:
                convs.append(a)
                recs.append(b)
        return [torch.cat(kvs, 0), torch.cat(convs, 0), torch.cat(recs, 0)]

    def forward(self, token_embedding, position_ids, kv, conv, rec):
        out = self.step(token_embedding, position_ids, *self.unpack(kv, conv, rec))
        return (out[0], *self.pack(out[1:]))
