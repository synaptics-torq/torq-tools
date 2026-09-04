# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Export LFM2.5-Encoder (Liquid bidirectional encoder) to Torq.

The encoder is the LFM2 hybrid backbone (gated short-conv + GQA blocks) with
full bidirectional attention and a masked-LM head whose weight is tied to the
token embeddings.  For Torq we export the *body* only, with a static sequence
length:

    token_embedding [1, S, 1024] fp32   (host does the embedding LUT lookup)
    attention_mask  [1, S]       fp32   (1.0 = real token, 0.0 = padding)
      -> hidden     [1, S, 1024] fp32   (final-norm output)

MLM logits at any position are then `hidden[pos] @ token_embeddings.T` on the
host — for zero-shot classification/routing only a handful of positions and
vocab rows are needed, so the 65536-row lm_head matmul stays off the chip.

Export-time model patches (see the classes below for details):
  * conv-block pad masking — upstream `apply_mask_to_padding_states` no-ops
    for batch-1 / 4D masks, letting pad garbage leak through the centered
    depthwise convs; each Lfm2ShortConv input is explicitly re-masked.
  * static additive attention mask — replaces the traced ScatterND/Where mask
    builder with `(mask - 1) * 1e9` broadcast to [1, 1, 1, S].
  * constant rotary — cos/sin for positions 0..S-1 are baked as constants.

The traced graph is then cleaned with onnxsim (folds the remaining traced
shape-math chains) and converted fp32 -> bf16 (+ int64 -> int32) with the
standard torq convert_dtype tooling.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Final

import numpy as np

from torq.utils.logging import configure_logging

from ...utils.compile import export_torq
from .export import LIQUID_TORQ_FLAGS

import logging

logger = logging.getLogger(__name__)


HF_REPO_ENCODER: Final[dict[str, str]] = {
    "230m": "LiquidAI/LFM2.5-Encoder-230M",
    "350m": "LiquidAI/LFM2.5-Encoder-350M",
}

# Host-side assets copied next to the vmfb for the runtime demo.
_TOKENIZER_ASSETS: Final[tuple[str, ...]] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _load_encoder(model_size: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    repo = HF_REPO_ENCODER[model_size]
    logger.info("Loading %s ...", repo)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    mlm = AutoModelForMaskedLM.from_pretrained(
        repo, trust_remote_code=True, dtype=torch.float32).eval()
    try:
        mlm.set_attn_implementation("eager")
    except Exception:
        mlm.config._attn_implementation = "eager"
    return tok, mlm


def _install_static_mask_builder():
    """Replace the repo's traced 4D-mask builder with a static-shape one."""
    import torch
    from transformers.models.lfm2 import modeling_lfm2 as _lfm2_mod

    def _simple_bidirectional_mask(config, input_embeds=None,
                                   attention_mask=None, **kwargs):
        if input_embeds is None:
            input_embeds = kwargs.get("inputs_embeds")
        if attention_mask is None:
            b, s = input_embeds.shape[:2]
            return torch.zeros(b, 1, 1, s, dtype=input_embeds.dtype,
                               device=input_embeds.device)
        m = attention_mask.to(input_embeds.dtype)
        return ((m - 1.0) * 1e9)[:, None, None, :]

    _lfm2_mod.create_causal_mask = _simple_bidirectional_mask


def _make_const_rotary(rot, seq_len):
    """Bake rotary cos/sin for positions 0..S-1 as constants."""
    import torch

    class ConstRotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pos = torch.arange(seq_len, dtype=torch.int64)[None]
            x = torch.zeros(1, seq_len, 1, dtype=torch.float32)
            with torch.no_grad():
                cos, sin = rot(x, pos)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, x, position_ids):
            n = x.shape[1]
            if n == self.cos.shape[1]:
                return self.cos, self.sin
            return self.cos[:, :n], self.sin[:, :n]

    return ConstRotary()


def _install_shift_mul_add_conv():
    """Replace the depthwise short-conv with a shift+mul+add chain.

    The centered depthwise Conv1D (k=3, groups=C) traces to a grouped Conv
    the SL2610 depthwise path has historically struggled with (see the LLM
    exporter's Conv1D→MatMul replacement).  For the full-sequence encoder the
    depthwise conv is just, per channel c:

        y[c, t] = sum_j w[c, j] * x_padded[c, t + j]

    i.e. k shifted elementwise mul-adds — Pad/Slice/Mul/Add ops that lower to
    plain NSS elementwise kernels.  Numerically this matches F.conv1d up to
    fp add ordering.
    """
    import torch.nn.functional as F
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    def slow_forward(self, hidden_states, past_key_values=None,
                     cache_position=None, attention_mask=None, **kwargs):
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x

        w = self.conv.weight  # [C, 1, k]
        k = w.shape[-1]
        pad = k // 2
        T = Bx.shape[-1]
        xp = F.pad(Bx, (pad, pad))
        y = w[:, 0, 0][None, :, None] * xp[..., 0:T]
        for j in range(1, k):
            y = y + w[:, 0, j][None, :, None] * xp[..., j:j + T]
        if self.conv.bias is not None:
            y = y + self.conv.bias[None, :, None]

        y = C * y
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)

    Lfm2ShortConv.slow_forward = slow_forward


def _make_body_wrapper(body):
    """Wrap the encoder body; re-mask each short-conv input.

    Upstream `apply_mask_to_padding_states` requires a 2D mask with batch>1
    (the layers receive the 4D additive mask with batch 1), so it never fires
    here and pad-position garbage would leak through the centered depthwise
    convs into real positions.  Zeroing the conv input at pad positions
    restores the training-time (FA2 unpadded) behavior.
    """
    import torch

    class EncoderBodyWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            # the mask holder lives on the body module so re-creating the
            # wrapper (e.g. for another seq_len) reuses the same conv wraps
            if not hasattr(m, "_torq_mask2d"):
                m._torq_mask2d = [None]
            self._mask2d = m._torq_mask2d
            for layer in m.layers:
                if not layer.is_attention_layer and \
                        not getattr(layer.conv, "_torq_mask_wrapped", False):
                    layer.conv.forward = self._wrap_conv(
                        layer.conv.forward, m._torq_mask2d)
                    layer.conv._torq_mask_wrapped = True

        @staticmethod
        def _wrap_conv(orig_forward, mask2d):
            def fwd(hidden_states, *args, **kwargs):
                mask = mask2d[0]
                if mask is not None:
                    hidden_states = hidden_states * mask[:, :, None]
                return orig_forward(hidden_states, *args, **kwargs)
            return fwd

        def forward(self, token_embedding, attention_mask):
            self._mask2d[0] = attention_mask.to(token_embedding.dtype)
            out = self.m(inputs_embeds=token_embedding,
                         attention_mask=attention_mask,
                         use_cache=False, return_dict=True)
            self._mask2d[0] = None
            return out.last_hidden_state

    return EncoderBodyWrapper(body)


def _simplify_onnx(path: Path):
    import onnx
    import onnxsim

    model = onnx.load(path)
    model, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim failed to simplify {path}")
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    # torch names the graph "main_graph"; the demo runners expect the vmfb
    # entrypoint (named after the graph) to be "main"
    model.graph.name = "main"
    onnx.save(model, path)


def _export_body_onnx(wrapper, seq_len: int, out_path: Path):
    import torch

    embeds = torch.zeros(1, seq_len, wrapper.m.config.hidden_size,
                         dtype=torch.float32)
    mask = torch.ones(1, seq_len, dtype=torch.float32)
    # trace with a partially-padded mask so the masked branches are taken
    mask[0, seq_len // 2:] = 0.0
    torch.onnx.export(
        wrapper, (embeds, mask), str(out_path),
        input_names=["token_embedding", "attention_mask"],
        output_names=["hidden"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    _simplify_onnx(out_path)
    logger.info("Exported %s (%.1f MB)", out_path,
                os.path.getsize(out_path) / 1e6)


def _validate_body_onnx(onnx_path: Path, tok, mlm, wrapper, seq_len: int):
    """ORT-vs-torch and padded-vs-unpadded checks on a fill-mask example."""
    import onnxruntime as ort
    import torch

    body = wrapper.m
    text = f"The capital of France is {tok.mask_token}."
    enc = tok(text, return_tensors="pt")
    n = enc["input_ids"].shape[1]
    if n > seq_len:
        raise ValueError("validation text longer than seq_len")
    ids = torch.zeros(1, seq_len, dtype=torch.int64)
    ids[0, :n] = enc["input_ids"][0]
    am = torch.zeros(1, seq_len, dtype=torch.float32)
    am[0, :n] = 1.0

    E = body.embed_tokens.weight.detach()
    embeds = E[ids]
    with torch.no_grad():
        ref = wrapper(embeds, am)
        ref_unpadded = body(input_ids=enc["input_ids"],
                            return_dict=True).last_hidden_state

    cos = torch.nn.functional.cosine_similarity(
        ref[0, :n], ref_unpadded[0], dim=-1).min()
    if cos < 0.9999:
        raise RuntimeError(
            f"padded-vs-unpadded cosine {cos:.6f} < 0.9999 — pad masking "
            "is leaking; check the conv-mask patch")

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"token_embedding": embeds.numpy(),
                          "attention_mask": am.numpy()})[0]
    err = float(np.abs(out - ref.numpy()).max())
    if err > 1e-2:
        raise RuntimeError(f"ORT-vs-torch max abs diff {err} too large")

    pos = int((ids[0] == tok.mask_token_id).nonzero()[0])
    logits = out[0, pos] @ E.numpy().T
    top = [tok.decode([t]).strip() for t in np.argsort(logits)[::-1][:5]]
    logger.info("validation ok: ORT max diff %.2e, min cosine %.6f, "
                "fill-mask top5 %s", err, cos, top)
    if top[0] != "Paris":
        logger.warning("fill-mask top-1 is %r (expected 'Paris')", top[0])


def _save_assets(tok, mlm, assets_dir: Path):
    import ml_dtypes

    assets_dir.mkdir(parents=True, exist_ok=True)
    E = mlm.lfm2.embed_tokens.weight.detach().numpy()
    np.save(assets_dir / "token_embeddings.npy",
            E.astype(ml_dtypes.bfloat16))
    repo_dir = Path(tok.name_or_path)
    if not repo_dir.is_dir():
        from huggingface_hub import snapshot_download
        repo_dir = Path(snapshot_download(
            tok.name_or_path, allow_patterns=list(_TOKENIZER_ASSETS)))
    for name in _TOKENIZER_ASSETS:
        src = repo_dir / name
        if src.exists():
            shutil.copy2(src, assets_dir / name)
    logger.info("Saved token embeddings + tokenizer assets to %s", assets_dir)


def export_liquid_encoder_from_args(args: argparse.Namespace):
    configure_logging(args.logging)

    from ...tools.convert_dtype.onnx import convert_model

    base = Path(args.models_dir) / f"liquid-encoder-{args.model_size}"
    fp32_dir = base / "export/onnx/fp32/static"
    bf16_dir = base / "export/onnx/bf16/static"
    iree_dir = base / "export/iree/bf16/static"
    assets_dir = base / "export/assets"
    for d in (fp32_dir, bf16_dir, iree_dir):
        d.mkdir(parents=True, exist_ok=True)

    tok, mlm = _load_encoder(args.model_size)
    _install_static_mask_builder()
    if not args.keep_conv1d:
        _install_shift_mul_add_conv()
    _save_assets(tok, mlm, assets_dir)
    orig_rotary = mlm.lfm2.pos_emb

    for seq_len in args.seq_len:
        name = f"body_s{seq_len}" + ("_conv" if args.keep_conv1d else "")
        fp32_path = fp32_dir / f"{name}.onnx"
        bf16_path = bf16_dir / f"{name}.onnx"

        mlm.lfm2.pos_emb = _make_const_rotary(orig_rotary, seq_len)
        wrapper = _make_body_wrapper(mlm.lfm2)

        _export_body_onnx(wrapper, seq_len, fp32_path)
        if not args.skip_validation:
            _validate_body_onnx(fp32_path, tok, mlm, wrapper, seq_len)

        logger.info("Converting %s to bf16 ...", fp32_path)
        convert_model(fp32_path, bf16_path, "bf16", convert_io=True)
        convert_model(bf16_path, bf16_path, "int32", convert_io=True)

        if args.skip_torq:
            continue
        logger.info("Compiling %s to Torq ...", bf16_path)
        export_torq(
            bf16_path,
            iree_dir,
            compiler_args=list(LIQUID_TORQ_FLAGS) + (args.compile_flags or []),
            local_compile=args.local_compile,
            use_binary=args.use_binary,
            compiler_path=args.compiler_path,
            opset=args.opset,
        )

    # write a small manifest the demo runner can read
    manifest = {
        "model": HF_REPO_ENCODER[args.model_size],
        "hidden_size": mlm.config.hidden_size,
        "seq_lens": args.seq_len,
        "mask_token_id": tok.mask_token_id,
        "pad_token_id": tok.pad_token_id,
        "inputs": ["token_embedding", "attention_mask"],
        "outputs": ["hidden"],
    }
    with open(assets_dir / "encoder_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Done.")


def main():
    from . import add_liquid_encoder_export_args

    parser = argparse.ArgumentParser(
        description="Export LFM2.5-Encoder (Liquid) to Torq")
    add_liquid_encoder_export_args(parser)
    export_liquid_encoder_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
