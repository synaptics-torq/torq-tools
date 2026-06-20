# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export Moonshine Streaming 2-split wrappers to TFLite via litert-torch.

Step 1 of the tflite pipeline: produce fp32 `.tflite` files for `decoder_kv` and
`fused_encoder` from the existing PyTorch wrapper modules (reused unchanged from
`export.py`).  Quantization (weight-only int8) and VMFB lowering are later steps.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

import litert_torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

from ...utils.logging import configure_logging
from .export import DecoderKVWrapper, StatefulFusedEncoderWrapper

logger = logging.getLogger("moonshine-tflite-export")


def _weight_only_int8(wrapper: torch.nn.Module, sample_args: tuple):
    """Apply per-channel dynamic-range PT2E quantization.

    Dynamic mode => Linear/Conv weights stored int8, activations quantized
    dynamically at runtime (no calibration corpus). This is the "int8 weights,
    float activations" scheme. Returns (quantized_module, quant_config).
    """
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    exported = torch.export.export(wrapper.eval(), sample_args).module()
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*sample_args)  # calibration pass (needed for per-channel scales)
    quantized = convert_pt2e(prepared, fold_quantize=False)
    return quantized, QuantConfig(pt2e_quantizer=quantizer)


def _load_model(model_size: str, hf_repo: str | None, models_dir: Path):
    from transformers import MoonshineStreamingForConditionalGeneration

    hf_repo = hf_repo or f"UsefulSensors/moonshine-streaming-{model_size}"
    local_dir = models_dir / hf_repo / "weights" / model_size
    if not (local_dir / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        logger.info("Downloading %s -> %s", hf_repo, local_dir)
        snapshot_download(
            repo_id=hf_repo, local_dir=str(local_dir), local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(local_dir), torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager",
    ).eval()
    return model, local_dir


def _convert(wrapper: torch.nn.Module, sample_args: tuple, out_path: Path, quantize: bool = False):
    if quantize:
        logger.info("Quantizing (weight-only int8) + converting -> %s", out_path.name)
        module, qcfg = _weight_only_int8(wrapper, sample_args)
        edge = litert_torch.convert(module, sample_args, quant_config=qcfg)
    else:
        logger.info("Converting (fp32) -> %s", out_path.name)
        edge = litert_torch.convert(wrapper.eval(), sample_args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(out_path))
    logger.info("Wrote %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)


def export_tflite(
    model_size: str = "tiny",
    hf_repo: str | None = None,
    models_dir: str = "models",
    chunk_len: int = 1280,
    max_audio_s: int = 5,
    max_tok_per_s: int = 6,
    out_dir: str | None = None,
    components: tuple[str, ...] = ("decoder_kv", "fused_encoder"),
    quantize: bool = False,
    extract_embeddings: bool = True,
):
    suffix = "_int8" if quantize else ""
    models_dir = Path(models_dir)
    config = AutoConfig.from_pretrained(hf_repo or f"UsefulSensors/moonshine-streaming-{model_size}")
    hidden = int(config.hidden_size)
    dec_heads = getattr(config, "num_attention_heads", 8)
    num_kv_heads = getattr(config, "num_key_value_heads", 8)
    head_dim = hidden // dec_heads
    n_layers = getattr(config, "decoder_num_hidden_layers",
                       getattr(config, "num_hidden_layers", 6))
    num_samples = max_audio_s * 16_000
    max_tokens = max_audio_s * max_tok_per_s
    max_memory_len = num_samples // 320

    out_dir = Path(out_dir) if out_dir else (
        models_dir / "export" / "tflite" / "2split_streaming_static"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model, local_dir = _load_model(model_size, hf_repo, models_dir)

    # Save host-side tables + configs alongside, mirroring the ONNX exporter.
    np.save(out_dir / "decoder_token_embeddings.npy",
            model.model.decoder.embed_tokens.weight.detach().cpu().numpy())
    np.save(out_dir / "adapter_pos_emb.npy",
            model.model.decoder.pos_emb.weight.detach().cpu().numpy())
    import shutil
    for fname in ("tokenizer.json", "config.json"):
        src = local_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    # ── decoder_kv ────────────────────────────────────────────────────────────
    if "decoder_kv" in components:
        decoder = DecoderKVWrapper(
            model, extract_embeddings=extract_embeddings, static_self_cache=True,
        ).eval()
        self_kv = [torch.zeros(1, num_kv_heads, max_tokens, head_dim) for _ in range(n_layers * 2)]
        cross_kv = [torch.zeros(1, num_kv_heads, max_memory_len, head_dim) for _ in range(n_layers * 2)]
        cross_attn_bias = torch.zeros(1, num_kv_heads, 1, max_memory_len)
        position_ids = torch.tensor([[0]], dtype=torch.long)
        # 4D additive self-attention mask over the fixed max_tokens self-KV buffer.
        self_attn_bias = torch.zeros(1, 1, 1, max_tokens)
        # Host-side embedding lookup => float inputs_embeds; avoids an in-graph
        # EMBEDDING_LOOKUP op (which int8 quant cannot satisfy: requires zp==0).
        if extract_embeddings:
            first = torch.zeros(1, 1, hidden)
        else:
            first = torch.ones(1, 1, dtype=torch.long)
        sample = (first, *self_kv, *cross_kv, cross_attn_bias, position_ids, self_attn_bias)
        _convert(decoder, sample, out_dir / f"decoder_kv{suffix}.tflite", quantize=quantize)

    # ── fused_encoder ─────────────────────────────────────────────────────────
    F = total_la = None
    if "fused_encoder" in components:
        fused = StatefulFusedEncoderWrapper(model, chunk_len).eval()
        F, total_la = fused.F, fused.total_la
        embedder = model.model.encoder.embedder
        enc_hidden = int(embedder.linear.out_features)
        c1 = int(embedder.conv1.out_channels)
        left_ctxs = fused.encoder._left_ctx
        sample = (
            torch.zeros(1, chunk_len),
            torch.zeros(1, enc_hidden, 4),
            torch.zeros(1, c1, 4),
            torch.zeros(1, total_la, enc_hidden),
            torch.zeros(1, F, enc_hidden),
            *[torch.zeros(1, lc, enc_hidden) for lc in left_ctxs],
        )
        _convert(fused, sample, out_dir / f"fused_encoder{suffix}.tflite", quantize=quantize)

    cfg = {
        "chunk_len": chunk_len, "feature_stride": F, "total_lookahead": total_la,
        "warmup_chunks": ((total_la + F - 1) // F) if F else None,
        "max_tokens": max_tokens, "max_memory_len": max_memory_len,
        "extract_embeddings": extract_embeddings,
    }
    with open(out_dir / "streaming_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Done. Outputs in %s", out_dir)


def main():
    p = argparse.ArgumentParser(description="Export Moonshine Streaming wrappers to TFLite")
    p.add_argument("-s", "--model-size", default="tiny", choices=["tiny", "small"])
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--chunk-len", type=int, default=1280)
    p.add_argument("--input-seconds", type=int, default=5)
    p.add_argument("--tokens-per-sec", type=int, default=6)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--components", nargs="+", default=["decoder_kv", "fused_encoder"],
                   choices=["decoder_kv", "fused_encoder"])
    p.add_argument("--int8", action="store_true",
                   help="Weight-only int8 quantization (per-channel dynamic-range)")
    p.add_argument("--in-graph-embeddings", action="store_true",
                   help="Keep embed_tokens in the decoder graph (token input) instead of "
                        "host-side inputs_embeds. Incompatible with --int8.")
    p.add_argument("--logging", default="INFO")
    args = p.parse_args()
    configure_logging(args.logging)
    export_tflite(
        model_size=args.model_size, hf_repo=args.hf_repo, models_dir=args.models_dir,
        chunk_len=args.chunk_len, max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec, out_dir=args.out_dir,
        components=tuple(args.components), quantize=args.int8,
        extract_embeddings=not args.in_graph_embeddings,
    )


if __name__ == "__main__":
    main()
