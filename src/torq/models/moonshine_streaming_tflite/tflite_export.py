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


# Modules kept in float (not quantized) for the static decoder. `proj_out` is the
# vocab projection: quantizing it wrecks logit magnitudes (SNR << 0) and flips the
# greedy argmax, which then diverges the self-KV cache. See static_quant_plan.md §3.3.
STATIC_EXCLUDE_MODULES: tuple[str, ...] = ("proj_out",)


def _static_int8(wrapper: torch.nn.Module, sample_args: tuple, calib_shards: list[Path],
                 exclude_modules: tuple[str, ...] = STATIC_EXCLUDE_MODULES):
    """Apply STATIC (full-integer) PT2E quantization.

    Static mode => weights int8 (per-channel) AND activations int8 with ranges
    fixed at export time from a representative calibration corpus (recorded by
    ``calibration.py``). This removes per-op runtime range estimation, which is the
    decode-latency win we're after. ``exclude_modules`` are kept in float (per
    ``static_quant_plan.md`` §3.3). Returns (quantized_module, quant_config).

    Shards are loaded lazily (one decode step at a time) so the calibration corpus
    never sits in RAM all at once.
    """
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    # Exclude modules by leaving them in float. The set_module_name() *setter* asserts
    # config != None, but the annotate() path handles a None config correctly: it
    # annotates nothing for that name AND the global config's filter
    # (_get_not_module_type_or_name_filter) skips any name in module_name_config. So we
    # populate the dict directly to get a true float exclusion.
    for name in exclude_modules:
        quantizer.module_name_config[name] = None
    exported = torch.export.export(wrapper.eval(), sample_args).module()
    prepared = prepare_pt2e(exported, quantizer)
    for sh in calib_shards:  # observers collect activation ranges over real feeds
        with np.load(sh) as data:
            feeds = tuple(torch.from_numpy(data[k]) for k in sorted(data.keys()))
        prepared(*feeds)
        del feeds
    quantized = convert_pt2e(prepared, fold_quantize=False)
    return quantized, QuantConfig(pt2e_quantizer=quantizer)


def _load_calib_shards(calib_dir: str, component: str, limit: int | None = None) -> list[Path]:
    """List recorded calibration shard paths (``<calib_dir>/<component>/*.npz``); each
    shard is one decode step. Returns paths only — feeds are loaded lazily at use."""
    shard_dir = Path(calib_dir) / component
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(
            f"No calibration shards in {shard_dir}. Record them first with "
            f"`python -m ...moonshine_streaming_tflite.calibration --out-dir {calib_dir}`."
        )
    return shards[:limit] if limit else shards


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


def _convert(wrapper: torch.nn.Module, sample_args: tuple, out_path: Path,
             quant_mode: str = "none", calib_shards: list[Path] | None = None):
    if quant_mode == "static":
        logger.info("Quantizing (STATIC int8, %d calib steps) -> %s",
                    len(calib_shards or []), out_path.name)
        module, qcfg = _static_int8(wrapper, sample_args, calib_shards)
        edge = litert_torch.convert(module, sample_args, quant_config=qcfg)
    elif quant_mode == "dynamic":
        logger.info("Quantizing (weight-only dynamic int8) -> %s", out_path.name)
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
    quant_mode: str = "none",
    calib_dir: str | None = None,
    calib_limit: int | None = None,
    extract_embeddings: bool = True,
):
    suffix = {"none": "", "dynamic": "_int8", "static": "_int8_static"}[quant_mode]

    # decoder_kv is the static-quant target; static for the encoder is out of scope
    # (see static_quant_plan.md §6), so it falls back to weight-only dynamic int8.
    if quant_mode == "static":
        dec_shards = _load_calib_shards(calib_dir, "decoder_kv", calib_limit)
        fused_mode = "dynamic"
        logger.info("static mode: decoder_kv=static (%d calib steps), fused_encoder=%s",
                    len(dec_shards), fused_mode)
    else:
        dec_shards = None
        fused_mode = quant_mode

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
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        # 4D additive self-attention mask over the fixed max_tokens self-KV buffer.
        self_attn_bias = torch.zeros(1, 1, 1, max_tokens)
        # Host-side embedding lookup => float inputs_embeds; avoids an in-graph
        # EMBEDDING_LOOKUP op (which int8 quant cannot satisfy: requires zp==0).
        if extract_embeddings:
            first = torch.zeros(1, 1, hidden)
        else:
            first = torch.ones(1, 1, dtype=torch.int32)
        sample = (first, *self_kv, *cross_kv, cross_attn_bias, position_ids, self_attn_bias)
        _convert(decoder, sample, out_dir / f"decoder_kv{suffix}.tflite",
                 quant_mode=quant_mode, calib_shards=dec_shards)

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
        _convert(fused, sample, out_dir / f"fused_encoder{suffix}.tflite", quant_mode=fused_mode)

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
    p.add_argument("--quant-mode", choices=["none", "dynamic", "static"], default="none",
                   help="none=fp32; dynamic=weight-only int8 (per-channel dynamic-range); "
                        "static=full-integer int8 for decoder_kv (needs --calib-dir; "
                        "fused_encoder falls back to dynamic).")
    p.add_argument("--int8", action="store_true",
                   help="Deprecated alias for --quant-mode dynamic.")
    p.add_argument("--calib-dir", default=None,
                   help="Directory of recorded calibration shards (for --quant-mode static).")
    p.add_argument("--calib-limit", type=int, default=None,
                   help="Cap the number of calibration steps used (default: all).")
    p.add_argument("--in-graph-embeddings", action="store_true",
                   help="Keep embed_tokens in the decoder graph (token input) instead of "
                        "host-side inputs_embeds. Incompatible with quantization.")
    p.add_argument("--logging", default="INFO")
    args = p.parse_args()
    configure_logging(args.logging)

    quant_mode = args.quant_mode
    if args.int8 and quant_mode == "none":
        quant_mode = "dynamic"
    if quant_mode == "static" and not args.calib_dir:
        p.error("--quant-mode static requires --calib-dir")
    if quant_mode != "none" and args.in_graph_embeddings:
        p.error("--in-graph-embeddings is incompatible with quantization")

    export_tflite(
        model_size=args.model_size, hf_repo=args.hf_repo, models_dir=args.models_dir,
        chunk_len=args.chunk_len, max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec, out_dir=args.out_dir,
        components=tuple(args.components), quant_mode=quant_mode,
        calib_dir=args.calib_dir, calib_limit=args.calib_limit,
        extract_embeddings=not args.in_graph_embeddings,
    )


if __name__ == "__main__":
    main()
