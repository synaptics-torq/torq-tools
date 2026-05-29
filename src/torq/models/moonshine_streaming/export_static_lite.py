# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Lightweight Moonshine Streaming → static ONNX export.

Produces four static-shape ONNX models:
    - preprocessor.onnx
    - encoder.onnx
    - decoder.onnx
    - decoder_with_past.onnx

plus:
    - decoder_token_embeddings.npy
    - tokenizer.json
    - config.json (if present)

This version preserves the same 4-stage split used by export_lite.py,
but removes dynamic shapes and instead exports a single fixed runtime
contract determined by:
    - batch size
    - audio chunk length
    - decoder self-attention past length

The encoder sequence length is derived automatically from the chosen
audio length by running the preprocessor wrapper once.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

# Allow ``python path/to/export_static.py`` (no package install) by making the
# script's directory importable so ``_wrappers`` resolves as a sibling.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _wrappers import (
        DecoderWithPastWrapper,
        DecoderWrapper,
        FullEncoderWrapper,
        PreprocessorWrapper,
        TransformerEncoderWrapper,
        kv_input_names,
        kv_output_names,
    )
else:
    from ._wrappers import (
        DecoderWithPastWrapper,
        DecoderWrapper,
        FullEncoderWrapper,
        PreprocessorWrapper,
        TransformerEncoderWrapper,
        kv_input_names,
        kv_output_names,
    )


# Safety net for older torch where asinh has no TorchScript symbolic.
def _asinh_symbolic(g, input):
    return g.op("Asinh", input)


torch.onnx.register_custom_op_symbolic("aten::asinh", _asinh_symbolic, opset_version=18)


# ── Download ────────────────────────────────────────────────────────────────

def download_model(model_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} → {local_dir.resolve()} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    return local_dir


# ── External-data consolidation ─────────────────────────────────────────────

def _consolidate_onnx(output_path: Path):
    """Embed external tensor data into the .onnx file and remove .data files."""
    from onnx.external_data_helper import convert_model_from_external_data

    data_path = Path(str(output_path) + ".data")
    if not data_path.exists():
        return

    model = onnx.load(str(output_path), load_external_data=True)
    convert_model_from_external_data(model)
    onnx.save(model, str(output_path))
    data_path.unlink()


# ── Shape helpers ───────────────────────────────────────────────────────────

def infer_encoder_seq_len(model, batch_size: int, audio_length: int) -> int:
    """
    Runs the preprocessor wrapper once on fixed-shape dummy audio to determine
    the encoder feature sequence length that the rest of the static pipeline
    must use.
    """
    wrapper = PreprocessorWrapper(model).eval()
    dummy_audio = torch.randn(batch_size, audio_length, dtype=torch.float32)
    dummy_mask = torch.ones(batch_size, audio_length, dtype=torch.long)

    with torch.no_grad():
        input_features, _padding_mask = wrapper(dummy_audio, dummy_mask)

    if input_features.ndim != 3:
        raise RuntimeError(
            f"Expected preprocessor output shape [B, T, H], got {tuple(input_features.shape)}"
        )

    enc_seq_len = int(input_features.shape[1])
    return enc_seq_len


# ── ONNX exports ────────────────────────────────────────────────────────────

def export_preprocessor(model, output_path: Path, batch_size: int, audio_length: int):
    wrapper = PreprocessorWrapper(model).eval()
    dummy_audio = torch.randn(batch_size, audio_length, dtype=torch.float32)
    dummy_mask = torch.ones(batch_size, audio_length, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_values", "attention_mask"],
            output_names=["input_features", "padding_mask"],
        )
    _consolidate_onnx(output_path)
    print(f"  preprocessor → {output_path}  [input_values: ({batch_size}, {audio_length})]")


def export_encoder(model, output_path: Path, batch_size: int, enc_seq_len: int, enc_hidden: int):
    wrapper = TransformerEncoderWrapper(model).eval()
    dummy_features = torch.randn(batch_size, enc_seq_len, enc_hidden, dtype=torch.float32)
    dummy_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.bool)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_features", "attention_mask"],
            output_names=["last_hidden_state"],
        )
    _consolidate_onnx(output_path)
    print(
        f"  encoder → {output_path}  "
        f"[input_features: ({batch_size}, {enc_seq_len}, {enc_hidden})]"
    )


def export_decoder(
    model,
    output_path: Path,
    batch_size: int,
    num_decoder_layers: int,
    enc_seq_len: int,
    enc_hidden: int,
):
    wrapper = DecoderWrapper(model, num_decoder_layers).eval()
    dummy_dec_ids = torch.ones(batch_size, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(batch_size, enc_seq_len, enc_hidden, dtype=torch.float32)
    dummy_enc_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask),
            str(output_path),
            dynamo=True,
            input_names=[
                "decoder_input_ids",
                "encoder_hidden_states",
                "encoder_attention_mask",
            ],
            output_names=["last_hidden_state"] + kv_output_names(num_decoder_layers),
        )
    _consolidate_onnx(output_path)
    print(
        f"  decoder → {output_path}  "
        f"[decoder_input_ids: ({batch_size}, 1), "
        f"encoder_hidden_states: ({batch_size}, {enc_seq_len}, {enc_hidden})]"
    )


def export_decoder_with_past(
    model,
    output_path: Path,
    batch_size: int,
    num_decoder_layers: int,
    num_heads: int,
    head_dim: int,
    enc_seq_len: int,
    past_seq_len: int,
    enc_hidden: int,
):
    wrapper = DecoderWithPastWrapper(model, num_decoder_layers).eval()

    dummy_dec_ids = torch.ones(batch_size, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(batch_size, enc_seq_len, enc_hidden, dtype=torch.float32)
    dummy_enc_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.long)

    # Fixed self-attention cache
    dummy_self_past = [
        (
            torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32),
            torch.randn(batch_size, num_heads, past_seq_len, head_dim, dtype=torch.float32),
        )
        for _ in range(num_decoder_layers)
    ]

    # Fixed cross-attention cache
    dummy_cross_past = [
        (
            torch.randn(batch_size, num_heads, enc_seq_len, head_dim, dtype=torch.float32),
            torch.randn(batch_size, num_heads, enc_seq_len, head_dim, dtype=torch.float32),
        )
        for _ in range(num_decoder_layers)
    ]

    flat_past = []
    for k, v in dummy_self_past:
        flat_past += [k, v]
    for k, v in dummy_cross_past:
        flat_past += [k, v]

    input_names = (
        ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"]
        + kv_input_names(num_decoder_layers)
    )
    output_names = ["last_hidden_state"] + kv_output_names(num_decoder_layers)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask, *flat_past),
            str(output_path),
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
        )
    _consolidate_onnx(output_path)
    print(
        f"  decoder_with_past → {output_path}  "
        f"[self_past_seq: {past_seq_len}, cross_past_seq: {enc_seq_len}]"
    )


def save_token_embeddings(model, output_path: Path):
    embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
    np.save(str(output_path), embeddings)
    print(
        f"  decoder_token_embeddings → {output_path}  "
        f"(shape {embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB)"
    )


# ── Validation ──────────────────────────────────────────────────────────────

def validate(model, output_dir: Path, batch_size: int, audio_length: int):
    """
    Compare split ONNX pipeline (preprocessor → encoder) against PyTorch using
    the single fixed static input shape.

    We vary only the number of valid samples via masking inside the same static
    [batch_size, audio_length] tensor shape.
    """
    import onnxruntime as ort

    full_wrapper = FullEncoderWrapper(model).eval()
    preproc_sess = ort.InferenceSession(str(output_dir / "preprocessor.onnx"))
    enc_sess = ort.InferenceSession(str(output_dir / "encoder.onnx"))

    valid_lengths = sorted(set([
        max(80, audio_length // 4),
        max(80, audio_length // 2),
        audio_length,
    ]))

    for valid_len in valid_lengths:
        dummy_audio = np.zeros((batch_size, audio_length), dtype=np.float32)
        dummy_mask = np.zeros((batch_size, audio_length), dtype=np.int64)

        dummy_audio[:, :valid_len] = np.random.randn(batch_size, valid_len).astype(np.float32)
        dummy_mask[:, :valid_len] = 1

        with torch.no_grad():
            pt_out = full_wrapper(
                torch.from_numpy(dummy_audio),
                torch.from_numpy(dummy_mask),
            ).numpy()

        preproc_outs = preproc_sess.run(
            None,
            {
                "input_values": dummy_audio,
                "attention_mask": dummy_mask,
            },
        )
        ort_out = enc_sess.run(
            None,
            {
                "input_features": preproc_outs[0],
                "attention_mask": preproc_outs[1],
            },
        )[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        duration_s = valid_len / 16000.0
        print(
            f"  valid {valid_len}/{audio_length} samples "
            f"({duration_s:.2f}s)  shape {ort_out.shape}  max diff: {max_diff:.6f}"
        )
        assert max_diff < 1e-4, f"Validation failed: max_diff={max_diff}"

    print("  ALL PASSED")


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "-s", "--model-size", choices=["tiny", "small"], default="tiny",
        help="Moonshine streaming variant (default: %(default)s)",
    )
    p.add_argument(
        "--hf-repo", default=None,
        help="HuggingFace repo (default: UsefulSensors/moonshine-streaming-{size})",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory "
             "(default: models/<hf-repo>/export/onnx/float/static)",
    )
    p.add_argument(
        "--weights-dir", default=None,
        help="Local weights cache "
             "(default: models/<hf-repo>/weights)",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Fixed batch size for static export (default: %(default)s)",
    )
    p.add_argument(
        "--audio-length", type=int, default=32000,
        help="Fixed raw audio length in samples for static export (default: %(default)s)",
    )
    p.add_argument(
        "--past-seq-len", type=int, default=64,
        help="Fixed self-attention cache length for decoder_with_past (default: %(default)s)",
    )
    p.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the post-export PyTorch-vs-ONNX numerical check",
    )
    return p.parse_args()


def main():
    args = parse_args()

    from transformers import (
        AutoConfig,
        MoonshineStreamingForConditionalGeneration,
    )

    hf_repo = args.hf_repo or f"UsefulSensors/moonshine-streaming-{args.model_size}"
    output_dir = Path(
        args.output_dir or f"models/{hf_repo}/export/onnx/float/static"
    )
    weights_dir = Path(args.weights_dir or f"models/{hf_repo}/weights")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not (weights_dir / "model.safetensors").exists():
        download_model(hf_repo, weights_dir)
    else:
        print(f"Using cached weights in {weights_dir.resolve()}")

    config = AutoConfig.from_pretrained(hf_repo)
    num_decoder_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    enc_hidden = config.encoder_hidden_size

    print(f"\nLoading model from {weights_dir} ...")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    model.config.use_cache = True

    # Derive the fixed encoder sequence length from the chosen static audio length.
    enc_seq_len = infer_encoder_seq_len(
        model=model,
        batch_size=args.batch_size,
        audio_length=args.audio_length,
    )

    print("\nStatic export configuration:")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  audio_length = {args.audio_length}")
    print(f"  enc_seq_len  = {enc_seq_len}")
    print(f"  past_seq_len = {args.past_seq_len}")

    print("\nExporting static ONNX models ...")
    export_preprocessor(
        model,
        output_dir / "preprocessor.onnx",
        batch_size=args.batch_size,
        audio_length=args.audio_length,
    )
    export_encoder(
        model,
        output_dir / "encoder.onnx",
        batch_size=args.batch_size,
        enc_seq_len=enc_seq_len,
        enc_hidden=enc_hidden,
    )
    export_decoder(
        model,
        output_dir / "decoder.onnx",
        batch_size=args.batch_size,
        num_decoder_layers=num_decoder_layers,
        enc_seq_len=enc_seq_len,
        enc_hidden=enc_hidden,
    )
    export_decoder_with_past(
        model,
        output_dir / "decoder_with_past.onnx",
        batch_size=args.batch_size,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        enc_seq_len=enc_seq_len,
        past_seq_len=args.past_seq_len,
        enc_hidden=enc_hidden,
    )
    save_token_embeddings(model, output_dir / "decoder_token_embeddings.npy")

    tok_src = weights_dir / "tokenizer.json"
    tok_dst = output_dir / "tokenizer.json"
    shutil.copy2(tok_src, tok_dst)
    print(f"  tokenizer → {tok_dst}")

    cfg_src = weights_dir / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, output_dir / "config.json")
        print(f"  config → {output_dir / 'config.json'}")

    if not args.skip_validation:
        print("\nValidating static split encoder pipeline ...")
        validate(
            model=model,
            output_dir=output_dir,
            batch_size=args.batch_size,
            audio_length=args.audio_length,
        )

    print(f"\nDone. Output in: {output_dir.resolve()}/")
    for f in sorted(output_dir.glob("*.onnx")) + sorted(output_dir.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:45s}  {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()