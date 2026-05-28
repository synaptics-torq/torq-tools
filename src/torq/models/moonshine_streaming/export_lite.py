# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Lightweight Moonshine Streaming → dynamic ONNX export.

Produces the same four dynamic-shape ONNX models as the full ``export.py``
pipeline (preprocessor / encoder / decoder / decoder_with_past) plus the
decoder token embeddings and tokenizer, without depending on the ``torq``
package or any of its graph-edit / static-shape conversion utilities.

This is suitable for downstream use that does not require the static-shape
ONNX (mic demo, HF comparison, custom downstream tooling). For the full
static-shape pipeline used by the Torq compiler, use the package CLI:

    torq-export-model moonshine-streaming -s tiny

Dependencies:
    pip install numpy onnx onnxruntime torch transformers huggingface_hub

Usage (from the repo root, with no ``pip install -e .`` required):

    python src/torq/models/moonshine_streaming/export_lite.py -s tiny
    python src/torq/models/moonshine_streaming/export_lite.py -s small \\
        --output-dir custom/path
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

# Allow ``python path/to/export_lite.py`` (no package install) by making the
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


# ── ONNX exports ────────────────────────────────────────────────────────────

def export_preprocessor(model, output_path: Path):
    wrapper = PreprocessorWrapper(model).eval()
    dummy_audio = torch.randn(1, 32000)
    dummy_mask = torch.ones(1, 32000, dtype=torch.long)

    batch = torch.export.Dim("batch", min=1)
    audio_len = torch.export.Dim("audio_length", min=80, max=960000)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_values", "attention_mask"],
            output_names=["input_features", "padding_mask"],
            dynamic_shapes={
                "input_values": {0: batch, 1: audio_len},
                "attention_mask": {0: batch, 1: audio_len},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  preprocessor → {output_path}")


def export_encoder(model, output_path: Path, enc_hidden: int):
    wrapper = TransformerEncoderWrapper(model).eval()
    dummy_features = torch.randn(1, 100, enc_hidden)
    dummy_mask = torch.ones(1, 100, dtype=torch.bool)

    batch = torch.export.Dim("batch", min=1)
    seq_len = torch.export.Dim("seq_length", min=1, max=3000)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_features", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_shapes={
                "input_features": {0: batch, 1: seq_len},
                "attention_mask": {0: batch, 1: seq_len},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  encoder → {output_path}")


def export_decoder(model, output_path: Path, num_decoder_layers: int, enc_hidden: int):
    wrapper = DecoderWrapper(model, num_decoder_layers).eval()
    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, enc_hidden)
    dummy_enc_mask = torch.ones(1, 50, dtype=torch.long)

    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask),
            str(output_path),
            dynamo=True,
            input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            output_names=["last_hidden_state"] + kv_output_names(num_decoder_layers),
            dynamic_shapes={
                "decoder_input_ids": {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc_seq},
                "encoder_attention_mask": {0: batch, 1: enc_seq},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  decoder → {output_path}")


def export_decoder_with_past(
    model, output_path: Path, num_decoder_layers: int,
    num_heads: int, head_dim: int, enc_hidden: int,
):
    wrapper = DecoderWithPastWrapper(model, num_decoder_layers).eval()
    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, enc_hidden)
    dummy_enc_mask = torch.ones(1, 50, dtype=torch.long)

    B, H, HEAD = 1, num_heads, head_dim
    dummy_self_past = [(torch.randn(B, H, 5, HEAD), torch.randn(B, H, 5, HEAD))
                       for _ in range(num_decoder_layers)]
    dummy_cross_past = [(torch.randn(B, H, 50, HEAD), torch.randn(B, H, 50, HEAD))
                        for _ in range(num_decoder_layers)]
    flat_past = []
    for k, v in dummy_self_past:
        flat_past += [k, v]
    for k, v in dummy_cross_past:
        flat_past += [k, v]

    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)
    past_seq = torch.export.Dim("past_seq", min=1)

    n = num_decoder_layers
    flat_past_shapes = []
    for _ in range(2 * n):
        flat_past_shapes.append({0: batch, 2: past_seq})
    for _ in range(2 * n):
        flat_past_shapes.append({0: batch, 2: enc_seq})

    dyn_shapes = {
        "decoder_input_ids": {0: batch},
        "encoder_hidden_states": {0: batch, 1: enc_seq},
        "encoder_attention_mask": {0: batch, 1: enc_seq},
        "flat_past": tuple(flat_past_shapes),
    }

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
            dynamic_shapes=dyn_shapes,
        )
    _consolidate_onnx(output_path)
    print(f"  decoder_with_past → {output_path}")


def save_token_embeddings(model, output_path: Path):
    embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
    np.save(str(output_path), embeddings)
    print(
        f"  decoder_token_embeddings → {output_path}  "
        f"(shape {embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB)"
    )


# ── Validation ──────────────────────────────────────────────────────────────

def validate(model, output_dir: Path):
    """Compare split ONNX pipeline (preprocessor → encoder) against PyTorch."""
    import onnxruntime as ort

    full_wrapper = FullEncoderWrapper(model).eval()
    preproc_sess = ort.InferenceSession(str(output_dir / "preprocessor.onnx"))
    enc_sess = ort.InferenceSession(str(output_dir / "encoder.onnx"))

    for audio_len in (16000, 48000, 80000):
        dummy_audio = np.random.randn(1, audio_len).astype(np.float32)
        dummy_mask = np.ones((1, audio_len), dtype=np.int64)

        with torch.no_grad():
            pt_out = full_wrapper(
                torch.from_numpy(dummy_audio),
                torch.from_numpy(dummy_mask),
            ).numpy()

        preproc_outs = preproc_sess.run(None, {
            "input_values": dummy_audio,
            "attention_mask": dummy_mask,
        })
        ort_out = enc_sess.run(None, {
            "input_features": preproc_outs[0],
            "attention_mask": preproc_outs[1],
        })[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        duration_s = audio_len / 16000
        print(
            f"  {duration_s:.0f}s audio  shape {ort_out.shape}  "
            f"max diff: {max_diff:.6f}"
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
             "(default: models/<hf-repo>/export/onnx/float/dynamic)",
    )
    p.add_argument(
        "--weights-dir", default=None,
        help="Local weights cache "
             "(default: models/<hf-repo>/weights)",
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
        args.output_dir or f"models/{hf_repo}/export/onnx/float/dynamic"
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

    print("\nExporting dynamic ONNX models ...")
    export_preprocessor(model, output_dir / "preprocessor.onnx")
    export_encoder(model, output_dir / "encoder.onnx", enc_hidden)
    export_decoder(model, output_dir / "decoder.onnx", num_decoder_layers, enc_hidden)
    export_decoder_with_past(
        model, output_dir / "decoder_with_past.onnx",
        num_decoder_layers, num_heads, head_dim, enc_hidden,
    )
    save_token_embeddings(model, output_dir / "decoder_token_embeddings.npy")

    tok_src = weights_dir / "tokenizer.json"
    tok_dst = output_dir / "tokenizer.json"
    shutil.copy2(tok_src, tok_dst)
    print(f"  tokenizer → {tok_dst}")

    cfg_src = weights_dir / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, output_dir / "config.json")

    if not args.skip_validation:
        print("\nValidating split encoder pipeline ...")
        validate(model, output_dir)

    print(f"\nDone. Output in: {output_dir.resolve()}/")
    for f in sorted(output_dir.glob("*.onnx")) + sorted(output_dir.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:45s}  {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()
