# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Export Moonshine Streaming models to ONNX.

Produces dynamic-shape ONNX models (split encoder architecture):
  preprocessor.onnx              — CNN embedder (audio → features)
  encoder.onnx                   — transformer layers (features → hidden states)
  decoder.onnx                   — first decode step (no KV cache)
  decoder_with_past.onnx         — subsequent decode steps (with KV cache)
  decoder_token_embeddings.npy   — vocab embedding matrix for external logit computation
  tokenizer.json                 — tokenizer for inference

The decoders output last_hidden_state (not logits). Logits are computed
externally: logits = last_hidden_state @ decoder_token_embeddings.T

Usage via CLI:
    torq-export-model moonshine-streaming -s tiny --skip-iree --extract-embeddings --split-encoder
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch

from torq.utils.logging import configure_logging

from ._wrappers import (
    FullEncoderWrapper,
    PreprocessorWrapper,
    TransformerEncoderWrapper,
    DecoderWrapper,
    DecoderWithPastWrapper,
    kv_output_names,
    kv_input_names,
)

logger = logging.getLogger(__name__)


# ── Model download ───────────────────────────────────────────────────────────

def download_model(model_id: str, local_dir: Path) -> Path:
    """Download model files from HuggingFace Hub into local_dir."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s ...", model_id, local_dir.resolve())
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    logger.info("Download complete.")
    return local_dir


# ── ONNX consolidation ──────────────────────────────────────────────────────

def _consolidate_onnx(output_path: Path):
    """Embed external tensor data into the .onnx file and remove .data files.

    The dynamo exporter sometimes creates external .onnx.data files for weights.
    This merges everything into a single self-contained .onnx protobuf.
    """
    from onnx.external_data_helper import convert_model_from_external_data

    data_path = Path(str(output_path) + ".data")
    if not data_path.exists():
        return

    model = onnx.load(str(output_path), load_external_data=True)
    convert_model_from_external_data(model)
    onnx.save(model, str(output_path))
    data_path.unlink()


# ── Export functions ─────────────────────────────────────────────────────────

# Register asinh symbolic for TorchScript compatibility (safety net for older torch)
def _asinh_symbolic(g, input):
    return g.op("Asinh", input)
torch.onnx.register_custom_op_symbolic("aten::asinh", _asinh_symbolic, opset_version=18)


def export_preprocessor(model, output_path: Path):
    """Export CNN embedder: raw audio → feature sequence + frame-level mask."""
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
    logger.info("preprocessor → %s", output_path)


def export_encoder(model, output_path: Path, enc_hidden: int):
    """Export transformer encoder layers: embedder features + mask → hidden states."""
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
    logger.info("encoder → %s", output_path)


def export_decoder(model, output_path: Path, num_decoder_layers: int, enc_hidden: int):
    """Export first decode step (no KV cache)."""
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
    logger.info("decoder → %s", output_path)


def export_decoder_with_past(
    model, output_path: Path, num_decoder_layers: int,
    num_heads: int, head_dim: int, enc_hidden: int,
):
    """Export cached decode step (with KV cache)."""
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
    for i in range(2 * n):
        flat_past_shapes.append({0: batch, 2: past_seq})
    for i in range(2 * n):
        flat_past_shapes.append({0: batch, 2: enc_seq})

    dyn_shapes = {
        "decoder_input_ids": {0: batch},
        "encoder_hidden_states": {0: batch, 1: enc_seq},
        "encoder_attention_mask": {0: batch, 1: enc_seq},
        "flat_past": tuple(flat_past_shapes),
    }

    input_names = ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"] + kv_input_names(num_decoder_layers)
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
    logger.info("decoder_with_past → %s", output_path)


def save_token_embeddings(model, output_path: Path):
    """Extract decoder token embeddings (tied to proj_out) and save as .npy."""
    embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
    np.save(str(output_path), embeddings)
    logger.info(
        "decoder_token_embeddings → %s  (shape %s, %.1f MB)",
        output_path, embeddings.shape, embeddings.nbytes / 1e6,
    )


# ── Validation ───────────────────────────────────────────────────────────────

def validate(model, output_dir: Path):
    """Numerical check: split ONNX pipeline (preprocessor → encoder) vs PyTorch full encoder."""
    import onnxruntime as ort

    full_wrapper = FullEncoderWrapper(model).eval()
    preproc_sess = ort.InferenceSession(str(output_dir / "preprocessor.onnx"))
    enc_sess = ort.InferenceSession(str(output_dir / "encoder.onnx"))

    test_lengths = [16000, 48000, 80000]
    print()

    for audio_len in test_lengths:
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
        features = preproc_outs[0]
        padding_mask = preproc_outs[1]

        ort_out = enc_sess.run(None, {
            "input_features": features,
            "attention_mask": padding_mask,
        })[0]

        max_diff = np.abs(pt_out - ort_out).max()
        duration_s = audio_len / 16000
        print(f"  Split pipeline validation ({duration_s:.0f}s audio) — "
              f"shape {ort_out.shape}, max diff: {max_diff:.6f}")
        assert max_diff < 1e-4, f"Validation failed! max_diff={max_diff}"

    print("  ALL PASSED")


# ── Static shape conversion ──────────────────────────────────────────────────

def compute_streaming_enc_seq_len(num_samples: int) -> int:
    """Compute encoder output sequence length for a given number of audio samples.

    The streaming preprocessor applies two strided-2 causal Conv1D layers
    after framing at 80 samples/frame::

        frames = num_samples // 80
        enc_seq_len = (frames - 1) // 4 + 1
    """
    frames = num_samples // 80
    return (frames - 1) // 4 + 1


def _make_preprocessor_static(
    model: onnx.ModelProto,
    num_samples: int,
    enc_seq_len: int,
) -> onnx.ModelProto:
    """Fix preprocessor I/O dims to static values."""
    from ._graph import MoonshineStreamingOnnxGraphEditor

    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model, "preprocessor")
    editor.fix_preprocessor_io(num_samples, enc_seq_len)
    return editor.to_onnx(override_ir=model.ir_version)


def _make_encoder_static(
    model: onnx.ModelProto,
    enc_seq_len: int,
) -> onnx.ModelProto:
    """Fix encoder I/O dims to static values."""
    from ._graph import MoonshineStreamingOnnxGraphEditor

    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model, "encoder")
    editor.fix_encoder_io(enc_seq_len)
    return editor.to_onnx(override_ir=model.ir_version)


def _make_decoder_static(
    model: onnx.ModelProto,
    enc_seq_len: int,
) -> onnx.ModelProto:
    """Fix first-step decoder I/O dims to static values (no KV cache edits)."""
    from ._graph import MoonshineStreamingOnnxGraphEditor

    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model, "decoder")
    editor.fix_decoder_io(enc_seq_len, max_tokens=0, with_past=False)
    return editor.to_onnx(override_ir=model.ir_version)


def _make_decoder_with_past_static(
    model: onnx.ModelProto,
    enc_seq_len: int,
    max_tokens: int,
) -> onnx.ModelProto:
    """Convert decoder_with_past to static shapes with fixed KV buffer.

    Applies the following transformations in order:
      1. Rename self-attention Softmax nodes (dynamo names → expected pattern)
      2. Fix all I/O dimensions to static values
      3. Add ``current_len`` as a new int64 model input
      4. Replace dynamic KV cache Concat with static Where-based blend
      5. Add causal attention mask before self-attention Softmax
    """
    import onnx_graphsurgeon as gs
    from ._graph import MoonshineStreamingOnnxGraphEditor

    export_dtype = onnx.TensorProto.FLOAT

    editor = MoonshineStreamingOnnxGraphEditor.from_onnx(
        model, "decoder_with_past", export_dtype
    )

    # 1. Rename before fix_io_dims (relies on symbolic dim strings)
    editor.rename_self_attn_softmax()

    # 2. Static I/O
    editor.fix_decoder_io(enc_seq_len, max_tokens, with_past=True)

    # 3. Add current_len input [1,1] → squeeze to [1]
    graph = editor.graph
    cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
    graph.inputs.append(cur_len_2d)
    cur_len = graph.layer(
        name="current_len_to_1d",
        op="Squeeze",
        inputs=[cur_len_2d, [0]],
        outputs=[
            gs.Variable(cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1])
        ],
    )[0]

    # 4. Replace Concat→output KV cache with static Where blend
    editor.replace_dynamic_kv_cache(cur_len, max_tokens)

    # 5. Causal attention mask
    editor.mask_future_attn_scores(cur_len, max_tokens)

    # 6. Replace Shape(past_self_key_*) → Squeeze seq-len with cur_len
    editor.replace_shape_seq_len(cur_len)

    return editor.to_onnx(override_ir=model.ir_version)


def make_static(
    dynamic_dir: Path,
    static_dir: Path,
    num_samples: int,
    max_tokens: int,
    enc_seq_len: int,
    decoder_enc_seq_len: int | None = None,
):
    """Load dynamic ONNX models, apply static-shape conversion, save to static_dir.

    Args:
        num_samples: encoder input audio samples (may be smaller for TTFT reduction).
        max_tokens: max decoder tokens.
        enc_seq_len: encoder output sequence length (from num_samples).
        decoder_enc_seq_len: decoder cross-attention size. If None, same as enc_seq_len.
            When set larger than enc_seq_len, the decoder expects padded encoder output
            with an encoder_attention_mask to indicate valid positions.
    """
    if decoder_enc_seq_len is None:
        decoder_enc_seq_len = enc_seq_len

    static_dir.mkdir(parents=True, exist_ok=True)

    components = {
        "preprocessor": lambda m: _make_preprocessor_static(m, num_samples, enc_seq_len),
        "encoder": lambda m: _make_encoder_static(m, enc_seq_len),
        "decoder": lambda m: _make_decoder_static(m, decoder_enc_seq_len),
        "decoder_with_past": lambda m: _make_decoder_with_past_static(m, decoder_enc_seq_len, max_tokens),
    }

    for comp_name, convert_fn in components.items():
        src = dynamic_dir / f"{comp_name}.onnx"
        dst = static_dir / f"{comp_name}.onnx"
        logger.info("(%s) Making graph static...", comp_name)
        model = onnx.load(str(src))
        static_model = convert_fn(model)
        onnx.save(static_model, str(dst))
        logger.info("(%s) → %s", comp_name, dst)

    # Copy non-ONNX assets
    for name in ("decoder_token_embeddings.npy", "tokenizer.json"):
        src = dynamic_dir / name
        dst = static_dir / name
        if src.exists():
            shutil.copy2(src, dst)

    logger.info("Static models saved to %s", static_dir)


# ── Main export pipeline ─────────────────────────────────────────────────────

def export_moonshine_streaming_from_args(args: argparse.Namespace):
    """Main entry point called by the CLI dispatcher."""
    configure_logging(args.logging)

    # Resolve HF repo
    hf_repo = args.hf_repo or f"UsefulSensors/moonshine-streaming-{args.model_size}"
    static_models = not args.dynamic_models

    # Load config to get model dimensions
    from transformers import AutoConfig, MoonshineStreamingForConditionalGeneration

    config = AutoConfig.from_pretrained(hf_repo)
    num_decoder_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    enc_hidden = config.encoder_hidden_size

    # Setup directories
    models_dir = Path(args.models_dir)
    dynamic_dir = models_dir / hf_repo / "export" / "onnx" / "float" / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    # Static shape parameters
    num_samples = args.input_seconds * 16000
    max_tokens = args.input_seconds * args.tokens_per_sec
    enc_seq_len = compute_streaming_enc_seq_len(num_samples)

    # Encoder chunk window for overlap-and-save incremental encoding.
    # If --chunk-seconds is given, the encoder processes only a bounded
    # window per step: overlap + chunk_frames + finalization_delay.
    # Otherwise (non-incremental), encoder takes the full audio at once.
    chunk_seconds = getattr(args, 'chunk_seconds', None)
    if chunk_seconds is not None:
        # Compute overlap and finalization from model config
        enc_cfg = getattr(config, 'encoder_config', config)
        sliding_windows = enc_cfg.sliding_windows
        finalization_delay = sum(right for _, right in sliding_windows)
        left_reach = sum(left for left, _ in sliding_windows)
        cnn_left = 3  # receptive field of two stride-2 kernel-5 causal convs
        overlap_frames = left_reach + cnn_left

        chunk_frames = int(chunk_seconds * 50)  # 50 Hz post-CNN rate
        encoder_window_frames = overlap_frames + chunk_frames + finalization_delay
        # Convert post-CNN frames → audio samples: need (frame-1)*4+1 pre-CNN frames
        encoder_pre_cnn = (encoder_window_frames - 1) * 4 + 1
        encoder_num_samples = encoder_pre_cnn * 80  # 80 samples per pre-CNN frame
        encoder_enc_seq_len = compute_streaming_enc_seq_len(encoder_num_samples)
    else:
        encoder_num_samples = num_samples
        encoder_enc_seq_len = enc_seq_len
        overlap_frames = None
        finalization_delay = None
        chunk_frames = None

    # Decoder cross-attention size is always the full input_seconds size
    decoder_enc_seq_len = enc_seq_len

    if static_models:
        static_dir = models_dir / hf_repo / "export" / "onnx" / "float" / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        output_dir = static_dir
    else:
        output_dir = dynamic_dir

    # Local weights directory (download if needed)
    weights_dir = models_dir / hf_repo / "weights"
    if not (weights_dir / "model.safetensors").exists():
        download_model(hf_repo, weights_dir)
    else:
        logger.info("Using cached weights in %s", weights_dir.resolve())

    # Load model
    print(f"\nLoading model from {weights_dir} ...")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    model.config.use_cache = True

    # Export dynamic ONNX models
    print("\nExporting dynamic ONNX models ...")
    export_preprocessor(model, dynamic_dir / "preprocessor.onnx")
    export_encoder(model, dynamic_dir / "encoder.onnx", enc_hidden)
    export_decoder(model, dynamic_dir / "decoder.onnx", num_decoder_layers, enc_hidden)
    export_decoder_with_past(
        model, dynamic_dir / "decoder_with_past.onnx",
        num_decoder_layers, num_heads, head_dim, enc_hidden,
    )
    save_token_embeddings(model, dynamic_dir / "decoder_token_embeddings.npy")

    # Copy tokenizer
    tok_src = weights_dir / "tokenizer.json"
    tok_dst = dynamic_dir / "tokenizer.json"
    shutil.copy2(tok_src, tok_dst)
    print(f"  tokenizer → {tok_dst}")

    # Make static
    if static_models:
        if chunk_seconds is not None:
            print(
                f"\nConverting to static shapes "
                f"(chunk={chunk_seconds}s → encoder window={encoder_num_samples} samples / "
                f"{encoder_enc_seq_len} frames "
                f"[overlap={overlap_frames} + chunk={chunk_frames} + fin={finalization_delay}], "
                f"decoder cross-attn={decoder_enc_seq_len} frames, "
                f"max_tokens={max_tokens}) ..."
            )
        else:
            print(
                f"\nConverting to static shapes "
                f"(encoder={args.input_seconds}s / {encoder_num_samples} samples / "
                f"{encoder_enc_seq_len} frames, "
                f"decoder cross-attn={decoder_enc_seq_len} frames, "
                f"max_tokens={max_tokens}) ..."
            )
        make_static(
            dynamic_dir, static_dir, encoder_num_samples, max_tokens,
            encoder_enc_seq_len, decoder_enc_seq_len,
        )

    # Validate split encoder pipeline (uses dynamic models)
    if not args.skip_validation:
        print("\nValidating split encoder pipeline ...")
        validate(model, dynamic_dir)

    # Summary
    print(f"\nDone. Output in: {output_dir.resolve()}/")
    for f in sorted(output_dir.glob("*.onnx")) + sorted(output_dir.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        tag = "numpy" if f.suffix == ".npy" else "fp32"
        print(f"  {f.name:50s}  {size_mb:7.1f} MB  [{tag}]")
    if static_models:
        if chunk_seconds is not None:
            print(
                f"\n  Static config: chunk={chunk_seconds}s, "
                f"encoder window={encoder_enc_seq_len} frames, "
                f"decoder buffer={decoder_enc_seq_len} frames, "
                f"max_tokens={max_tokens}"
            )
        else:
            print(
                f"\n  Static config: encoder={args.input_seconds}s "
                f"({encoder_enc_seq_len} frames), "
                f"decoder={decoder_enc_seq_len} frames, "
                f"max_tokens={max_tokens}"
            )


def main():
    """Standalone entry point for direct script execution."""
    from . import add_moonshine_streaming_export_args
    parser = argparse.ArgumentParser(description="Export Moonshine Streaming to ONNX")
    add_moonshine_streaming_export_args(parser)
    export_moonshine_streaming_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
