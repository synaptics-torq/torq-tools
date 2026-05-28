#!/usr/bin/env python3
"""Test incremental overlap-and-save encoding with static ONNX models.

Simulates the real streaming pipeline:
  1. Audio arrives in steps (--step-seconds at a time)
  2. Each step, the encoder processes a bounded window (overlap + new + finalization)
  3. Newly finalized frames are placed into the decoder's encoder_hidden_state buffer
  4. The decoder cross-attends to the buffer via encoder_attention_mask
  5. Compare decoded text against the HuggingFace full-encode baseline

This validates that the static ONNX models with --chunk-seconds produce
correct output when used in the overlap-and-save pattern.

Usage:
    python src/torq/models/moonshine_streaming/test_incremental_static.py \
        --model-dir moonshine_streaming_tiny --n-samples 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
FRAME_SAMPLES = 80
BOS_TOKEN = 1
EOS_TOKEN = 2
MAX_DECODE_TOKENS = 30


# ── Frame conversion helpers ─────────────────────────────────────────────────

def samples_to_post_cnn(n_samples: int) -> int:
    """Audio samples → post-CNN encoder output frames."""
    pre_cnn = n_samples // FRAME_SAMPLES
    after_conv1 = (pre_cnn - 1) // 2 + 1 if pre_cnn > 0 else 0
    after_conv2 = (after_conv1 - 1) // 2 + 1 if after_conv1 > 0 else 0
    return after_conv2


def post_cnn_to_samples(post_cnn_frame: int) -> int:
    """Post-CNN frame index → audio sample offset."""
    return post_cnn_frame * 4 * FRAME_SAMPLES


# ── Incremental encoder ─────────────────────────────────────────────────────

class IncrementalEncoder:
    """Overlap-and-save encoder using ONNX preprocessor + encoder sessions."""

    def __init__(self, preproc_sess, encoder_sess, overlap_frames: int, finalization_delay: int):
        self.preproc = preproc_sess
        self.encoder = encoder_sess
        self.overlap_frames = overlap_frames
        self.finalization_delay = finalization_delay

        # State
        self.cached_frames = None  # (1, N, hidden)
        self.n_finalized = 0
        self.n_samples_encoded = 0

    def reset(self):
        self.cached_frames = None
        self.n_finalized = 0
        self.n_samples_encoded = 0

    def _encode_window(self, audio: np.ndarray) -> np.ndarray:
        """Run preprocessor + encoder on an audio window."""
        orig_len = len(audio)
        remainder = len(audio) % FRAME_SAMPLES
        if remainder:
            audio = np.pad(audio, (0, FRAME_SAMPLES - remainder))
        inp = audio[np.newaxis, :].astype(np.float32)
        mask = np.zeros((1, len(audio)), dtype=np.int64)
        mask[0, :orig_len] = 1

        features, padding_mask = self.preproc.run(None, {
            "input_values": inp,
            "attention_mask": mask,
        })
        (enc_out,) = self.encoder.run(None, {
            "input_features": features,
            "attention_mask": padding_mask,
        })
        return enc_out  # (1, T, hidden)

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """Incremental encode. Returns (all_frames, n_finalized).

        Uses overlap-and-save: re-encodes only the recent window,
        splices cached finalized frames with fresh output.
        """
        n_total = len(audio)

        if n_total <= self.n_samples_encoded and self.cached_frames is not None:
            return self.cached_frames, self.n_finalized

        total_post_cnn = samples_to_post_cnn(n_total)

        # First call or short buffer: encode everything
        if self.cached_frames is None or total_post_cnn <= self.overlap_frames + self.finalization_delay:
            enc_out = self._encode_window(audio)
            self.cached_frames = enc_out
            self.n_samples_encoded = n_total
            self.n_finalized = max(0, enc_out.shape[1] - self.finalization_delay)
            return enc_out, self.n_finalized

        # Overlap-and-save
        splice_frame = self.n_finalized
        reenc_start_frame = max(0, splice_frame - self.overlap_frames)
        reenc_start_sample = post_cnn_to_samples(reenc_start_frame)

        chunk_audio = audio[reenc_start_sample:]
        chunk_enc = self._encode_window(chunk_audio)

        overlap_output_frames = splice_frame - reenc_start_frame
        if overlap_output_frames < chunk_enc.shape[1]:
            fresh = chunk_enc[:, overlap_output_frames:, :]
        else:
            fresh = chunk_enc[:, chunk_enc.shape[1]:, :]

        if splice_frame > 0:
            enc_out = np.concatenate([
                self.cached_frames[:, :splice_frame, :],
                fresh,
            ], axis=1)
        else:
            enc_out = fresh

        self.cached_frames = enc_out
        self.n_samples_encoded = n_total
        self.n_finalized = max(0, enc_out.shape[1] - self.finalization_delay)
        return enc_out, self.n_finalized


# ── Decoder helper ───────────────────────────────────────────────────────────

def decode_from_buffer(decoder_sess, decoder_wp_sess, embeddings: np.ndarray,
                       enc_buffer: np.ndarray, enc_mask: np.ndarray) -> list[int]:
    """Run autoregressive decoding from encoder hidden state buffer."""
    # First step
    dec_out = decoder_sess.run(None, {
        "decoder_input_ids": np.array([[BOS_TOKEN]], dtype=np.int64),
        "encoder_hidden_states": enc_buffer,
        "encoder_attention_mask": enc_mask,
    })
    hidden = dec_out[0]
    logits = hidden[0, 0] @ embeddings.T
    token = int(np.argmax(logits))

    if token == EOS_TOKEN:
        return []

    tokens = [token]

    # Get KV caches and figure out names
    dec_out_names = [o.name for o in decoder_sess.get_outputs()]
    dec_wp_in_names = [inp.name for inp in decoder_wp_sess.get_inputs()]

    # Build KV mapping (present_X → past_X)
    kv_feeds = {}
    for name, tensor in zip(dec_out_names[1:], dec_out[1:]):
        past_name = name.replace("present_", "past_", 1)
        if past_name in dec_wp_in_names:
            kv_feeds[past_name] = tensor

    # Check if decoder_with_past needs current_len and has static KV buffers
    needs_current_len = "current_len" in dec_wp_in_names
    needs_enc_hs = "encoder_hidden_states" in dec_wp_in_names
    if needs_current_len:
        # Static model: pad self-KV to buffer size
        max_tokens = None
        for inp in decoder_wp_sess.get_inputs():
            if inp.name.startswith("past_self_key"):
                max_tokens = inp.shape[2]
                break
        if max_tokens:
            for name in list(kv_feeds.keys()):
                if "self" in name:
                    kv = kv_feeds[name]
                    B, H, S, D = kv.shape
                    if S < max_tokens:
                        padded = np.zeros((B, H, max_tokens, D), dtype=kv.dtype)
                        padded[:, :, :S, :] = kv
                        kv_feeds[name] = padded

    # Subsequent steps
    for step in range(MAX_DECODE_TOKENS - 1):
        feeds = {
            "decoder_input_ids": np.array([[tokens[-1]]], dtype=np.int64),
            "encoder_attention_mask": enc_mask,
        }
        if needs_enc_hs:
            feeds["encoder_hidden_states"] = enc_buffer
        feeds.update(kv_feeds)
        if needs_current_len:
            feeds["current_len"] = np.array([[step + 1]], dtype=np.int64)

        dec_wp_out_names = [o.name for o in decoder_wp_sess.get_outputs()]
        outs = decoder_wp_sess.run(None, feeds)
        hidden = outs[0]
        logits = hidden[0, 0] @ embeddings.T
        tok = int(np.argmax(logits))

        if tok == EOS_TOKEN:
            break
        tokens.append(tok)

        # Update KV
        kv_feeds = {}
        for name, tensor in zip(dec_wp_out_names[1:], outs[1:]):
            past_name = name.replace("present_", "past_", 1)
            if past_name in dec_wp_in_names:
                kv_feeds[past_name] = tensor

    return tokens


# ── Main test ────────────────────────────────────────────────────────────────

def load_model_config(model_dir: Path) -> dict:
    """Load config.json and extract streaming parameters."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        # Try weights subdir
        cfg_path = model_dir / "weights" / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    enc_cfg = cfg.get("encoder_config", cfg)
    windows = enc_cfg.get("sliding_windows", [])
    finalization_delay = sum(right for _, right in windows)
    left_reach = sum(left for left, _ in windows)
    overlap_frames = left_reach + 3  # +3 for CNN receptive field
    return {
        "finalization_delay": finalization_delay,
        "overlap_frames": overlap_frames,
        "sliding_windows": windows,
    }


def run_test(args):
    import onnxruntime as ort
    import torch
    from tokenizers import Tokenizer
    from transformers import MoonshineStreamingForConditionalGeneration
    from datasets import load_dataset

    model_dir = Path(args.model_dir)
    static_dir = Path(args.static_dir) if args.static_dir else None

    # Load config
    cfg = load_model_config(model_dir)
    overlap_frames = cfg["overlap_frames"]
    finalization_delay = cfg["finalization_delay"]

    # Load ONNX models (dynamic for incremental, static shapes not needed here)
    dynamic_dir = static_dir or model_dir
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    preproc = ort.InferenceSession(str(dynamic_dir / "preprocessor.onnx"), sess_options=opts)
    encoder = ort.InferenceSession(str(dynamic_dir / "encoder.onnx"), sess_options=opts)
    decoder = ort.InferenceSession(str(dynamic_dir / "decoder.onnx"), sess_options=opts)
    decoder_wp = ort.InferenceSession(str(dynamic_dir / "decoder_with_past.onnx"), sess_options=opts)
    embeddings = np.load(str(dynamic_dir / "decoder_token_embeddings.npy"))
    tokenizer = Tokenizer.from_file(str(dynamic_dir / "tokenizer.json"))

    # Decoder buffer size (from decoder model input shape)
    dec_inputs = {inp.name: inp for inp in decoder.get_inputs()}
    enc_hs_shape = dec_inputs["encoder_hidden_states"].shape
    buffer_frames = enc_hs_shape[1] if isinstance(enc_hs_shape[1], int) else 250
    hidden_dim = enc_hs_shape[2] if isinstance(enc_hs_shape[2], int) else 320

    # Load HF model for baseline
    weights_dir = model_dir / "weights" if (model_dir / "weights").exists() else model_dir
    print(f"Loading HF model from {weights_dir} ...")
    hf_model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(weights_dir), torch_dtype=torch.float32,
        local_files_only=True, attn_implementation="eager",
    ).eval()

    # Load test audio
    print("Loading LibriSpeech samples ...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    step_sec = args.step_seconds
    step_samples = int(step_sec * SAMPLE_RATE)

    print(f"\nConfig: overlap={overlap_frames}, finalization_delay={finalization_delay}, "
          f"buffer={buffer_frames}, step={step_sec}s")
    print(f"{'=' * 100}")

    all_match = []
    for sample_idx in range(min(args.n_samples, len(ds))):
        audio = np.array(ds[sample_idx]["audio"]["array"], dtype=np.float32)
        gt_text = ds[sample_idx]["text"]
        total_sec = len(audio) / SAMPLE_RATE

        # HF baseline (full audio, full encode)
        with torch.no_grad():
            input_values = torch.from_numpy(audio).unsqueeze(0)
            attention_mask = torch.ones(1, len(audio), dtype=torch.long)
            hf_out = hf_model.generate(
                input_values=input_values, attention_mask=attention_mask,
                max_new_tokens=MAX_DECODE_TOKENS,
            )
            hf_tokens = [t for t in hf_out[0].tolist() if t not in (BOS_TOKEN, EOS_TOKEN)]
            hf_text = tokenizer.decode(hf_tokens)

        # Incremental ONNX pipeline
        inc_encoder = IncrementalEncoder(preproc, encoder, overlap_frames, finalization_delay)

        # Process audio in steps, decode after each step
        n_steps = max(1, len(audio) // step_samples)
        step_results = []

        for step in range(1, n_steps + 1):
            end_sample = min(step * step_samples, len(audio))
            chunk = audio[:end_sample]

            t0 = time.perf_counter()
            enc_frames, n_finalized = inc_encoder.encode(chunk)
            enc_ms = (time.perf_counter() - t0) * 1000

            # Place finalized frames into decoder buffer
            total_frames = enc_frames.shape[1]
            fill_frames = min(total_frames, buffer_frames)

            enc_buffer = np.zeros((1, buffer_frames, hidden_dim), dtype=np.float32)
            enc_buffer[:, :fill_frames, :] = enc_frames[:, :fill_frames, :]
            enc_mask = np.zeros((1, buffer_frames), dtype=np.int64)
            enc_mask[:, :fill_frames] = 1

            # Decode
            t0 = time.perf_counter()
            tokens = decode_from_buffer(decoder, decoder_wp, embeddings, enc_buffer, enc_mask)
            dec_ms = (time.perf_counter() - t0) * 1000

            text = tokenizer.decode(tokens)
            step_results.append({
                "step": step,
                "audio_sec": end_sample / SAMPLE_RATE,
                "total_frames": total_frames,
                "finalized": n_finalized,
                "fill": fill_frames,
                "enc_ms": enc_ms,
                "dec_ms": dec_ms,
                "text": text,
            })

        # Final result
        final = step_results[-1]
        match = final["text"].strip() == hf_text.strip()
        all_match.append(match)

        print(f"\nSample {sample_idx} — {total_sec:.1f}s, GT: \"{gt_text[:50]}\"")
        print(f"  HF:   \"{hf_text[:60]}\"")
        print(f"  ONNX: \"{final['text'][:60]}\" {'✓ MATCH' if match else '✗ DIFF'}")
        print(f"  {'Step':>4s}  {'Sec':>4s}  {'Frm':>4s}  {'Fin':>4s}  {'Fill':>4s}  "
              f"{'Enc':>5s}  {'Dec':>5s}  Text")
        print(f"  {'─' * 80}")
        for r in step_results:
            print(f"  {r['step']:4d}  {r['audio_sec']:4.1f}  {r['total_frames']:4d}  "
                  f"{r['finalized']:4d}  {r['fill']:4d}  "
                  f"{r['enc_ms']:4.0f}ms {r['dec_ms']:4.0f}ms "
                  f"\"{r['text'][:45]}\"")

    # Summary
    n_match = sum(all_match)
    print(f"\n{'=' * 100}")
    print(f"SUMMARY: {n_match}/{len(all_match)} samples match HF baseline "
          f"({100*n_match/len(all_match):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test incremental encoding with static ONNX models")
    parser.add_argument("--model-dir", type=str, default="moonshine_streaming_tiny",
                        help="Model weights directory")
    parser.add_argument("--static-dir", type=str, default=None,
                        help="Static ONNX models directory (default: uses dynamic from model-dir export)")
    parser.add_argument("--step-seconds", type=float, default=1.0,
                        help="Audio step size in seconds (default: 1.0)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of test samples")
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
