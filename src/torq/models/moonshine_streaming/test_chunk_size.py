#!/usr/bin/env python3
"""Find minimum encoder input size for correct TTFT on streaming Moonshine.

Tests three decoding strategies for each audio prefix:
  1. Full chunk — decode from all encoder frames (variable length, no padding)
  2. Padded + masked — pad encoder output to full 250 frames with zeros, pass
     cross-attention mask to decoder so it ignores padding positions.
  3. Finalized + padded + masked — same but only use finalized frames before padding.

Strategy 2 (pad+mask) is the key test: it proves whether we can decouple the
encoder and decoder static shapes. If pad+mask produces the same tokens as the
full-chunk approach, we can use a smaller encoder (less audio) while the decoder
always cross-attends to a fixed [1, 250, hidden] tensor.

Usage:
    python src/torq/models/moonshine_streaming/test_chunk_size.py \
        --model-dir moonshine_streaming_tiny
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────

BOS_TOKEN = 1
EOS_TOKEN = 2
MAX_DECODE_TOKENS = 30
SAMPLE_RATE = 16000
FULL_ENC_FRAMES = 250  # encoder output frames for 5s audio (decoder target size)

# Chunk sizes to sweep (seconds)
CHUNK_SIZES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_enc_seq_len(num_samples: int) -> int:
    """Encoder output frames for given audio samples."""
    frames = num_samples // 80
    return (frames - 1) // 4 + 1


def compute_finalization_delay(config: dict) -> int:
    """Total right-context frames that are provisional (not yet stable)."""
    enc_cfg = config.get("encoder_config", config)
    sliding_windows = enc_cfg["sliding_windows"]
    return sum(right for _, right in sliding_windows)


def greedy_decode(model, encoder_hidden_states: torch.Tensor,
                  encoder_attention_mask: torch.Tensor = None) -> list[int]:
    """Greedy autoregressive decode from encoder output.

    Args:
        encoder_hidden_states: [1, enc_seq_len, hidden]
        encoder_attention_mask: Optional [1, enc_seq_len] with 1=real, 0=padding.
            Passed to decoder cross-attention. If None, all frames are attended.

    Returns token IDs (excluding BOS/EOS).
    """
    # IMPORTANT: decoder.forward() mutates encoder_hidden_states in-place
    # (pos_emb += and proj), so we must clone to avoid corrupting the caller's tensor.
    encoder_hidden_states = encoder_hidden_states.clone()

    device = encoder_hidden_states.device
    decoder = model.model.decoder
    vocab_weight = decoder.embed_tokens.weight  # [vocab_size, hidden]

    input_ids = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
    past_key_values = None
    tokens = []

    for step in range(MAX_DECODE_TOKENS):
        with torch.no_grad():
            outputs = decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        hidden = outputs.last_hidden_state[:, -1, :]  # [1, hidden]
        past_key_values = outputs.past_key_values

        logits = hidden @ vocab_weight.T  # [1, vocab]
        token_id = int(logits.argmax(dim=-1).item())

        if token_id == EOS_TOKEN:
            break

        tokens.append(token_id)
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

    return tokens


def pad_encoder_output(enc_out: torch.Tensor, target_len: int):
    """Pad encoder output to target_len frames with zeros. Returns (padded, mask).

    The mask has 1 for real frames and 0 for padding, suitable for cross-attention.
    """
    B, T, H = enc_out.shape
    if T >= target_len:
        mask = torch.ones(B, target_len, dtype=torch.long, device=enc_out.device)
        return enc_out[:, :target_len, :], mask

    pad = torch.zeros(B, target_len - T, H, dtype=enc_out.dtype, device=enc_out.device)
    padded = torch.cat([enc_out, pad], dim=1)  # [B, target_len, H]
    mask = torch.zeros(B, target_len, dtype=torch.long, device=enc_out.device)
    mask[:, :T] = 1
    return padded, mask


def encode_audio(model, audio: np.ndarray) -> torch.Tensor:
    """Encode raw audio through full encoder (embedder + transformer). Returns hidden states."""
    device = next(model.parameters()).device
    input_values = torch.from_numpy(audio).unsqueeze(0).to(device)  # [1, samples]
    attention_mask = torch.ones(1, audio.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_out = model.model.encoder(
            input_values, attention_mask=attention_mask, return_dict=True
        )
    return enc_out.last_hidden_state  # [1, enc_seq_len, hidden]


def prefix_match_len(tokens: list[int], reference: list[int]) -> int:
    """Number of leading tokens that match the reference."""
    n = min(len(tokens), len(reference))
    for i in range(n):
        if tokens[i] != reference[i]:
            return i
    return n


# ── Main test ────────────────────────────────────────────────────────────────

def load_test_audio(n_samples: int = 5) -> list[tuple[np.ndarray, str]]:
    """Load speech samples from librispeech dummy dataset. Returns (audio, text) pairs."""
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
    )

    samples = []
    for i in range(min(n_samples, len(ds))):
        audio_array = np.array(ds[i]["audio"]["array"], dtype=np.float32)
        text = ds[i]["text"]
        samples.append((audio_array, text))

    return samples


def run_sweep(model, audio: np.ndarray, tokenizer, finalization_delay: int):
    """Run chunk-size sweep on a single audio sample. Returns results list."""
    total_samples = len(audio)
    total_seconds = total_samples / SAMPLE_RATE

    # Baseline: full audio, no mask
    enc_full = encode_audio(model, audio)
    baseline_tokens = greedy_decode(model, enc_full)
    baseline_text = tokenizer.decode(baseline_tokens)

    # Sanity: full audio with all-ones mask should give same result
    full_mask = torch.ones(1, enc_full.shape[1], dtype=torch.long, device=enc_full.device)
    baseline_masked_tokens = greedy_decode(model, enc_full, full_mask)

    results = []
    for chunk_s in CHUNK_SIZES:
        if chunk_s > total_seconds:
            break

        chunk_samples = int(chunk_s * SAMPLE_RATE)
        chunk_audio = audio[:chunk_samples]
        enc_frames = compute_enc_seq_len(chunk_samples)

        # Encode the chunk
        t0 = time.perf_counter()
        enc_out = encode_audio(model, chunk_audio)
        enc_ms = (time.perf_counter() - t0) * 1000

        # Strategy 1: Full chunk decode (variable-length, no mask)
        tokens_full = greedy_decode(model, enc_out)
        prefix_full = prefix_match_len(tokens_full, baseline_tokens)

        # Strategy 2: Pad to 250 frames + cross-attention mask
        padded_enc, enc_mask = pad_encoder_output(enc_out, FULL_ENC_FRAMES)
        tokens_padded = greedy_decode(model, padded_enc, enc_mask)
        text_padded = tokenizer.decode(tokens_padded)
        prefix_padded = prefix_match_len(tokens_padded, baseline_tokens)

        # Strategy 3: Finalized frames only, padded + masked
        finalized_frames = max(1, enc_frames - finalization_delay)
        padded_fin, fin_mask = pad_encoder_output(
            enc_out[:, :finalized_frames, :], FULL_ENC_FRAMES
        )
        tokens_pad_fin = greedy_decode(model, padded_fin, fin_mask)
        prefix_pad_fin = prefix_match_len(tokens_pad_fin, baseline_tokens)

        results.append({
            "chunk_s": chunk_s,
            "enc_frames": enc_frames,
            "finalized_frames": finalized_frames,
            "enc_ms": enc_ms,
            "prefix_full": prefix_full,
            "n_tokens_padded": len(tokens_padded),
            "prefix_padded": prefix_padded,
            "text_padded": text_padded,
            "prefix_pad_fin": prefix_pad_fin,
        })

    return baseline_tokens, baseline_text, baseline_masked_tokens, results


def print_results(sample_idx: int, audio: np.ndarray, baseline_text: str,
                  baseline_tokens: list[int], baseline_masked_tokens: list[int],
                  results: list[dict]):
    """Print formatted results table for one sample."""
    duration = len(audio) / SAMPLE_RATE
    mask_ok = baseline_tokens == baseline_masked_tokens
    n = len(baseline_tokens)
    print(f"\n{'═' * 100}")
    print(f"  Sample {sample_idx} — {duration:.1f}s, {n} tokens, "
          f"mask sanity={'OK' if mask_ok else 'MISMATCH!'}")
    print(f"  Baseline: \"{baseline_text}\"")
    print(f"{'═' * 100}")
    print(f"  {'Chunk':>5s}  {'Frm':>4s}  {'Fin':>4s}  "
          f"{'ms':>4s}  "
          f"{'Full':>6s}  {'Pad+M':>6s}  {'Fin+M':>6s}  "
          f"Text (padded+masked)")
    print(f"  {'─' * 90}")

    for r in results:
        padded_match = "✓" if r["prefix_padded"] == n else " "
        print(
            f"  {r['chunk_s']:5.2f}s  "
            f"{r['enc_frames']:4d}  "
            f"{r['finalized_frames']:4d}  "
            f"{r['enc_ms']:4.0f}  "
            f"{r['prefix_full']:2d}/{n:<2d}  "
            f"{r['prefix_padded']:2d}/{n:<2d}{padded_match}"
            f"{r['prefix_pad_fin']:2d}/{n:<2d}  "
            f"\"{r['text_padded'][:48]}\""
        )


def print_summary(all_results: list[list[dict]], all_baseline_tokens: list[list[int]]):
    """Print aggregate summary across all samples."""
    print(f"\n{'═' * 100}")
    print("  SUMMARY — Average prefix match % by strategy")
    print(f"{'═' * 100}")

    chunk_sizes_seen = sorted(set(r["chunk_s"] for results in all_results for r in results))

    print(f"  {'Chunk':>5s}  {'Full':>6s}  {'Pad+Mask':>8s}  {'Fin+Mask':>8s}  "
          f"{'Enc ms':>6s}  {'All Pad+M?':>10s}")
    print(f"  {'─' * 65}")

    best_chunk = None
    for cs in chunk_sizes_seen:
        ratios_full = []
        ratios_padded = []
        ratios_pad_fin = []
        enc_times = []
        all_padded_match = True

        for i, results in enumerate(all_results):
            for r in results:
                if r["chunk_s"] == cs:
                    n_base = len(all_baseline_tokens[i])
                    if n_base > 0:
                        ratios_full.append(r["prefix_full"] / n_base)
                        ratios_padded.append(r["prefix_padded"] / n_base)
                        ratios_pad_fin.append(r["prefix_pad_fin"] / n_base)
                    enc_times.append(r["enc_ms"])
                    if r["prefix_padded"] < n_base:
                        all_padded_match = False

        if ratios_full:
            avg_full = np.mean(ratios_full) * 100
            avg_padded = np.mean(ratios_padded) * 100
            avg_pad_fin = np.mean(ratios_pad_fin) * 100
            avg_enc = np.mean(enc_times)
            match_str = "YES" if all_padded_match else "no"
            if all_padded_match and best_chunk is None:
                best_chunk = cs
                match_str = "YES ★"
            print(f"  {cs:5.2f}s  {avg_full:5.1f}%  {avg_padded:7.1f}%  "
                  f"{avg_pad_fin:7.1f}%  {avg_enc:5.0f}ms  {match_str:>10s}")

    if best_chunk is not None:
        enc_frames = compute_enc_seq_len(int(best_chunk * SAMPLE_RATE))
        savings = (1 - best_chunk / 5.0) * 100
        print(f"\n  ★ Recommended --encoder-seconds: {best_chunk}s "
              f"({enc_frames} frames, {savings:.0f}% encoder reduction)")
    else:
        print("\n  ⚠ No chunk size achieved full prefix match with pad+mask on all samples")


def main():
    parser = argparse.ArgumentParser(description="Find minimum encoder chunk size for TTFT")
    parser.add_argument(
        "--model-dir", type=str, default="moonshine_streaming_tiny",
        help="Path to model weights directory (default: moonshine_streaming_tiny)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of test audio samples (default: 5)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    weights_dir = model_dir / "weights" if (model_dir / "weights").exists() else model_dir

    # Load config
    config_path = weights_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    finalization_delay = compute_finalization_delay(config)
    print(f"Model: {model_dir}")
    print(f"Finalization delay: {finalization_delay} frames ({finalization_delay * 20}ms)")
    print(f"Decoder cross-attn target: {FULL_ENC_FRAMES} frames (fixed)")

    # Load model
    print("Loading model ...")
    from transformers import MoonshineStreamingForConditionalGeneration
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()

    # Load tokenizer
    from tokenizers import Tokenizer
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        tok_path = weights_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Load test audio
    print(f"Loading {args.n_samples} test samples from librispeech_asr_dummy ...")
    samples = load_test_audio(args.n_samples)
    durations = [f"{len(a)/SAMPLE_RATE:.1f}s" for a, _ in samples]
    print(f"Loaded {len(samples)} samples ({', '.join(durations)})")

    # Run sweep
    all_results = []
    all_baseline_tokens = []

    for i, (audio, ref_text) in enumerate(samples):
        # Pad or trim to 5s for consistency
        target_len = 5 * SAMPLE_RATE
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        baseline_tokens, baseline_text, baseline_masked, results = run_sweep(
            model, audio, tokenizer, finalization_delay
        )

        all_results.append(results)
        all_baseline_tokens.append(baseline_tokens)

        print_results(i, audio, baseline_text, baseline_tokens, baseline_masked, results)

    # Summary
    print_summary(all_results, all_baseline_tokens)


if __name__ == "__main__":
    main()
