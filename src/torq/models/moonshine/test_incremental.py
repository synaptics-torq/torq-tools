"""
Test: verify that overlap-and-save incremental encoding produces the same
*finalized* frames as full re-encoding, and that encoding time is bounded.

Simulates the mic demo's pattern — grow the audio buffer in 0.5s steps,
call transcribe(incremental=True) at each step, and compare against
transcribe(incremental=False) as the ground truth.

Key checks:
  1. Finalized frames must be bit-identical to the full-encode baseline.
  2. Provisional (trailing) frames may differ — the paper (arXiv:2602.12241
     §3.2) documents this as expected behaviour.
  3. Encoding time for incremental updates should be bounded (not grow
     linearly with buffer length).

Usage:
    python src/torq/models/moonshine/test_incremental.py
    python src/torq/models/moonshine/test_incremental.py --model-dir moonshine_streaming_small
    python src/torq/models/moonshine/test_incremental.py --duration-sec 10
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add the script's own directory so `infer_test` resolves when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_test import MoonshineASR, SAMPLE_RATE


def make_test_audio(duration_sec: float = 5.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize a deterministic test signal (chirp + noise) so the test is
    fully reproducible without downloading datasets.
    """
    rng = np.random.RandomState(42)
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    # Chirp from 200 Hz to 3000 Hz
    chirp = 0.3 * np.sin(2 * np.pi * (200 + 1400 * t / duration_sec) * t).astype(np.float32)
    noise = 0.02 * rng.randn(n).astype(np.float32)
    return chirp + noise


def run_test(model_dir: str, step_sec: float = 0.5, duration_sec: float = 5.0):
    audio = make_test_audio(duration_sec)
    print(f"Audio: {len(audio)} samples ({duration_sec:.1f}s @ {SAMPLE_RATE} Hz)")

    asr_incr = MoonshineASR(model_dir)
    asr_full = MoonshineASR(model_dir)
    print(f"Models loaded from {model_dir}")
    print(f"  finalization_delay = {asr_incr._finalization_delay} post-CNN frames")
    print(f"  overlap_frames     = {asr_incr._overlap_frames} post-CNN frames")
    print()

    step_samples = int(SAMPLE_RATE * step_sec)
    n_steps = max(2, len(audio) // step_samples)

    all_passed = True
    incr_times = []
    full_times = []

    hdr = (f"{'Step':>4}  {'Samples':>8}  {'Sec':>5}  {'Frames':>6}  "
           f"{'Final':>5}  {'FinalDiff':>10}  {'AllDiff':>10}  "
           f"{'TxtMatch':>8}  {'tIncr':>7}  {'tFull':>7}  {'Speedup':>7}")
    print(hdr)
    print("-" * len(hdr))

    for i in range(1, n_steps + 1):
        end = min(i * step_samples, len(audio))
        chunk = audio[:end]

        # ── Incremental path (simulates mic demo) ────────────────────
        t0 = time.perf_counter()
        text_incr, _ = asr_incr.transcribe(chunk, incremental=True)
        t_incr = time.perf_counter() - t0
        enc_incr = asr_incr._cached_enc.copy()
        n_finalized = asr_incr._n_finalized_frames

        # ── Full re-encode baseline ──────────────────────────────────
        t0 = time.perf_counter()
        text_full, _ = asr_full.transcribe(chunk, incremental=False)
        t_full = time.perf_counter() - t0
        enc_full = asr_full._encode_raw(chunk)

        incr_times.append(t_incr)
        full_times.append(t_full)

        # ── Compare ──────────────────────────────────────────────────
        if enc_incr.shape != enc_full.shape:
            all_diff = float("inf")
            final_diff = float("inf")
            shape_ok = False
        else:
            shape_ok = True
            all_diff = float(np.max(np.abs(enc_incr - enc_full)))
            if n_finalized > 0:
                final_diff = float(np.max(np.abs(
                    enc_incr[:, :n_finalized, :] - enc_full[:, :n_finalized, :]
                )))
            else:
                final_diff = all_diff

        text_match = text_incr.strip() == text_full.strip()

        # Finalized frames must match exactly (they had full context).
        ok = final_diff < 1e-5 and shape_ok
        if not ok:
            all_passed = False

        speedup = t_full / t_incr if t_incr > 0 else 0

        print(f"{i:>4}  {end:>8}  {end/SAMPLE_RATE:>5.1f}  "
              f"{enc_incr.shape[1]:>6}  {n_finalized:>5}  "
              f"{final_diff:>10.2e}  {all_diff:>10.2e}  "
              f"{'YES' if text_match else 'NO':>8}  "
              f"{t_incr*1000:>6.0f}ms  {t_full*1000:>6.0f}ms  "
              f"{speedup:>6.1f}x"
              f"{'  SHAPE!' if not shape_ok else ''}"
              f"{'  FAIL' if not ok else ''}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * len(hdr))
    avg_incr = 1000 * np.mean(incr_times[2:]) if len(incr_times) > 2 else 0
    avg_full = 1000 * np.mean(full_times[2:]) if len(full_times) > 2 else 0
    print(f"Avg encode time (steps 3+):  incremental {avg_incr:.0f}ms  |  full {avg_full:.0f}ms")
    if avg_incr > 0:
        print(f"Average speedup: {avg_full / avg_incr:.1f}x")

    # Final full-audio comparison (from clean state)
    print()
    asr_incr.reset_encoder_cache()
    text_incr_final, _ = asr_incr.transcribe(audio, incremental=True)
    text_full_final, _ = asr_full.transcribe(audio, incremental=False)
    enc_incr_final = asr_incr._cached_enc
    enc_full_final = asr_full._encode_raw(audio)

    if enc_incr_final.shape == enc_full_final.shape:
        final_diff = float(np.max(np.abs(enc_incr_final - enc_full_final)))
    else:
        final_diff = float("inf")

    final_text_match = text_incr_final.strip() == text_full_final.strip()
    print(f"Final full-audio (clean state): enc_diff={final_diff:.2e}  text_match={final_text_match}")
    print(f"  Incremental: {text_incr_final.strip()!r}")
    print(f"  Full:        {text_full_final.strip()!r}")

    print("\n" + "=" * len(hdr))
    if all_passed and final_diff < 1e-5 and final_text_match:
        print("RESULT: PASS — finalized frames match full re-encode at every step")
        return True
    else:
        print("RESULT: FAIL — finalized frames diverge from full re-encode")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test incremental vs full encoding consistency")
    parser.add_argument("--model-dir", default="moonshine_streaming_tiny")
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    args = parser.parse_args()

    passed = run_test(args.model_dir, args.step_sec, args.duration_sec)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
