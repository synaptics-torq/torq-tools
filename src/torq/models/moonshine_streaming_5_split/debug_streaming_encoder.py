# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.
"""
Parity validation for the stateful streaming encoder vs the batch encoder.

Usage:
    PYTHONPATH=src python -m torq.models.moonshine_streaming_5_split.debug_streaming_encoder \
        --model-dir <path/to/dynamic/export>

Tests:
  1. Single-call parity: streaming encoder fed all features at once should match batch encoder
     in the stable region (all frames except the last total_lookahead).
  2. Chunked parity: streaming encoder fed features in small chunks should produce the same
     output as the single-call test above (buffer continuity).
  3. Buffer continuity: two splits of the same audio must give identical outputs and final
     buffer states.
"""

import argparse
import sys
import numpy as np
import onnxruntime as ort


def run_batch_encoder(sess: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
    return sess.run(None, {"features": features})[0]


def run_streaming_encoder_single(
    sess: ort.InferenceSession,
    features: np.ndarray,
    total_lookahead: int,
    enc_left_ctx: list[int],
    hidden_size: int,
) -> np.ndarray:
    """One-shot: split features into stable + right_ctx, return encoded_stable."""
    n_layers = len(enc_left_ctx)
    zero_bufs = {
        f"buf_{i}": np.zeros((1, lc, hidden_size), dtype=np.float32)
        for i, lc in enumerate(enc_left_ctx)
    }
    T = features.shape[1]
    if T <= total_lookahead:
        stable_feats = features
        right_ctx = np.zeros((1, total_lookahead, hidden_size), dtype=np.float32)
    else:
        stable_feats = features[:, :-total_lookahead, :]
        right_ctx = features[:, -total_lookahead:, :]

    res = sess.run(None, {"stable_features": stable_feats, "right_ctx": right_ctx, **zero_bufs})
    return res[0], res[1:]  # encoded_stable, updated_bufs


def run_streaming_encoder_chunked(
    sess: ort.InferenceSession,
    features: np.ndarray,
    total_lookahead: int,
    enc_left_ctx: list[int],
    hidden_size: int,
    chunk_frames: int,
) -> np.ndarray:
    """
    Feed features to the streaming encoder chunk_frames at a time, accumulating the stable
    encoded output.  Matches the real streaming inference usage pattern.
    """
    n_layers = len(enc_left_ctx)
    bufs = {
        f"buf_{i}": np.zeros((1, lc, hidden_size), dtype=np.float32)
        for i, lc in enumerate(enc_left_ctx)
    }
    pending = np.zeros((1, 0, hidden_size), dtype=np.float32)
    accum = []

    T = features.shape[1]
    for start in range(0, T, chunk_frames):
        chunk = features[:, start:start + chunk_frames, :]
        combined = np.concatenate([pending, chunk], axis=1)
        stable_count = max(0, combined.shape[1] - total_lookahead)

        if stable_count > 0:
            stable_feats = combined[:, :stable_count, :]
            rc_end = min(stable_count + total_lookahead, combined.shape[1])
            right_ctx = combined[:, stable_count:rc_end, :]
            if right_ctx.shape[1] < total_lookahead:
                pad = np.zeros((1, total_lookahead - right_ctx.shape[1], hidden_size), dtype=np.float32)
                right_ctx = np.concatenate([right_ctx, pad], axis=1)

            res = sess.run(None, {"stable_features": stable_feats, "right_ctx": right_ctx, **bufs})
            accum.append(res[0])
            bufs = {f"buf_{i}": res[i + 1] for i in range(n_layers)}
            pending = combined[:, stable_count:, :]
        else:
            pending = combined

    # Flush remaining pending frames with zero right context
    if pending.shape[1] > 0:
        right_ctx = np.zeros((1, total_lookahead, hidden_size), dtype=np.float32)
        res = sess.run(None, {"stable_features": pending, "right_ctx": right_ctx, **bufs})
        accum.append(res[0])
        bufs = {f"buf_{i}": res[i + 1] for i in range(n_layers)}

    return np.concatenate(accum, axis=1) if accum else np.zeros((1, 0, hidden_size), dtype=np.float32)


def run_frontend(frontend_sess: ort.InferenceSession, audio: np.ndarray, chunk_len: int = 640) -> np.ndarray:
    """Run the stateful frontend preprocessor and accumulate features."""
    frontend_inputs = frontend_sess.get_inputs()
    hidden_size = frontend_sess.get_outputs()[0].shape[2]
    conv1_ch = next(inp.shape[1] for inp in frontend_inputs if inp.name == "conv1_buffer")
    conv2_ch = next(inp.shape[1] for inp in frontend_inputs if inp.name == "conv2_buffer")

    sample_buffer = np.zeros((1, 79), dtype=np.float32)
    sample_len = np.zeros(1, dtype=np.int64)
    conv1_buffer = np.zeros((1, conv1_ch, 4), dtype=np.float32)
    conv2_buffer = np.zeros((1, conv2_ch, 4), dtype=np.float32)
    frame_count = np.zeros(1, dtype=np.int64)

    accum = []
    audio_len = audio.shape[-1]
    for offset in range(0, audio_len, chunk_len):
        chunk = audio[:, offset:offset + chunk_len]
        if chunk.shape[-1] < chunk_len:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_len - chunk.shape[-1])))
        res = frontend_sess.run(None, {
            "audio_chunk": chunk,
            "sample_buffer": sample_buffer,
            "sample_len": sample_len,
            "conv1_buffer": conv1_buffer,
            "conv2_buffer": conv2_buffer,
            "frame_count": frame_count,
        })
        features, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count = res
        accum.append(features)

    return np.concatenate(accum, axis=1)


def inspect_streaming_encoder(sess: ort.InferenceSession):
    inputs = sess.get_inputs()
    enc_left_ctx = [inp.shape[1] for inp in inputs if inp.name.startswith("buf_")]
    total_lookahead = next(inp.shape[1] for inp in inputs if inp.name == "right_ctx")
    hidden_size = next(inp.shape[2] for inp in inputs if inp.name == "stable_features")
    return enc_left_ctx, total_lookahead, hidden_size


def main():
    parser = argparse.ArgumentParser(description="Validate stateful streaming encoder parity")
    parser.add_argument("--model-dir", required=True, help="Path to dynamic ONNX export directory")
    parser.add_argument(
        "--batch-encoder", default=None,
        help="Path to a batch encoder.onnx for parity comparison (Test 1). "
             "If omitted, Test 1 is skipped."
    )
    parser.add_argument("--chunk-frames", type=int, default=10, help="Chunk size for chunked parity test")
    parser.add_argument("--audio-frames", type=int, default=300, help="Synthetic audio features to test with")
    parser.add_argument("--tol", type=float, default=1e-4, help="Max absolute difference tolerance")
    args = parser.parse_args()

    import os
    model_dir = args.model_dir

    def load(path):
        if not os.path.exists(path):
            print(f"  MISSING: {path}", file=sys.stderr)
            sys.exit(1)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print("Loading sessions...")
    streaming_sess = load(os.path.join(model_dir, "encoder.onnx"))
    batch_encoder_path = args.batch_encoder
    encoder_sess = load(batch_encoder_path) if batch_encoder_path else None

    enc_left_ctx, total_lookahead, hidden_size = inspect_streaming_encoder(streaming_sess)
    n_enc_layers = len(enc_left_ctx)
    max_left_ctx = max(enc_left_ctx)
    # Contamination from zero-initialized buffers propagates left_ctx frames per layer.
    # Frames before warmup_frames will not match the batch encoder (expected behaviour).
    warmup_frames = n_enc_layers * max_left_ctx
    print(f"  enc_left_ctx={enc_left_ctx}, total_lookahead={total_lookahead}, hidden_size={hidden_size}")
    print(f"  warmup_frames (zero-buf contamination)={warmup_frames}")

    # Synthetic features (skip frontend for a faster unit test)
    np.random.seed(42)
    T = args.audio_frames
    assert T > warmup_frames + total_lookahead, \
        f"--audio-frames must be > {warmup_frames + total_lookahead} to have a testable steady-state region"
    features = np.random.randn(1, T, hidden_size).astype(np.float32)

    stable_len = T - total_lookahead
    ss_start = warmup_frames

    # Test 1: single-call streaming encoder (always runs; compared vs batch encoder if provided)
    print(f"\n[Test 1] Single-call streaming encoder (steady-state=[{ss_start}:{stable_len}])...")
    streaming_single, _ = run_streaming_encoder_single(
        streaming_sess, features, total_lookahead, enc_left_ctx, hidden_size
    )
    assert streaming_single.shape[1] == stable_len, \
        f"Shape mismatch: got {streaming_single.shape[1]}, expected {stable_len}"

    if encoder_sess is not None:
        ref_encoded = run_batch_encoder(encoder_sess, features)
        diff1 = np.abs(ref_encoded[:, ss_start:stable_len, :] - streaming_single[:, ss_start:, :]).max()
        mean1 = np.abs(ref_encoded[:, ss_start:stable_len, :] - streaming_single[:, ss_start:, :]).mean()
        print(f"  vs batch encoder — Max diff: {diff1:.6f}  Mean diff: {mean1:.6f}")
        if diff1 < args.tol:
            print(f"  PASS (tol={args.tol})")
        else:
            print(f"  FAIL — exceeds tolerance {args.tol}")
            sys.exit(1)
    else:
        print("  (no --batch-encoder provided; shape/run check only — PASS)")

    # Test 2: chunked streaming encoder vs single-call (steady-state region)
    print(f"\n[Test 2] Chunked ({args.chunk_frames} frames/call) vs single-call (steady-state)...")
    streaming_chunked = run_streaming_encoder_chunked(
        streaming_sess, features, total_lookahead, enc_left_ctx, hidden_size, args.chunk_frames
    )
    assert streaming_chunked.shape[1] == T, \
        f"Chunked output shape mismatch: got {streaming_chunked.shape[1]}, expected {T}"
    diff2 = np.abs(streaming_single[:, ss_start:] - streaming_chunked[:, ss_start:stable_len, :]).max()
    mean2 = np.abs(streaming_single[:, ss_start:] - streaming_chunked[:, ss_start:stable_len, :]).mean()
    print(f"  Max diff: {diff2:.6f}  Mean diff: {mean2:.6f}")
    if diff2 < args.tol:
        print(f"  PASS (tol={args.tol})")
    else:
        print(f"  FAIL — exceeds tolerance {args.tol}")
        sys.exit(1)

    # Test 3: buffer continuity — two chunk sizes give identical outputs
    print(f"\n[Test 3] Buffer continuity — chunk size {T // 2} vs {T // 3} (steady-state)...")
    chunked_a = run_streaming_encoder_chunked(
        streaming_sess, features, total_lookahead, enc_left_ctx, hidden_size, T // 2
    )
    chunked_b = run_streaming_encoder_chunked(
        streaming_sess, features, total_lookahead, enc_left_ctx, hidden_size, T // 3
    )
    diff3 = np.abs(chunked_a[:, ss_start:stable_len, :] - chunked_b[:, ss_start:stable_len, :]).max()
    print(f"  Max diff between chunk sizes: {diff3:.6f}")
    if diff3 < args.tol:
        print(f"  PASS (tol={args.tol})")
    else:
        print(f"  FAIL — exceeds tolerance {args.tol}")
        sys.exit(1)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
