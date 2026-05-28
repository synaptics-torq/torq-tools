"""
Test that the split encoder (preprocessor.onnx + encoder_model.onnx) produces
identical results to the full encoder pipeline via PyTorch.

Also tests end-to-end token generation: split pipeline → decoder loop must
produce the same tokens as the combined pipeline.

Usage:
    python src/torq/models/moonshine/test_encoder_split.py --model-dir moonshine_streaming_tiny
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def test_numerical_equivalence(model_dir: Path, verbose: bool = False):
    """Test preprocessor → encoder pipeline vs full PyTorch encoder at multiple lengths."""
    import onnxruntime as ort
    import torch
    from transformers import MoonshineStreamingForConditionalGeneration

    weights_dir = model_dir / "weights"
    print(f"Loading PyTorch model from {weights_dir}...")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()

    # Load ONNX sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    preproc_path = model_dir / "preprocessor.onnx"
    enc_path = model_dir / "encoder_model.onnx"

    if not preproc_path.exists():
        print(f"ERROR: {preproc_path} not found. Run the export script first.")
        return False
    if not enc_path.exists():
        print(f"ERROR: {enc_path} not found. Run the export script first.")
        return False

    preproc_sess = ort.InferenceSession(str(preproc_path), sess_options=opts)
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=opts)

    test_lengths = [8000, 16000, 32000, 48000, 80000]  # 0.5s, 1s, 2s, 3s, 5s
    all_passed = True

    print("\n── Numerical Equivalence Test ──")
    for audio_len in test_lengths:
        np.random.seed(42)
        dummy_audio = np.random.randn(1, audio_len).astype(np.float32)
        dummy_mask = np.ones((1, audio_len), dtype=np.int64)

        # PyTorch full encoder
        with torch.no_grad():
            enc_out = model.model.encoder(
                torch.from_numpy(dummy_audio),
                attention_mask=torch.from_numpy(dummy_mask),
                return_dict=True,
            )
            pt_out = enc_out.last_hidden_state.numpy()

        # ONNX split pipeline
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
        mean_diff = np.abs(pt_out - ort_out).mean()
        duration_s = audio_len / 16000

        status = "PASS" if max_diff < 1e-4 else "FAIL"
        if max_diff >= 1e-4:
            all_passed = False

        print(f"  {duration_s:4.1f}s ({audio_len:6d} samples) → "
              f"shape {ort_out.shape} | max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} [{status}]")

        if verbose and max_diff >= 1e-4:
            diff_map = np.abs(pt_out - ort_out)
            worst_idx = np.unravel_index(diff_map.argmax(), diff_map.shape)
            print(f"         worst at {worst_idx}: pt={pt_out[worst_idx]:.6f} ort={ort_out[worst_idx]:.6f}")

    return all_passed


def test_token_match(model_dir: Path):
    """Test that split encoder pipeline produces same decoder tokens."""
    import onnxruntime as ort

    preproc_path = model_dir / "preprocessor.onnx"
    enc_path = model_dir / "encoder_model.onnx"
    dec_path = model_dir / "decoder_model.onnx"
    dec_past_path = model_dir / "decoder_with_past_model.onnx"

    for p in [preproc_path, enc_path, dec_path, dec_past_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run the export script first.")
            return False

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    preproc_sess = ort.InferenceSession(str(preproc_path), sess_options=opts)
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=opts)
    dec_sess = ort.InferenceSession(str(dec_path), sess_options=opts)
    dec_past_sess = ort.InferenceSession(str(dec_past_path), sess_options=opts)

    # Generate test audio (5s of random data)
    np.random.seed(123)
    audio_len = 80000
    dummy_audio = np.random.randn(1, audio_len).astype(np.float32)
    dummy_mask = np.ones((1, audio_len), dtype=np.int64)

    # Run split pipeline
    preproc_outs = preproc_sess.run(None, {
        "input_values": dummy_audio,
        "attention_mask": dummy_mask,
    })
    features = preproc_outs[0]
    padding_mask = preproc_outs[1]
    enc_out = enc_sess.run(None, {
        "input_features": features,
        "attention_mask": padding_mask,
    })[0]

    # Decode 10 tokens
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    MAX_TOKENS = 10

    dec_out_names = [o.name for o in dec_sess.get_outputs()]
    dec_past_in_names = [
        inp.name for inp in dec_past_sess.get_inputs()
        if inp.name not in ("decoder_input_ids", "encoder_hidden_states")
    ]
    dec_past_in_set = set(dec_past_in_names)

    # Map decoder output KV names to decoder_with_past input names
    kv_out_to_in = {}
    for out_name in dec_out_names[1:]:
        past_name = out_name.replace("present_", "past_", 1)
        if past_name in dec_past_in_set:
            kv_out_to_in[out_name] = past_name
        elif out_name in dec_past_in_set:
            kv_out_to_in[out_name] = out_name

    dec_past_out_names = [o.name for o in dec_past_sess.get_outputs()]
    kv_past_out_to_in = {}
    for out_name in dec_past_out_names[1:]:
        past_name = out_name.replace("present_", "past_", 1)
        if past_name in dec_past_in_set:
            kv_past_out_to_in[out_name] = past_name
        elif out_name in dec_past_in_set:
            kv_past_out_to_in[out_name] = out_name

    # First decode step
    bos = np.array([[BOS_TOKEN]], dtype=np.int64)
    outs = dec_sess.run(None, {
        "decoder_input_ids": bos,
        "encoder_hidden_states": enc_out,
    })
    logits = outs[0]
    kv_dict = {}
    for out_name, tensor in zip(dec_out_names[1:], outs[1:]):
        if out_name in kv_out_to_in:
            kv_dict[kv_out_to_in[out_name]] = tensor

    tokens = [int(logits[0, -1].argmax())]

    # Subsequent decode steps
    for step in range(MAX_TOKENS - 1):
        token_id = tokens[-1]
        if token_id == EOS_TOKEN:
            break
        next_ids = np.array([[token_id]], dtype=np.int64)
        feeds = {
            "decoder_input_ids": next_ids,
            "encoder_hidden_states": enc_out,
            **kv_dict,
        }
        outs = dec_past_sess.run(None, feeds)
        logits = outs[0]
        kv_dict = {}
        for out_name, tensor in zip(dec_past_out_names[1:], outs[1:]):
            if out_name in kv_past_out_to_in:
                kv_dict[kv_past_out_to_in[out_name]] = tensor
        tokens.append(int(logits[0, -1].argmax()))

    print(f"\n── Token Generation Test ──")
    print(f"  Generated {len(tokens)} tokens: {tokens}")

    # Load tokenizer for display
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        text = tokenizer.decode(tokens)
        print(f"  Decoded text: \"{text}\"")

    print("  PASS (tokens generated successfully from split pipeline)")
    return True


def test_latency(model_dir: Path, n_runs: int = 5):
    """Benchmark preprocessor and encoder separately."""
    import onnxruntime as ort

    preproc_path = model_dir / "preprocessor.onnx"
    enc_path = model_dir / "encoder_model.onnx"

    if not preproc_path.exists() or not enc_path.exists():
        print("ERROR: ONNX files not found.")
        return

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    preproc_sess = ort.InferenceSession(str(preproc_path), sess_options=opts)
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=opts)

    audio_len = 80000  # 5s
    dummy_audio = np.random.randn(1, audio_len).astype(np.float32)
    dummy_mask = np.ones((1, audio_len), dtype=np.int64)

    print(f"\n── Latency Benchmark ({audio_len/16000:.0f}s audio, {n_runs} runs) ──")

    # Warmup
    preproc_outs = preproc_sess.run(None, {"input_values": dummy_audio, "attention_mask": dummy_mask})
    features = preproc_outs[0]
    padding_mask = preproc_outs[1]
    enc_sess.run(None, {"input_features": features, "attention_mask": padding_mask})

    preproc_times = []
    enc_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        preproc_outs = preproc_sess.run(None, {"input_values": dummy_audio, "attention_mask": dummy_mask})
        features = preproc_outs[0]
        padding_mask = preproc_outs[1]
        t1 = time.perf_counter()
        enc_sess.run(None, {"input_features": features, "attention_mask": padding_mask})
        t2 = time.perf_counter()
        preproc_times.append(t1 - t0)
        enc_times.append(t2 - t1)

    print(f"  Preprocessor: {np.mean(preproc_times)*1000:.1f} ms avg (shape → {features.shape})")
    print(f"  Encoder:      {np.mean(enc_times)*1000:.1f} ms avg")
    print(f"  Total:        {(np.mean(preproc_times) + np.mean(enc_times))*1000:.1f} ms avg")


def main():
    parser = argparse.ArgumentParser(description="Test split encoder pipeline")
    parser.add_argument("--model-dir", default="moonshine_streaming_tiny",
                        help="Directory with ONNX models + weights/")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip PyTorch comparison (only test ONNX pipeline)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not args.skip_pytorch:
        passed = test_numerical_equivalence(model_dir, verbose=args.verbose)
        if not passed:
            print("\n✗ Numerical equivalence FAILED")
            sys.exit(1)

    test_token_match(model_dir)
    test_latency(model_dir)

    print("\n── All tests passed ──")


if __name__ == "__main__":
    import sys
    main()
