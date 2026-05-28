# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Compare WER between ONNX FP32 and HuggingFace Transformers streaming models
on LibriSpeech test-clean.

Usage:
    python -m torq.models.moonshine.validate_hf --model-dir moonshine_streaming_tiny
    python -m torq.models.moonshine.validate_hf --model-dir moonshine_streaming_tiny --max-samples 50
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset as load_ds
from jiwer import wer
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from infer_test import MoonshineASR, SAMPLE_RATE

# Map model directory stems to HuggingFace Hub repo IDs
_REPO_MAP = {
    "moonshine_streaming_tiny": "UsefulSensors/moonshine-streaming-tiny",
    "moonshine_streaming_small": "UsefulSensors/moonshine-streaming-small",
}

import io
import os
from pathlib import Path

def _candidate_audio_paths(audio_info, dataset=None):
    """
    Build candidate absolute paths for an audio file.

    HF datasets may return:
      - an absolute path
      - a relative path
      - bytes content
    """
    raw_path = audio_info.get("path")
    if not raw_path:
        return []

    p = Path(raw_path)

    # Already absolute and exists
    if p.is_absolute():
        return [p]

    candidates = []

    # 1. relative to current working dir
    candidates.append(Path.cwd() / p)

    # 2. relative to HF datasets cache
    hf_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_cache:
        candidates.append(Path(hf_cache) / p)

    # 3. default HF cache roots
    home = Path.home()
    candidates.append(home / ".cache" / "huggingface" / "datasets" / p)
    candidates.append(home / ".cache" / "huggingface" / "hub" / p)

    # 4. relative to dataset cache files, if available
    if dataset is not None:
        for cf in getattr(dataset, "cache_files", []):
            filename = cf.get("filename")
            if filename:
                cache_dir = Path(filename).resolve().parent
                candidates.append(cache_dir / p)
                # sometimes audio lives a few levels above/below parquet shards
                for parent in cache_dir.parents[:4]:
                    candidates.append(parent / p)

    # de-dup while preserving order
    out = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            out.append(c)
    return out


def _read_audio_with_soundfile(audio_info, dataset=None):
    import soundfile as sf

    data_bytes = audio_info.get("bytes")
    raw_path = audio_info.get("path")

    # Best case: bytes are embedded, no path resolution needed
    if data_bytes is not None:
        audio, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), sr

    # Otherwise resolve path
    for cand in _candidate_audio_paths(audio_info, dataset=dataset):
        if cand.exists():
            audio, sr = sf.read(str(cand), dtype="float32", always_2d=False)
            return np.asarray(audio, dtype=np.float32), sr

    raise FileNotFoundError(
        f"Could not resolve audio path: raw_path={raw_path!r}. "
        f"Tried: {[str(p) for p in _candidate_audio_paths(audio_info, dataset=dataset)]}"
    )

def _load_audio_manual(audio_info, dataset=None) -> np.ndarray:
    import io
    import soundfile as sf

    data_bytes = audio_info.get("bytes")
    if data_bytes is None:
        raise RuntimeError(
            f"Expected embedded audio bytes, but got audio_info={audio_info}"
        )

    audio, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=-1)

    if sr != SAMPLE_RATE:
        audio = _resample_linear(audio, sr, SAMPLE_RATE)

    return audio

class MoonshineASRHuggingFace:
    """HuggingFace Transformers backend for Moonshine streaming models."""

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

        self.model_dir = Path(model_dir)
        stem = self.model_dir.stem

        repo_id = _REPO_MAP.get(stem)
        if repo_id is None:
            raise ValueError(
                f"Unknown model directory stem '{stem}'. "
                f"Expected one of: {list(_REPO_MAP)}"
            )

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.float32
        )
        self.model.eval()
        self.load_time = time.perf_counter() - t0
        self._torch = torch

    def transcribe(self, audio: np.ndarray):
        """Transcribe audio array -> (text, stats)."""
        t0 = time.perf_counter()
        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        with self._torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(generated[0], skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        stats = {
            "total_ms": elapsed * 1000,
            "audio_sec": len(audio) / SAMPLE_RATE,
        }
        return text, stats


def _load_librispeech():
    """Load LibriSpeech test-clean without automatic audio decoding."""
    ds = load_ds(
        path="hf-audio/esb-datasets-test-only-sorted",
        name="librispeech",
        split="test.clean",
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds

def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple numpy resampler to avoid extra dependencies."""
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(src_sr)
    new_len = int(round(duration * dst_sr))
    if new_len <= 1:
        return np.asarray(audio[:1], dtype=np.float32)

    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    out = np.interp(x_new, x_old, audio).astype(np.float32)
    return out

def _filter_samples(dataset, max_audio_sec: float, max_samples: int | None):
    """Return list of (audio_array, reference_text) pairs that fit length limit."""
    max_len = int(max_audio_sec * SAMPLE_RATE)
    samples = []
    skipped = 0

    for example in dataset:
        audio = _load_audio_manual(example["audio"], dataset=dataset)

        if len(audio) > max_len:
            skipped += 1
            continue

        samples.append((audio, example["text"].strip()))
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Compare ONNX vs HuggingFace WER on LibriSpeech test-clean"
    )
    parser.add_argument(
        "--model-dir",
        default="moonshine_streaming_tiny",
        help="ONNX model directory (default: moonshine_streaming_tiny)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to process (default: all)",
    )
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        default=31.0,
        help="Skip samples longer than this (seconds, default: 31)",
    )
    args = parser.parse_args()

    normalizer = EnglishTextNormalizer()

    print("Loading LibriSpeech test-clean...")
    dataset = _load_librispeech()
    first = dataset[0]
    print("First audio record:")
    print("  path =", first["audio"].get("path"))
    print("  has_bytes =", first["audio"].get("bytes") is not None)
    print("  cache_files =", [cf.get("filename") for cf in getattr(dataset, "cache_files", [])[:3]])
    samples, skipped = _filter_samples(dataset, args.max_audio_sec, args.max_samples)
    print(f"  {len(samples)} samples selected, {skipped} skipped (>{args.max_audio_sec}s)")

    if not samples:
        print("No samples to process!")
        return

    print("\nLoading ONNX model...")
    onnx_asr = MoonshineASR(args.model_dir)
    print(f"  ONNX loaded in {onnx_asr.load_time:.1f}s")

    print("Loading HuggingFace model...")
    hf_asr = MoonshineASRHuggingFace(args.model_dir)
    print(f"  HF loaded in {hf_asr.load_time:.1f}s")

    refs = []
    onnx_hyps = []
    hf_hyps = []
    onnx_total_ms = 0.0
    hf_total_ms = 0.0
    mismatches = []

    print(f"\nProcessing {len(samples)} samples...\n")
    for i, (audio, ref_text) in enumerate(tqdm(samples, desc="Samples")):
        onnx_text, onnx_stats = onnx_asr.transcribe(audio, incremental=False)
        onnx_text = onnx_text.strip()
        onnx_total_ms += onnx_stats["total_ms"]

        hf_text, hf_stats = hf_asr.transcribe(audio)
        hf_text = hf_text.strip()
        hf_total_ms += hf_stats["total_ms"]

        refs.append(ref_text)
        onnx_hyps.append(onnx_text)
        hf_hyps.append(hf_text)

        if normalizer(onnx_text) != normalizer(hf_text):
            mismatches.append({
                "idx": i,
                "ref": ref_text,
                "onnx": onnx_text,
                "hf": hf_text,
            })

    refs_norm = [normalizer(r) for r in refs]
    onnx_norm = [normalizer(h) for h in onnx_hyps]
    hf_norm = [normalizer(h) for h in hf_hyps]

    wer_onnx = wer(refs_norm, onnx_norm)
    wer_hf = wer(refs_norm, hf_norm)

    if mismatches:
        print(f"\n{'='*70}")
        print(f"MISMATCHES between ONNX and HF ({len(mismatches)} of {len(samples)})")
        print(f"{'='*70}")
        for m in mismatches[:20]:
            print(f"\n  Sample {m['idx']}:")
            print(f"    REF:  {m['ref']}")
            print(f"    ONNX: {m['onnx']}")
            print(f"    HF:   {m['hf']}")
        if len(mismatches) > 20:
            print(f"\n  ... and {len(mismatches) - 20} more")

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Samples:    {len(samples)}")
    print(f"  Skipped:    {skipped} (>{args.max_audio_sec}s)")
    print(f"  Mismatches: {len(mismatches)}")
    print()
    print(f"  ONNX  WER:  {100 * wer_onnx:.2f}%   ({onnx_total_ms / 1000:.1f}s total inference)")
    print(f"  HF    WER:  {100 * wer_hf:.2f}%   ({hf_total_ms / 1000:.1f}s total inference)")
    print()
    if abs(wer_onnx - wer_hf) < 0.001:
        print("  MATCH: ONNX and HF produce identical WER")
    else:
        print(f"  DIFF:  {100 * abs(wer_onnx - wer_hf):.2f}% WER difference")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()