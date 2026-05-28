"""Real-time streaming ASR CLI using Moonshine v2 models.

Supports three modes:
  --hf      Use HuggingFace Transformers (downloads from Hub)
  --both    Run ONNX *and* HF side-by-side and compare outputs
  (default) Use local ONNX FP32 models

Uses continuous audio accumulation with incremental encoding — the encoder
output is cached and only new audio is encoded each update, keeping latency
constant regardless of total buffer length. The decoder re-decodes from
scratch using the full concatenated encoder output.

Usage:
    python infer_test.py [--model-dir models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic]
    python infer_test.py --hf
    python infer_test.py --both

Speaks into your mic, transcribes in real-time.  Ctrl+C to stop.
"""

import argparse
import json
import os
import sys
import time
import resource
import numpy as np
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
FRAME_SAMPLES = 80          # 5ms @ 16kHz — encoder frame size
BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
MAX_DECODE_TOKENS = 256     # safety limit per transcription
BLOCK_MS = 200              # audio block size in ms
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

# Streaming defaults
UPDATE_INTERVAL = 0.5       # re-run inference every 0.5s
SILENCE_SEC = 1.5           # finalize line after 1.5s of silence
SPEECH_THRESHOLD = 0.01     # RMS energy threshold for speech
MAX_BUFFER_SEC = 10.0       # max audio buffer before force-flush (keeps RAM bounded)
MIN_AUDIO_SEC = 0.3         # minimum audio to bother processing


# ── Model wrapper ────────────────────────────────────────────────────────────

class MoonshineASR:
    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)

        t0 = time.perf_counter()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"),
            sess_options=opts, providers=providers,
        )
        # Require preprocessor (split encoder pipeline)
        preproc_path = self.model_dir / "preprocessor.onnx"
        if not preproc_path.exists():
            raise FileNotFoundError(
                f"preprocessor.onnx not found in {self.model_dir}. "
                f"Re-export with the latest export script to produce split encoder models."
            )
        self.preprocessor = ort.InferenceSession(
            str(preproc_path), sess_options=opts, providers=providers,
        )
        # Load token embeddings for external logit computation
        emb_path = self.model_dir / "decoder_token_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"decoder_token_embeddings.npy not found in {self.model_dir}. "
                f"Re-export with the latest export script."
            )
        self._embeddings = np.load(str(emb_path))  # (vocab_size, hidden_dim)
        self.decoder = ort.InferenceSession(
            str(self.model_dir / "decoder.onnx"),
            sess_options=opts, providers=providers,
        )
        self.decoder_past = ort.InferenceSession(
            str(self.model_dir / "decoder_with_past.onnx"),
            sess_options=opts, providers=providers,
        )
        self.load_time = time.perf_counter() - t0

        tok_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_path))

        # Build KV cache name mappings
        dec_out_names = [o.name for o in self.decoder.get_outputs()][1:]
        dec_past_in_names = [
            inp.name for inp in self.decoder_past.get_inputs()
            if inp.name not in (
                "decoder_input_ids",
                "encoder_hidden_states",
                "encoder_attention_mask",
                "current_len",  # static decoder_with_past adds this
            )
        ]
        dec_past_in_set = {n: n for n in dec_past_in_names}

        self._kv_out_to_in = {}
        for out_name in dec_out_names:
            past_name = out_name.replace("present_", "past_", 1)
            if past_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = past_name
            elif out_name + "_orig" in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name + "_orig"
            elif out_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name

        dec_past_out_names = [o.name for o in self.decoder_past.get_outputs()][1:]
        self._kv_past_out_to_in = {}
        for out_name in dec_past_out_names:
            past_name = out_name.replace("present_", "past_", 1)
            if past_name in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = past_name
            elif out_name + "_orig" in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = out_name + "_orig"
            elif out_name in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = out_name

        # Moonshine streaming: 16kHz input, 50Hz encoder output = 320 samples/frame
        self._samples_per_frame = 320.0

        # ── Sliding-window overlap parameters (from config.json) ─────────
        # The encoder uses per-layer sliding-window attention.  To produce
        # a *finalized* hidden state at frame t the encoder needs up to
        # `finalization_delay` frames of future audio (right context that
        # cascades through layers).  For correct overlap-and-save we also
        # need enough left context so the splice-point frames match the
        # full-encode baseline.
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            enc_cfg = cfg.get("encoder_config", cfg)
            windows = enc_cfg.get("sliding_windows", [])
            # Cascaded right-reach: how many future frames are transitively
            # incorporated through the layer stack.
            right_reach = 0
            for _left, right in windows:
                right_reach += right
            self._finalization_delay = right_reach  # post-CNN frames

            # Cascaded left-reach: maximum past context a frame depends on.
            left_reach = 0
            for left, _right in windows:
                left_reach += left
            # Add CNN receptive field in post-CNN frame units (~3 frames for
            # two stride-2 kernel-5 causal convs).
            cnn_left = 3
            self._overlap_frames = left_reach + cnn_left
        else:
            # Safe fallback when config.json is absent.
            self._finalization_delay = 16
            self._overlap_frames = 99  # 96 (6×16) + 3 CNN

        # Incremental encoding state
        self._cached_enc = None       # (1, T_cached, hidden)
        self._n_samples_encoded = 0   # how many audio samples the cache covers
        self._n_finalized_frames = 0  # frames whose output won't change

    def _encode_raw(self, audio: np.ndarray) -> np.ndarray:
        """Run encoder on audio array (preprocessor → encoder). No caching."""
        orig_len = len(audio)
        remainder = len(audio) % FRAME_SAMPLES
        if remainder:
            audio = np.pad(audio, (0, FRAME_SAMPLES - remainder))
        inp = audio[np.newaxis, :].astype(np.float32)
        mask = np.zeros((1, len(audio)), dtype=np.int64)
        mask[0, :orig_len] = 1

        preproc_outs = self.preprocessor.run(None, {
            "input_values": inp,
            "attention_mask": mask,
        })
        features = preproc_outs[0]
        padding_mask = preproc_outs[1]
        (enc_out,) = self.encoder.run(None, {
            "input_features": features,
            "attention_mask": padding_mask,
        })
        return enc_out

    def encode_incremental(self, audio: np.ndarray) -> np.ndarray:
        """
        Overlap-and-save incremental encoding.

        The Moonshine v2 encoder uses sliding-window attention (no global
        attention), so each output frame depends only on a bounded
        neighbourhood.  We exploit this by:

        1. Tracking which cached frames are *finalized* (will not change
           when more audio arrives — i.e. they are far enough from the
           right edge that all future-context layers have seen enough).
        2. When new audio arrives, re-encoding only from
           ``(finalized - overlap) × stride`` to the end.
        3. Splicing: keep ``cached[:, :finalized, :]`` and append the
           freshly-encoded frames starting at the splice point.

        The re-encode region is bounded regardless of total buffer length,
        giving O(1) encoder cost per update.
        """
        n_total = len(audio)

        # No new audio since last encode — return cached result.
        if self._cached_enc is not None and n_total <= self._n_samples_encoded:
            return self._cached_enc

        # First call or very short buffer — encode everything.
        total_input_frames = n_total // FRAME_SAMPLES
        total_post_cnn = self._input_frames_to_post_cnn(total_input_frames)
        if self._cached_enc is None or total_post_cnn <= self._overlap_frames + self._finalization_delay:
            enc_out = self._encode_raw(audio)
            self._cached_enc = enc_out
            self._n_samples_encoded = n_total
            self._n_finalized_frames = max(0, enc_out.shape[1] - self._finalization_delay)
            return enc_out

        # ── Overlap-and-save ─────────────────────────────────────────
        # Splice point: the first cached frame we will *replace*.
        splice_frame = self._n_finalized_frames

        # Re-encode start in audio samples: go back `overlap_frames`
        # post-CNN frames before the splice point to give the encoder
        # enough left context.
        reenc_start_frame = max(0, splice_frame - self._overlap_frames)
        reenc_start_sample = self._post_cnn_to_samples(reenc_start_frame)

        chunk_audio = audio[reenc_start_sample:]
        chunk_enc = self._encode_raw(chunk_audio)  # (1, T_chunk, hidden)

        # How many frames of the chunk correspond to the overlap region
        # (i.e. duplicates of cached frames that we discard).
        overlap_output_frames = splice_frame - reenc_start_frame

        if overlap_output_frames < chunk_enc.shape[1]:
            fresh = chunk_enc[:, overlap_output_frames:, :]
        else:
            # Edge case: chunk is entirely overlap — nothing new.
            fresh = chunk_enc[:, chunk_enc.shape[1]:, :]

        # Splice cached finalized + fresh
        if splice_frame > 0:
            enc_out = np.concatenate([
                self._cached_enc[:, :splice_frame, :],
                fresh,
            ], axis=1)
        else:
            enc_out = fresh

        self._cached_enc = enc_out
        self._n_samples_encoded = n_total
        self._n_finalized_frames = max(0, enc_out.shape[1] - self._finalization_delay)
        return enc_out

    # ── Helpers for frame ↔ sample conversion ────────────────────────────

    @staticmethod
    def _input_frames_to_post_cnn(n_input_frames: int) -> int:
        """Number of post-CNN (encoder output) frames from input frames.

        Two stride-2 causal convs: each does ``(N + pad - kernel) // stride + 1``
        with left_pad=4, kernel=5 → ``(N - 1) // 2 + 1``.
        """
        after_conv1 = (n_input_frames - 1) // 2 + 1 if n_input_frames > 0 else 0
        after_conv2 = (after_conv1 - 1) // 2 + 1 if after_conv1 > 0 else 0
        return after_conv2

    def _post_cnn_to_samples(self, post_cnn_frame: int) -> int:
        """Convert a post-CNN frame index to the audio sample offset.

        Each post-CNN frame spans 4 input frames (stride 2 × 2).
        Each input frame = FRAME_SAMPLES (80) audio samples.
        """
        input_frame = post_cnn_frame * 4
        return input_frame * FRAME_SAMPLES

    def reset_encoder_cache(self):
        """Clear encoder cache (call when starting a new utterance)."""
        self._cached_enc = None
        self._n_samples_encoded = 0
        self._n_finalized_frames = 0

    def decode_first(self, enc_out: np.ndarray):
        bos = np.array([[BOS_TOKEN]], dtype=np.int64)
        enc_mask = np.ones((enc_out.shape[0], enc_out.shape[1]), dtype=np.int64)
        outs = self.decoder.run(None, {
            "decoder_input_ids": bos,
            "encoder_hidden_states": enc_out,
            "encoder_attention_mask": enc_mask,
        })
        hidden = outs[0]  # (1, 1, hidden_dim)
        dec_out_names = [o.name for o in self.decoder.get_outputs()]
        kv_dict = {}
        for out_name, tensor in zip(dec_out_names[1:], outs[1:]):
            if out_name in self._kv_out_to_in:
                kv_dict[self._kv_out_to_in[out_name]] = tensor
        # External logit computation: logits = hidden @ embeddings.T
        logits = hidden[0, -1, :] @ self._embeddings.T
        token_id = int(np.argmax(logits))
        return token_id, kv_dict

    def decode_next(self, token_id: int, enc_out: np.ndarray, kv_dict: dict):
        enc_mask = np.ones((enc_out.shape[0], enc_out.shape[1]), dtype=np.int64)
        inputs = {
            "decoder_input_ids": np.array([[token_id]], dtype=np.int64),
            "encoder_hidden_states": enc_out,
            "encoder_attention_mask": enc_mask,
        }
        inputs.update(kv_dict)

        dec_past_out_names = [o.name for o in self.decoder_past.get_outputs()]
        outs = self.decoder_past.run(None, inputs)
        hidden = outs[0]  # (1, 1, hidden_dim)

        new_kv = {}
        for out_name, tensor in zip(dec_past_out_names[1:], outs[1:]):
            if out_name in self._kv_past_out_to_in:
                new_kv[self._kv_past_out_to_in[out_name]] = tensor
        # External logit computation: logits = hidden @ embeddings.T
        logits = hidden[0, -1, :] @ self._embeddings.T
        next_id = int(np.argmax(logits))
        return next_id, new_kv

    def transcribe(self, audio: np.ndarray, incremental: bool = False):
        """
        Transcribe audio → (text, stats).
        If incremental=True, reuse cached encoder output for previously-seen audio.
        """
        t_enc_start = time.perf_counter()
        if incremental:
            enc_out = self.encode_incremental(audio)
        else:
            enc_out = self._encode_raw(audio)
        t_enc = time.perf_counter() - t_enc_start

        t_dec_start = time.perf_counter()
        token_id, past_kvs = self.decode_first(enc_out)
        token_ids = [token_id]

        while token_id != EOS_TOKEN and len(token_ids) < MAX_DECODE_TOKENS:
            token_id, past_kvs = self.decode_next(token_id, enc_out, past_kvs)
            token_ids.append(token_id)

        t_dec = time.perf_counter() - t_dec_start
        text = self.tokenizer.decode(token_ids)

        stats = {
            "encode_ms": t_enc * 1000,
            "decode_ms": t_dec * 1000,
            "total_ms": (t_enc + t_dec) * 1000,
            "n_tokens": len(token_ids),
            "tokens_per_sec": len(token_ids) / t_dec if t_dec > 0 else 0,
            "audio_sec": len(audio) / SAMPLE_RATE,
            "rtf": (t_enc + t_dec) / (len(audio) / SAMPLE_RATE),
        }
        return text, stats


class MoonshineASRHuggingFace:
    """HuggingFace Transformers backend — downloads model from the Hub."""

    _REPO_MAP = {
        "moonshine_streaming_tiny": "UsefulSensors/moonshine-streaming-tiny",
        "moonshine_streaming_small": "UsefulSensors/moonshine-streaming-small",
        "dynamic": None,  # resolved from parent path
    }

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

        self._torch = torch
        model_path = Path(model_dir)
        dir_stem = model_path.name
        repo_id = self._REPO_MAP.get(dir_stem)
        if dir_stem == "dynamic" or dir_stem == "static":
            # Infer repo from parent path: .../UsefulSensors/moonshine-streaming-tiny/export/...
            parts = model_path.resolve().parts
            for i, p in enumerate(parts):
                if p == "UsefulSensors" and i + 1 < len(parts):
                    repo_id = f"UsefulSensors/{parts[i + 1]}"
                    break
        if repo_id is None:
            raise ValueError(
                f"Unknown model-dir '{model_dir}' for HF mode. "
                f"Expected a path containing UsefulSensors/<model-name> or one of: {list(self._REPO_MAP)}"
            )

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            repo_id,
        ).to(self._device).to(self._dtype)
        self.load_time = time.perf_counter() - t0

        self._sampling_rate = self.processor.feature_extractor.sampling_rate
        self._token_limit_factor = 6.5 / self._sampling_rate

    def reset_encoder_cache(self):
        """No-op — HF model manages its own state."""
        pass

    def transcribe(self, audio: np.ndarray, incremental: bool = False):
        """
        Transcribe audio → (text, stats).
        `incremental` is accepted but ignored (not supported in HF mode).
        """
        torch = self._torch

        t_start = time.perf_counter()
        inputs = self.processor(
            audio, return_tensors="pt", sampling_rate=self._sampling_rate,
        )
        inputs = inputs.to(self._device, self._dtype)

        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = int((seq_lens * self._token_limit_factor).max().item())
        max_length = max(max_length, 2)

        t_gen_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=max_length)
        t_gen = time.perf_counter() - t_gen_start

        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        t_total = time.perf_counter() - t_start

        n_tokens = generated_ids.shape[-1]
        audio_sec = len(audio) / SAMPLE_RATE
        stats = {
            "encode_ms": 0.0,
            "decode_ms": t_gen * 1000,
            "total_ms": t_total * 1000,
            "n_tokens": n_tokens,
            "tokens_per_sec": n_tokens / t_gen if t_gen > 0 else 0,
            "audio_sec": audio_sec,
            "rtf": t_total / audio_sec if audio_sec > 0 else 0,
        }
        return text, stats


# ── Utilities ────────────────────────────────────────────────────────────────

def get_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / 1e6
    return ru.ru_maxrss / 1e3


def rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Moonshine v2 Streaming ASR — real-time mic transcription")
    parser.add_argument("--model-dir", default="models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic", help="ONNX model directory")
    parser.add_argument("--update-interval", type=float, default=UPDATE_INTERVAL,
                        help="Seconds between re-inference updates (default: 0.5)")
    parser.add_argument("--silence-sec", type=float, default=SILENCE_SEC,
                        help="Seconds of silence to finalize a line (default: 1.5)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--hf", action="store_true",
                            help="Use HuggingFace model from the Hub instead of local ONNX")
    mode_group.add_argument("--both", action="store_true",
                            help="Run ONNX and HF side-by-side and compare outputs")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    compare_mode = args.both
    model_dir = Path(args.model_dir)

    # ── Load models ──────────────────────────────────────────────────────
    asr_onnx = None
    asr_hf = None

    if args.hf or compare_mode:
        print("=" * 60)
        if compare_mode:
            print("  Moonshine v2 Streaming ASR  (ONNX vs HF comparison)")
        else:
            print("  Moonshine v2 Streaming ASR  (HuggingFace)")
        print("=" * 60)

        rss_before = get_rss_mb()
        print(f"\n  Loading HF model from Hub...", end="", flush=True)
        asr_hf = MoonshineASRHuggingFace(str(model_dir))
        rss_after = get_rss_mb()
        print(f" done ({asr_hf.load_time:.1f}s, +{rss_after - rss_before:.0f} MB)")

    if not args.hf or compare_mode:
        if not compare_mode:
            print("=" * 60)
            print("  Moonshine v2 Streaming ASR  (ONNX FP32)")
            print("=" * 60)

        enc_mb = model_size_mb(model_dir / "encoder.onnx")
        dec_mb = model_size_mb(model_dir / "decoder.onnx")
        dec_past_mb = model_size_mb(model_dir / "decoder_with_past.onnx")
        preproc_mb = model_size_mb(model_dir / "preprocessor.onnx") if (model_dir / "preprocessor.onnx").exists() else 0
        total_mb = enc_mb + dec_mb + dec_past_mb + preproc_mb
        if preproc_mb:
            print(f"  ONNX:    preproc {preproc_mb:.0f} MB | encoder {enc_mb:.0f} MB | decoder {dec_mb:.0f} MB | "
                  f"decoder_past {dec_past_mb:.0f} MB  ({total_mb:.0f} MB total)")
        else:
            print(f"  ONNX:    encoder {enc_mb:.0f} MB | decoder {dec_mb:.0f} MB | "
                  f"decoder_past {dec_past_mb:.0f} MB  ({total_mb:.0f} MB total)")

        rss_before = get_rss_mb()
        print(f"  Loading ONNX models...", end="", flush=True)
        asr_onnx = MoonshineASR(str(model_dir))
        rss_after = get_rss_mb()
        print(f" done ({asr_onnx.load_time:.1f}s, +{rss_after - rss_before:.0f} MB)")

    print(f"  RAM:     {get_rss_mb():.0f} MB total")

    dev_info = sd.query_devices(args.device or sd.default.device[0], "input")
    print(f"  Mic:     {dev_info['name']}  ({int(dev_info['default_samplerate'])} Hz native)")
    print(f"  Update:  every {args.update_interval}s  |  Silence flush: {args.silence_sec}s")
    print()
    print("  Speak into your microphone. Ctrl+C to stop.")
    print("-" * 60)

    # ── Streaming state ─────────────────────────────────────────────────
    audio_buffer = []
    speech_detected = False
    silence_duration = 0.0

    current_onnx_text = ""
    current_hf_text = ""

    total_audio_sec = 0.0
    total_onnx_ms = 0.0
    total_hf_ms = 0.0
    total_tokens = 0
    n_lines = 0
    n_matches = 0

    try:
        term_cols = os.get_terminal_size().columns
    except OSError:
        term_cols = 80
    prev_display_lines = [0]

    def clear_display():
        for _ in range(prev_display_lines[0]):
            sys.stdout.write("\033[A\033[K")
        sys.stdout.write("\r\033[K")
        prev_display_lines[0] = 0

    def _wrapped_lines(text):
        if not text:
            return 0
        return max(0, (len(text) - 1) // term_cols)

    def show_live():
        clear_display()
        lines = 0
        if compare_mode:
            onnx_line = f"  ONNX> {current_onnx_text}"
            hf_line   = f"  HF  > {current_hf_text}"
            sys.stdout.write(onnx_line + "\n")
            lines += 1 + _wrapped_lines(onnx_line)
            sys.stdout.write(hf_line)
            lines += _wrapped_lines(hf_line)
        else:
            text = current_onnx_text or current_hf_text
            line = f"  > {text}"
            sys.stdout.write(line)
            lines += _wrapped_lines(line)
        prev_display_lines[0] = lines
        sys.stdout.flush()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio: {status}]", file=sys.stderr)
        audio_buffer.append(indata[:, 0].copy())

    last_update_time = time.perf_counter()
    last_displayed = ("", "")
    level_tick = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.05)

                if not audio_buffer:
                    continue
                last_block = audio_buffer[-1]
                dur = sum(len(b) for b in audio_buffer) / SAMPLE_RATE

                # ── VAD ──────────────────────────────────────────────
                last_rms = rms_energy(last_block)
                if last_rms >= SPEECH_THRESHOLD:
                    speech_detected = True
                    silence_duration = 0.0
                elif speech_detected:
                    silence_duration += 0.05

                # ── Display ──────────────────────────────────────────
                level_tick += 1
                cur = (current_onnx_text, current_hf_text)
                has_text = current_onnx_text or current_hf_text
                if has_text and cur != last_displayed:
                    show_live()
                    last_displayed = cur
                elif level_tick % 10 == 0 and not has_text:
                    bar_len = min(int(last_rms * 200), 40)
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    state = "SPEECH" if speech_detected else "quiet"
                    sys.stdout.write(f"\r\033[K  [{bar}] rms={last_rms:.4f} {state}")
                    sys.stdout.flush()

                # ── Discard background noise ─────────────────────────
                if not speech_detected and dur > 5.0:
                    audio_buffer.clear()
                    silence_duration = 0.0
                    continue

                # ── Finalize on silence or max buffer ────────────────
                should_finalize = (
                    (speech_detected and silence_duration >= args.silence_sec)
                    or dur >= MAX_BUFFER_SEC
                )

                if should_finalize:
                    text_onnx = text_hf = ""
                    stats_onnx = stats_hf = None

                    if dur >= MIN_AUDIO_SEC:
                        audio_snapshot = np.concatenate(audio_buffer).astype(np.float32)
                        if asr_onnx:
                            text_onnx, stats_onnx = asr_onnx.transcribe(audio_snapshot, incremental=True)
                            text_onnx = text_onnx.strip()
                        if asr_hf:
                            text_hf, stats_hf = asr_hf.transcribe(audio_snapshot)
                            text_hf = text_hf.strip()

                    any_text = text_onnx or text_hf
                    if any_text:
                        clear_display()
                        audio_sec = (stats_onnx or stats_hf)["audio_sec"]

                        if compare_mode:
                            match = text_onnx == text_hf
                            tag = "MATCH" if match else "DIFF"
                            n_matches += int(match)
                            onnx_ms = stats_onnx["total_ms"] if stats_onnx else 0
                            hf_ms = stats_hf["total_ms"] if stats_hf else 0
                            print(f"  [{audio_sec:.1f}s | ONNX {onnx_ms:.0f}ms | HF {hf_ms:.0f}ms | {tag}]")
                            print(f"  ONNX>> {text_onnx}")
                            print(f"  HF  >> {text_hf}")
                            total_onnx_ms += onnx_ms
                            total_hf_ms += hf_ms
                        elif stats_onnx:
                            print(f"  [{audio_sec:.1f}s | enc {stats_onnx['encode_ms']:.0f}ms | "
                                  f"dec {stats_onnx['decode_ms']:.0f}ms | "
                                  f"{stats_onnx['n_tokens']} tok | "
                                  f"RTF {stats_onnx['rtf']:.2f}]")
                            print(f"  >> {text_onnx}")
                            total_onnx_ms += stats_onnx["total_ms"]
                        else:
                            print(f"  [{audio_sec:.1f}s | {stats_hf['total_ms']:.0f}ms | "
                                  f"{stats_hf['n_tokens']} tok | "
                                  f"RTF {stats_hf['rtf']:.2f}]")
                            print(f"  >> {text_hf}")
                            total_hf_ms += stats_hf["total_ms"]
                        print()

                        total_audio_sec += audio_sec
                        total_tokens += (stats_onnx or stats_hf)["n_tokens"]
                        n_lines += 1

                    audio_buffer.clear()
                    current_onnx_text = ""
                    current_hf_text = ""
                    if asr_onnx:
                        asr_onnx.reset_encoder_cache()
                    if asr_hf:
                        asr_hf.reset_encoder_cache()
                    speech_detected = False
                    silence_duration = 0.0
                    last_displayed = ("", "")
                    last_update_time = time.perf_counter()
                    continue

                # ── Periodic inference (incremental) ─────────────────
                elapsed = time.perf_counter() - last_update_time
                if speech_detected and elapsed >= args.update_interval and dur >= MIN_AUDIO_SEC:
                    audio_snapshot = np.concatenate(audio_buffer).astype(np.float32)
                    try:
                        if asr_onnx:
                            t_onnx, _ = asr_onnx.transcribe(audio_snapshot, incremental=True)
                            current_onnx_text = t_onnx.strip()
                        if asr_hf:
                            t_hf, _ = asr_hf.transcribe(audio_snapshot)
                            current_hf_text = t_hf.strip()
                        last_update_time = time.perf_counter()
                        cur = (current_onnx_text, current_hf_text)
                        if cur != last_displayed:
                            show_live()
                            last_displayed = cur
                    except Exception as e:
                        print(f"\n  [inference error: {e}]", file=sys.stderr)

    except KeyboardInterrupt:
        pass

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Lines transcribed:   {n_lines}")
    print(f"  Total audio:         {total_audio_sec:.1f}s")
    if asr_onnx:
        total_onnx_sec = total_onnx_ms / 1000
        print(f"  ONNX inference:      {total_onnx_sec:.1f}s")
        if total_audio_sec > 0:
            print(f"  ONNX RTF:            {total_onnx_sec / total_audio_sec:.2f}x")
    if asr_hf:
        total_hf_sec = total_hf_ms / 1000
        print(f"  HF inference:        {total_hf_sec:.1f}s")
        if total_audio_sec > 0:
            print(f"  HF RTF:              {total_hf_sec / total_audio_sec:.2f}x")
    if compare_mode and n_lines > 0:
        print(f"  Matches:             {n_matches}/{n_lines} ({100 * n_matches / n_lines:.0f}%)")
    print(f"  Total tokens:        {total_tokens}")
    print(f"  Peak RAM:            {get_rss_mb():.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()