#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoConfig, AutoProcessor


# ============================================================
# ORT / Utility helpers
# ============================================================

def make_session(model_path: Path, use_cuda: bool = True):
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)


def print_session_io(name: str, sess: ort.InferenceSession):
    print(f"\n{name}")
    print("  Inputs:")
    for x in sess.get_inputs():
        print(f"    {x.name:35s} shape={x.shape} type={x.type}")
    print("  Outputs:")
    for x in sess.get_outputs():
        print(f"    {x.name:35s} shape={x.shape} type={x.type}")


def np_dtype_from_ort(ort_type: str):
    ort_type = ort_type.lower()
    if "bool" in ort_type:
        return np.bool_
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type or "double" in ort_type:
        return np.float32
    return np.float32


def cast_for_session_input(sess: ort.InferenceSession, input_name: str, array: np.ndarray) -> np.ndarray:
    for inp in sess.get_inputs():
        if inp.name == input_name:
            return array.astype(np_dtype_from_ort(inp.type), copy=False)
    raise KeyError(f"Input '{input_name}' not found in session")


def normalize_name(name: str) -> str:
    return name.lower().strip()


def hidden_to_logits(last_hidden_state: np.ndarray, token_embeddings: np.ndarray) -> np.ndarray:
    return np.matmul(last_hidden_state.astype(np.float32), token_embeddings.T.astype(np.float32))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def load_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(audio_path))
    audio = normalize_audio(audio)
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz ...")
        audio = resample_poly(audio, up=target_sr, down=sr).astype(np.float32)
    return audio


def get_audio_input_shape(preprocessor_sess: ort.InferenceSession):
    for inp in preprocessor_sess.get_inputs():
        if inp.name == "input_values":
            return inp.shape
    raise RuntimeError("Could not determine input_values shape")


def get_batch_size(preprocessor_sess: ort.InferenceSession) -> int | None:
    shape = get_audio_input_shape(preprocessor_sess)
    d0 = shape[0]
    return int(d0) if isinstance(d0, int) else None


def get_static_audio_length(preprocessor_sess: ort.InferenceSession) -> int | None:
    shape = get_audio_input_shape(preprocessor_sess)
    d1 = shape[1]
    return int(d1) if isinstance(d1, int) else None


def get_eos_set(config):
    eos = config.eos_token_id
    if eos is None:
        return None
    if isinstance(eos, (list, tuple, set)):
        return set(int(x) for x in eos)
    return {int(eos)}


def load_token_embeddings(model_dir: Path):
    path = model_dir / "decoder_token_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path).astype(np.float32)


# ============================================================
# Audio chunking
# ============================================================

def chunk_audio(audio: np.ndarray, chunk_size: int, hop_size: int | None = None):
    if hop_size is None:
        hop_size = chunk_size

    n = len(audio)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = audio[start:end]
        valid_len = len(chunk)
        is_last = end >= n

        if valid_len < chunk_size:
            padded = np.zeros((chunk_size,), dtype=np.float32)
            padded[:valid_len] = chunk
            chunk = padded
        else:
            chunk = chunk.astype(np.float32)

        yield chunk, valid_len, is_last, start, end
        start += hop_size


# ============================================================
# Shared preprocessor / encoder
# ============================================================

def run_preprocessor(preprocessor_sess, chunk_audio_1d: np.ndarray, valid_len: int):
    audio_len = len(chunk_audio_1d)

    input_values = chunk_audio_1d[None, :].astype(np.float32)
    attention_mask = np.zeros((1, audio_len), dtype=np.int64)
    attention_mask[:, :valid_len] = 1

    input_values = cast_for_session_input(preprocessor_sess, "input_values", input_values)
    attention_mask = cast_for_session_input(preprocessor_sess, "attention_mask", attention_mask)

    outputs = preprocessor_sess.run(
        None,
        {
            "input_values": input_values,
            "attention_mask": attention_mask,
        },
    )

    output_names = [o.name for o in preprocessor_sess.get_outputs()]
    output_map = dict(zip(output_names, outputs))
    return output_map["input_features"], output_map["padding_mask"]


def run_encoder(encoder_sess, input_features: np.ndarray, padding_mask: np.ndarray):
    input_features = cast_for_session_input(encoder_sess, "input_features", input_features)
    attention_mask = cast_for_session_input(encoder_sess, "attention_mask", padding_mask)

    outputs = encoder_sess.run(
        None,
        {
            "input_features": input_features,
            "attention_mask": attention_mask,
        },
    )
    return outputs[0]


# ============================================================
# Static merged decoder path
# ============================================================

def initialize_decoder_cache_merged(decoder_sess: ort.InferenceSession):
    cache = {}
    for inp in decoder_sess.get_inputs():
        if inp.name in {
            "decoder_input_ids",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "past_valid_len",
        }:
            continue

        shape = []
        for d in inp.shape:
            if not isinstance(d, int):
                raise RuntimeError(
                    f"Static merged decoder expected static cache shape for {inp.name}, got {inp.shape}"
                )
            shape.append(d)

        cache[inp.name] = np.zeros(shape, dtype=np_dtype_from_ort(inp.type))
    return cache


def split_decoder_outputs_merged(decoder_sess: ort.InferenceSession, ort_outs):
    output_names = [o.name for o in decoder_sess.get_outputs()]
    output_map = dict(zip(output_names, ort_outs))

    last_hidden_state = output_map["last_hidden_state"]
    updated_past_valid_len = output_map["updated_past_valid_len"]

    cache_outputs = {
        k: v for k, v in output_map.items()
        if k not in {"last_hidden_state", "updated_past_valid_len"}
    }
    return last_hidden_state, updated_past_valid_len, cache_outputs


def decode_one_chunk_static_merged(
    decoder_sess,
    token_embeddings: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray,
    config,
    max_tokens: int,
):
    start_token_id = config.decoder_start_token_id
    if start_token_id is None:
        start_token_id = config.bos_token_id
    if start_token_id is None:
        raise RuntimeError("No decoder_start_token_id or bos_token_id in config")

    eos_set = get_eos_set(config)

    cache_inputs = initialize_decoder_cache_merged(decoder_sess)
    past_valid_len = np.array([0], dtype=np.int64)

    generated = [int(start_token_id)]
    next_token_id = int(start_token_id)

    enc_hidden = cast_for_session_input(decoder_sess, "encoder_hidden_states", encoder_hidden_states)
    enc_mask = cast_for_session_input(decoder_sess, "encoder_attention_mask", encoder_attention_mask)

    for _ in range(max_tokens):
        decoder_input_ids = np.array([[next_token_id]], dtype=np.int64)
        decoder_input_ids = cast_for_session_input(decoder_sess, "decoder_input_ids", decoder_input_ids)
        past_valid_len_cast = cast_for_session_input(decoder_sess, "past_valid_len", past_valid_len)

        feeds = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_hidden,
            "encoder_attention_mask": enc_mask,
            "past_valid_len": past_valid_len_cast,
        }

        for name, tensor in cache_inputs.items():
            feeds[name] = cast_for_session_input(decoder_sess, name, tensor)

        ort_outs = decoder_sess.run(None, feeds)

        last_hidden_state, updated_past_valid_len, cache_outputs = split_decoder_outputs_merged(
            decoder_sess, ort_outs
        )

        logits = hidden_to_logits(last_hidden_state, token_embeddings)
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(next_token_id)

        cache_inputs = {
            name.replace("present_", "past_", 1): value
            for name, value in cache_outputs.items()
        }
        past_valid_len = updated_past_valid_len.astype(np.int64, copy=False)

        if eos_set is not None and next_token_id in eos_set:
            break

    return generated


# ============================================================
# Dynamic split decoder path
#   decoder.onnx
#   decoder_with_past.onnx
# ============================================================

def is_past_input_name(name: str):
    n = normalize_name(name)
    return "past" in n or "key_values" in n or "cache" in n


def build_decoder_feeds_split(
    decoder_sess,
    decoder_input_ids,
    encoder_hidden_states,
    encoder_attention_mask,
    past_key_values=None,
):
    feeds = {}
    past_iter = iter([] if past_key_values is None else past_key_values)

    for inp in decoder_sess.get_inputs():
        name = normalize_name(inp.name)
        dtype = np_dtype_from_ort(inp.type)

        if inp.name == "decoder_input_ids" or "decoder_input_ids" in name:
            feeds[inp.name] = decoder_input_ids.astype(dtype)

        elif inp.name == "encoder_hidden_states" or "encoder_hidden_states" in name:
            feeds[inp.name] = encoder_hidden_states.astype(dtype)

        elif inp.name == "encoder_attention_mask" or "encoder_attention_mask" in name:
            feeds[inp.name] = encoder_attention_mask.astype(dtype)

        elif is_past_input_name(inp.name):
            try:
                pkv = next(past_iter)
            except StopIteration:
                raise ValueError(f"Missing past tensor for input: {inp.name}")
            feeds[inp.name] = pkv.astype(dtype)

        elif "use_cache_branch" in name:
            feeds[inp.name] = np.array([True], dtype=np.bool_)

        else:
            raise ValueError(f"Unexpected decoder input: {inp.name}")

    return feeds


def parse_decoder_outputs_split(ort_outputs, token_embeddings):
    """
    Export returns:
      [last_hidden_state] + kv outputs

    Convert hidden states -> logits using tied token embeddings.
    """
    if len(ort_outputs) == 0:
        raise ValueError("Decoder returned no outputs")

    last_hidden_state = ort_outputs[0]
    logits = hidden_to_logits(last_hidden_state, token_embeddings)
    present = ort_outputs[1:]
    return logits, present


def decode_one_chunk_dynamic_split(
    decoder_sess,
    decoder_with_past_sess,
    token_embeddings: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray,
    config,
    max_tokens: int,
):
    start_token_id = config.decoder_start_token_id
    if start_token_id is None:
        start_token_id = config.bos_token_id
    if start_token_id is None:
        raise RuntimeError("No decoder_start_token_id or bos_token_id in config")

    eos_set = get_eos_set(config)

    generated = [int(start_token_id)]

    # ---- first step: decoder.onnx ----
    decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)

    feeds = build_decoder_feeds_split(
        decoder_sess=decoder_sess,
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=None,
    )

    ort_outputs = decoder_sess.run(None, feeds)
    logits, past_key_values = parse_decoder_outputs_split(ort_outputs, token_embeddings)

    next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    generated.append(next_token_id)

    if eos_set is not None and next_token_id in eos_set:
        return generated

    # ---- subsequent steps: decoder_with_past.onnx ----
    while len(generated) < max_tokens:
        decoder_input_ids = np.array([[next_token_id]], dtype=np.int64)

        feeds = build_decoder_feeds_split(
            decoder_sess=decoder_with_past_sess,
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
        )

        ort_outputs = decoder_with_past_sess.run(None, feeds)
        logits, past_key_values = parse_decoder_outputs_split(ort_outputs, token_embeddings)

        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(next_token_id)

        if eos_set is not None and next_token_id in eos_set:
            break

    return generated


# ============================================================
# Model bundle loaders
# ============================================================

@dataclass
class StaticBundle:
    name: str
    model_dir: Path
    preprocessor_sess: ort.InferenceSession
    encoder_sess: ort.InferenceSession
    decoder_merged_sess: ort.InferenceSession
    config: object
    processor: object
    token_embeddings: np.ndarray
    batch_size: int | None
    static_audio_length: int | None


@dataclass
class DynamicBundle:
    name: str
    model_dir: Path
    preprocessor_sess: ort.InferenceSession
    encoder_sess: ort.InferenceSession
    decoder_sess: ort.InferenceSession
    decoder_with_past_sess: ort.InferenceSession
    config: object
    processor: object
    token_embeddings: np.ndarray
    batch_size: int | None
    static_audio_length: int | None


def load_static_bundle(name: str, model_dir: Path, repo_or_path: str | None, use_cuda: bool) -> StaticBundle:
    preprocessor_sess = make_session(model_dir / "preprocessor.onnx", use_cuda=use_cuda)
    encoder_sess = make_session(model_dir / "encoder.onnx", use_cuda=use_cuda)
    decoder_merged_sess = make_session(model_dir / "decoder_merged.onnx", use_cuda=use_cuda)

    print_session_io(f"{name}/preprocessor.onnx", preprocessor_sess)
    print_session_io(f"{name}/encoder.onnx", encoder_sess)
    print_session_io(f"{name}/decoder_merged.onnx", decoder_merged_sess)

    repo = repo_or_path if repo_or_path is not None else str(model_dir)
    config = AutoConfig.from_pretrained(repo)
    processor = AutoProcessor.from_pretrained(repo)
    token_embeddings = load_token_embeddings(model_dir)

    batch_size = get_batch_size(preprocessor_sess)
    static_audio_length = get_static_audio_length(preprocessor_sess)

    print(f"\n{name} model contract:")
    print(f"  batch_size       = {batch_size if batch_size is not None else 'dynamic'}")
    print(f"  audio_length     = {static_audio_length if static_audio_length is not None else 'dynamic'}")
    print(f"  sampling_rate    = {processor.feature_extractor.sampling_rate}")
    print(f"  token_embeddings = {token_embeddings.shape}")

    if batch_size is not None and batch_size != 1:
        raise RuntimeError(f"{name}: batch_size=1 expected, got {batch_size}")

    return StaticBundle(
        name=name,
        model_dir=model_dir,
        preprocessor_sess=preprocessor_sess,
        encoder_sess=encoder_sess,
        decoder_merged_sess=decoder_merged_sess,
        config=config,
        processor=processor,
        token_embeddings=token_embeddings,
        batch_size=batch_size,
        static_audio_length=static_audio_length,
    )


def load_dynamic_bundle(name: str, model_dir: Path, repo_or_path: str | None, use_cuda: bool) -> DynamicBundle:
    preprocessor_sess = make_session(model_dir / "preprocessor.onnx", use_cuda=use_cuda)
    encoder_sess = make_session(model_dir / "encoder.onnx", use_cuda=use_cuda)
    decoder_sess = make_session(model_dir / "decoder.onnx", use_cuda=use_cuda)
    decoder_with_past_sess = make_session(model_dir / "decoder_with_past.onnx", use_cuda=use_cuda)

    print_session_io(f"{name}/preprocessor.onnx", preprocessor_sess)
    print_session_io(f"{name}/encoder.onnx", encoder_sess)
    print_session_io(f"{name}/decoder.onnx", decoder_sess)
    print_session_io(f"{name}/decoder_with_past.onnx", decoder_with_past_sess)

    repo = repo_or_path if repo_or_path is not None else str(model_dir)
    config = AutoConfig.from_pretrained(repo)
    processor = AutoProcessor.from_pretrained(repo)
    token_embeddings = load_token_embeddings(model_dir)

    batch_size = get_batch_size(preprocessor_sess)
    static_audio_length = get_static_audio_length(preprocessor_sess)

    print(f"\n{name} model contract:")
    print(f"  batch_size       = {batch_size if batch_size is not None else 'dynamic'}")
    print(f"  audio_length     = {static_audio_length if static_audio_length is not None else 'dynamic'}")
    print(f"  sampling_rate    = {processor.feature_extractor.sampling_rate}")
    print(f"  token_embeddings = {token_embeddings.shape}")

    if batch_size is not None and batch_size != 1:
        raise RuntimeError(f"{name}: batch_size=1 expected, got {batch_size}")

    return DynamicBundle(
        name=name,
        model_dir=model_dir,
        preprocessor_sess=preprocessor_sess,
        encoder_sess=encoder_sess,
        decoder_sess=decoder_sess,
        decoder_with_past_sess=decoder_with_past_sess,
        config=config,
        processor=processor,
        token_embeddings=token_embeddings,
        batch_size=batch_size,
        static_audio_length=static_audio_length,
    )


# ============================================================
# Generic chunk transcription
# ============================================================

def transcribe_long_audio_static_bundle(
    bundle: StaticBundle,
    audio: np.ndarray,
    chunk_size_samples: int,
    hop_size_samples: int | None = None,
    max_tokens_per_chunk: int | None = None,
    verbose: bool = True,
):
    sampling_rate = bundle.processor.feature_extractor.sampling_rate

    if max_tokens_per_chunk is None:
        token_limit_factor = 6.5 / sampling_rate
        max_tokens_per_chunk = max(2, int(chunk_size_samples * token_limit_factor))

    if verbose:
        print("\nValidation settings (static):")
        print(f"  chunk_size_samples   = {chunk_size_samples}")
        print(f"  hop_size_samples     = {hop_size_samples if hop_size_samples is not None else chunk_size_samples}")
        print(f"  max_tokens_per_chunk = {max_tokens_per_chunk}")

    chunk_results = []

    for chunk_idx, (chunk_audio_1d, valid_len, is_last, start, end) in enumerate(
        chunk_audio(audio, chunk_size_samples, hop_size_samples)
    ):
        input_features, padding_mask = run_preprocessor(bundle.preprocessor_sess, chunk_audio_1d, valid_len)
        encoder_hidden_states = run_encoder(bundle.encoder_sess, input_features, padding_mask)

        token_ids = decode_one_chunk_static_merged(
            decoder_sess=bundle.decoder_merged_sess,
            token_embeddings=bundle.token_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=padding_mask,
            config=bundle.config,
            max_tokens=max_tokens_per_chunk,
        )

        text = bundle.processor.decode(token_ids, skip_special_tokens=True)

        info = {
            "chunk_index": chunk_idx,
            "sample_range": (start, end),
            "valid_len": valid_len,
            "is_last_chunk": is_last,
            "token_ids": token_ids,
            "text": text,
        }
        chunk_results.append(info)

        if verbose:
            print(
                f"[static chunk {chunk_idx:03d}] "
                f"samples {start}:{end} "
                f"({start / sampling_rate:.2f}s → {end / sampling_rate:.2f}s), "
                f"valid_len={valid_len}, tokens={len(token_ids)}"
            )
            print(f"  text: {text}")

    full_text = " ".join(x["text"].strip() for x in chunk_results if x["text"].strip()).strip()
    return full_text, chunk_results


def transcribe_long_audio_dynamic_bundle(
    bundle: DynamicBundle,
    audio: np.ndarray,
    chunk_size_samples: int,
    hop_size_samples: int | None = None,
    max_tokens_per_chunk: int | None = None,
    verbose: bool = True,
):
    sampling_rate = bundle.processor.feature_extractor.sampling_rate

    if max_tokens_per_chunk is None:
        token_limit_factor = 6.5 / sampling_rate
        max_tokens_per_chunk = max(2, int(chunk_size_samples * token_limit_factor))

    if verbose:
        print("\nValidation settings (dynamic):")
        print(f"  chunk_size_samples   = {chunk_size_samples}")
        print(f"  hop_size_samples     = {hop_size_samples if hop_size_samples is not None else chunk_size_samples}")
        print(f"  max_tokens_per_chunk = {max_tokens_per_chunk}")

    chunk_results = []

    for chunk_idx, (chunk_audio_1d, valid_len, is_last, start, end) in enumerate(
        chunk_audio(audio, chunk_size_samples, hop_size_samples)
    ):
        input_features, padding_mask = run_preprocessor(bundle.preprocessor_sess, chunk_audio_1d, valid_len)
        encoder_hidden_states = run_encoder(bundle.encoder_sess, input_features, padding_mask)

        token_ids = decode_one_chunk_dynamic_split(
            decoder_sess=bundle.decoder_sess,
            decoder_with_past_sess=bundle.decoder_with_past_sess,
            token_embeddings=bundle.token_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=padding_mask,
            config=bundle.config,
            max_tokens=max_tokens_per_chunk,
        )

        text = bundle.processor.decode(token_ids, skip_special_tokens=True)

        info = {
            "chunk_index": chunk_idx,
            "sample_range": (start, end),
            "valid_len": valid_len,
            "is_last_chunk": is_last,
            "token_ids": token_ids,
            "text": text,
        }
        chunk_results.append(info)

        if verbose:
            print(
                f"[dynamic chunk {chunk_idx:03d}] "
                f"samples {start}:{end} "
                f"({start / sampling_rate:.2f}s → {end / sampling_rate:.2f}s), "
                f"valid_len={valid_len}, tokens={len(token_ids)}"
            )
            print(f"  text: {text}")

    full_text = " ".join(x["text"].strip() for x in chunk_results if x["text"].strip()).strip()
    return full_text, chunk_results


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate Moonshine streaming ONNX static merged vs dynamic split-decoder on the same audio"
    )
    p.add_argument(
        "--static-model-dir",
        type=str,
        default="/home/yhtet/projects/moonshine-streaming/torq-tools-dev/src/torq/models/moonshine_streaming/models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/static_merged",
    )
    p.add_argument(
        "--dynamic-model-dir",
        type=str,
        default="/home/yhtet/projects/moonshine-streaming/torq-tools-dev/models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic",
    )
    p.add_argument(
        "--audio",
        type=str,
        default="/home/yhtet/projects/moonshine-streaming/torq-tools-dev/src/torq/models/moonshine_streaming/OSR_us_000_0010_8k.wav",
    )
    p.add_argument("--hf-repo", type=str, default="UsefulSensors/moonshine-streaming-tiny")
    p.add_argument("--dynamic-hf-repo", type=str, default=None)
    p.add_argument("--hop-size-samples", type=int, default=None)
    p.add_argument(
        "--dynamic-chunk-size-samples",
        type=int,
        default=None,
        help="Chunk size for the dynamic model. Defaults to the static model's chunk size.",
    )
    p.add_argument("--max-tokens-per-chunk", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    use_cuda = not args.cpu

    static_bundle = load_static_bundle(
        name="static",
        model_dir=Path(args.static_model_dir),
        repo_or_path=args.hf_repo,
        use_cuda=use_cuda,
    )

    dynamic_bundle = load_dynamic_bundle(
        name="dynamic",
        model_dir=Path(args.dynamic_model_dir),
        repo_or_path=args.dynamic_hf_repo if args.dynamic_hf_repo is not None else args.hf_repo,
        use_cuda=use_cuda,
    )

    static_sr = static_bundle.processor.feature_extractor.sampling_rate
    dynamic_sr = dynamic_bundle.processor.feature_extractor.sampling_rate
    if static_sr != dynamic_sr:
        raise RuntimeError(f"Sampling-rate mismatch: static={static_sr}, dynamic={dynamic_sr}")

    if static_bundle.static_audio_length is None:
        raise RuntimeError(
            "Static preprocessor input_values length is dynamic, but this harness expects "
            "the static bundle to define the reference chunk size."
        )

    static_chunk_size = static_bundle.static_audio_length
    dynamic_chunk_size = args.dynamic_chunk_size_samples or static_chunk_size

    audio_path = Path(args.audio)
    audio = load_audio(audio_path, static_sr)
    duration_s = len(audio) / static_sr

    print(f"\nLoaded audio: {audio_path}")
    print(f"  samples   = {len(audio)}")
    print(f"  duration  = {duration_s:.2f}s")

    print("\n" + "=" * 30)
    print("Running STATIC merged model")
    print("=" * 30)

    static_text, static_chunk_results = transcribe_long_audio_static_bundle(
        bundle=static_bundle,
        audio=audio,
        chunk_size_samples=static_chunk_size,
        hop_size_samples=args.hop_size_samples,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        verbose=not args.quiet,
    )

    print("\n==============================")
    print("STATIC validation summary")
    print("==============================")
    print(f"Chunks processed: {len(static_chunk_results)}")
    print("\nConcatenated text:")
    print(static_text)

    print("\n" + "=" * 30)
    print("Running DYNAMIC split-decoder model")
    print("=" * 30)

    dynamic_text, dynamic_chunk_results = transcribe_long_audio_dynamic_bundle(
        bundle=dynamic_bundle,
        audio=audio,
        chunk_size_samples=dynamic_chunk_size,
        hop_size_samples=args.hop_size_samples,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        verbose=not args.quiet,
    )

    print("\n==============================")
    print("DYNAMIC validation summary")
    print("==============================")
    print(f"Chunks processed: {len(dynamic_chunk_results)}")
    print("\nConcatenated text:")
    print(dynamic_text)

    print("\n==============================")
    print("STATIC vs DYNAMIC comparison")
    print("==============================")
    print(f"Static chunks      : {len(static_chunk_results)}")
    print(f"Dynamic chunks     : {len(dynamic_chunk_results)}")
    print(f"Same chunk size    : {static_chunk_size == dynamic_chunk_size}")
    print(f"Exact text match   : {static_text == dynamic_text}")

    static_ids_flat = [tid for c in static_chunk_results for tid in c["token_ids"]]
    dynamic_ids_flat = [tid for c in dynamic_chunk_results for tid in c["token_ids"]]

    print(f"Static token count : {len(static_ids_flat)}")
    print(f"Dynamic token count: {len(dynamic_ids_flat)}")
    print(f"Exact token match  : {static_ids_flat == dynamic_ids_flat}")

    if static_text != dynamic_text:
        print("\n--- STATIC TEXT ---")
        print(static_text)
        print("\n--- DYNAMIC TEXT ---")
        print(dynamic_text)


if __name__ == "__main__":
    main()