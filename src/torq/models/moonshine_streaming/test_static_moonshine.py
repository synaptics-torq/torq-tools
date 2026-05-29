#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoConfig, AutoProcessor


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


def get_static_audio_length(preprocessor_sess: ort.InferenceSession) -> int:
    for inp in preprocessor_sess.get_inputs():
        if inp.name == "input_values":
            return int(inp.shape[1])
    raise RuntimeError("Could not determine static audio length")


def get_static_batch_size(preprocessor_sess: ort.InferenceSession) -> int:
    for inp in preprocessor_sess.get_inputs():
        if inp.name == "input_values":
            return int(inp.shape[0])
    raise RuntimeError("Could not determine static batch size")


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


def initialize_decoder_cache(decoder_sess: ort.InferenceSession):
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
                raise RuntimeError(f"Expected static input shape for {inp.name}, got {inp.shape}")
            shape.append(d)

        cache[inp.name] = np.zeros(shape, dtype=np_dtype_from_ort(inp.type))
    return cache


def split_decoder_outputs(decoder_sess: ort.InferenceSession, ort_outs):
    output_names = [o.name for o in decoder_sess.get_outputs()]
    output_map = dict(zip(output_names, ort_outs))

    last_hidden_state = output_map["last_hidden_state"]
    updated_past_valid_len = output_map["updated_past_valid_len"]

    cache_outputs = {
        k: v for k, v in output_map.items()
        if k not in {"last_hidden_state", "updated_past_valid_len"}
    }
    return last_hidden_state, updated_past_valid_len, cache_outputs


def decode_one_chunk_merged(
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

    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        eos_set = None
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_set = set(int(x) for x in eos_token_id)
    else:
        eos_set = {int(eos_token_id)}

    cache_inputs = initialize_decoder_cache(decoder_sess)
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

        last_hidden_state, updated_past_valid_len, cache_outputs = split_decoder_outputs(decoder_sess, ort_outs)

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


def transcribe_long_audio_static(
    audio: np.ndarray,
    sampling_rate: int,
    processor,
    config,
    preprocessor_sess,
    encoder_sess,
    decoder_sess,
    token_embeddings: np.ndarray,
    hop_size_samples: int | None = None,
    max_tokens_per_chunk: int | None = None,
    verbose: bool = True,
):
    chunk_size_samples = get_static_audio_length(preprocessor_sess)

    if max_tokens_per_chunk is None:
        token_limit_factor = 6.5 / sampling_rate
        max_tokens_per_chunk = max(2, int(chunk_size_samples * token_limit_factor))

    if verbose:
        print("\nValidation settings:")
        print(f"  chunk_size_samples   = {chunk_size_samples}")
        print(f"  hop_size_samples     = {hop_size_samples if hop_size_samples is not None else chunk_size_samples}")
        print(f"  max_tokens_per_chunk = {max_tokens_per_chunk}")

    chunk_results = []

    for chunk_idx, (chunk_audio_1d, valid_len, is_last, start, end) in enumerate(
        chunk_audio(audio, chunk_size_samples, hop_size_samples)
    ):
        input_features, padding_mask = run_preprocessor(
            preprocessor_sess,
            chunk_audio_1d,
            valid_len,
        )
        encoder_hidden_states = run_encoder(
            encoder_sess,
            input_features,
            padding_mask,
        )
        token_ids = decode_one_chunk_merged(
            decoder_sess=decoder_sess,
            token_embeddings=token_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=padding_mask,
            config=config,
            max_tokens=max_tokens_per_chunk,
        )

        text = processor.decode(token_ids, skip_special_tokens=True)

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
            start_s = start / sampling_rate
            end_s = end / sampling_rate
            print(
                f"[chunk {chunk_idx:03d}] "
                f"samples {start}:{end} "
                f"({start_s:.2f}s → {end_s:.2f}s), "
                f"valid_len={valid_len}, "
                f"tokens={len(token_ids)}"
            )
            print(f"  text: {text}")

    full_text = " ".join(
        x["text"].strip()
        for x in chunk_results
        if x["text"].strip()
    ).strip()

    return full_text, chunk_results


def parse_args():
    p = argparse.ArgumentParser(description="Static Moonshine Streaming ONNX merged-decoder validation harness")
    p.add_argument("--model-dir", type=str, default="/home/yhtet/projects/moonshine-streaming/torq-tools-dev/src/torq/models/moonshine_streaming/models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/static_merged")
    p.add_argument("--audio", type=str, default="/home/yhtet/projects/moonshine-streaming/torq-tools-dev/src/torq/models/moonshine_streaming/OSR_us_000_0010_8k.wav")
    p.add_argument("--hf-repo", type=str, default="UsefulSensors/moonshine-streaming-tiny")
    p.add_argument("--hop-size-samples", type=int, default=None)
    p.add_argument("--max-tokens-per-chunk", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    audio_path = Path(args.audio)

    preprocessor_sess = make_session(model_dir / "preprocessor.onnx", use_cuda=not args.cpu)
    encoder_sess = make_session(model_dir / "encoder.onnx", use_cuda=not args.cpu)
    decoder_sess = make_session(model_dir / "decoder_merged.onnx", use_cuda=not args.cpu)

    print_session_io("preprocessor.onnx", preprocessor_sess)
    print_session_io("encoder.onnx", encoder_sess)
    print_session_io("decoder_merged.onnx", decoder_sess)

    repo_or_path = args.hf_repo if args.hf_repo is not None else str(model_dir)

    config = AutoConfig.from_pretrained(repo_or_path)
    processor = AutoProcessor.from_pretrained(repo_or_path)

    token_embeddings_path = model_dir / "decoder_token_embeddings.npy"
    if not token_embeddings_path.exists():
        raise FileNotFoundError(f"Missing {token_embeddings_path}")
    token_embeddings = np.load(token_embeddings_path).astype(np.float32)

    static_batch_size = get_static_batch_size(preprocessor_sess)
    static_audio_length = get_static_audio_length(preprocessor_sess)

    print("\nStatic model contract:")
    print(f"  batch_size       = {static_batch_size}")
    print(f"  audio_length     = {static_audio_length}")
    print(f"  sampling_rate    = {processor.feature_extractor.sampling_rate}")
    print(f"  token_embeddings = {token_embeddings.shape}")

    if static_batch_size != 1:
        raise RuntimeError(
            f"This validation harness currently expects batch_size=1, but model has batch_size={static_batch_size}"
        )

    audio = load_audio(audio_path, processor.feature_extractor.sampling_rate)
    duration_s = len(audio) / processor.feature_extractor.sampling_rate

    print(f"\nLoaded audio: {audio_path}")
    print(f"  samples   = {len(audio)}")
    print(f"  duration  = {duration_s:.2f}s")

    full_text, chunk_results = transcribe_long_audio_static(
        audio=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        processor=processor,
        config=config,
        preprocessor_sess=preprocessor_sess,
        encoder_sess=encoder_sess,
        decoder_sess=decoder_sess,
        token_embeddings=token_embeddings,
        hop_size_samples=args.hop_size_samples,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        verbose=not args.quiet,
    )

    print("\n==============================")
    print("Validation summary")
    print("==============================")
    print(f"Chunks processed: {len(chunk_results)}")
    print("\nConcatenated text:")
    print(full_text)


if __name__ == "__main__":
    main()