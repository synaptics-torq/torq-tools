# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tokenizers import Tokenizer

from . import add_moonshine_streaming_infer_args
from ._inference import load_moonshine_streaming
from ...utils.demo import format_answer
from ...utils.logging import configure_logging


def _load_audio_16k(wav: str | os.PathLike) -> np.ndarray:
    """Load a WAV as mono float32 at 16 kHz, shaped [1, num_samples]."""
    data, sr = sf.read(wav, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = resample_poly(data, up=16000, down=sr).astype(np.float32)
    return data.astype(np.float32)[np.newaxis, :]


def _transcribe(wav: str | os.PathLike, runner, tokenizer: Tokenizer) -> str:
    tokens = runner.run(_load_audio_16k(wav))
    return tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]


def infer_moonshine_streaming(args: argparse.Namespace):
    configure_logging(args.logging)
    logger = logging.getLogger("MoonshineStreaming")
    logger.info("Starting demo...")
    runner = load_moonshine_streaming(args.model_dir, args.model_size, args.threads)

    tokenizer_path = Path(args.model_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"'tokenizer.json' not found in model dir '{args.model_dir}'")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    try:
        for wav in args.inputs:
            transcribed = _transcribe(wav, runner, tokenizer)
            print(format_answer(transcribed, runner.last_infer_time, agent_name="Transcribed"))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser("Run Moonshine Streaming inference")
    add_moonshine_streaming_infer_args(parser)
    infer_moonshine_streaming(parser.parse_args())


if __name__ == "__main__":
    main()
