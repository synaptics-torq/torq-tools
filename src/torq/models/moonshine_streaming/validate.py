# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging
from pathlib import Path
import os
import numpy as np
from tokenizers import Tokenizer

try:
    from ._inference import MoonshineStreaming
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from _inference import MoonshineStreaming


def validate(models_dir: str, model_size: str = "tiny"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("validate_streaming")

    p = Path(models_dir)
    fused_encoder = p / "encoder.onnx"
    decoder       = p / "decoder.onnx"

    for model_path in (fused_encoder, decoder):
        if not model_path.exists():
            raise FileNotFoundError(f"Missing required model: {model_path}")

    logger.info("Initialising MoonshineStreaming runner...")
    runner = MoonshineStreaming.from_onnx(
        encoder_model=fused_encoder,
        decoder_model=decoder,
        model_size=model_size,
    )

    tokenizer_path = p / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists() else None

    try:
        from datasets import load_dataset, Audio
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        ).cast_column("audio", Audio(16_000))
        audio = dataset[0]["audio"]["array"].astype(np.float32)[np.newaxis, :]
        logger.info("Loaded validation sample from librispeech_asr_dummy")
    except Exception as exc:
        logger.warning("Could not load validation dataset (%s); using dummy audio", exc)
        audio = np.random.randn(1, 80_000).astype(np.float32)

    tokens = runner.run(audio)
    logger.info("=" * 80)
    if tokenizer is not None:
        logger.info("Transcription: '%s'", tokenizer.decode_batch(tokens, skip_special_tokens=True)[0])
    else:
        logger.info("Tokens: %s", tokens.tolist())
    logger.info("=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate Moonshine Streaming.")
    parser.add_argument("-m", "--model-dir", required=True,
                        help="Directory containing encoder.onnx and decoder.onnx")
    parser.add_argument("-s", "--model-size", default="tiny",
                        choices=["tiny", "small"], help="Model size")
    return parser.parse_args()


def main():
    args = parse_arguments()
    validate(args.model_dir, args.model_size)


if __name__ == "__main__":
    main()
