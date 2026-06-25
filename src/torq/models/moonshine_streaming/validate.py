# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging
from pathlib import Path
import os
import numpy as np

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
        fused_encoder_model=fused_encoder,
        decoder_model=decoder,
        model_size=model_size,
    )

    wav_path = Path(__file__).parent.parent / "moonshine_streaming" / "OSR_us_000_0010_8k.wav"
    if wav_path.exists():
        import soundfile as sf
        from scipy.signal import resample_poly
        from tokenizers import Tokenizer

        logger.info("Loading test audio '%s'...", wav_path.name)
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            data = resample_poly(data, up=16000, down=sr).astype(np.float32)

        tokens = runner.run(data[np.newaxis, :])
        tokenizer_path = p / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            transcribed = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
            logger.info("=" * 80)
            logger.info("Validation successful!")
            logger.info("Transcription: '%s'", transcribed)
            logger.info("=" * 80)
        else:
            logger.info("Validation successful, tokens: %s", str(tokens))
    else:
        logger.warning("Test audio not found — running with dummy audio...")
        dummy_audio = np.random.randn(1, 80000).astype(np.float32)
        tokens = runner.run(dummy_audio)
        logger.info("Validation successful on dummy audio, tokens: %s", str(tokens))


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
