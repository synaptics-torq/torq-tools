# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging
from pathlib import Path
import os
import numpy as np

# Import 5-split inference class
try:
    from ._inference import MoonshineStreaming5Split
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from _inference import MoonshineStreaming5Split


def validate(models_dir: str, model_size: str = "tiny"):
    """
    Validate the exported 5-split ONNX models using the MoonshineStreaming5Split runner.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("validate_5_split")

    p = Path(models_dir)
    frontend = p / "frontend.onnx"
    encoder = p / "encoder.onnx"
    adapter = p / "adapter.onnx"
    cross_kv = p / "cross_kv.onnx"
    decoder = p / "decoder_kv.onnx"

    for model_path in (frontend, encoder, adapter, cross_kv, decoder):
        if not model_path.exists():
            raise FileNotFoundError(f"Missing required model for validation: {model_path}")

    logger.info("Initializing MoonshineStreaming5Split runner...")
    runner = MoonshineStreaming5Split.from_onnx(
        frontend_model=frontend,
        encoder_model=encoder,
        adapter_model=adapter,
        cross_kv_model=cross_kv,
        decoder_model=decoder,
        model_size=model_size,
    )

    wav_path = Path(__file__).parent.parent / "moonshine_streaming" / "OSR_us_000_0010_8k.wav"
    if wav_path.exists():
        logger.info("Loading test audio file '%s' for validation...", wav_path.name)
        import soundfile as sf
        from scipy.signal import resample_poly
        from tokenizers import Tokenizer

        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            data = resample_poly(data, up=16000, down=sr).astype(np.float32)

        speech = data.astype(np.float32)[np.newaxis, :]
        tokens = runner.run(speech)

        tokenizer_path = p / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            transcribed = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
            logger.info(f"{'='*80}")
            logger.info("Validation successful!")
            logger.info("Transcribed text: '%s'", transcribed)
            logger.info(f"{'='*80}")
        else:
            logger.info("Validation successful, tokens: %s", str(tokens))
    else:
        logger.warning("Test audio file '%s' not found, running with dummy audio...", wav_path)
        dummy_audio = np.random.randn(1, 80000).astype(np.float32)
        tokens = runner.run(dummy_audio)
        logger.info("Validation successful on dummy audio, tokens: %s", str(tokens))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate 5-Split Moonshine Streaming.")
    parser.add_argument("-m", "--model-dir", required=True, help="Directory containing the 5-split ONNX models")
    parser.add_argument("-s", "--model-size", default="tiny", choices=["tiny", "small"], help="Model size")
    return parser.parse_args()


def main():
    args = parse_arguments()
    validate(args.model_dir, args.model_size)


if __name__ == "__main__":
    main()
