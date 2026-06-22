# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate exported TFLite models on the sample audio (transcription smoke test).

Mirrors the 2-split ONNX validation: runs the fused_encoder + decoder_kv streaming
pipeline on the bundled OSR speech clip and prints the decoded text. The export is
sized for max_memory_len (~5 s), so — as with the ONNX validation — the transcription
covers roughly the first few sentences of the clip.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from ._inference import MoonshineStreamingTFLite
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from _inference import MoonshineStreamingTFLite

_SAMPLE_WAV = Path(__file__).parent.parent / "moonshine_streaming" / "OSR_us_000_0010_8k.wav"


_SUFFIX = {"none": "", "dynamic": "_int8", "static": "_int8_static"}


def validate(model_dir: str, model_size: str = "tiny", quant_mode: str = "none",
             wav: str | None = None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("validate_tflite")

    p = Path(model_dir)
    suffix = _SUFFIX[quant_mode]
    fused_encoder = p / f"fused_encoder{suffix}.tflite"
    decoder = p / f"decoder_kv{suffix}.tflite"
    for mp in (fused_encoder, decoder):
        if not mp.exists():
            raise FileNotFoundError(f"Missing required model: {mp}")

    logger.info("Loading TFLite runner (%s)...", quant_mode)
    runner = MoonshineStreamingTFLite.from_tflite(fused_encoder, decoder, model_size)

    wav_path = Path(wav) if wav else _SAMPLE_WAV
    if not wav_path.exists():
        logger.warning("Test audio not found at %s — using random audio", wav_path)
        tokens = runner.run(np.random.randn(1, 80000).astype(np.float32))
        logger.info("Tokens (dummy audio): %s", tokens)
        return tokens

    import soundfile as sf
    from scipy.signal import resample_poly

    logger.info("Loading test audio '%s'...", wav_path.name)
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = resample_poly(data, up=16000, down=sr).astype(np.float32)

    tokens = runner.run(data[np.newaxis, :])

    tokenizer_path = p / "tokenizer.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        text = tok.decode_batch(tokens, skip_special_tokens=True)[0]
        logger.info("=" * 80)
        logger.info("Validation successful (%s)!", quant_mode)
        logger.info("Transcription: '%s'", text)
        logger.info("=" * 80)
        return text
    logger.info("Validation successful, tokens: %s", tokens)
    return tokens


def main():
    ap = argparse.ArgumentParser(description="Validate TFLite Moonshine Streaming on sample audio.")
    ap.add_argument("-m", "--model-dir", required=True,
                    help="Directory with fused_encoder{,_int8,_int8_static}.tflite + decoder_kv*.tflite")
    ap.add_argument("-s", "--model-size", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--quant-mode", choices=["none", "dynamic", "static"], default="none",
                    help="Which exported variant to validate: none=fp32, dynamic=_int8, "
                         "static=_int8_static.")
    ap.add_argument("--int8", action="store_true", help="Deprecated alias for --quant-mode dynamic.")
    ap.add_argument("--wav", default=None, help="Override the input WAV (default: bundled OSR clip)")
    args = ap.parse_args()
    quant_mode = args.quant_mode
    if args.int8 and quant_mode == "none":
        quant_mode = "dynamic"
    validate(args.model_dir, args.model_size, quant_mode=quant_mode, wav=args.wav)


if __name__ == "__main__":
    main()
