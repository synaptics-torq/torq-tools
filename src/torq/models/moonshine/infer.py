# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import argparse
import logging

import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from torq.utils.logging import configure_logging

from . import add_moonshine_infer_args
from ._inference import load_moonshine
from ...utils.demo import format_answer


def _transcribe(wav: str | os.PathLike, runner, tokenizer) -> str:
    data, _ = sf.read(wav, dtype="float32")
    speech = data.astype(np.float32)[np.newaxis, :]
    tokens = runner.run(speech)
    text = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
    return text


def infer_moonshine(args: argparse.Namespace):
    configure_logging(args.logging)
    logger = logging.getLogger("Moonshine")
    logger.info("Starting demo...")
    runner = load_moonshine(args.model_dir, args.model_size, args.max_inp_len, args.max_dec_len, args.threads)
    tokenizer_file = hf_hub_download(f"UsefulSensors/moonshine-{args.model_size}", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    try:
        for wav in args.inputs:
            transcribed = _transcribe(wav, runner, tokenizer)
            print(format_answer(transcribed, runner.last_infer_time, agent_name="Transcribed"))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser("Run Moonshine inference")
    add_moonshine_infer_args(parser)
    infer_moonshine(parser.parse_args)


if __name__ == "__main__":
    main()
