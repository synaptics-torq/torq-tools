# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging

from torq.utils.logging import configure_logging
from transformers import AutoTokenizer

from . import HF_TOKENIZER_REPO, add_washingbert_infer_args
from ._inference import WashingBERTRunner, LabelMap
from ...utils.demo import format_answer


def infer_washingbert(args: argparse.Namespace):
    configure_logging(args.logging)
    logger = logging.getLogger("WashingBERT")
    logger.info("Starting inference...")

    label_map = LabelMap.from_dir(args.model_dir)
    runner = WashingBERTRunner.from_onnx(
        model_path=args.model_dir,
        max_seq_len=args.max_seq_len,
        label_map=label_map,
        threads=args.threads,
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO, trust_remote_code=True)

    try:
        for text in args.inputs:
            encoded = tokenizer(
                text,
                max_length=args.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            result = runner.run(encoded["input_ids"], encoded["attention_mask"])
            print(format_answer(str(result), runner.last_infer_time, agent_name="WashingBERT"))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser("Run WashingBERT inference")
    add_washingbert_infer_args(parser)
    infer_washingbert(parser.parse_args())


if __name__ == "__main__":
    main()
