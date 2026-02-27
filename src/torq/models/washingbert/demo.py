# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
WashingBERT inference demo script.

Usage:
    PYTHONPATH="src:../python" python3 -m torq.models.washingbert.demo

    # Custom model path:
    PYTHONPATH="src:../python" python3 -m torq.models.washingbert.demo \
        --model models/WashingBERT/source/onnx/model.onnx

    # With your own Japanese text:
    PYTHONPATH="src:../python" python3 -m torq.models.washingbert.demo \
        --text "白いシャツの黄ばみを落としたい" "カビ取りをしたい"

    # Using pre-tokenized sample_inputs.json (no tokenizer deps required):
    PYTHONPATH="src:../python" python3 -m torq.models.washingbert.demo --use-samples
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import HF_TOKENIZER_REPO, DEFAULT_MAX_SEQ_LEN
from ._inference import WashingBERTRunner, LabelMap


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
logger = logging.getLogger("WashingBERT-demo")

DEFAULT_MODEL_PATH = "models/WashingBERT/source/onnx/model.onnx"

SAMPLE_SENTENCES: list[tuple[str, str]] = [
    ("白いシャツの黄ばみを落としたい",      "Remove yellowing from white shirt"),
    ("洗濯機の掃除方法を教えてください",     "How to clean the washing machine"),
    ("デリケートな衣類の洗い方は？",         "How to wash delicate clothes?"),
    ("タオルをふわふわにしたい",             "Want fluffy towels"),
    ("汗のニオイが取れない",                 "Cannot remove sweat smell"),
    ("布団を洗いたい",                       "Want to wash bedding"),
    ("カビ取りをしたい",                     "Want to remove mold"),
]


def _print_result(text: str, translation: str | None, result, infer_ms: float):
    print(f"{'─' * 70}")
    print(f"  Input : {text}")
    if translation:
        print(f"  (EN)  : {translation}")
    print(f"  Intent: {result.intent} ({result.intent_confidence:.3f})")
    if result.type1_labels:
        t1 = ", ".join(
            f"{l} ({s:.3f})" for l, s in zip(result.type1_labels, result.type1_scores)
        )
        print(f"  Type1 : {t1}")
    if result.type2_labels:
        t2 = ", ".join(
            f"{l} ({s:.3f})" for l, s in zip(result.type2_labels, result.type2_scores)
        )
        print(f"  Type2 : {t2}")
    print(f"  Time  : {infer_ms:.1f} ms")


def run_with_tokenizer(runner: WashingBERTRunner, texts: list[str], max_seq_len: int):
    """Run inference using the HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO, trust_remote_code=True)

    translations = {jp: en for jp, en in SAMPLE_SENTENCES}

    for text in texts:
        encoded = tokenizer(
            text,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        result = runner.run(encoded["input_ids"], encoded["attention_mask"])
        _print_result(text, translations.get(text), result, runner.last_infer_time * 1000)


def run_with_samples(runner: WashingBERTRunner, samples_path: Path):
    """Run inference using pre-tokenized sample_inputs.json (no tokenizer needed)."""
    if not samples_path.exists():
        print(f"Error: sample_inputs.json not found at '{samples_path}'", file=sys.stderr)
        sys.exit(1)

    with open(samples_path) as f:
        samples = json.load(f)

    for sample in samples:
        input_ids = np.array([sample["input_ids"]], dtype=np.int64)
        attention_mask = np.array([sample["attention_mask"]], dtype=np.int64)
        result = runner.run(input_ids, attention_mask)
        _print_result(sample["text"], None, result, runner.last_infer_time * 1000)


def main():
    parser = argparse.ArgumentParser(
        description="WashingBERT inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to WashingBERT ONNX model (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum sequence length (default: %(default)d)",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        metavar="TEXT",
        help="Japanese text(s) to classify (uses built-in samples if omitted)",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use pre-tokenized sample_inputs.json instead of the HF tokenizer",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads for inference",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)

    runner = WashingBERTRunner.from_onnx(
        model_path,
        max_seq_len=args.max_seq_len,
        threads=args.threads,
    )

    print(f"\nWashingBERT Inference Demo")
    print(f"Model: {model_path}")
    print(f"Max sequence length: {args.max_seq_len}")

    if args.use_samples:
        samples_path = model_path.parent / "sample_inputs.json"
        run_with_samples(runner, samples_path)
    else:
        texts = args.text or [jp for jp, _ in SAMPLE_SENTENCES]
        run_with_tokenizer(runner, texts, args.max_seq_len)

    print(f"{'─' * 70}")
    print(f"Avg inference time: {runner.avg_infer_time * 1000:.1f} ms")
    print()


if __name__ == "__main__":
    main()
