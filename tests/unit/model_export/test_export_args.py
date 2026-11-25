# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Defaults and opt-outs defined by the shared argument helpers."""

import argparse

from torq.utils.onnx import add_decoder_args, add_onnx_args


def test_extract_embeddings_is_on_by_default():
    parser = argparse.ArgumentParser()
    add_onnx_args(parser, extract_embeddings=True)
    assert parser.parse_args([]).extract_embeddings is True
    assert parser.parse_args(["--no-extract-embeddings"]).extract_embeddings is False


def test_split_lm_head_on_and_batch_prefill_default():
    parser = argparse.ArgumentParser()
    add_decoder_args(parser, model_name="test", split_lm_head=True, batch_prefill=True)
    defaults = parser.parse_args([])
    assert defaults.split_lm_head is True
    assert defaults.batch_prefill == 64
    off = parser.parse_args(["--no-split-lm-head", "--batch-prefill", "0"])
    assert off.split_lm_head is False
    assert off.batch_prefill == 0
