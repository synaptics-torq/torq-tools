# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from typing import Final

from torq.utils.logging import add_logging_args

from ...utils.compile import add_torq_args

from ...utils.demo import add_common_args
from ...utils.onnx import add_onnx_args, add_llm_args
from ...graph_edit.harness import add_graph_edit_harness_args


DEFAULT_MODEL_SIZE: Final[str] = "350m"
DEFAULT_GEN_TOKENS: Final[int] = 256
DEFAULT_IS_INSTRUCT: Final[bool] = False
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
MODEL_SIZES: Final[list[str]] = ["350m", "230m"]
MODEL_DTYPES: Final[list[str]] = ["fp32"]


def add_liquid_export_args(parser: argparse.ArgumentParser):
    add_llm_args(
        parser,
        model_name="LFM2.5 (Liquid)",
        model_sizes=MODEL_SIZES,
        default_model_size=DEFAULT_MODEL_SIZE,
        max_gen_tokens=DEFAULT_GEN_TOKENS,
        split_lm_head=True,
        batch_prefill=True,
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=MODEL_DTYPES,
        default="fp32",
        help="Model data type (default: %(default)s)",
    )
    add_onnx_args(
        parser,
        convert_dtypes=["bf16", "fp16"],
        allow_no_opt=False,
        extract_embeddings=True,
        dynamic_models=True,
        keep_individual_kv_io=True,
    )
    parser.add_argument(
        "--simulate-bf16",
        action="store_true",
        default=False,
        help="Simulate bf16 inference by sandwiching each op with fp32→bf16→fp32 casts (for measuring quantization impact)",
    )
    parser.add_argument(
        "--keep-conv1d",
        action="store_true",
        default=False,
        help=(
            "Keep the original depthwise Conv1D nodes (default: replace with a "
            "bit-exact batched-MatMul chain). The SL2610's depthwise-conv path "
            "crashes torq-compile; use only for CPU/ORT targets."
        ),
    )
    parser.add_argument(
        "--chunk-lm-head",
        action="store_true",
        default=False,
        help=(
            "Split the lm_head MatMul into 512 chunks of [1024, 128] (default: "
            "a single [1024, 65536] MatMul; tile-and-fuse handles it)."
        ),
    )
    add_graph_edit_harness_args(parser)
    add_logging_args(parser)
    add_torq_args(parser, skip=True)


def add_liquid_vl_export_args(parser: argparse.ArgumentParser):
    """Export args for LFM2-VL-450M (vision-language).

    Mirrors :func:`add_liquid_export_args` (the text-only LFM2.5 flags) and
    adds ``--compile-vision``.  The model size / dtype are fixed for VL, so
    those selectors are omitted.
    """
    add_llm_args(
        parser,
        model_name="LFM2-VL",
        max_gen_tokens=DEFAULT_GEN_TOKENS,
        split_lm_head=True,
        batch_prefill=True,
    )
    add_onnx_args(
        parser,
        convert_dtypes=["bf16", "fp16"],
        allow_no_opt=False,
        dynamic_models=True,
        keep_individual_kv_io=True,
    )
    parser.add_argument(
        "--compile-vision",
        action="store_true",
        default=False,
        help=(
            "Also compile the SigLIP vision encoder to a vmfb (experimental: "
            "it has dynamic shapes and exotic ops; off by default)."
        ),
    )
    parser.add_argument(
        "--vision-res",
        type=int,
        choices=[128, 256],
        default=None,
        help=(
            "Build + compile a static single-resolution SigLIP vision encoder "
            "(vision_encoder_<res>.vmfb): 128 -> 16 image tokens, 256 -> 64. "
            "Replaces the dynamic encoder with the board's static build."
        ),
    )
    parser.add_argument(
        "--image-decoder-parts",
        type=int,
        nargs="?",
        const=2,
        choices=[2, 3, 5],
        default=None,
        help=(
            "Build + compile the one-shot image-prefill decoder, split into N "
            "layer-boundary parts (decoder_image_<N>part_<A..>.vmfb). Bare flag "
            "= 2-part (the shipping split on HuggingFace); 3 / 5 are alternates. "
            "Runs all 64 image tokens through the decoder in one shot for lower "
            "TTFT."
        ),
    )
    parser.add_argument(
        "--simulate-bf16",
        action="store_true",
        default=False,
        help="Simulate bf16 inference by sandwiching each op with fp32→bf16→fp32 casts",
    )
    parser.add_argument(
        "--keep-conv1d",
        action="store_true",
        default=False,
        help=(
            "Keep the original depthwise Conv1D nodes (default: replace with a "
            "bit-exact batched-MatMul chain). The SL2610's depthwise-conv path "
            "crashes torq-compile; use only for CPU/ORT targets."
        ),
    )
    parser.add_argument(
        "--chunk-lm-head",
        action="store_true",
        default=False,
        help=(
            "Split the lm_head MatMul into 512 chunks of [1024, 128] (default: "
            "a single [1024, 65536] MatMul; tile-and-fuse handles it)."
        ),
    )
    add_graph_edit_harness_args(parser)
    add_logging_args(parser)
    add_torq_args(parser, skip=True)


def add_liquid_infer_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input prompts (space-separated).",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        metavar=".onnx | .vmfb",
        help="Path to Liquid LFM2.5 model",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="LFM2.5 model size (default: %(default)s)"
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-inp-len",
        type=int,
        help="Maximum input length",
    )
    parser.add_argument(
        "--instruct-model",
        action="store_true",
        default=False,
        help="Is instruct model"
    )
    parser.add_argument(
        "--dynamic-model",
        action="store_true",
        default=False,
        help="Is dynamic model"
    )
    add_common_args(parser)
    add_logging_args(parser)
