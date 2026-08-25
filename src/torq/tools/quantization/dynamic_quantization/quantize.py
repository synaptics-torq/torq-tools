# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import os

logger = logging.getLogger(__name__)

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.preprocess import quant_pre_process

from ....utils.cli import (
    parse_remainder_args_to_dict,
)
from ....utils.logging import (
    add_logging_args,
    configure_logging,
)


def dynamic_quantize_model(
    model_input_path: str | os.PathLike | onnx.ModelProto,
    model_output_path: str | os.PathLike,
    *,
    quantize_only_ops: list[str] | None = None,
    quantize_only_nodes: list[str] | None = None,
    skip_preprocess: bool = False,
    uint8_weights: bool = False,
    per_tensor: bool = False,
    **quantize_kwargs
):
    logger.debug("Dynamically quantizing '%s' with the following parameters:", model_input_path)
    logger.debug("  Weight dtype = %s", "UINT8" if uint8_weights else "INT8")
    logger.debug("  Per-tensor   = %s", str(per_tensor))
    logger.debug("  Op types     = %s", "all" if quantize_only_ops is None else ", ".join(quantize_only_ops))
    logger.debug("  Graph nodes  = %s", "all" if quantize_only_nodes is None else ", ".join(quantize_only_nodes))
    if not skip_preprocess:
        quant_pre_process(model_input_path, model_output_path)
        logger.debug("Preprocessed model '%s' before quantization", str(model_output_path))
    quantize_dynamic(
        model_input_path if skip_preprocess else model_output_path,
        model_output_path,
        op_types_to_quantize=quantize_only_ops,
        nodes_to_quantize=quantize_only_nodes,
        weight_type=QuantType.QUInt8 if uint8_weights else QuantType.QInt8,
        per_channel=not per_tensor,
        **quantize_kwargs
    )
    model = onnx.load(model_output_path)
    model = onnx.shape_inference.infer_shapes(model, True, True, True)
    onnx.save(model, model_output_path)
    logger.debug("Saved dynamically quantized model to '%s'", model_output_path)


def add_dynamic_quantize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input fp32 ONNX model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output quantized ONNX model path",
    )
    parser.add_argument(
        "--quantize-only-ops",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified op types; must be valid ONNX op types",
    )
    parser.add_argument(
        "--quantize-only-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified nodes; must be valid node names from graph",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=False,
        help="Skip pre-processing steps that may improve quantization quality",
    )
    parser.add_argument(
        "--uint8-weights",
        action="store_true",
        default=False,
        help="Generate unsigned integer weights during quantization",
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        default=False,
        help="Quantize weights per channel",
    )
    parser.add_argument(
        "--extra-quant-args",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra quantization args for `onnxruntime.quantization.dynamic_quantize`. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    add_logging_args(parser)


def dynamic_quantize_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    extra_quant_args = parse_remainder_args_to_dict(
        args.extra_quant_args,
        "--extra_quant_args"
    )
    dynamic_quantize_model(
        args.input, args.output,
        skip_preprocess=args.skip_preprocess,
        uint8_weights=args.uint8_weights,
        per_tensor=args.per_tensor,
        **extra_quant_args
    )
