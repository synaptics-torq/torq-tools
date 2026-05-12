# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""CLI for ONNX weight quantization.

Usage
-----
Quantize all weights uniformly::

    python -m torq.tools.quantization quantize \\
        -i model.onnx -o model_int8.onnx --bits 8

Quantize with per-layer config from sensitivity analysis::

    python -m torq.tools.quantization quantize \\
        -i model.onnx -o model_mixed.onnx --config quant_config.json

Produce a bf16 model with quantization error baked in::

    python -m torq.tools.quantization quantize \\
        -i model.onnx -o model_bf16.onnx --bits 8 --dequantize-weights

Run sensitivity analysis::

    python -m torq.tools.quantization analyze \\
        -i model.onnx -o sensitivity.json --config-output quant_config.json

Convert to bf16 only (delegates to torq.tools.convert_dtype)::

    python -m torq.tools.quantization quantize \\
        -i model.onnx -o model_bf16.onnx --bits 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# quantize subcommand
# ---------------------------------------------------------------------------


def _add_quantize_args(parser: argparse.ArgumentParser) -> None:
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
        "--bits",
        type=int,
        choices=[4, 8, 16],
        default=None,
        help="Uniform quantization bit-width (4=int4, 8=int8, 16=bf16). "
        "Ignored when --config is provided.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for block quantization (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to quantization config JSON (from sensitivity analysis). "
        "Overrides --bits for per-layer mixed quantization.",
    )
    parser.add_argument(
        "--dequantize-weights",
        action="store_true",
        help="Dequantize weights and output a single bf16 model "
        "ready for IREE compilation (no DQL nodes).",
    )
    parser.add_argument(
        "--skip-layers",
        type=str,
        nargs="*",
        default=[],
        help="Layer name substrings to skip (e.g. lm_head)",
    )


def _run_quantize(args: argparse.Namespace) -> None:
    from .config import QuantizationConfig
    from .quantize import WeightQuantizer

    if args.config:
        config = QuantizationConfig.load(args.config)
        logger.info("Loaded per-layer config from %s", args.config)
    elif args.bits is not None:
        config = QuantizationConfig.uniform(args.bits, args.block_size)
        logger.info("Uniform %d-bit quantization, block_size=%d", args.bits, args.block_size)
    else:
        parser_error("Either --bits or --config is required for quantize")
        return

    if args.bits == 16 and not args.dequantize_weights and not args.config:
        # bf16 only — use WeightQuantizer with bits=16 and dequantize_weights
        # (bits=16 layers are skipped in quantization, then _convert_to_bf16
        # converts the whole model to bf16)
        pass  # fall through to WeightQuantizer below

    quantizer = WeightQuantizer(
        model_path=args.input,
        output_path=args.output,
        skip_layers=args.skip_layers,
    )
    quantizer.quantize(config, dequantize_weights=args.dequantize_weights)


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
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
        help="Output sensitivity results JSON path",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default=None,
        help="Output quantization config JSON path (derived from sensitivity)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json for prompt tokenization (required)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to token_embeddings.npy for embedding lookup (required)",
    )
    parser.add_argument(
        "--token-lut",
        type=str,
        default=None,
        help="Path to token_id_lut.npy for reduced-vocab models (maps reduced index → full vocab ID)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=18,
        help="Number of transformer layers for KV cache init (default: %(default)s)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Bit-widths to test (default: 4 8 16)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=20,
        help="Number of output tokens to evaluate per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=None,
        help="Calibration prompts (text strings). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with list of calibration prompt strings",
    )
    parser.add_argument(
        "--bf16-threshold",
        type=float,
        default=0.1,
        help="KL divergence threshold above which layers stay bf16 (default: %(default)s)",
    )
    parser.add_argument(
        "--int8-threshold",
        type=float,
        default=0.01,
        help="KL divergence threshold above which layers use int8 (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-layers",
        type=str,
        nargs="*",
        default=[],
        help="Layer name substrings to skip (e.g. lm_head)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for chat template (default: Gemma-3 assistant prompt)",
    )


def _run_analyze(args: argparse.Namespace) -> None:
    from .sensitivity import LayerSensitivityAnalyzer

    prompts = args.prompts
    if args.prompts_file:
        prompts = json.loads(open(args.prompts_file).read())

    analyzer = LayerSensitivityAnalyzer(
        model_path=args.input,
        embeddings_path=args.embeddings,
        tokenizer_path=args.tokenizer,
        token_lut_path=args.token_lut,
        calibration_prompts=prompts,
        system_prompt=args.system_prompt,
        num_tokens=args.num_tokens,
        num_layers=args.num_layers,
        skip_layers=args.skip_layers,
    )
    analyzer.analyze(
        bits_options=args.bits,
        output_path=args.output,
        config_output_path=args.config_output,
        bf16_threshold=args.bf16_threshold,
        int8_threshold=args.int8_threshold,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parser_error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="torq.tools.quantization",
        description="ONNX weight quantization tool — int4/int8/bf16 with per-layer sensitivity analysis",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    q_parser = sub.add_parser(
        "quantize",
        help="Quantize MatMul weights in an fp32 ONNX model",
    )
    _add_quantize_args(q_parser)

    a_parser = sub.add_parser(
        "analyze",
        help="Run per-layer quantization sensitivity analysis",
    )
    _add_analyze_args(a_parser)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging.upper()),
        format="[%(levelname)-8s] %(message)s",
    )

    if args.command == "quantize":
        _run_quantize(args)
    elif args.command == "analyze":
        _run_analyze(args)


if __name__ == "__main__":
    main()
