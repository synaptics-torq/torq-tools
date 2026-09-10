import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_onnx_args


DEFAULT_HF_REPO: Final[str] = "Qwen/Qwen3-0.6B"
DEFAULT_GEN_TOKENS: Final[int] = 256


def add_qwen_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--max-gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Token generation limit (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Qwen Hugging Face repository (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-repo-subdir",
        type=str,
        metavar="DIR",
        help="Subdirectory within the Hugging Face repository containing model files",
    )

    add_onnx_args(
        parser,
        convert_dtypes=["bf16", "fp16"],
        allow_no_opt=False,
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and exported models (default: %(default)s)",
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export a dynamic model instead of producing a static model",
    )
    parser.add_argument(
        "--skip-torq",
        action="store_true",
        default=False,
        help="Skip Torq compilation",
    )
    parser.add_argument(
        "--keep-individual-kv-io",
        action="store_true",
        default=False,
        help="Keep key/value cache tensors separate instead of combining them",
    )

    parser.add_argument(
        "--split-model",
        action="store_true",
        default=False,
        help=(
            "Split the static Qwen transformer into two parts and "
            "export a separate LM head for memory-constrained Torq targets"
        ),
    )

    parser.add_argument(
        "--weight-quantization",
        choices=["int8"],
        default=None,
        help=(
            "Apply weight quantization to split Qwen components. "
            "Currently supported: int8."
        ),
    )

    add_logging_args(parser)
    add_torq_args(parser)


def add_qwen_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input prompts",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        metavar=".onnx | .vmfb",
        help="Path to the Qwen model",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max-inp-len",
        type=int,
        help="Maximum input length",
    )
    parser.add_argument(
        "--dynamic-model",
        action="store_true",
        default=False,
        help="Run a dynamic model",
    )
    parser.add_argument(
        "--split-vmfb-dir",
        type=str,
        help="Self-contained directory containing split Qwen VMFBs and runtime assets",
    )

    add_common_args(parser)
    add_logging_args(parser)
