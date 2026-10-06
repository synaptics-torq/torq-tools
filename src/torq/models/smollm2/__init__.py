import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_onnx_args, add_llm_args
from ...graph_edit.harness import add_graph_edit_harness_args


DEFAULT_MODEL_SIZE: Final[str] = "135M"
DEFAULT_GEN_TOKENS: Final[int] = 64
DEFAULT_IS_INSTRUCT: Final[bool] = False
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
MODEL_SIZES: Final[list[str]] = ["135M", "360M", "1.7B"]


def add_smollm2_export_args(parser: argparse.ArgumentParser):
    add_llm_args(
        parser,
        model_name="SmolLM2",
        model_sizes=MODEL_SIZES,
        default_model_size=DEFAULT_MODEL_SIZE,
        max_gen_tokens=DEFAULT_GEN_TOKENS,
        instruct=True,
    )
    add_onnx_args(
        parser,
        convert_dtypes=["bf16", "fp16"],
        allow_no_opt=False,
        extract_embeddings=True,
        dynamic_models=True,
        keep_individual_kv_io=True,
        replace_int_bf16_cast=True,
    )
    add_graph_edit_harness_args(parser)
    add_logging_args(parser)
    add_torq_args(parser, skip=True)


def add_smollm2_infer_args(parser: argparse.ArgumentParser):
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
        help="Path to SmolLM2 model",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="SmolLM2 model size (default: %(default)s)"
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
