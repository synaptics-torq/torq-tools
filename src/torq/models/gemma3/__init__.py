import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_onnx_args, add_llm_args
from ...graph_edit.harness import add_graph_edit_harness_args


DEFAULT_MODEL_SIZE: Final[str] = "270m"
DEFAULT_GEN_TOKENS: Final[int] = 256
DEFAULT_IS_INSTRUCT: Final[bool] = False
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
MODEL_SIZES: Final[list[str]] = ["270m", "1b"]
TRIM_VOCAB_GROUPS: Final[list[str]] = ["latin", "punct", "digits", "digits-non-latin", "other"]


def add_gemma3_export_args(parser: argparse.ArgumentParser):
    add_llm_args(
        parser,
        model_name="Gemma3",
        model_sizes=MODEL_SIZES,
        default_model_size=DEFAULT_MODEL_SIZE,
        max_gen_tokens=DEFAULT_GEN_TOKENS,
        instruct=True,
        split_lm_head=True,
        batch_prefill=True,
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="Custom Gemma3 HuggingFace repository"
    )
    parser.add_argument(
        "--hf-repo-subdir",
        type=str,
        metavar="DIR",
        help="Sub-directory within the HuggingFace repository containing the model files"
    )
    add_onnx_args(
        parser,
        dynamic_quantize=True,
        convert_dtypes=True,
        allow_no_opt=False,
        extract_embeddings=True,
        dynamic_models=True,
        keep_individual_kv_io=True,
        replace_int_bf16_cast=True,
    )
    parser.add_argument(
        "--trim-vocab",
        action="store_true",
        default=False,
        help="Trim static export vocab to selected token groups plus required safety tokens (static exports only)"
    )
    parser.add_argument(
        "--trim-vocab-groups",
        type=str,
        nargs="+",
        choices=TRIM_VOCAB_GROUPS,
        default=["latin", "punct", "digits"],
        metavar="GROUP",
        help="Token groups to retain when --trim-vocab is enabled (default: %(default)s)",
    )
    parser.add_argument(
        "--trim-byte-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain byte fallback tokens when --trim-vocab is enabled (default: %(default)s)",
    )
    add_graph_edit_harness_args(parser)
    add_logging_args(parser)
    add_torq_args(parser, skip=True)


def add_gemma3_infer_args(parser: argparse.ArgumentParser):
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
        help="Path to Gemma3 model",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="Gemma3 model size (default: %(default)s)"
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
