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
# The hosted GPTQ INT4 weights belong to this source only (270m, instruct).
GPTQ_SOURCE_REPO: Final[str] = "google/gemma-3-270m-it"
GPTQ_HF_REPO: Final[str] = "Synaptics/gemma-3-270-GPTQ-int4-it-torq"
FUNCTIONGEMMA_HF_REPO: Final[str] = "google/functiongemma-270m-it"


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
        native_batch_prefill=True,
    )
    parser.add_argument(
        "--functiongemma",
        action="store_true",
        help=f"Export FunctionGemma (270m only; default source: {FUNCTIONGEMMA_HF_REPO}). "
             "Fine-tune for your function-calling task for best accuracy and override the source "
             "with --hf-repo or --onnx-source-dir",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="Custom Gemma3 HuggingFace repository"
    )
    parser.add_argument(
        "--weight-quantization",
        choices=["int4", "int8", "none"],
        default="int4",
        help="Quantize the MatMul weights to INT4 or INT8 in blocks of 32; the activations "
             "stay bf16 (W4A16 / W8A16). 'none' keeps the float weights. --dynamic-quantize "
             "turns it off (default: %(default)s)"
    )
    parser.add_argument(
        "--gptq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --weight-quantization int4 and the default 270m instruct source, take the "
            "INT4 weights from the GPTQ checkpoint in the gptq/ folder of "
            f"{GPTQ_HF_REPO}; other sources use round-to-nearest (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--gptq-checkpoint",
        type=str,
        metavar="DIR",
        help="Local GPTQ INT4 checkpoint (compressed-tensors) to take the INT4 weights from, "
             "with --weight-quantization int4; overrides the hosted one"
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
        read_only_kv_cache=True,
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
