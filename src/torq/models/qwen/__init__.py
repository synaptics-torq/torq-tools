import argparse
from typing import Final

from ...utils.compile import add_torq_args
from ...utils.demo import add_common_args
from ...utils.logging import add_logging_args
from ...utils.onnx import add_llm_args, add_onnx_args


# -s/--model-size choices. Each one sets the source repository and the GPTQ
# INT4 weights that the default one-command export uses.
QWEN_MODELS: Final[dict[str, dict[str, str]]] = {
    "qwen3-0.6b": {
        "hf_repo": "Qwen/Qwen3-0.6B",
        "gptq": "Synaptics/Qwen3-0.6B-GPTQ-int4-torq/gptq",
    },
    "qwen2.5-0.5b-instruct": {
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "gptq": "Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq/gptq",
    },
}
DEFAULT_MODEL: Final[str] = "qwen3-0.6b"
DEFAULT_HF_REPO: Final[str] = QWEN_MODELS[DEFAULT_MODEL]["hf_repo"]
DEFAULT_GEN_TOKENS: Final[int] = 256


def add_qwen_export_args(parser: argparse.ArgumentParser) -> None:
    add_llm_args(
        parser,
        model_name="Qwen",
        model_sizes=list(QWEN_MODELS),
        default_model_size=DEFAULT_MODEL,
        max_gen_tokens=DEFAULT_GEN_TOKENS,
        split_lm_head=True,
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Override the Hugging Face source repository of --model-size",
    )
    parser.add_argument(
        "--hf-repo-subdir",
        type=str,
        metavar="DIR",
        help="Subdirectory within the Hugging Face repository containing model files",
    )

    add_onnx_args(
        parser,
        dynamic_quantize=False,
        convert_dtypes=True,
        allow_no_opt=False,
        dynamic_models=True,
        keep_individual_kv_io=True,
        read_only_kv_cache=True,
    )

    parser.add_argument(
        "--split-model",
        action="store_true",
        default=False,
        help=(
            "Also split the transformer into two parts (Part A and Part B) "
            "for memory-constrained Torq targets"
        ),
    )
    parser.add_argument(
        "--weight-quantization",
        choices=["int4", "int8", "none"],
        default="int4",
        help=(
            "Block-size-32 DequantizeLinear weight quantization of the transformer and "
            "LM head MatMuls; requires --split-lm-head (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--gptq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the GPTQ INT4 weights of --model-size from Hugging Face instead of "
            "round-to-nearest (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--gptq-checkpoint",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Local compressed-tensors GPTQ INT4 checkpoint (int4quant/run_gptq.py); "
            "overrides the Hugging Face default"
        ),
    )

    add_logging_args(parser)
    add_torq_args(parser, skip=True)


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
