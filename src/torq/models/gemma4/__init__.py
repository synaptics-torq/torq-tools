import argparse
from typing import Final

from ...utils.logging import add_logging_args


DEFAULT_HF_REPO: Final[str] = "principled-intelligence/gemma-4-E2B-it-text-only"
DEFAULT_DTYPE: Final[str] = "fp32"
DTYPES: Final[list[str]] = ["fp32", "bf16"]

# int4 (quantized) source export defaults -- see export_int4.py.
DEFAULT_HF_REPO_INT4: Final[str] = "tss-deposium/gemma-4-E2B-text-only-onnx-int4"
DEFAULT_TEMPLATE_REPO_INT4: Final[str] = "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX"
DEFAULT_MAX_KV_LEN: Final[int] = 256
# RoPE / sliding-window upper bounds used to mark the dynamic axes of the
# exported ONNX graph -- see `export.py` for how these map onto the two
# distinct KV-cache lengths (sliding-window layers cap out at
# `config.sliding_window`; full-attention layers keep growing).
DEFAULT_MAX_SEQ_LEN: Final[int] = 4096


def add_gemma4_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="HuggingFace repository holding the safetensors checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        choices=DTYPES,
        default=DEFAULT_DTYPE,
        help=(
            "Torch / ONNX export dtype (default: %(default)s). 'fp32' is required to "
            "validate the exported graph with onnxruntime on CPU, but roughly doubles "
            "peak memory versus the checkpoint's native bf16."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Upper bound for the dynamic full-attention KV-cache / position axis (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip ORT validation of the exported ONNX against the PyTorch model",
    )
    add_logging_args(parser)


def add_gemma4_int4_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO_INT4,
        help="HuggingFace repository holding the pre-quantized (int4) ONNX source (default: %(default)s)",
    )
    parser.add_argument(
        "--template-repo",
        type=str,
        default=DEFAULT_TEMPLATE_REPO_INT4,
        help="HuggingFace repository to pull chat_template.jinja from, since the int4 "
             "repo doesn't ship one (default: %(default)s)",
    )
    parser.add_argument(
        "--max-kv-len",
        type=int,
        default=DEFAULT_MAX_KV_LEN,
        help="Fixed KV-cache length for the static decoder graph (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--onnx-source-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Load the source ONNX pair from this local directory instead of downloading "
             "(must contain decoder_model_merged_q4.onnx(+_data) and embed_tokens_q4.onnx(+_data))",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip schema/shape validation of the exported static ONNX",
    )
    parser.add_argument(
        "--convert-dtypes",
        action="store_true",
        default=False,
        help="Convert the exported static ONNX to bf16 then int32 (mirrors gemma3's "
             "conversion step; runs after export, before --skip-torq would apply)",
    )
    parser.add_argument(
        "--preserve-io-dtypes",
        action="store_true",
        default=False,
        help="Preserve model input/output dtypes by adding runtime casts (only "
             "relevant with --convert-dtypes)",
    )
    add_logging_args(parser)
