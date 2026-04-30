import argparse
from typing import Final

from torq.compile import add_iree_args
from torq.utils.logging import add_logging_args

from ...utils.demo import add_common_args
from ...utils.onnx import add_onnx_args


DEFAULT_MODEL_SIZE: Final[str] = "270m"
DEFAULT_GEN_TOKENS: Final[int] = 256
DEFAULT_IS_INSTRUCT: Final[bool] = False
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
MODEL_SIZES: Final[list[str]] = ["270m", "1b"]
MODEL_DTYPES: Final[list[str]] = ["fp32", "int4", "int8"]


def add_gemma3_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-t",
        "--max-gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Input audio length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="Gemma3 model size to export (default: %(default)s)",
    )
    parser.add_argument(
        "--instruct-model",
        action="store_true",
        default=False,
        help="Export instruct model variant"
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=MODEL_DTYPES,
        default="fp32",
        help="Model data type / quantization (default: %(default)s)",
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
        help="Base directory for source and export models (default: %(default)s)",
    )
    parser.add_argument(
        "--extract-embeddings",
        action="store_true",
        default=False,
        help="Extract large embeddings tables into external .npy data"
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export dynamic models for CPU"
    )
    parser.add_argument(
        "--skip-iree",
        action="store_true",
        default=False,
        help="Skip exporting to IREE"
    )
    parser.add_argument(
        "--replace-int-bf16-cast",
        action="store_true",
        default=False,
        help="Replace int64 -> bf16 casts with a look-up table"
    )
    parser.add_argument(
        "--keep-individual-kv-io",
        action="store_true",
        default=False,
        help="Keep KV I/O as separate key, value tensors instead of combining"
    )
    parser.add_argument(
        "--broadcast-ops",
        type=str,
        metavar="OP",
        nargs="*",
        default=None,
        help="Broadcast op inputs: specify ops or pass with no args to broadcast for all ops",
    )
    parser.add_argument(
        "--dequantize-weights",
        action="store_true",
        default=False,
        help="Dequantize int4 MatMulNBits weights to fp32 MatMul (int4 only, default: keep quantized)",
    )
    parser.add_argument(
        "--dequantize-weights-linear",
        action="store_true",
        default=False,
        help="Replace MatMulNBits with DequantizeLinear+Reshape+MatMul (int4 only, runtime dequantization)",
    )
    parser.add_argument(
        "--simulate-bf16",
        action="store_true",
        default=False,
        help="Simulate bf16 inference by sandwiching each op with fp32→bf16→fp32 casts (for measuring quantization impact)",
    )
    add_logging_args(parser)
    add_iree_args(parser)


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
