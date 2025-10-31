import argparse
from typing import Final

from torq.compile import add_iree_args
from torq.utils.logging import add_logging_args

from ...utils.demo import add_common_args
from ...utils.onnx import add_onnx_args


DEFAULT_INPUT_AUDIO_S: Final[int] = 5
DEFAULT_DEC_TOK_PER_SEC: Final[int] = 6
DEFAULT_MODEL_SIZE: Final[str] = "tiny"
ONNX_DTYPES: Final[list[str]] = ["float", "quantized", "quantized_4bit"]
OPTIMUM_DTYPES: Final[list[str]] = ["fp32", "fp16", "bf16"]
STATIC_MODEL_COMPONENTS: Final[list[str]] = ["encoder", "decoder", "decoder_with_past"]


def add_moonshine_export_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input-seconds",
        type=int,
        default=DEFAULT_INPUT_AUDIO_S,
        help="Input audio length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tokens-per-sec",
        type=int,
        default=DEFAULT_DEC_TOK_PER_SEC,
        help="Max number of tokens decoded per second (default: %(default)d)",
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        choices=["base", "tiny"],
        default=DEFAULT_MODEL_SIZE,
        help="Moonshine model size to export (default: %(default)s)",
    )
    add_onnx_args(
        parser,
        model_dtypes=ONNX_DTYPES + OPTIMUM_DTYPES,
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
        "--split-encoder",
        action="store_true",
        default=False,
        help="Split merged encoder into preprocessor and encoder models"
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export dynamic models for CPU"
    )
    parser.add_argument(
        "--use-optimum",
        action="store_true",
        default=False,
        help="Use optimum-cli to generate ONNX models rather than loading prebuilt ones"
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
        "--skip-export",
        type=str,
        nargs="+",
        choices=["encoder", "decoder", "decoder_with_past", "decoder_merged"],
        help="Skip export of specific components"
    )
    add_logging_args(parser)
    add_iree_args(parser)


def add_moonshine_infer_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "inputs",
        type=str,
        metavar="WAV",
        nargs="+",
        help="WAV files for inference",
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to Moonshine model directory",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        required=True,
        choices=["base", "tiny"],
        help="Moonshine model size"
    )
    parser.add_argument(
        "--max-inp-len",
        type=int,
        help="Maximum input length (required for static VMFB models)",
    )
    parser.add_argument(
        "--max-dec-len",
        type=int,
        help="Maximum decoder length (required for static VMFB models)",
    )
    add_common_args(parser)
    add_logging_args(parser)
