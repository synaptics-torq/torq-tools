import argparse
from pathlib import Path

from ._inference import SmolLMDynamic


def main():
    inputs = args.inputs
    smollm = SmolLMDynamic.from_onnx(args.model, instruct_model=args.instruct_model)
    for inp in inputs:
        out = smollm.run(inp, 64)
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolLM2 inference.")
    parser.add_argument(
        "-m", "--model",
        type=Path,
        required=True,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--instruct-model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input prompts (space-separated).",
    )
    args = parser.parse_args()

    main()
