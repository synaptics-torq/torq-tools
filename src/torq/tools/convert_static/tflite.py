import argparse
from pathlib import Path
import flatbuffers
import os

from tensorflow.lite.python import schema_py_generated as schema_fb


def convert_model(
    input_model: str | os.PathLike,
    output_model: str | os.PathLike,
):
    input_path = Path(input_model)
    output_path = Path(output_model)

    buf = input_path.read_bytes()

    model = schema_fb.Model.GetRootAsModel(buf, 0)
    model_t = schema_fb.ModelT.InitFromObj(model)

    converted = 0
    for subgraph in model_t.subgraphs:
        for tensor in subgraph.tensors:
            if (
                tensor.hasRank
                and tensor.shapeSignature is not None
                and -1 in tensor.shapeSignature
            ):
                tensor.shapeSignature = None
                converted += 1

    builder = flatbuffers.Builder(1024)
    builder.Finish(model_t.Pack(builder), b"TFL3")

    output_path.write_bytes(builder.Output())

    print(f"\nWrote {output_path}")
    print(f"Removed dynamic shape signatures from {converted} tensor(s)")

def add_tflite_static_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input TFLite model path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output TFLite model path"
    )

def tflite_static_convert_from_args(args: argparse.Namespace):
    convert_model(
        args.input,
        args.output
    )

def main():
    parser = argparse.ArgumentParser()
    add_tflite_static_convert_args(parser)
    tflite_static_convert_from_args(parser.parse_args())

if __name__ == "__main__":
    main()