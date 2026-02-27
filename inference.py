#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_shape(shape_str: str):
    return tuple(int(x) for x in shape_str.split(","))


def make_input(shape, dtype_str="float32", fill="random"):
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    dtype = dtype_map[dtype_str]

    if fill == "zeros":
        return np.zeros(shape, dtype=dtype)
    if fill == "ones":
        return np.ones(shape, dtype=dtype)

    if np.issubdtype(dtype, np.floating):
        return np.random.randn(*shape).astype(dtype)
    return np.random.randint(0, 10, size=shape, dtype=dtype)


def print_model_io(session: ort.InferenceSession):
    print("\nInputs:")
    for x in session.get_inputs():
        print(f"  name={x.name}, shape={x.shape}, type={x.type}")

    print("\nOutputs:")
    for y in session.get_outputs():
        print(f"  name={y.name}, shape={y.shape}, type={y.type}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on an ONNX model.")
    parser.add_argument("model", type=str, help="Path to .onnx model")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Input spec in the form "
            "name:dim0,dim1,...:dtype:fill "
            "Example: in_frame_mag:1,2,1,256:float32:random"
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["CPUExecutionProvider"],
        help="Execution providers, e.g. CPUExecutionProvider CUDAExecutionProvider",
    )
    parser.add_argument(
        "--save_outputs",
        type=str,
        default=None,
        help="Optional path to save outputs as .npz",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=args.providers)

    print_model_io(session)

    feed = {}

    if args.input:
        for spec in args.input:
            try:
                name, shape_str, dtype_str, fill = spec.split(":")
            except ValueError:
                raise ValueError(
                    f"Bad --input format: {spec}\n"
                    "Expected: name:dim0,dim1,...:dtype:fill"
                )

            shape = parse_shape(shape_str)
            arr = make_input(shape, dtype_str=dtype_str, fill=fill)
            feed[name] = arr
    else:
        print("\nNo --input provided. Generating default random inputs from model metadata.")
        for inp in session.get_inputs():
            if any(dim is None or isinstance(dim, str) for dim in inp.shape):
                raise ValueError(
                    f"Input {inp.name} has dynamic shape {inp.shape}. "
                    "Please provide it explicitly with --input."
                )

            onnx_type = inp.type
            if "float" in onnx_type:
                dtype_str = "float32"
            elif "int64" in onnx_type:
                dtype_str = "int64"
            elif "int32" in onnx_type:
                dtype_str = "int32"
            else:
                raise ValueError(
                    f"Unsupported input type {onnx_type} for input {inp.name}. "
                    "Provide it explicitly with --input."
                )

            shape = tuple(int(d) for d in inp.shape)
            feed[inp.name] = make_input(shape, dtype_str=dtype_str, fill="random")

    print("\nFeeding inputs:")
    for k, v in feed.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, feed)

    print("\nInference done.\n")
    for name, value in zip(output_names, outputs):
        print(
            f"Output {name}: shape={value.shape}, dtype={value.dtype}, "
            f"min={value.min():.6f}, max={value.max():.6f}, mean={value.mean():.6f}"
        )

    if args.save_outputs:
        out_path = Path(args.save_outputs)
        np.savez(out_path, **{n: v for n, v in zip(output_names, outputs)})
        print(f"\nSaved outputs to {out_path}")


if __name__ == "__main__":
    main()
