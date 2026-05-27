# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
from pathlib import Path

from . import add_liquid_infer_args
from ._inference import LiquidDynamic, LiquidStatic


def infer_liquid(args: argparse.Namespace):
    inputs = args.inputs
    model_args = {
        "model_path": args.model,
        "max_inp_len": args.max_inp_len,
        "n_threads": args.threads,
        "instruct_model": args.instruct_model,
    }

    # If a local tokenizer / config lives next to the model, prefer it over HF
    model_parent = Path(args.model).resolve().parent
    for parent in (model_parent, *model_parent.parents):
        cfg = parent / "config.json"
        tok = parent / "tokenizer.json"
        if cfg.exists() and tok.exists():
            model_args["config_path"] = str(cfg)
            model_args["tokenizer_path"] = str(tok)
            break

    is_vmfb = str(args.model).endswith(".vmfb")
    if not args.dynamic_model:
        if not args.max_gen_tokens:
            raise ValueError("`--max-gen-tokens` is required for static models")
        model_args["max_gen_tokens"] = args.max_gen_tokens
        model_cls = LiquidStatic
    else:
        model_cls = LiquidDynamic
        model_args["max_gen_tokens"] = args.max_gen_tokens

    loader = model_cls.from_vmfb if is_vmfb else model_cls.from_onnx
    liquid = loader(**model_args)
    for inp in inputs:
        out = liquid.run(inp, args.max_gen_tokens)
        print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LFM2.5 (Liquid) inference.")
    add_liquid_infer_args(parser)
    infer_liquid(parser.parse_args())
