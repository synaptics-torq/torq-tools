import argparse

from . import add_smollm2_infer_args
from ._inference import SmolLM2Dynamic, SmolLM2Static


def infer_smollm2(args: argparse.Namespace):
    inputs = args.inputs
    model_args = {
        "model_path": args.model,
        "max_inp_len": args.max_inp_len,
        "n_threads": args.threads,
        "instruct_model": args.instruct_model,
    }
    if not args.dynamic_model:
        if not args.max_gen_tokens:
            raise ValueError("`--max-gen-tokens` is required for static models")
        model_args["max_gen_tokens"] = args.max_gen_tokens
        model_cls = SmolLM2Static
    else:
        model_cls = SmolLM2Dynamic

    smollm = model_cls.from_onnx(**model_args)
    for inp in inputs:
        out = smollm.run(inp, args.max_gen_tokens)
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolLM2 inference.")
    add_smollm2_infer_args(parser)
    infer_smollm2(parser.parse_args())
