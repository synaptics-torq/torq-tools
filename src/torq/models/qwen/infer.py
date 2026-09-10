import argparse

from . import add_qwen_infer_args
from ._inference import QwenDynamic, QwenStatic


def infer_qwen(args: argparse.Namespace):
    if args.split_vmfb_dir:
        if args.dynamic_model:
            raise ValueError("--dynamic-model cannot be used with --split-vmfb-dir")
        if not args.max_gen_tokens:
            raise ValueError("--max-gen-tokens is required for split VMFB inference")
        qwen = QwenStatic.from_split_vmfb(args.split_vmfb_dir, max_gen_tokens=args.max_gen_tokens, max_inp_len=args.max_inp_len, repo_id=None)
    else:
        if not args.model:
            raise ValueError("--model is required unless --split-vmfb-dir is used")
        model_args = {"model_path": args.model, "max_inp_len": args.max_inp_len, "repo_id": None}
        if args.dynamic_model:
            qwen = QwenDynamic.from_vmfb(**model_args)
        else:
            if not args.max_gen_tokens:
                raise ValueError("--max-gen-tokens is required for static models")
            qwen = QwenStatic.from_vmfb(max_gen_tokens=args.max_gen_tokens, **model_args)
    for inp in args.inputs:
        print(qwen.run(inp, args.max_gen_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen inference.")
    add_qwen_infer_args(parser)
    args = parser.parse_args()
    infer_qwen(args)
