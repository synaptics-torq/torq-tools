import argparse
from pathlib import Path

from . import add_smollm2_infer_args
from ._inference import SmolLM2Dynamic, SmolLM2Static, _repo_id_for_size


def _find_local_assets(model_path: str) -> tuple[Path | None, Path | None]:
    """Nearest (config.json, tokenizer.json) staged next to the model.

    Every exporter stages both into the variant dir — the model's own dir
    (ONNX) or its parent (the compiled/ vmfb) — so a deploy laid out as the
    whole variant dir needs no HF download. The walk is bounded to those two
    dirs: an unbounded one could pick up a sibling model's assets from a
    higher-level dir and build the KV cache from the wrong layer count.
    """
    model_parent = Path(model_path).resolve().parent
    for parent in (model_parent, model_parent.parent):
        cfg = parent / "config.json"
        tok = parent / "tokenizer.json"
        if cfg.exists() and tok.exists():
            return cfg, tok
    return None, None


def infer_smollm2(args: argparse.Namespace):
    inputs = args.inputs
    model_args = {
        "model_path": args.model,
        "max_inp_len": args.max_inp_len,
        "n_threads": args.threads,
        "instruct_model": args.instruct_model,
        # -s selects the matching HF repo (e.g. the 360M layer counts, not
        # the 135M default) for config/tokenizer resolution.
        "repo_id": _repo_id_for_size(args.model_size, args.instruct_model),
    }
    # Prefer config/tokenizer staged next to the model over an HF download.
    cfg, tok = _find_local_assets(args.model)
    if cfg is not None:
        model_args["config_path"] = str(cfg)
    if tok is not None:
        model_args["tokenizer_path"] = str(tok)

    if not args.dynamic_model:
        if not args.max_gen_tokens:
            raise ValueError("`--max-gen-tokens` is required for static models")
        model_args["max_gen_tokens"] = args.max_gen_tokens
        model_cls = SmolLM2Static
    else:
        model_cls = SmolLM2Dynamic

    is_vmfb = str(args.model).endswith(".vmfb")
    loader = model_cls.from_vmfb if is_vmfb else model_cls.from_onnx
    smollm = loader(**model_args)
    for inp in inputs:
        out = smollm.run(inp, args.max_gen_tokens)
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolLM2 inference.")
    add_smollm2_infer_args(parser)
    infer_smollm2(parser.parse_args())
