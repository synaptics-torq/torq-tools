#!/usr/bin/env python3
"""Compare INT4 weight quantization (RTN vs GPTQ) on a static split Qwen export.

Takes the fp32 static ``model.onnx`` + ``lm_head.onnx`` produced by
``torq-export-model qwen --split-lm-head`` and builds fake-quantized copies whose
MatMul weights are replaced by their INT4 dequantization
``(q - zp) * bf16(scale)`` — exactly what the DequantizeLinear nodes of the Torq
model compute. All variants are run teacher-forced over chat prompts with ONNX
Runtime and compared against fp32 logits with the metrics used for Gemma 3:
KL divergence, cosine similarity, max absolute error, plus top-1 agreement.

Example:
    python compare_static_quant.py --static-dir models/.../onnx/fp32/static \
        --gptq-checkpoint gptq-W4A16-G32-infeat-256 --out results.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer

from torq.lab.quantization.onnx.weights import GPTQCheckpoint, dequantize_weight, quantize_weight
from torq.lab.quantization.onnx.weights._gptq import module_key_from_matmul_name

PROMPTS = [
    "What causes rainbows?",
    "Who wrote Romeo and Juliet?",
    "Write one sentence about embedded systems.",
    "Explain what a neural processing unit does in two sentences.",
    "List three fruits that are red.",
    "12345+54321=",
]


def fake_quantize(src: Path, dst: Path, method: str, gptq: GPTQCheckpoint | None, lut: np.ndarray | None) -> int:
    model = onnx.load(src)
    inits = {i.name: i for i in model.graph.initializer}
    n = 0
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        init = inits.get(node.input[1])
        if init is None or len(init.dims) != 2 or init.data_type != onnx.TensorProto.FLOAT:
            continue
        key = module_key_from_matmul_name(node.name)
        if key is None:  # e.g. one-hot rotary table lookups, not weights
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        if method == "gptq":
            qw = gptq.load(key, lut if key == "lm_head" else None)
        else:
            qw = quantize_weight(w, 4, 32)
        wq = dequantize_weight(qw)  # bf16-truncated scales, as on hardware
        assert wq.shape == w.shape, (node.name, wq.shape, w.shape)
        init.CopyFrom(numpy_helper.from_array(wq, init.name))
        n += 1
    onnx.save(model, dst, save_as_external_data=True, location=dst.name + "_data")
    return n


class Static:
    def __init__(self, model: Path, lm_head: Path, threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.m = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.h = ort.InferenceSession(str(lm_head), opts, providers=["CPUExecutionProvider"])
        self.kv = {i.name: i.shape for i in self.m.get_inputs() if i.name.startswith("past_key_values")}

    def logits(self, emb: np.ndarray, tokens: list[int]) -> np.ndarray:
        cache = {k: np.zeros(s, np.float32) for k, s in self.kv.items()}
        names = list(cache)
        out = []
        for pos, tok in enumerate(tokens):
            feed = {"token_embedding": emb[tok][None, None].astype(np.float32),
                    "position_ids": np.array([[pos]], np.int64), **cache}
            res = self.m.run(None, feed)
            hidden = next(r for r in res if r.ndim == 3)
            cache = dict(zip(names, [r for r in res if r.ndim == 4]))
            out.append(self.h.run(None, {self.h.get_inputs()[0].name: hidden})[0][0, -1])
        return np.stack(out)


def metrics(ref: np.ndarray, q: np.ndarray) -> dict:
    def logsm(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    lp, lq = logsm(ref.astype(np.float64)), logsm(q.astype(np.float64))
    kl = float((np.exp(lp) * (lp - lq)).sum(-1).mean())
    cos = float(np.mean((ref * q).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(q, axis=-1))))
    return {"kl": kl, "cosine": cos, "max_abs_error": float(np.abs(ref - q).max()),
            "top1_agreement": float(np.mean(ref.argmax(-1) == q.argmax(-1))),
            "class": "LOW" if kl <= 0.01 else "MEDIUM" if kl <= 0.1 else "HIGH" if kl <= 1.0 else "CRITICAL"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-dir", required=True, type=Path)
    ap.add_argument("--gptq-checkpoint", type=Path)
    ap.add_argument("--max-tokens", type=int, default=48, help="tokens per prompt (prompt + reference continuation)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--methods", nargs="+", choices=["rtn", "gptq"], default=None,
                    help="default: rtn, plus gptq when --gptq-checkpoint is given")
    ap.add_argument("--components", nargs="+", choices=["model", "lm_head"], default=["model", "lm_head"],
                    help="components to quantize; the others stay fp32 (ablation)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    d = args.static_dir
    emb = np.load(d / "token_embeddings.npy", mmap_mode="r")
    lut = np.load(d / "token_id_lut.npy")
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    gptq = GPTQCheckpoint(args.gptq_checkpoint) if args.gptq_checkpoint else None

    fp32 = Static(d / "model.onnx", d / "lm_head.onnx", args.threads)

    # Teacher-forced sequences: chat prompt followed by the fp32 greedy continuation.
    seqs = []
    for p in PROMPTS:
        ids = tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n").ids
        while len(ids) < args.max_tokens:
            ids.append(int(lut[fp32.logits(emb, ids)[-1].argmax()]))
        seqs.append(ids[: args.max_tokens])
    ref = np.concatenate([fp32.logits(emb, s) for s in seqs])

    results = {}
    methods = args.methods or (["rtn"] + (["gptq"] if gptq else []))
    with tempfile.TemporaryDirectory() as tmp:
        for method in methods:
            paths = {}
            for comp in ("model", "lm_head"):
                if comp not in args.components:
                    paths[comp] = d / f"{comp}.onnx"
                    continue
                paths[comp] = Path(tmp) / f"{comp}_{method}.onnx"
                n = fake_quantize(d / f"{comp}.onnx", paths[comp], method, gptq, lut)
                print(f"[{method}] fake-quantized {n} MatMuls in {comp}")
            q = Static(paths["model"], paths["lm_head"], args.threads)
            out = np.concatenate([q.logits(emb, s) for s in seqs])
            results[method] = metrics(ref, out)
            print(f"[{method}] {results[method]}", flush=True)

    results["_setup"] = {"components": args.components,"prompts": len(PROMPTS), "positions": int(ref.shape[0]), "vocab": int(ref.shape[1]),
                         "gptq_checkpoint": str(args.gptq_checkpoint) if gptq else None}
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
