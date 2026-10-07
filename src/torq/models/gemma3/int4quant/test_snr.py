"""
Single-step logit SNR / cosine-similarity test.

Runs the baseline (fp32 base ONNX) and the GPTQ QDQ ONNX (in/out_feature agnostic)
on the same input and measures the logit difference as
mean_abs / max_abs / mean_rel / max_rel / cos_sim / snr_db.

Since both models may be bf16, they are converted to fp32 via
quant_utils.convert_model_to_fp32() and then run on the onnxruntime CPU EP.

Example:
    python test_snr.py --orig onnx/model.onnx --quant onnx/gptq_int4_in_feature.onnx
"""

import argparse
import os

import numpy as np
import onnxruntime as ort

from quant_utils import convert_model_to_fp32


def build_feed(num_layers=18, hidden=640, past_len=256, head_dim=256, seed=42):
    rng = np.random.default_rng(seed)
    feed = {
        "token_embedding": rng.standard_normal((1, 1, hidden)).astype(np.float32),
        "position_ids": np.array([[0]], dtype=np.int32),
    }
    for i in range(num_layers):
        feed[f"past_key_values.{i}.key_value"] = np.zeros(
            (1, 2, past_len, head_dim),
            dtype=np.float32,
        )
    return feed


def compute_metrics(out_orig, out_quant):
    diff = np.abs(out_orig - out_quant)
    norm_o = np.linalg.norm(out_orig)
    norm_q = np.linalg.norm(out_quant)
    sig_pow = np.mean(out_orig ** 2)
    nse_pow = np.mean((out_orig - out_quant) ** 2)
    return {
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "mean_rel": float(np.mean(diff / np.maximum(np.abs(out_orig), 1e-12))),
        "max_rel": float(np.max(diff / np.maximum(np.abs(out_orig), 1e-12))),
        "cos_sim": float(np.dot(out_orig, out_quant) / (norm_o * norm_q)),
        "snr_db": float(10 * np.log10(sig_pow / max(nse_pow, 1e-30))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", default="model.onnx", help="Baseline fp32/bf16 base ONNX")
    parser.add_argument(
        "--quant",
        default="model_outfeat_qdq.onnx",
        help="out_feature GPTQ QDQ ONNX to evaluate",
    )
    parser.add_argument("--num-layers", type=int, default=18)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--past-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("origin : {}, quant : {}".format(args.orig, args.quant))
    orig_rt = convert_model_to_fp32(args.orig)
    quant_rt = convert_model_to_fp32(args.quant)

    feed = build_feed(
        num_layers=args.num_layers,
        hidden=args.hidden,
        past_len=args.past_len,
        head_dim=args.head_dim,
        seed=args.seed,
    )

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    sess_orig = ort.InferenceSession(orig_rt, opts, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(quant_rt, opts, providers=["CPUExecutionProvider"])

    out_orig = sess_orig.run(None, feed)[0].astype(np.float64).flatten()
    out_quant = sess_quant.run(None, feed)[0].astype(np.float64).flatten()

    print(compute_metrics(out_orig, out_quant))

    for rt, src in ((orig_rt, args.orig), (quant_rt, args.quant)):
        if rt is not None and rt != src:
            os.unlink(rt)


if __name__ == "__main__":
    main()
