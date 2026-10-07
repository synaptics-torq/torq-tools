"""
GPTQ has large run-to-run variance (GPU non-determinism + 4-bit rounding sensitivity).
So the normal workflow is to run it several times and pick the model with the highest
cos_sim vs. base.

This script automates that process:
  1) Run GPTQ N times (distributed across a GPU pool) → save compressed safetensors each
  2) Convert each safetensors to QDQ ONNX (safetensors_to_onnx_qdq.py)
  3) Measure single-step logit cos_sim vs. the base ONNX
  4) Copy the highest-cos_sim model to --out and print the ranking table

Example (default in_feature):
    python select_best_gptq.py --num-runs 8 --gpus 0,1,2,3,4,5,6,7 \
        --base onnx/model.onnx --template onnx/model_q4k.onnx \
        --out onnx/gptq_int4_in_feature.onnx
    # For out_feature: --grouping out --out onnx/gptq_int4_out_feature.onnx

The recipe branches on --grouping into in (group_size=32, default) / out
(block_structure="32x1"), identical to run_gptq.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# GPTQ worker (run as a separate process — CUDA is isolated to 1 GPU per process)
# ---------------------------------------------------------------------------

def _worker():
    gpu = os.environ["GPU"]
    seed = int(os.environ["SEED"])
    save = os.environ["SAVE"]
    n_calib = int(os.environ["NCALIB"])
    max_seq = int(os.environ["MAXSEQ"])
    grouping = os.environ.get("GROUPING", "out")
    dev_choice = os.environ.get("DEVICE", "cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    if dev_choice == "auto":
        dev_choice = "cuda" if torch.cuda.is_available() else "cpu"
    device = dev_choice
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device != "cuda":
        # hide MPS/accelerator so calibration + save stay on plain CPU
        # (compressed_tensors has no MPS offload; see run_gptq.py for details)
        torch.accelerator.is_available = lambda *a, **k: False
        torch.accelerator.current_accelerator = lambda *a, **k: torch.device("cpu")
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    model_id = "google/gemma-3-270m-it"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{n_calib}]").shuffle(seed=seed)
    ds = ds.map(lambda e: {"text": tokenizer.apply_chat_template(
        e["messages"], tokenize=False, add_generation_prompt=False)})
    ds = ds.map(lambda s: tokenizer(s["text"], padding=False, max_length=max_seq,
                                    truncation=True, add_special_tokens=False),
                remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) >= 32)

    if grouping == "in":
        weights_cfg = {"num_bits": 4, "type": "int", "symmetric": False,
                       "strategy": "group", "group_size": 32}
    else:
        weights_cfg = {"num_bits": 4, "type": "int", "symmetric": False,
                       "strategy": "block", "block_structure": "32x1"}
    recipe = GPTQModifier(
        ignore=[],
        config_groups={"group_0": {"targets": ["Linear"], "weights": weights_cfg}},
    )
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=max_seq,
            num_calibration_samples=min(n_calib, len(ds)))
    if device != "cuda":
        model.to("cpu")  # ensure save's from_accelerate infers cpu (not mps)
    model.save_pretrained(save, save_compressed=True)
    tokenizer.save_pretrained(save)


# ---------------------------------------------------------------------------
# convert + measure
# ---------------------------------------------------------------------------

def convert(safetensors, template, out, group_size, base_onnx=None, grouping="out"):
    cmd = [sys.executable, os.path.join(HERE, "safetensors_to_onnx_qdq.py"),
           "--onnx-template", template, "--safetensors", safetensors,
           "--out", out, "--expected-group-size", str(group_size),
           "--grouping", grouping]
    if base_onnx:
        cmd += ["--base-onnx", base_onnx]  # auto-generate from base if no template
    subprocess.run(cmd, check=True, capture_output=True)


def cos_sim_vs_base(base_logits, onnx_path):
    import numpy as np
    import onnxruntime as ort
    from quant_utils import convert_model_to_fp32
    rt = convert_model_to_fp32(onnx_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(rt, opts, providers=["CPUExecutionProvider"])
    from test_snr import build_feed
    q = sess.run(None, build_feed())[0].astype(np.float64).ravel()
    if rt != onnx_path and os.path.exists(rt):
        os.unlink(rt)
    b = base_logits
    return float(b @ q / (np.linalg.norm(b) * np.linalg.norm(q)))


def base_logits(base_path):
    import numpy as np
    import onnxruntime as ort
    from quant_utils import convert_model_to_fp32
    from test_snr import build_feed
    rt = convert_model_to_fp32(base_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(rt, opts, providers=["CPUExecutionProvider"])
    out = sess.run(None, build_feed())[0].astype(np.float64).ravel()
    if rt != base_path and os.path.exists(rt):
        os.unlink(rt)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-runs", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Compute device for GPTQ. 'auto' (default) picks cuda if present, "
                             "else cpu. With cpu there is a single device, so runs go "
                             "sequentially (--gpus is ignored). MPS is unsupported; a Mac uses cpu.")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU id pool (e.g. 0,1,2,3); CUDA only")
    parser.add_argument("--base", default="onnx/model.onnx")
    parser.add_argument("--template", default="onnx/model_q4k.onnx")
    parser.add_argument("--out", default="onnx/gptq_int4_in_feature.onnx")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--grouping", choices=["out", "in"], default="in",
                        help="'in' (group_size 32, default) or 'out' (block 32x1).")
    parser.add_argument("--base-seed", type=int, default=5436)
    parser.add_argument("--work-dir", default=None, help="temp location for safetensors/onnx (default: a temp dir)")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _worker()
        return

    # Resolve device up front so we know whether a GPU pool applies.
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # cpu = a single device → run sequentially (one process at a time).
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if device == "cuda" else ["0"]
    work = args.work_dir or tempfile.mkdtemp(prefix="gptq_select_")
    os.makedirs(work, exist_ok=True)
    pool_desc = f"GPU pool: {gpus}" if device == "cuda" else f"device: {device} (sequential)"
    print(f"work dir: {work}\n{pool_desc}\nruns: {args.num_runs}\n")

    # 1) GPTQ N times — run in parallel waves of GPU-pool size
    save_dirs = []
    pending = list(range(args.num_runs))
    while pending:
        wave, pending = pending[:len(gpus)], pending[len(gpus):]
        procs = []
        for slot, run_idx in enumerate(wave):
            save = os.path.join(work, f"st_run{run_idx}")
            save_dirs.append((run_idx, save))
            env = dict(os.environ,
                       GPU=gpus[slot], SEED=str(args.base_seed + run_idx), SAVE=save,
                       NCALIB=str(args.calibration_samples), MAXSEQ=str(args.max_seq_length),
                       GROUPING=args.grouping, DEVICE=device)
            log = open(os.path.join(work, f"run{run_idx}.log"), "w")
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--_worker"],
                env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"  launched runs {wave} on GPUs {gpus[:len(wave)]} ...")
        for p in procs:
            p.wait()

    # 2)+3) convert + measure
    print("\nmeasuring cos_sim vs base ...")
    b = base_logits(args.base)
    results = []
    for run_idx, save in save_dirs:
        st = os.path.join(save, "model.safetensors")
        if not os.path.exists(st):
            print(f"  run{run_idx}: FAILED (no safetensors)")
            continue
        onnx_out = os.path.join(work, f"run{run_idx}.onnx")
        convert(st, args.template, onnx_out, args.group_size, base_onnx=args.base,
                grouping=args.grouping)
        c = cos_sim_vs_base(b, onnx_out)
        results.append((run_idx, c, onnx_out))
        print(f"  run{run_idx}: cos_sim={c:.4f}")

    if not results:
        raise RuntimeError("all runs failed")

    # 4) select the best
    results.sort(key=lambda r: r[1], reverse=True)
    best_idx, best_cos, best_onnx = results[0]
    import shutil
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    shutil.copy(best_onnx, args.out)

    cs = [c for _, c, _ in results]
    print("\n=== ranking ===")
    for run_idx, c, _ in results:
        print(f"  run{run_idx:<3d} cos_sim={c:.4f}" + ("   <- BEST" if run_idx == best_idx else ""))
    print(f"\nN={len(cs)}  min={min(cs):.4f}  mean={sum(cs)/len(cs):.4f}  max={max(cs):.4f}")
    print(f"best run{best_idx} (cos_sim={best_cos:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
