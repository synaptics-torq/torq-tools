"""
Per-layer quantization sensitivity analysis via logit divergence.

For each MatMul layer, swaps that one layer's weight from int8 -> int4 in the
reference model, runs calibration prompts, and measures how much the output
logits diverge from the pure int8 baseline.

Usage:
  cd torq-tools-dev
  python layer_sensitivity.py
  python layer_sensitivity.py --n-prompts 3 --n-tokens 3
"""

import argparse
import json
import time
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort

BASE = Path("/home/kshanmug/synpu_compiler/torq-tools-dev/models/google/gemma-3-270m-it/export/onnx")
INT8_MODEL = BASE / "int8_converted" / "static" / "model.onnx"
INT4_MODEL = BASE / "converted" / "static" / "model.onnx"

CALIBRATION_PROMPTS = [
    "What is the capital of France?",
    "What is photosynthesis?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "Explain why the sky is blue.",
    "What is the largest planet in our solar system?",
    "What is DNA?",
    "Who invented the telephone?",
    "What is gravity?",
    "What is the speed of light?",
]

BOS_ID = 2
EOS_ID = 1
END_TURN_ID = 106
N_LAYERS = 18
HEAD_DIM = 256
MAX_SEQ = 256
SYS_PROMPT = "You are a helpful AI assistant named Gemma. Answer in 1-2 sentences. No lists, no bullet points, no repetition."


def convert_proto_to_fp32(model):
    """Convert a bf16 model proto to fp32 in-memory."""
    has_bf16 = any(
        init.data_type == TensorProto.BFLOAT16 for init in model.graph.initializer
    ) or any(
        inp.type.tensor_type.elem_type == TensorProto.BFLOAT16 for inp in model.graph.input
    )
    if not has_bf16:
        return model

    new_inits = []
    for init in model.graph.initializer:
        if init.data_type == TensorProto.BFLOAT16:
            dims = list(init.dims)
            n_elems = 1
            for d in dims:
                n_elems *= d
            raw = init.raw_data
            if len(raw) == n_elems * 2:
                u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
                fp32 = np.zeros(dims, dtype=np.float32)
                fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
            else:
                fp32 = np.frombuffer(raw, dtype=np.float32).reshape(dims)
            new_inits.append(numpy_helper.from_array(fp32, name=init.name))
        else:
            new_inits.append(init)

    new_inputs = []
    for inp in model.graph.input:
        t = inp.type.tensor_type
        if t.elem_type == TensorProto.BFLOAT16:
            shape = [d.dim_value if d.HasField("dim_value") else d.dim_param
                     for d in t.shape.dim] if t.HasField("shape") else None
            new_inputs.append(helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, shape))
        else:
            new_inputs.append(inp)

    new_outputs = []
    for out in model.graph.output:
        t = out.type.tensor_type
        if t.elem_type == TensorProto.BFLOAT16:
            shape = [d.dim_value if d.HasField("dim_value") else d.dim_param
                     for d in t.shape.dim] if t.HasField("shape") else None
            new_outputs.append(helper.make_tensor_value_info(out.name, TensorProto.FLOAT, shape))
        else:
            new_outputs.append(out)

    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "Cast":
            new_attrs = []
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    new_attr = onnx.AttributeProto()
                    new_attr.name = "to"
                    new_attr.type = onnx.AttributeProto.INT
                    new_attr.i = TensorProto.FLOAT
                    new_attrs.append(new_attr)
                else:
                    new_attrs.append(attr)
            new_node = helper.make_node(
                node.op_type, inputs=list(node.input),
                outputs=list(node.output), name=node.name,
            )
            new_node.attribute.extend(new_attrs)
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes=new_nodes, name=model.graph.name,
        inputs=new_inputs, outputs=new_outputs, initializer=new_inits,
    )
    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    del new_model.opset_import[:]
    for op in model.opset_import:
        new_model.opset_import.append(op)
    return new_model


def load_tokenizer():
    from tokenizers import Tokenizer
    tok_path = BASE / "int8_converted" / "static" / "tokenizer.json"
    if not tok_path.exists():
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download("google/gemma-3-270m-it", "tokenizer.json")
    return Tokenizer.from_file(str(tok_path))


def tokenize_prompt(tokenizer, text):
    start_token = tokenizer.decode([105], skip_special_tokens=False)
    end_token = tokenizer.decode([106], skip_special_tokens=False)

    def encode_turn(content, role):
        if role == 'model':
            ids = tokenizer.encode(start_token + 'model\n').ids
        else:
            ids = tokenizer.encode(start_token + role + '\n' + content + end_token + '\n').ids
        return ids[1:] if ids and ids[0] == BOS_ID else ids

    sys_toks = encode_turn(SYS_PROMPT, 'system')
    user_toks = encode_turn(text, 'user')
    model_toks = encode_turn('', 'model')
    return [BOS_ID] + sys_toks + user_toks + model_toks


class GemmaInference:
    def __init__(self, sess, embeddings):
        self.sess = sess
        self.embeddings = embeddings
        self.output_names = [o.name for o in sess.get_outputs()]
        self.kv = {
            f"past_key_values.{i}.key_value": np.zeros((1, 2, MAX_SEQ, HEAD_DIM), dtype=np.float32)
            for i in range(N_LAYERS)
        }

    def reset(self):
        for k in self.kv:
            self.kv[k][:] = 0

    def step(self, token_id, pos):
        emb = self.embeddings[token_id].astype(np.float32).reshape(1, 1, -1)
        feeds = {"token_embedding": emb, "position_ids": np.array([[pos]], dtype=np.int32)}
        feeds.update(self.kv)
        outputs = self.sess.run(None, feeds)
        for idx, name in enumerate(self.output_names[1:], 1):
            self.kv[name.replace("present.", "past_key_values.")] = outputs[idx]
        return outputs[0][0, -1]

    def run_prompt_collect_logits(self, token_ids, n_gen_tokens, reference_tokens=None):
        """Run prompt and collect logits.
        
        If reference_tokens is provided (teacher forcing), feed those tokens
        instead of the model's own argmax predictions.
        """
        self.reset()
        pos = 0
        for tok in token_ids:
            logits = self.step(tok, pos)
            pos += 1

        results = []
        next_token = int(logits.argmax())
        results.append((next_token, logits.copy()))

        for i in range(n_gen_tokens - 1):
            # Teacher forcing: feed baseline's token, not our own argmax
            if reference_tokens is not None:
                feed_token = reference_tokens[i] if i < len(reference_tokens) else next_token
            else:
                feed_token = next_token
            if feed_token in (EOS_ID, END_TURN_ID):
                break
            logits = self.step(feed_token, pos)
            pos += 1
            next_token = int(logits.argmax())
            results.append((next_token, logits.copy()))
        return results


def kl_divergence(logits_p, logits_q, temperature=1.0):
    p = logits_p.astype(np.float64) / temperature
    q = logits_q.astype(np.float64) / temperature
    p = p - p.max()
    q = q - q.max()
    log_p = p - np.log(np.sum(np.exp(p)))
    log_q = q - np.log(np.sum(np.exp(q)))
    p_probs = np.exp(log_p)
    kl = np.sum(p_probs * (log_p - log_q))
    return float(max(kl, 0.0))


def top_k_overlap(logits_a, logits_b, k=5):
    top_a = set(np.argsort(logits_a)[-k:])
    top_b = set(np.argsort(logits_b)[-k:])
    return len(top_a & top_b) / k


def cross_entropy_at_token(logits, target_token):
    """Compute -log P(target_token) from logits (negative log-likelihood)."""
    logits_f64 = logits.astype(np.float64)
    logits_f64 = logits_f64 - logits_f64.max()
    log_probs = logits_f64 - np.log(np.sum(np.exp(logits_f64)))
    return float(-log_probs[target_token])


def compute_sensitivity_metrics(baseline_logits, modified_logits):
    n = min(len(baseline_logits), len(modified_logits))
    if n == 0:
        return {"kl_div": 0.0, "top1_match": 1.0, "top5_overlap": 1.0, "logit_mse": 0.0,
                "cross_entropy": 0.0, "perplexity": 1.0}

    kl_divs, top1_matches, top5_overlaps, mses, ce_losses = [], [], [], [], []
    for i in range(n):
        base_tok, base_logit = baseline_logits[i]
        mod_tok, mod_logit = modified_logits[i]
        kl_divs.append(kl_divergence(base_logit, mod_logit))
        top1_matches.append(1.0 if base_tok == mod_tok else 0.0)
        top5_overlaps.append(top_k_overlap(base_logit, mod_logit, k=5))
        mses.append(float(np.mean((base_logit.astype(np.float64) - mod_logit.astype(np.float64)) ** 2)))
        # Cross-entropy: how likely is the baseline's token under the modified model?
        ce_losses.append(cross_entropy_at_token(mod_logit, base_tok))

    mean_ce = float(np.mean(ce_losses))
    return {
        "kl_div": float(np.mean(kl_divs)),
        "kl_div_max": float(np.max(kl_divs)),
        "top1_match": float(np.mean(top1_matches)),
        "top5_overlap": float(np.mean(top5_overlaps)),
        "logit_mse": float(np.mean(mses)),
        "cross_entropy": mean_ce,
        "perplexity": float(np.exp(mean_ce)),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-layer quantization sensitivity analysis")
    parser.add_argument("--ref-model", default=str(INT8_MODEL))
    parser.add_argument("--test-model", default=str(INT4_MODEL))
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_prompts = min(args.n_prompts, len(CALIBRATION_PROMPTS))
    prompts = CALIBRATION_PROMPTS[:n_prompts]

    print(f"{'='*100}")
    print(f"PER-LAYER QUANTIZATION SENSITIVITY ANALYSIS")
    print(f"{'='*100}")
    print(f"  Reference model : {args.ref_model}")
    print(f"  Test model      : {args.test_model}")
    print(f"  Prompts         : {n_prompts}")
    print(f"  Tokens/prompt   : {args.n_tokens}")
    print()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    emb_path = Path(args.ref_model).parent / "token_embeddings.npy"
    print(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(str(emb_path))
    if embeddings.dtype != np.float32:
        u16 = embeddings.view(np.uint16)
        fp32 = np.zeros(u16.shape, dtype=np.float32)
        fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
        embeddings = fp32

    print("Tokenizing prompts...")
    all_token_ids = [tokenize_prompt(tokenizer, p) for p in prompts]
    for i, (p, ids) in enumerate(zip(prompts, all_token_ids)):
        print(f"  [{i+1}] {p} ({len(ids)} tokens)")
    print()

    print("Loading reference model (int8)...")
    t0 = time.time()
    ref_model_proto = onnx.load(args.ref_model)
    print(f"  Converting bf16 -> fp32...")
    ref_fp32_proto = convert_proto_to_fp32(ref_model_proto)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("  Creating ORT session for baseline...")
    ref_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(ref_fp32_proto, ref_tmp.name)
    ref_tmp.close()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 4
    ref_sess = ort.InferenceSession(ref_tmp.name, opts, providers=["CPUExecutionProvider"])
    ref_engine = GemmaInference(ref_sess, embeddings)

    print(f"\n{'-'*100}")
    print("Phase 1: Collecting baseline logits...")
    baseline_all = []
    for i, ids in enumerate(all_token_ids):
        t1 = time.time()
        logits_seq = ref_engine.run_prompt_collect_logits(ids, args.n_tokens)
        ms = (time.time() - t1) * 1000
        tokens_str = tokenizer.decode([t for t, _ in logits_seq])
        print(f"  [{i+1}/{n_prompts}] {ms:.0f}ms -- \"{tokens_str[:50]}\"")
        baseline_all.append(logits_seq)
    print(f"  Baseline collection complete.\n")

    print("Loading test model (int4) weights...")
    t0 = time.time()
    test_model_proto = onnx.load(args.test_model)
    test_init_map = {init.name: init for init in test_model_proto.graph.initializer}
    print(f"  Loaded in {time.time()-t0:.1f}s")

    ref_init_map = {init.name: init for init in ref_model_proto.graph.initializer}

    # Build test model weight map by node name (handles naming differences like _bf16 suffix)
    test_node_weight_map = {}
    for node in test_model_proto.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            for inp in node.input:
                if inp in test_init_map:
                    test_node_weight_map[node.name] = inp
                    break

    matmul_layers = []
    for i, node in enumerate(ref_model_proto.graph.node):
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        wt_name = node.input[1]
        if wt_name not in ref_init_map:
            continue
        # Match by node name, not weight name
        test_wt_name = test_node_weight_map.get(node.name)
        if test_wt_name is None or test_wt_name not in test_init_map:
            continue
        matmul_layers.append({"node_index": i, "weight_name": wt_name, "test_weight_name": test_wt_name, "shape": list(ref_init_map[wt_name].dims)})

    print(f"  Found {len(matmul_layers)} MatMul layers to evaluate\n")

    print(f"{'-'*100}")
    print(f"Phase 2: Measuring per-layer sensitivity ({len(matmul_layers)} layers)...")
    print(f"{'-'*100}")

    # Release baseline ORT session to free memory before per-layer loop
    del ref_sess, ref_engine

    results = []
    total_layers = len(matmul_layers)

    for layer_idx, layer_info in enumerate(matmul_layers):
        wt_name = layer_info["weight_name"]
        short_name = wt_name[-55:] if len(wt_name) > 55 else wt_name

        fp32_init_idx = None
        for idx, init in enumerate(ref_fp32_proto.graph.initializer):
            if init.name == wt_name:
                fp32_init_idx = idx
                break

        if fp32_init_idx is None:
            print(f"  [{layer_idx+1}/{total_layers}] SKIP -- {short_name}")
            continue

        orig_init = ref_fp32_proto.graph.initializer[fp32_init_idx]
        orig_raw = orig_init.raw_data
        orig_dtype = orig_init.data_type
        orig_dims = list(orig_init.dims)

        test_init = test_init_map[layer_info.get("test_weight_name", wt_name)]
        if test_init.data_type == TensorProto.BFLOAT16:
            dims = list(test_init.dims)
            n_elems = 1
            for d in dims:
                n_elems *= d
            raw = test_init.raw_data
            if len(raw) == n_elems * 2:
                u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
                fp32_arr = np.zeros(dims, dtype=np.float32)
                fp32_arr.view(np.uint32)[...] = u16.astype(np.uint32) << 16
            else:
                fp32_arr = np.frombuffer(raw, dtype=np.float32).reshape(dims)
        else:
            fp32_arr = numpy_helper.to_array(test_init).astype(np.float32)

        new_init = numpy_helper.from_array(fp32_arr, name=wt_name)
        ref_fp32_proto.graph.initializer[fp32_init_idx].CopyFrom(new_init)

        t_start = time.time()
        mod_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx.save(ref_fp32_proto, mod_tmp.name)
        mod_tmp.close()

        try:
            mod_sess = ort.InferenceSession(mod_tmp.name, opts, providers=["CPUExecutionProvider"])
            mod_engine = GemmaInference(mod_sess, embeddings)

            prompt_metrics = []
            for p_idx, ids in enumerate(all_token_ids):
                # Teacher forcing: feed baseline's generated tokens to the modified model
                ref_tokens = [t for t, _ in baseline_all[p_idx]]
                mod_logits = mod_engine.run_prompt_collect_logits(ids, args.n_tokens, reference_tokens=ref_tokens)
                metrics = compute_sensitivity_metrics(baseline_all[p_idx], mod_logits)
                prompt_metrics.append(metrics)

            avg_kl = float(np.mean([m["kl_div"] for m in prompt_metrics]))
            max_kl = float(np.max([m["kl_div_max"] for m in prompt_metrics]))
            avg_top1 = float(np.mean([m["top1_match"] for m in prompt_metrics]))
            avg_top5 = float(np.mean([m["top5_overlap"] for m in prompt_metrics]))
            avg_mse = float(np.mean([m["logit_mse"] for m in prompt_metrics]))
            avg_ce = float(np.mean([m["cross_entropy"] for m in prompt_metrics]))
            avg_ppl = float(np.exp(avg_ce))

            elapsed = time.time() - t_start
            result = {
                "weight_name": wt_name, "node_index": layer_info["node_index"],
                "shape": layer_info["shape"], "kl_div": avg_kl, "kl_div_max": max_kl,
                "top1_match": avg_top1, "top5_overlap": avg_top5, "logit_mse": avg_mse,
                "cross_entropy": avg_ce, "perplexity": avg_ppl,
            }
            results.append(result)

            severity = "CRITICAL" if avg_kl > 1.0 else "HIGH" if avg_kl > 0.1 else "MEDIUM" if avg_kl > 0.01 else "LOW"
            print(f"  [{layer_idx+1:3d}/{total_layers}] KL={avg_kl:.6f} ppl={avg_ppl:.2f} top1={avg_top1:.2f} "
                  f"[{severity:>8}] {elapsed:.1f}s  {short_name}")

            del mod_sess, mod_engine

        except Exception as e:
            print(f"  [{layer_idx+1:3d}/{total_layers}] ERROR: {e}  {short_name}")

        finally:
            orig_restore = onnx.TensorProto()
            orig_restore.name = wt_name
            orig_restore.data_type = orig_dtype
            orig_restore.dims.extend(orig_dims)
            orig_restore.raw_data = orig_raw
            ref_fp32_proto.graph.initializer[fp32_init_idx].CopyFrom(orig_restore)
            try:
                os.unlink(mod_tmp.name)
            except OSError:
                pass

    # Summary
    results.sort(key=lambda r: r["kl_div"], reverse=True)

    print(f"\n{'='*100}")
    print(f"SENSITIVITY RANKING (sorted by KL divergence, highest first)")
    print(f"  Teacher forcing: modified model fed baseline's tokens (perplexity w.r.t. int8 GT)")
    print(f"{'='*100}")
    print(f"  {'Rank':>4}  {'KL Div':>10}  {'PPL':>8}  {'CE':>7}  {'Top1':>5}  "
          f"{'Top5':>5}  {'MSE':>12}  {'Severity':>8}  Weight Name")

    for rank, r in enumerate(results):
        kl = r["kl_div"]
        severity = "CRITICAL" if kl > 1.0 else "HIGH" if kl > 0.1 else "MEDIUM" if kl > 0.01 else "LOW"
        wn = r["weight_name"][-55:] if len(r["weight_name"]) > 55 else r["weight_name"]
        print(f"  {rank+1:>4}  {r['kl_div']:>10.6f}  {r['perplexity']:>8.2f}  "
              f"{r['cross_entropy']:>7.4f}  {r['top1_match']:>5.2f}  {r['top5_overlap']:>5.2f}  "
              f"{r['logit_mse']:>12.6e}  {severity:>8}  {wn}")

    if results:
        n_critical = sum(1 for r in results if r["kl_div"] > 1.0)
        n_high = sum(1 for r in results if 0.1 < r["kl_div"] <= 1.0)
        n_medium = sum(1 for r in results if 0.01 < r["kl_div"] <= 0.1)
        n_low = sum(1 for r in results if r["kl_div"] <= 0.01)

        print(f"\n  SEVERITY BREAKDOWN:")
        print(f"    CRITICAL (KL > 1.0)     : {n_critical:3d} / {len(results)}  -- MUST keep at int8")
        print(f"    HIGH     (KL > 0.1)     : {n_high:3d} / {len(results)}  -- Strongly recommend int8")
        print(f"    MEDIUM   (KL > 0.01)    : {n_medium:3d} / {len(results)}  -- Consider int8")
        print(f"    LOW      (KL <= 0.01)   : {n_low:3d} / {len(results)}  -- Safe to use int4")
        print(f"\n  RECOMMENDATION:")
        print(f"    Keep top {n_critical + n_high} layers (CRITICAL+HIGH) at int8.")
        print(f"    The remaining {n_medium + n_low} layers can safely use int4.")

    print(f"\n{'='*100}\n")

    out_path = args.output or str(BASE / "layer_sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    try:
        os.unlink(ref_tmp.name)
    except OSError:
        pass


if __name__ == "__main__":
    main()
