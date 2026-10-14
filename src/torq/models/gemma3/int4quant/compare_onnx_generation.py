import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, numpy_helper
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def ort_numpy_dtype(type_name):
    if type_name == "tensor(float16)":
        return np.float16
    return np.float32


def load_embeddings(model_dir):
    safetensors_path = Path(model_dir) / "model.safetensors"
    if not safetensors_path.exists():
        print(f"embedding safetensors not found at {safetensors_path}; loading via transformers")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        del model
        return emb

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        if "model.embed_tokens.weight" not in handle.keys():
            raise KeyError("Missing safetensors key: model.embed_tokens.weight")
        emb = handle.get_tensor("model.embed_tokens.weight")

    return emb.float().cpu().numpy()


def unpack_int4_from_int32(packed):
    packed = packed.to(torch.int32)
    parts = []
    for offset in range(8):
        parts.append(((packed >> (4 * offset)) & 0xF).to(torch.uint8))
    return torch.stack(parts, dim=-1).flatten(-2)


def unpack_int4_signed_from_int32(packed):
    return (unpack_int4_from_int32(packed).to(torch.int16) - 8).to(torch.int8)


def unpack_zero_point_if_needed(zero_point, expected_shape):
    if tuple(zero_point.shape) == tuple(expected_shape):
        zp = zero_point.to(torch.int8)
        if torch.any((zp < -8) | (zp > 7)):
            raise ValueError("zero_point values must be in signed int4 range [-8, 7]")
        return zp

    zero_point = zero_point.to(torch.int32)
    rows, cols = expected_shape
    zp = torch.empty((rows, cols), dtype=torch.int8)
    for offset in range(8):
        target_rows = torch.arange(offset, rows, 8)
        if target_rows.numel() == 0:
            continue
        zp[target_rows] = (
            ((zero_point[: target_rows.numel(), :] >> (4 * offset)) & 0xF).to(torch.int16)
            - 8
        ).to(torch.int8)
    return zp


def load_quant_lm_head_embeddings(model_dir):
    safetensors_path = Path(model_dir) / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Quantized lm_head embedding source requires a local safetensors file: "
            f"{safetensors_path}"
        )

    prefix = "lm_head.weight"
    names = {
        "packed": f"{prefix}_packed",
        "scale": f"{prefix}_scale",
        "shape": f"{prefix}_shape",
        "zero_point": f"{prefix}_zero_point",
    }
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [name for name in names.values() if name not in keys]
        if missing:
            raise KeyError(f"Missing quantized lm_head safetensors keys: {missing}")
        packed = handle.get_tensor(names["packed"])
        scale = handle.get_tensor(names["scale"]).float()
        shape = handle.get_tensor(names["shape"]).to(torch.int64).tolist()
        zero_point = handle.get_tensor(names["zero_point"])

    vocab_size = int(shape[0])
    hidden_size = int(shape[1])
    expected_packed = (vocab_size, (hidden_size + 7) // 8)
    if tuple(packed.shape) != expected_packed:
        raise ValueError(
            f"lm_head.weight_packed shape {tuple(packed.shape)} does not match "
            f"expected {expected_packed}"
        )
    if scale.shape[1] != hidden_size:
        raise ValueError(
            f"lm_head.weight_scale shape {tuple(scale.shape)} is not out-feature "
            f"grouped for hidden_size={hidden_size}"
        )

    out_groups = int(scale.shape[0])
    if vocab_size % out_groups != 0:
        raise ValueError(f"vocab_size={vocab_size} is not divisible by out_groups={out_groups}")
    group_size = vocab_size // out_groups

    q = unpack_int4_signed_from_int32(packed)[:, :hidden_size].float()
    zp = unpack_zero_point_if_needed(zero_point, (out_groups, hidden_size)).float()
    group_ids = torch.arange(vocab_size, dtype=torch.long) // group_size
    emb = (q - zp[group_ids]) * scale[group_ids]
    return emb.cpu().numpy()


def find_embeddings_in_onnx(onnx_path):
    model = onnx.load(str(onnx_path))
    candidates = [
        initializer
        for initializer in model.graph.initializer
        if "embed_tokens" in initializer.name and len(initializer.dims) == 2
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda initializer: int(np.prod(initializer.dims)), reverse=True)
    emb = numpy_helper.to_array(candidates[0])
    return emb.astype(np.float32)


def load_embeddings_for_model(onnx_path, model_dir, label, embedding_source):
    if embedding_source == "quant-lm-head":
        print(f"{label} embeddings: loaded from quantized lm_head in {model_dir}")
        return load_quant_lm_head_embeddings(model_dir)

    emb = find_embeddings_in_onnx(onnx_path)
    if emb is not None:
        print(f"{label} embeddings: loaded from ONNX initializer")
        return emb

    print(f"{label} embeddings: loaded from {model_dir}")
    return load_embeddings(model_dir)


def bf16_tensor_to_float32_array(tensor):
    shape = tuple(tensor.dims)
    if tensor.raw_data:
        bits = np.frombuffer(tensor.raw_data, dtype=np.uint16).copy()
    elif tensor.int32_data:
        bits = np.asarray(tensor.int32_data, dtype=np.uint16)
    else:
        array = numpy_helper.to_array(tensor)
        return array.astype(np.float32).reshape(shape)

    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def convert_tensor_bf16_to_fp32(tensor):
    if tensor.data_type != TensorProto.BFLOAT16:
        return tensor
    return numpy_helper.from_array(bf16_tensor_to_float32_array(tensor), name=tensor.name)


def convert_type_proto_bf16_to_fp32(value_info):
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == TensorProto.BFLOAT16:
        tensor_type.elem_type = TensorProto.FLOAT


def convert_model_to_fp32(path):
    model = onnx.load(str(path))

    for idx, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == TensorProto.BFLOAT16:
            model.graph.initializer[idx].CopyFrom(convert_tensor_bf16_to_fp32(initializer))

    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        convert_type_proto_bf16_to_fp32(value_info)

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.BFLOAT16:
                attr.t.CopyFrom(convert_tensor_bf16_to_fp32(attr.t))
            elif attr.type == onnx.AttributeProto.TENSORS:
                for idx, tensor in enumerate(attr.tensors):
                    if tensor.data_type == TensorProto.BFLOAT16:
                        attr.tensors[idx].CopyFrom(convert_tensor_bf16_to_fp32(tensor))

        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    attr.i = TensorProto.FLOAT

    fd, tmp_path = tempfile.mkstemp(suffix=".fp32.onnx")
    os.close(fd)
    onnx.save(model, tmp_path)
    return tmp_path


def make_session(path, provider):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available. Available providers: {available}"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), opts, providers=providers)


def make_initial_cache(num_layers, past_len, head_dim, dtype):
    return {
        f"past_key_values.{idx}.key_value": np.zeros(
            (1, 2, past_len, head_dim), dtype=dtype
        )
        for idx in range(num_layers)
    }


def build_feed(token_id, position_id, embeddings, caches, dtype):
    token_embedding = embeddings[token_id].reshape(1, 1, -1).astype(dtype)
    feed = {
        "token_embedding": token_embedding,
        "position_ids": np.array([[position_id]], dtype=np.int32),
    }
    feed.update(caches)
    return feed


def update_cache(outputs, num_layers, past_len, dtype):
    caches = {}
    for idx in range(num_layers):
        present = outputs[idx + 1].astype(dtype, copy=False)
        caches[f"past_key_values.{idx}.key_value"] = present[:, :, -past_len:, :]
    return caches


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def topk(logits, k):
    row = logits.reshape(-1)
    idx = np.argpartition(-row, kth=min(k, row.size - 1))[:k]
    idx = idx[np.argsort(-row[idx])]
    return [(int(i), float(row[i])) for i in idx]


def get_stop_token_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for token in ("<end_of_turn>", "<eos>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_ids.add(int(token_id))

    return stop_ids


def run_one_step(sess, token_id, position_id, embeddings, caches, dtype, num_layers, past_len):
    feed = build_feed(token_id, position_id, embeddings, caches, dtype)
    outputs = sess.run(None, feed)
    if len(outputs) < num_layers + 1:
        output_names = [output.name for output in sess.get_outputs()]
        raise RuntimeError(
            f"Expected at least {num_layers + 1} outputs, got {len(outputs)}: {output_names}"
        )
    logits = outputs[0][:, -1, :]
    next_token = int(np.argmax(logits, axis=-1)[0])
    next_caches = update_cache(outputs, num_layers, past_len, dtype)
    return logits, next_token, next_caches


def main():
    parser = argparse.ArgumentParser(
        description="Run side-by-side greedy generation for two Gemma decode ONNX models."
    )
    parser.add_argument("--onnx-a", required=True, help="Baseline ONNX path")
    parser.add_argument("--onnx-b", required=True, help="Candidate ONNX path")
    parser.add_argument("--model-dir", required=True, help="HF model dir with tokenizer and model.safetensors")
    parser.add_argument("--model-dir-a", default=None, help="Optional HF model dir for onnx-a embeddings")
    parser.add_argument("--model-dir-b", default=None, help="Optional HF model dir for onnx-b embeddings")
    parser.add_argument(
        "--embedding-source",
        choices=["dense", "quant-lm-head"],
        default="dense",
        help="Embedding source for both models unless overridden.",
    )
    parser.add_argument(
        "--embedding-source-a",
        choices=["dense", "quant-lm-head"],
        default=None,
        help="Embedding source for onnx-a.",
    )
    parser.add_argument(
        "--embedding-source-b",
        choices=["dense", "quant-lm-head"],
        default=None,
        help="Embedding source for onnx-b.",
    )
    parser.add_argument("--prompt", default="Introduce yourself in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=18)
    parser.add_argument("--past-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-fp32-convert",
        action="store_true",
        help="Open ONNX files as-is. By default BF16 models are converted to FP32 temp files for CPUExecutionProvider.",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNX Runtime execution provider.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model_dir_a = args.model_dir_a or args.model_dir
    model_dir_b = args.model_dir_b or args.model_dir
    embedding_source_a = args.embedding_source_a or args.embedding_source
    embedding_source_b = args.embedding_source_b or args.embedding_source
    embeddings_a = load_embeddings_for_model(
        args.onnx_a, model_dir_a, "onnx-a", embedding_source_a
    )
    embeddings_b = load_embeddings_for_model(
        args.onnx_b, model_dir_b, "onnx-b", embedding_source_b
    )

    tmp_paths = []
    onnx_a = args.onnx_a
    onnx_b = args.onnx_b
    if not args.no_fp32_convert:
        onnx_a = convert_model_to_fp32(args.onnx_a)
        onnx_b = convert_model_to_fp32(args.onnx_b)
        tmp_paths.extend([onnx_a, onnx_b])

    try:
        sess_a = make_session(onnx_a, args.provider)
        sess_b = make_session(onnx_b, args.provider)
        dtype = ort_numpy_dtype(sess_a.get_inputs()[0].type)

        prompt_text = args.prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not input_ids:
            raise ValueError("Tokenizer produced no input ids")

        caches_a = make_initial_cache(args.num_layers, args.past_len, args.head_dim, dtype)
        caches_b = make_initial_cache(args.num_layers, args.past_len, args.head_dim, dtype)

        generated_a = list(input_ids)
        generated_b = list(input_ids)
        current_a = input_ids[0]
        current_b = input_ids[0]

        print("prompt_text:")
        print(prompt_text)
        print("prompt_ids:", input_ids)
        print()

        # Warm up both models through the prompt. This keeps each model's cache on its
        # own generated path only after prompt processing is finished.
        last_logits_a = None
        last_logits_b = None
        for pos, token_id in enumerate(input_ids):
            last_logits_a, current_a, caches_a = run_one_step(
                sess_a, token_id, pos, embeddings_a, caches_a, dtype, args.num_layers, args.past_len
            )
            last_logits_b, current_b, caches_b = run_one_step(
                sess_b, token_id, pos, embeddings_b, caches_b, dtype, args.num_layers, args.past_len
            )

        print("after prompt:")
        print("  logits_cos:", cosine(last_logits_a, last_logits_b))
        print("  a_top:", [(tokenizer.decode([i]), i, v) for i, v in topk(last_logits_a, args.top_k)])
        print("  b_top:", [(tokenizer.decode([i]), i, v) for i, v in topk(last_logits_b, args.top_k)])
        print()

        stop_token_ids = get_stop_token_ids(tokenizer)
        next_a = current_a
        next_b = current_b
        active_a = next_a not in stop_token_ids
        active_b = next_b not in stop_token_ids

        generated_a.append(next_a)
        generated_b.append(next_b)
        print(
            f"step=00 cos={cosine(last_logits_a, last_logits_b):.6f} "
            f"a={next_a}:{tokenizer.decode([next_a])!r} "
            f"b={next_b}:{tokenizer.decode([next_b])!r}"
        )

        for step in range(1, args.max_new_tokens):
            # The first generated token is appended at sequence position len(input_ids).
            # step starts at 1 here because step=0 was produced from the prompt logits.
            pos = len(input_ids) + step - 1
            logits_a = None
            logits_b = None

            if active_a:
                logits_a, next_a, caches_a = run_one_step(
                    sess_a, next_a, pos, embeddings_a, caches_a, dtype, args.num_layers, args.past_len
                )
                generated_a.append(next_a)
                active_a = next_a not in stop_token_ids

            if active_b:
                logits_b, next_b, caches_b = run_one_step(
                    sess_b, next_b, pos, embeddings_b, caches_b, dtype, args.num_layers, args.past_len
                )
                generated_b.append(next_b)
                active_b = next_b not in stop_token_ids

            cos_text = "ended"
            if logits_a is not None and logits_b is not None:
                cos_text = f"{cosine(logits_a, logits_b):.6f}"

            token_a_text = "ended" if logits_a is None else f"{next_a}:{tokenizer.decode([next_a])!r}"
            token_b_text = "ended" if logits_b is None else f"{next_b}:{tokenizer.decode([next_b])!r}"
            print(f"step={step:02d} cos={cos_text} a={token_a_text} b={token_b_text}")

            if not active_a and not active_b:
                break

        print()
        print("decoded_a:")
        print(tokenizer.decode(generated_a, skip_special_tokens=False))
        print()
        print("decoded_b:")
        print(tokenizer.decode(generated_b, skip_special_tokens=False))
    finally:
        for tmp_path in tmp_paths:
            if tmp_path not in (args.onnx_a, args.onnx_b) and os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
