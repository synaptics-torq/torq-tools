# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export Qwen3.5 (text model) to static ONNX and Torq VMFBs.

optimum has no Qwen3.5 export, so this exporter builds a static single-token
decode step in PyTorch (see :mod:`._step`) and exports it with
``torch.onnx.export``. The steps:

1. Build the step from the Hugging Face weights and compare its logits with the
   Hugging Face model on a short chat prompt.
2. Export ``transformer.onnx``: token embedding, position and the three packed
   states in; hidden state and the updated states out.
3. Trim the output vocabulary to Latin text, punctuation and digits (as for
   Qwen2.5 and Qwen3) and write ``lm_head.onnx`` from the tied embeddings, plus
   ``token_embeddings.npy`` and ``token_id_lut.npy`` for the host.
4. Convert both models to bf16 and quantize their MatMul weights to INT4
   DequantizeLinear (round-to-nearest, block 32, bf16 scales, INT4 zero points).
5. Compile ``transformer.vmfb`` and ``lm_head.vmfb`` with ``compile_flags.json``.

Needs transformers 5.x (``requirements.txt``), which conflicts with the optimum
pin of the base package: use a separate virtualenv.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from huggingface_hub import snapshot_download
from onnx import TensorProto, helper, numpy_helper, shape_inference
from tokenizers import Tokenizer

from ...utils.compile import export_torq, read_compile_flags_file
from ...utils.logging import configure_logging
from ..qwen._trim_vocab import build_qwen_trimmed_vocab_spec
from . import add_qwen35_export_args

logger = logging.getLogger("Qwen3.5.export")

COMPILE_FLAGS_FILE = Path(__file__).parent / "compile_flags.json"
# The compact vocabulary is padded with repeats of its last token to a multiple
# of 64: an odd INT4 LM head width puts tile rows mid-byte.
VOCAB_PAD_MULTIPLE = 64
QUANT_BLOCK = 32
HOST_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _load_hf(hf_repo: str):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        import transformers.models.qwen3_5  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Qwen3.5 export needs transformers 5.x and torch: install "
            "src/torq/models/qwen3_5/requirements.txt in a separate virtualenv"
        ) from e
    tok = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForImageTextToText.from_pretrained(hf_repo, dtype=torch.float32).eval()
    return tok, model


def _validate(step, model, tok) -> None:
    import torch

    text_model = model.model.language_model
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = model(input_ids=ids).logits[0]
        states = step.init_states()
        max_diff = 0.0
        for t in range(ids.shape[1]):
            out = step(text_model.embed_tokens(ids[:, t : t + 1]), torch.tensor([[t]]), *states)
            states = list(out[1:])
            logits = model.lm_head(out[0])[0, 0]
            max_diff = max(max_diff, (logits - ref[t]).abs().max().item())
        gen, nxt, pos = [], int(logits.argmax()), ids.shape[1]
        for _ in range(12):
            gen.append(nxt)
            out = step(text_model.embed_tokens(torch.tensor([[nxt]])), torch.tensor([[pos]]), *states)
            states, pos = list(out[1:]), pos + 1
            nxt = int(model.lm_head(out[0])[0, 0].argmax())
    logger.info("Static step vs Hugging Face: max |logit diff| %.3e over %d tokens", max_diff, ids.shape[1])
    logger.info("Static step greedy answer: %r", tok.decode(gen))
    if max_diff > 1e-2:
        raise RuntimeError(f"Static decode step does not match the Hugging Face model (max logit diff {max_diff:.3e})")


def _fold_shapes_and_strip(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Shape nodes of static tensors with constants and drop value_info.

    torch.onnx.export writes fp32 value_info for every intermediate tensor; it
    becomes stale after the bf16 conversion, so drop it.
    """
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[vi.name] = [d.dim_value for d in dims]
    for i, node in enumerate(list(model.graph.node)):
        if node.op_type == "Shape" and node.input[0] in shapes:
            value = numpy_helper.from_array(np.array(shapes[node.input[0]], dtype=np.int64))
            model.graph.node.remove(node)
            model.graph.node.insert(i, helper.make_node("Constant", [], [node.output[0]], value=value, name=node.name))
    del model.graph.value_info[:]
    model.graph.name = "main"
    return model


def _to_bf16(model: onnx.ModelProto) -> onnx.ModelProto:
    from torq.lab.model_tools.dtype_conversion.onnx import FP32Converter, Int64Converter

    model = FP32Converter("bf16", convert_io=True).convert_model(model)
    model = Int64Converter("int32", convert_io=True).convert_model(model)
    del model.graph.value_info[:]
    return model


def _tensor_np(t: onnx.TensorProto) -> np.ndarray:
    if t.data_type == TensorProto.BFLOAT16:
        raw = np.frombuffer(t.raw_data, dtype=np.uint16).view(ml_dtypes.bfloat16)
        return raw.astype(np.float32).reshape(tuple(t.dims))
    return numpy_helper.to_array(t).astype(np.float32)


def _int4_tensor(name: str, a: np.ndarray) -> onnx.TensorProto:
    # ONNX INT4 raw_data: two values per byte, first value in the low nibble.
    u = (a.astype(np.int8).ravel() & 0xF).astype(np.uint8)
    if u.size % 2:
        u = np.append(u, 0)
    t = onnx.TensorProto(name=name, data_type=TensorProto.INT4, dims=a.shape)
    t.raw_data = (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()
    return t


def _bf16_tensor(name: str, a: np.ndarray) -> onnx.TensorProto:
    t = onnx.TensorProto(name=name, data_type=TensorProto.BFLOAT16, dims=a.shape)
    t.raw_data = a.astype(ml_dtypes.bfloat16).view(np.uint16).tobytes()
    return t


def _quantize_int4(model: onnx.ModelProto) -> int:
    """Replace each constant MatMul weight [K, N] of a bf16 model with INT4 DequantizeLinear.

    Round-to-nearest, asymmetric, block 32 along K: bf16 scales and INT4 zero
    points of shape [K / 32, N], the format of the Qwen2.5 and Qwen3 exports.
    The torq.lab weight quantizer is not used: its bf16 conversion fails on the
    torch-exported graph, so the graph is converted to bf16 first.
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    count = 0
    for node in list(g.node):
        if node.op_type != "MatMul" or node.input[1] not in inits:
            continue
        w = _tensor_np(inits[node.input[1]])
        if w.ndim != 2 or w.shape[0] % QUANT_BLOCK:
            continue
        k, n = w.shape
        wb = w.reshape(k // QUANT_BLOCK, QUANT_BLOCK, n)
        lo, hi = np.minimum(wb.min(1), 0), np.maximum(wb.max(1), 0)
        scale = (hi - lo) / 15.0
        scale = np.where(scale == 0, 1.0, scale).astype(ml_dtypes.bfloat16).astype(np.float32)
        zp = np.clip(np.round(-8 - lo / scale), -8, 7)
        q = np.clip(np.round(wb / scale[:, None, :]) + zp[:, None, :], -8, 7).reshape(k, n)
        base = node.input[1]
        g.initializer.extend([_int4_tensor(f"{base}_q", q), _bf16_tensor(f"{base}_s", scale), _int4_tensor(f"{base}_z", zp)])
        g.node.insert(
            list(g.node).index(node),
            helper.make_node("DequantizeLinear", [f"{base}_q", f"{base}_s", f"{base}_z"], [f"{base}_dq"],
                             axis=0, block_size=QUANT_BLOCK, name=f"{base}_DQL"),
        )
        node.input[1] = f"{base}_dq"
        g.initializer.remove(inits[base])
        count += 1
    return count


def _flat_text_config(cfg: dict, tok) -> dict:
    """config.json for the runners: the text config at the top level, with the token IDs they read."""
    flat = dict(cfg.get("text_config", cfg))
    endoftext = tok.convert_tokens_to_ids("<|endoftext|>")
    flat.update({
        "model_type": flat.get("model_type", "qwen3_5_text"),
        "bos_token_id": endoftext,
        "eos_token_id": tok.convert_tokens_to_ids("<|im_end|>"),
        "pad_token_id": endoftext,
        "tie_word_embeddings": True,
    })
    return flat


def _write_lm_head_and_host_files(model, tok, src_dir: Path, onnx_dir: Path, torq_dir: Path) -> None:
    import torch

    cfg = json.loads((src_dir / "config.json").read_text())
    flat = _flat_text_config(cfg, tok)
    tok_json = json.loads((src_dir / "tokenizer.json").read_text())
    spec = build_qwen_trimmed_vocab_spec(
        Tokenizer.from_file(str(src_dir / "tokenizer.json")), tok_json, flat,
        selected_groups=("latin", "punct", "digits"), byte_fallback=True,
    )
    kept = list(spec.kept_model_ids) + list(spec.extra_token_ids)
    n_kept = len(kept)
    kept += [kept[-1]] * (-len(kept) % VOCAB_PAD_MULTIPLE)
    logger.info("Vocabulary %d -> %d tokens (padded to %d)", flat["vocab_size"], n_kept, len(kept))

    with torch.no_grad():
        emb = model.model.language_model.embed_tokens.weight.float().numpy()
    np.save(torq_dir / "token_embeddings.npy", emb.astype(ml_dtypes.bfloat16))
    np.save(torq_dir / "token_id_lut.npy", np.array(kept, dtype=np.int32))
    (torq_dir / "config.json").write_text(json.dumps(flat, indent=2) + "\n")
    for name in HOST_FILES:
        if (src_dir / name).exists():
            shutil.copy(src_dir / name, torq_dir / name)

    w = emb[np.array(kept)].T.copy()  # tied LM head, [hidden, V_trim]
    hidden, vocab = w.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["hidden_states", "lm_head.weight"], ["logits"], name="/lm_head/MatMul")],
        "main",
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 1, hidden])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab])],
        [numpy_helper.from_array(w, "lm_head.weight")],
    )
    lm_head = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    lm_head.ir_version = 10
    onnx.save(lm_head, onnx_dir / "lm_head.onnx")


def _export_transformer(step_model, onnx_path: Path) -> None:
    import torch
    from ._step import PackedStep

    packed = PackedStep(step_model).eval()
    inputs = (
        torch.zeros(1, 1, step_model.cfg.hidden_size),
        torch.tensor([[0]], dtype=torch.int64),
        *packed.pack(step_model.init_states()),
    )
    with torch.no_grad():
        torch.onnx.export(
            packed, inputs, str(onnx_path),
            input_names=PackedStep.INPUT_NAMES, output_names=PackedStep.OUTPUT_NAMES,
            opset_version=21, dynamo=False, do_constant_folding=True,
        )


def _compiler_options(compiler_path: str | None) -> set[str]:
    """Option names that the torq-compile binary knows (empty if it cannot run)."""
    import os

    compiler = str(compiler_path or os.environ.get("TORQ_COMPILER_PATH", "torq-compile"))
    try:
        result = subprocess.run([compiler, "--help-hidden"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(re.findall(r"^\s+(--[\w-]+)", result.stdout, flags=re.MULTILINE))


def export_qwen35_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    out_root = Path(args.models_dir) / args.hf_repo / "export"
    onnx_dir, torq_dir = out_root / "onnx", out_root / "torq"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    torq_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(snapshot_download(args.hf_repo, allow_patterns=["*.json", "*.jinja", "*.safetensors"]))
    tok, model = _load_hf(args.hf_repo)
    from ._step import StaticStep

    text_model = model.model.language_model
    step_model = StaticStep(text_model, text_model.config, args.context).eval()
    if not args.skip_validation:
        _validate(step_model, model, tok)

    logger.info("Exporting the static decode step to ONNX...")
    _export_transformer(step_model, onnx_dir / "transformer.onnx")
    _write_lm_head_and_host_files(model, tok, src_dir, onnx_dir, torq_dir)
    del model, text_model, step_model

    for name in ("transformer", "lm_head"):
        path = onnx_dir / f"{name}.onnx"
        m = _to_bf16(_fold_shapes_and_strip(onnx.load(path)))
        if args.weight_quantization == "int4":
            logger.info("%s: %d MatMul weights quantized to INT4", name, _quantize_int4(m))
        onnx.save(m, path)
        onnx.checker.check_model(str(path))
        logger.info("Wrote %s", path)

    if args.skip_torq:
        return
    flags_file = read_compile_flags_file(COMPILE_FLAGS_FILE)
    known = _compiler_options(args.compiler_path)
    for name in ("transformer", "lm_head"):
        flags = list(flags_file.get("flags") or []) + list(flags_file.get(name) or [])
        flags += [f for f in flags_file.get(f"{name}_if_supported") or [] if f.split("=", 1)[0] in known]
        flags += list(args.compile_flags or [])
        logger.info("Compiling %s with: %s", name, " ".join(flags))
        export_torq(
            onnx_dir / f"{name}.onnx", torq_dir, compiler_args=flags,
            local_compile=args.local_compile, use_binary=args.use_binary,
            compiler_path=args.compiler_path, opset=21,
        )
        (torq_dir / f"{name}.mlir").unlink(missing_ok=True)
        logger.info("Wrote %s", torq_dir / f"{name}.vmfb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3.5 to ONNX and Torq")
    add_qwen35_export_args(parser)
    export_qwen35_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
