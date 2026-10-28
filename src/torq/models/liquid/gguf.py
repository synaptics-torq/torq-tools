# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Build LFM2.5 Torq models from a llama.cpp GGUF checkpoint.

Some LFM2.5 checkpoints are published only as GGUF, e.g. the quantization-aware
distilled ``LFM2.5-350M-QAD-Q4_0.gguf`` in ``LiquidAI/LFM2.5-350M-GGUF``. Two
steps turn one into the exporter's inputs and outputs:

1. :func:`gguf_to_source_onnx` rebuilds the fp32 source ONNX: the upstream
   ``LiquidAI/LFM2.5-*-ONNX`` graph is the template, and every weight
   initializer is replaced by the matching GGUF tensor, dequantized. Q4_0
   dequantizes exactly (fp16 scale x 4-bit code fits fp32), so the regular
   static export of this graph computes with the GGUF's weights.
2. :func:`apply_gguf_q4_0` puts the GGUF's own 4-bit weights back into an
   exported graph: each MatMul whose weight is a Q4_0 tensor reads it through
   ``DequantizeLinear(INT4, axis=0, block_size=32, scale fp32, zero_point 0)``
   with the codes and scales unpacked from the GGUF blocks. Nothing is
   re-quantized; codes x scales must equal the fp32 weight bit for bit.

3. :func:`apply_gguf_q6_k_lm_head` does the same for a Q6_K tied embedding used
   as the lm_head: INT8 codes in [-32, 31] with one fp32 scale per 16 values
   (fp16 d x int8 sub-scale, exact), read through
   ``DequantizeLinear(INT8, axis=0, block_size=16)``.

Tensors of other GGUF types (F32 norms, conv kernels) stay in fp32 and are
converted to bf16 by the compiler like any other float weight.
"""

import logging
import os
import re
import shutil
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper

logger = logging.getLogger("Torq-tools")

Q4_0_BLOCK = 32
Q6_K_BLOCK, Q6_K_SUPER = 16, 256

# ONNX initializer suffix (per layer) -> GGUF tensor suffix
_LAYER_MAP = {
    "operator_layernorm.weight": "attn_norm.weight",
    "ffn_layernorm.weight": "ffn_norm.weight",
    "conv.in_proj.MatMul.weight": "shortconv.in_proj.weight",
    "conv.conv.weight": "shortconv.conv.weight",
    "conv.out_proj.MatMul.weight": "shortconv.out_proj.weight",
    "attn.q_proj.MatMul.weight": "attn_q.weight",
    "attn.k_proj.MatMul.weight": "attn_k.weight",
    "attn.v_proj.MatMul.weight": "attn_v.weight",
    "attn.o_proj.MatMul.weight": "attn_output.weight",
    "attn.q_norm.layernorm.weight": "attn_q_norm.weight",
    "attn.k_norm.layernorm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.MatMul.weight": "ffn_gate.weight",
    "mlp.up_proj.MatMul.weight": "ffn_up.weight",
    "mlp.down_proj.MatMul.weight": "ffn_down.weight",
}


def _gguf():
    try:
        import gguf
    except ImportError as e:
        raise ImportError("GGUF support needs the `gguf` package: pip install gguf") from e
    return gguf


def resolve_gguf(spec: str | os.PathLike) -> Path:
    """A local ``.gguf`` path, or ``<hf_repo>:<filename>`` downloaded from the Hub."""
    path = Path(spec)
    if path.exists():
        return path
    repo, sep, filename = str(spec).partition(":")
    if not sep or not filename.endswith(".gguf"):
        raise FileNotFoundError(f"GGUF '{spec}' not found (expected a path or '<hf_repo>:<file>.gguf')")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename))


def gguf_tensor_name(onnx_name: str, n_layers: int) -> str | None:
    """GGUF tensor name for an upstream LFM2.5 ONNX initializer, or None if it has none."""
    if onnx_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if onnx_name == f"model.layers.{n_layers}.final_norm_layernorm.weight":
        return "token_embd_norm.weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", onnx_name)
    if m and m.group(2) in _LAYER_MAP:
        return f"blk.{m.group(1)}.{_LAYER_MAP[m.group(2)]}"
    return None


def _read_gguf(gguf_path: Path):
    gguf = _gguf()
    reader = gguf.GGUFReader(str(gguf_path))
    arch = reader.fields["general.architecture"].contents()
    if arch != "lfm2":
        raise ValueError(f"{gguf_path.name}: architecture '{arch}', expected 'lfm2'")
    n_layers = int(reader.fields["lfm2.block_count"].contents())
    return reader, {t.name: t for t in reader.tensors}, n_layers


def gguf_to_source_onnx(
    gguf_path: str | os.PathLike,
    template_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
) -> Path:
    """Write ``out_dir/model.onnx`` (+ config/tokenizer) with the GGUF's weights in the template graph."""
    gguf = _gguf()
    gguf_path, template_dir, out_dir = Path(gguf_path), Path(template_dir), Path(out_dir)
    _, tensors, n_layers = _read_gguf(gguf_path)
    model = onnx.load(str(template_dir / "model.onnx"))
    used = set()
    for i, init in enumerate(model.graph.initializer):
        gname = gguf_tensor_name(init.name, n_layers)
        if gname is None:
            continue
        if gname not in tensors:
            raise KeyError(f"{init.name}: {gguf_path.name} has no tensor '{gname}'")
        t = tensors[gname]
        base = numpy_helper.to_array(init)
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        if ".MatMul.weight" in init.name:            # ONNX MatMul B is [in, out]
            w = w.T
        w = np.ascontiguousarray(w).reshape(base.shape)
        # The GGUF is a fine-tune / quantization of the template's weights, so a
        # wrong mapping or layout shows up as a low cosine.
        cos = float(w.ravel() @ base.ravel() / (np.linalg.norm(w) * np.linalg.norm(base) + 1e-30))
        if cos < 0.5:
            raise ValueError(f"{init.name} <- {gname}: cosine {cos:.3f} to the template weight")
        model.graph.initializer[i].CopyFrom(numpy_helper.from_array(w, init.name))
        used.add(gname)
    if unused := sorted(set(tensors) - used):
        raise ValueError(f"{gguf_path.name}: tensors not mapped into the graph: {unused}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("model.onnx*"):
        f.unlink()
    onnx.save(model, str(out_dir / "model.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (template_dir / name).exists():
            shutil.copy(template_dir / name, out_dir / name)
    logger.info("(gguf) Wrote '%s' with %d tensors from '%s'", out_dir / "model.onnx", len(used), gguf_path.name)
    return out_dir / "model.onnx"


def unpack_q4_0(tensor) -> tuple[np.ndarray, np.ndarray]:
    """GGUF Q4_0 tensor [out, in] -> (codes int8 [in, out] in [-8, 7], scales fp32 [in/32, out]).

    ggml ``block_q4_0``: fp16 ``d``, then 16 bytes holding codes ``j`` (low
    nibble) and ``j + 16`` (high nibble); value = (nibble - 8) * d.
    """
    n_in, n_out = (int(x) for x in tensor.shape)       # ggml ne order: ne0 = in features
    blocks = np.asarray(tensor.data, np.uint8).reshape(n_out, n_in // Q4_0_BLOCK, 2 + Q4_0_BLOCK // 2)
    d = blocks[..., :2].copy().view(np.float16)[..., 0]
    qs = blocks[..., 2:]
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).astype(np.int8) - 8
    return codes.reshape(n_out, n_in).T.copy(), d.T.astype(np.float32)


def apply_gguf_q4_0(
    gguf_path: str | os.PathLike,
    src: str | os.PathLike,
    dst: str | os.PathLike,
) -> int:
    """Copy ``src`` to ``dst`` with every Q4_0 MatMul weight read through an INT4 DequantizeLinear.

    Returns the number of weights carried as Q4_0.
    """
    gguf = _gguf()
    gguf_path, src, dst = Path(gguf_path), Path(src), Path(dst)
    _, tensors, n_layers = _read_gguf(gguf_path)
    model = onnx.load(str(src))
    graph = model.graph
    if not any(o.domain in ("", "ai.onnx") and o.version >= 21 for o in model.opset_import):
        raise ValueError(f"{src.name}: INT4 blocked DequantizeLinear needs opset >= 21")
    inits = {t.name: t for t in graph.initializer}
    nodes, new_inits, replaced = [], [], set()
    for node in graph.node:
        name = node.input[1] if node.op_type == "MatMul" and len(node.input) > 1 else None
        t = tensors.get(gguf_tensor_name(name, n_layers)) if name in inits else None
        if t is not None and t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            dq = f"{name}/dequantized"
            if name not in replaced:
                codes, scale = unpack_q4_0(t)
                deq = (codes.reshape(-1, Q4_0_BLOCK, codes.shape[1]) * scale[:, None, :]).reshape(codes.shape)
                if not np.array_equal(deq.astype(np.float32), numpy_helper.to_array(inits[name])):
                    raise ValueError(f"{src.name}: '{name}' does not match the GGUF Q4_0 tensor '{t.name}'")
                qn, sn, zn = f"{name}/int4", f"{name}/scale", f"{name}/zero_point"
                new_inits += [
                    numpy_helper.from_array(codes.astype(ml_dtypes.int4), qn),
                    numpy_helper.from_array(scale, sn),
                    numpy_helper.from_array(np.zeros_like(scale, dtype=ml_dtypes.int4), zn),
                ]
                nodes.append(helper.make_node("DequantizeLinear", [qn, sn, zn], [dq],
                                              name=f"{name}/DequantizeLinear",
                                              axis=0, block_size=Q4_0_BLOCK))
                replaced.add(name)
            node.input[1] = dq
        nodes.append(node)
    keep = [t for t in graph.initializer if t.name not in replaced]
    del graph.initializer[:]
    graph.initializer.extend(keep + new_inits)
    del graph.node[:]
    graph.node.extend(nodes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for f in dst.parent.glob(dst.name + "*"):
        f.unlink()
    onnx.save(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.name + ".data", size_threshold=1024)
    logger.info("(gguf) '%s': %d MatMul weights carried as GGUF Q4_0 (INT4 DequantizeLinear)",
                dst.name, len(replaced))
    return len(replaced)


def unpack_q6_k(tensor) -> tuple[np.ndarray, np.ndarray]:
    """GGUF Q6_K tensor [rows, n] -> (codes int8 [rows, n] in [-32, 31], scales fp32 [rows, n/16]).

    ggml ``block_q6_K`` (210 B per 256 values): ql[128] low 4 bits, qh[64] high
    2 bits, scales[16] int8, fp16 ``d``; value = d * scales[j] * (q - 32).
    """
    n, rows = (int(x) for x in tensor.shape)              # ggml ne order: ne0 = n
    nb = n // Q6_K_SUPER
    b = np.asarray(tensor.data, np.uint8).reshape(rows, nb, 210)
    ql, qh = b[..., :128].astype(np.int16), b[..., 128:192].astype(np.int16)
    sc = b[..., 192:208].copy().view(np.int8).astype(np.float32)
    d = b[..., 208:210].copy().view(np.float16).astype(np.float32)
    q = np.empty((rows, nb, Q6_K_SUPER), np.int16)
    for half in range(2):                                  # two 128-value halves per block
        lo, hi, o = ql[..., 64 * half:64 * half + 64], qh[..., 32 * half:32 * half + 32], 128 * half
        q[..., o + 0:o + 32] = (lo[..., :32] & 0xF) | (((hi >> 0) & 3) << 4)
        q[..., o + 32:o + 64] = (lo[..., 32:] & 0xF) | (((hi >> 2) & 3) << 4)
        q[..., o + 64:o + 96] = (lo[..., :32] >> 4) | (((hi >> 4) & 3) << 4)
        q[..., o + 96:o + 128] = (lo[..., 32:] >> 4) | (((hi >> 6) & 3) << 4)
    codes = (q - 32).astype(np.int8).reshape(rows, n)
    return codes, (d * sc).reshape(rows, n // Q6_K_BLOCK)  # fp16 x int8 is exact in fp32


def apply_gguf_q6_k_lm_head(
    gguf_path: str | os.PathLike,
    src: str | os.PathLike,
    dst: str | os.PathLike,
) -> bool:
    """Copy lm_head.onnx ``src`` to ``dst`` with a Q6_K token_embd weight carried as INT8 DequantizeLinear.

    Returns False (and leaves ``dst`` unwritten) when the GGUF embedding is not Q6_K.
    """
    gguf = _gguf()
    gguf_path, src, dst = Path(gguf_path), Path(src), Path(dst)
    _, tensors, _ = _read_gguf(gguf_path)
    t = tensors.get("token_embd.weight")
    if t is None or t.tensor_type != gguf.GGMLQuantizationType.Q6_K:
        return False
    codes, scales = unpack_q6_k(t)                         # [vocab, hidden], [vocab, hidden/16]
    model = onnx.load(str(src))
    graph = model.graph
    matmuls = [n for n in graph.node if n.op_type == "MatMul"]
    inits = {i.name: i for i in graph.initializer}
    if len(matmuls) != 1 or matmuls[0].input[1] not in inits:
        raise ValueError(f"{src.name}: expected a single MatMul against a constant lm_head weight")
    mm = matmuls[0]
    name = mm.input[1]
    deq = (codes.reshape(codes.shape[0], -1, Q6_K_BLOCK) * scales[..., None]).reshape(codes.shape)
    if not np.array_equal(deq.T.astype(np.float32), numpy_helper.to_array(inits[name])):
        raise ValueError(f"{src.name}: '{name}' does not match the GGUF Q6_K token_embd")
    qn, sn, zn = f"{name}/int8", f"{name}/scale", f"{name}/zero_point"
    keep = [i for i in graph.initializer if i.name != name]
    del graph.initializer[:]
    graph.initializer.extend(keep + [
        numpy_helper.from_array(np.ascontiguousarray(codes.T), qn),       # [hidden, vocab]
        numpy_helper.from_array(np.ascontiguousarray(scales.T), sn),      # [hidden/16, vocab]
        numpy_helper.from_array(np.zeros(scales.T.shape, np.int8), zn),
    ])
    graph.node.insert(0, helper.make_node("DequantizeLinear", [qn, sn, zn], [f"{name}/dequantized"],
                                          name=f"{name}/DequantizeLinear", axis=0, block_size=Q6_K_BLOCK))
    mm.input[1] = f"{name}/dequantized"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for f in dst.parent.glob(dst.name + "*"):
        f.unlink()
    onnx.save(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.name + ".data", size_threshold=1024)
    logger.info("(gguf) '%s': lm_head carried as GGUF Q6_K (INT8 DequantizeLinear, block %d)",
                dst.name, Q6_K_BLOCK)
    return True
