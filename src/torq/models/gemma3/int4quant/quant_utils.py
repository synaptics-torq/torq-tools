"""
Shared quantization utilities for ONNX MatMul weight-only quantization.

Contains:
  - BF16 <-> FP32 helpers
  - Quantization math (per-channel, per-tensor, per-block, Q4_0, Q4_K)
  - ONNX graph building helpers (DequantizeLinear subgraph, bf16->fp32 conversion)
  - Accuracy testing utilities
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto, shape_inference


# ═══════════════════════════════════════════════════════════════════════════════
# BF16 helpers
# ═══════════════════════════════════════════════════════════════════════════════

def bf16_raw_to_fp32(raw: bytes, dims: list) -> np.ndarray:
    """Convert raw bf16 bytes to fp32 numpy array."""
    n_elems = 1
    for d in dims:
        n_elems *= d
    if len(raw) == n_elems * 2:
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
        fp32 = np.zeros(dims, dtype=np.float32)
        fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
        return fp32
    else:
        return np.frombuffer(raw, dtype=np.float32).reshape(dims)


def fp32_to_bf16_proto(name: str, fp32_array: np.ndarray, dims: list) -> onnx.TensorProto:
    """Create an ONNX TensorProto with bf16 dtype from fp32 numpy array."""
    f32 = fp32_array.astype(np.float32)
    bf16_u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    proto = onnx.TensorProto()
    proto.name = name
    proto.data_type = TensorProto.BFLOAT16
    proto.dims.extend(dims)
    proto.raw_data = bf16_u16.tobytes()
    return proto


def read_weight(wt_proto, dims: list) -> np.ndarray:
    """Read an ONNX weight tensor as fp32, handling bf16 transparently."""
    if wt_proto.data_type == TensorProto.BFLOAT16:
        return bf16_raw_to_fp32(wt_proto.raw_data, dims)
    return numpy_helper.to_array(wt_proto)


def scale_to_bf16_fp32(scale: np.ndarray) -> np.ndarray:
    """Round scale values to bf16 precision but store as fp32 dtype.

    The compiler stack can't lower bf16 scale in DequantizeLinear, so we keep
    fp32 dtype while ensuring the actual values are bf16-representable.
    """
    scale_f32 = scale.astype(np.float32)
    bf16_u16 = (scale_f32.view(np.uint32) >> 16).astype(np.uint16)
    result = np.zeros_like(scale_f32)
    result.view(np.uint32)[:] = bf16_u16.astype(np.uint32) << 16
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Quantization math
# ═══════════════════════════════════════════════════════════════════════════════

def qrange(bits: int) -> tuple:
    """Return (qmin, qmax, max_positive) for symmetric quantization."""
    if bits == 4:
        return -7, 7, 7
    else:
        return -127, 127, 127


def quant_per_channel(W: np.ndarray, bits: int = 8):
    """
    Symmetric per-output-channel quantization.
    W shape: [K, N]  -> scale shape: [N]
    Returns (W_q, scale, zero_point)
    """
    qmin, qmax, qp = qrange(bits)
    W = W.astype(np.float32)
    scale = np.max(np.abs(W), axis=0) / float(qp)
    scale = np.where(scale == 0, 1e-8, scale)
    W_q = np.clip(np.round(W / scale[np.newaxis, :]), qmin, qmax).astype(np.int8)
    zp = np.zeros(scale.shape, dtype=np.int8)
    return W_q, scale.astype(np.float32), zp


def quant_per_tensor(W: np.ndarray, bits: int = 8):
    """
    Symmetric per-tensor quantization.
    Returns (W_q, scale scalar, zero_point scalar)
    """
    qmin, qmax, qp = qrange(bits)
    W = W.astype(np.float32)
    scale = float(np.max(np.abs(W))) / float(qp)
    scale = max(scale, 1e-8)
    W_q = np.clip(np.round(W / scale), qmin, qmax).astype(np.int8)
    return W_q, np.array(scale, dtype=np.float32), np.array(0, dtype=np.int8)


def quant_per_block(W: np.ndarray, block_size: int = 32, bits: int = 8):
    """
    Per-block symmetric quantization (Q8_0 style).
    Falls back to per-channel if W cannot be evenly blocked.
    Returns (W_q, scale, zp, block_meta) or (W_q, scale, zp) on fallback.
    """
    qmin, qmax, qp = qrange(bits)
    W_fp = W.astype(np.float32)
    orig_shape = W_fp.shape
    flat = W_fp.flatten()
    n = len(flat)
    if n % block_size != 0:
        print(f"    Warning: {n} elements not divisible by block_size={block_size}; "
              f"falling back to per-channel.")
        return quant_per_channel(W, bits)
    num_blocks = n // block_size
    blocks = flat.reshape(num_blocks, block_size)
    scale = np.max(np.abs(blocks), axis=1) / float(qp)
    scale = np.where(scale == 0, 1e-8, scale)
    W_q = np.clip(
        np.round(blocks / scale[:, np.newaxis]), qmin, qmax
    ).astype(np.int8).reshape(orig_shape)
    W_block = W_q.flatten().reshape(num_blocks, block_size)
    zp = np.zeros(scale.shape, dtype=np.int8)
    return W_block, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


def quant_q4_0(W: np.ndarray, block_size: int = 32):
    """
    Q4_0: per-block 4-bit symmetric quantization.
      scale = max(|block|) / 7, range [-8, 7], zp = 0

    Returns (W_q [num_blocks, block_size], scale [num_blocks], zp [num_blocks],
             block_meta (num_blocks, block_size, orig_shape))
    """
    W_fp = W.astype(np.float32)
    orig_shape = W_fp.shape
    flat = W_fp.flatten()
    n = len(flat)

    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        n = len(flat)

    num_blocks = n // block_size
    blocks = flat.reshape(num_blocks, block_size)

    amax = np.max(np.abs(blocks), axis=1)
    scale = amax / 7.0
    scale = np.where(scale == 0, 1e-8, scale)

    W_q = np.clip(np.round(blocks / scale[:, np.newaxis]), -8, 7).astype(np.int8)
    zp = np.zeros(num_blocks, dtype=np.int8)

    return W_q, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


def quant_q4_k(W: np.ndarray, block_size: int = 32):
    """
    Q4_K: per-block 4-bit asymmetric quantization.
      scale = (max - min) / 15, range [0, 15], zp per block

    Returns (W_q [num_blocks, block_size] uint8, scale [num_blocks] fp32,
             zp [num_blocks] uint8, block_meta)
    """
    W_fp = W.astype(np.float32)
    orig_shape = W_fp.shape
    flat = W_fp.flatten()
    n = len(flat)

    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        n = len(flat)

    num_blocks = n // block_size
    blocks = flat.reshape(num_blocks, block_size)

    block_min = np.min(blocks, axis=1)
    block_max = np.max(blocks, axis=1)

    scale = (block_max - block_min) / 15.0
    scale = np.where(scale == 0, 1e-8, scale)

    zp = np.clip(np.round(-block_min / scale), 0, 15).astype(np.uint8)

    W_q = np.clip(
        np.round(blocks / scale[:, np.newaxis] + zp[:, np.newaxis].astype(np.float32)),
        0, 15
    ).astype(np.uint8)

    return W_q, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX graph helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_unique_name(base: str, existing: set) -> str:
    """Generate a unique name by appending _N if base already exists."""
    name, i = base, 0
    while name in existing:
        name = f"{base}_{i}"
        i += 1
    existing.add(name)
    return name


def activation_elem_type(node, vi_map: dict) -> int:
    """Return the ONNX elem_type of the activation input (input[0]) of a MatMul node."""
    inp = node.input[0]
    if inp in vi_map:
        return vi_map[inp].type.tensor_type.elem_type
    return TensorProto.FLOAT


def build_dequant_subgraph(W_q, scale, zp, block_meta, dq_axis, act_type,
                           scale_dtype="bf16", base_prefix="weight"):
    """Build DequantizeLinear -> (Reshape) -> (Cast) subgraph nodes and initializers.

    Returns (nodes, initializers, final_weight_name) where final_weight_name
    is the tensor name to feed into the MatMul weight input.
    """
    nodes = []
    inits = []

    int8_init = numpy_helper.from_array(W_q, name=f"{base_prefix}_int8")
    inits.append(int8_init)

    # Scale: round to bf16 precision, store as fp32
    if scale_dtype == "bf16":
        scale_arr = scale_to_bf16_fp32(scale)
    elif scale_dtype == "fp16":
        scale_arr = scale.astype(np.float16)
    else:
        scale_arr = scale.astype(np.float32)
    scale_init = numpy_helper.from_array(scale_arr, name=f"{base_prefix}_scale")
    inits.append(scale_init)

    zp_init = numpy_helper.from_array(zp, name=f"{base_prefix}_zp")
    inits.append(zp_init)

    # DequantizeLinear
    dq_kwargs = {}
    if dq_axis is not None:
        dq_kwargs["axis"] = dq_axis
    dq_out = f"{base_prefix}_dq"
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[f"{base_prefix}_int8", f"{base_prefix}_scale", f"{base_prefix}_zp"],
        outputs=[dq_out], name="DQL", **dq_kwargs,
    )
    nodes.append(dq_node)

    mm_weight = dq_out

    # Reshape for per-block modes
    if block_meta is not None:
        num_blocks, bs, orig_shape = block_meta
        target_shape = np.array(list(orig_shape), dtype=np.int64)
        shape_init = numpy_helper.from_array(target_shape, name="target_shape")
        inits.append(shape_init)
        reshape_node = helper.make_node(
            "Reshape", inputs=[dq_out, "target_shape"],
            outputs=[f"{base_prefix}_reshaped"], name="Reshape_dq",
        )
        nodes.append(reshape_node)
        mm_weight = f"{base_prefix}_reshaped"

    # Cast to activation dtype if needed
    if act_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
        cast_node = helper.make_node(
            "Cast", inputs=[mm_weight], outputs=[f"{base_prefix}_cast"],
            name="Cast_weight", to=act_type,
        )
        nodes.append(cast_node)
        mm_weight = f"{base_prefix}_cast"

    return nodes, inits, mm_weight


def quantize_weight(W_raw, strategy, bits, block_size):
    """Dispatch to the appropriate quantization function.

    Returns (W_q, scale, zp, dq_axis, block_meta).
    block_meta is None for non-block strategies.
    """
    block_meta = None
    if strategy == "q4_k":
        W_q, scale, zp, block_meta = quant_q4_k(W_raw, block_size)
        dq_axis = 0
    elif strategy == "q4_0":
        W_q, scale, zp, block_meta = quant_q4_0(W_raw, block_size)
        dq_axis = 0
    elif strategy == "per_channel":
        W_q, scale, zp = quant_per_channel(W_raw, bits)
        dq_axis = 1
    elif strategy == "per_tensor":
        W_q, scale, zp = quant_per_tensor(W_raw, bits)
        dq_axis = None
    else:  # per_block
        result = quant_per_block(W_raw, block_size, bits)
        if len(result) == 4:
            W_q, scale, zp, block_meta = result
            dq_axis = 0
        else:
            W_q, scale, zp = result
            dq_axis = 1
    return W_q, scale, zp, dq_axis, block_meta


def compression_bytes(W_raw, W_q, scale, strategy, bits, block_size, scale_dtype="bf16"):
    """Compute original and quantized byte sizes for compression ratio."""
    scale_elem_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[scale_dtype]
    orig_bytes = W_raw.size * 4  # fp32 equivalent
    if strategy in ("q4_0", "q4_k"):
        n_blocks = W_q.shape[0]
        zp_bytes = 1 if strategy == "q4_k" else 0
        quant_bytes = n_blocks * (block_size // 2 + scale_elem_bytes + zp_bytes)
    else:
        wt_bytes = W_q.size if bits == 8 else W_q.size // 2
        quant_bytes = wt_bytes + scale.size * scale_elem_bytes
    return orig_bytes, quant_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# Model conversion for onnxruntime (bf16 -> fp32)
# ═══════════════════════════════════════════════════════════════════════════════

def convert_model_to_fp32(model_or_path) -> str:
    """Convert bf16 model to fp32 for onnxruntime CPU. Returns path to use.

    model_or_path: either a file path string or an onnx.ModelProto.
    If no bf16 tensors, returns the original path unchanged.
    """
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
        orig_path = model_or_path
    else:
        model = model_or_path
        orig_path = None

    has_bf16 = False
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            has_bf16 = True
    for init in model.graph.initializer:
        if init.data_type == TensorProto.BFLOAT16:
            has_bf16 = True
    if not has_bf16:
        return orig_path if orig_path else None

    new_inits = []
    for init in model.graph.initializer:
        if init.data_type == TensorProto.BFLOAT16:
            fp32 = bf16_raw_to_fp32(init.raw_data, list(init.dims))
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

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(new_model, tmp.name)
    tmp.close()
    return tmp.name


def test_accuracy(orig_path: str, quant_path: str, n_samples: int, seed: int = 42):
    """Run both models through onnxruntime and return error metrics dict."""
    orig_rt = convert_model_to_fp32(orig_path)
    quant_rt = convert_model_to_fp32(quant_path)

    model = onnx.load(orig_rt)
    init_names = {i.name for i in model.graph.initializer}
    rng = np.random.default_rng(seed)

    feeds = []
    for _ in range(n_samples):
        feed = {}
        for inp in model.graph.input:
            if inp.name in init_names:
                continue
            t = inp.type.tensor_type
            dims = []
            for d in t.shape.dim:
                if d.HasField("dim_value") and d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(1)
            feed[inp.name] = rng.standard_normal(dims).astype(np.float32)
        feeds.append(feed)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    sess_orig = ort.InferenceSession(orig_rt, opts, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(quant_rt, opts, providers=["CPUExecutionProvider"])

    all_orig = []
    all_quant = []
    for feed in feeds:
        all_orig.append(sess_orig.run(None, feed)[0].astype(np.float64).flatten())
        all_quant.append(sess_quant.run(None, feed)[0].astype(np.float64).flatten())

    orig_cat = np.concatenate(all_orig)
    quant_cat = np.concatenate(all_quant)
    diff = np.abs(orig_cat - quant_cat)

    mean_abs = float(np.mean(diff))
    max_abs = float(np.max(diff))
    denom = np.maximum(np.abs(orig_cat), 1e-12)
    mean_rel = float(np.mean(diff / denom))
    max_rel = float(np.max(diff / denom))

    norm_o = np.linalg.norm(orig_cat)
    norm_q = np.linalg.norm(quant_cat)
    cos_sim = float(np.dot(orig_cat, quant_cat) / (norm_o * norm_q)) if norm_o > 0 and norm_q > 0 else 1.0

    sig_pow = np.mean(orig_cat ** 2)
    nse_pow = np.mean((orig_cat - quant_cat) ** 2)
    snr_db = float(10 * np.log10(sig_pow / max(nse_pow, 1e-30)))

    # Cleanup temp files
    for p in (orig_rt, quant_rt):
        if p != orig_path and p != quant_path:
            os.unlink(p)

    return {
        "mean_abs": mean_abs, "max_abs": max_abs,
        "mean_rel": mean_rel, "max_rel": max_rel,
        "cos_sim": cos_sim, "snr_db": snr_db,
    }
