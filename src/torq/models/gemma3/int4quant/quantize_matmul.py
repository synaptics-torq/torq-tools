"""
Weight-Only Quantization for ONNX MatMul layers (INT8 or INT4)
================================================================

Supports --bits 8 (W8A16) and --bits 4 (W4A16):

  INT8: W_q = clip(round(W / scale), -127, 127)     scale = max(|W|) / 127
  INT4: W_q = clip(round(W / scale),   -7,   7)     scale = max(|W|) /   7

  At runtime: DequantizeLinear(W_q, scale) -> W_float -> MatMul(activation, W)

INT4 notes:
  - Stored in int8 container (values restricted to [-7, 7]) for ONNX compat
  - Actual 4-bit packing happens at deployment (NPU runtime / MLIR lowering)
  - 8x compression vs FP32, 4x vs FP16, 2x vs INT8

Usage:
  python quantize_matmul.py model.onnx --bits 8              # default W8A16
  python quantize_matmul.py model.onnx --bits 4              # W4A16
  python quantize_matmul.py model.onnx --bits 4 --out m4.onnx
  python quantize_matmul.py model.onnx --granularity per_block --block-size 32
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto, shape_inference

from quant_utils import (
    read_weight, qrange, quantize_weight, compression_bytes,
    make_unique_name, activation_elem_type, scale_to_bf16_fp32,
)


# ── main quantization pass ────────────────────────────────────────────────────

def quantize_model(model_path: str, out_path: str, granularity: str, block_size: int,
                   scale_dtype: str = "bf16", bits: int = 8):
    print(f"Loading: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Shape inference so we know activation dtypes
    try:
        model = shape_inference.infer_shapes(model)
        graph = model.graph
    except Exception as e:
        print(f"  Warning: shape inference failed ({e})")

    vi_map = {vi.name: vi for vi in graph.value_info}
    for x in list(graph.input) + list(graph.output):
        vi_map[x.name] = x

    init_map = {init.name: init for init in graph.initializer}

    # Collect names that are already used (to generate unique names)
    used_names: set = {n.name for n in graph.node if n.name}
    used_names.update(init_map.keys())
    used_names.update(vi_map.keys())

    # Accumulators for new nodes / initializers / removed initializers
    new_initializers: list = []
    nodes_to_replace: dict = {}   # node index -> list of replacement nodes
    removed_inits: set = set()

    stats = defaultdict(int)

    if granularity in ("q4_0", "q4_k"):
        print(f"\nQuantizing MatMul weights  ({granularity.upper()}, block_size={block_size}"
              f", scale_dtype={scale_dtype})\n")
    else:
        bits_label = f"W{bits}A16"
        print(f"\nQuantizing MatMul weights  ({bits_label}, granularity={granularity}"
              + (f", block_size={block_size}" if granularity == "per_block" else "")
              + f", scale_dtype={scale_dtype})\n")

    for idx, node in enumerate(graph.node):
        if node.op_type not in ("MatMul", "Gemm"):
            continue

        wt_name = node.input[1]
        if wt_name not in init_map:
            stats["skipped_no_const"] += 1
            continue

        wt_proto = init_map[wt_name]
        dims = list(wt_proto.dims)
        W_raw = read_weight(wt_proto, dims)
        orig_dtype = W_raw.dtype

        # Must be 2-D for standard MatMul weight
        if W_raw.ndim != 2:
            print(f"  [SKIP] {node.name or wt_name}: weight ndim={W_raw.ndim} (need 2D)")
            stats["skipped_non_2d"] += 1
            continue

        K, N = W_raw.shape
        act_type = activation_elem_type(node, vi_map)
        act_type_name = TensorProto.DataType.Name(act_type)

        print(f"  MatMul  weight={wt_name}  shape=[{K},{N}]  "
              f"orig_dtype={orig_dtype}  act={act_type_name}")

        # ── quantize ─────────────────────────────────────────────────────────
        W_q, scale, zp, dq_axis, block_meta = quantize_weight(
            W_raw, granularity, bits, block_size)

        if granularity in ("q4_0", "q4_k"):
            bits = 4

        # Compression stats
        orig_bytes, quant_bytes = compression_bytes(
            W_raw, W_q, scale, granularity, bits, block_size, scale_dtype)
        ratio = orig_bytes / quant_bytes
        if granularity == "q4_0":
            range_str = "[-8,7] symmetric"
        elif granularity == "q4_k":
            range_str = "[0,15] asymmetric"
        else:
            qmin, qmax, _ = qrange(bits)
            range_str = f"[{qmin},{qmax}]"
        print(f"    -> INT{bits}  range={range_str}  scale_shape={list(scale.shape)}  "
              f"scale_dtype={scale_dtype}  "
              f"size {orig_bytes/1024:.1f}KB -> {quant_bytes/1024:.1f}KB  "
              f"compression={ratio:.1f}x")

        # ── build new initializer names ───────────────────────────────────────
        base = wt_name.replace("/", "_").replace(".", "_")
        wt_int8_name  = make_unique_name(f"{base}_int8",  used_names)
        scale_name    = make_unique_name(f"{base}_scale", used_names)
        zp_name       = make_unique_name(f"{base}_zp",    used_names)
        dq_out_name   = make_unique_name(f"{base}_dq",    used_names)

        # ── add new initializers ──────────────────────────────────────────────
        new_initializers.append(numpy_helper.from_array(W_q, name=wt_int8_name))

        # Store scale in requested dtype
        if scale_dtype == "bf16":
            new_initializers.append(numpy_helper.from_array(
                scale_to_bf16_fp32(scale), name=scale_name))
        elif scale_dtype == "fp16":
            new_initializers.append(numpy_helper.from_array(
                scale.astype(np.float16), name=scale_name))
        else:  # fp32
            new_initializers.append(numpy_helper.from_array(scale, name=scale_name))

        new_initializers.append(numpy_helper.from_array(zp, name=zp_name))
        removed_inits.add(wt_name)

        # ── DequantizeLinear node ─────────────────────────────────────────────
        dq_kwargs = {}
        if dq_axis is not None:
            dq_kwargs["axis"] = dq_axis
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[wt_int8_name, scale_name, zp_name],
            outputs=[dq_out_name],
            name=make_unique_name(f"DQL_{base}", used_names),
            **dq_kwargs,
        )

        # ── optional Reshape back to [K, N] for per-block mode ────────────────
        reshape_nodes = []
        matmul_weight_input = dq_out_name
        if block_meta is not None:
            num_blocks, bs, orig_shape = block_meta
            target_shape_name = make_unique_name(f"{base}_target_shape", used_names)
            reshape_out_name  = make_unique_name(f"{base}_reshaped",      used_names)
            new_initializers.append(numpy_helper.from_array(
                np.array(list(orig_shape), dtype=np.int64), name=target_shape_name))
            reshape_nodes.append(helper.make_node(
                "Reshape",
                inputs=[dq_out_name, target_shape_name],
                outputs=[reshape_out_name],
                name=make_unique_name(f"Reshape_{base}", used_names),
            ))
            matmul_weight_input = reshape_out_name

        # ── Cast dequantized weight to activation dtype if needed ─────────────
        cast_nodes = []
        mm_weight_input = matmul_weight_input
        if act_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
            cast_out_name = make_unique_name(f"{base}_cast", used_names)
            cast_nodes.append(helper.make_node(
                "Cast",
                inputs=[matmul_weight_input],
                outputs=[cast_out_name],
                name=make_unique_name(f"Cast_{base}", used_names),
                to=act_type,
            ))
            mm_weight_input = cast_out_name

        # ── rebuild the MatMul/Gemm node with new weight input ────────────────
        new_inputs = list(node.input)
        new_inputs[1] = mm_weight_input
        new_matmul = helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name or make_unique_name(f"MM_{base}", used_names),
        )
        for attr in node.attribute:
            new_matmul.attribute.append(attr)

        nodes_to_replace[idx] = [dq_node] + reshape_nodes + cast_nodes + [new_matmul]
        stats["quantized"] += 1

    # ── rebuild graph ─────────────────────────────────────────────────────────
    new_nodes = []
    for idx, node in enumerate(graph.node):
        if idx in nodes_to_replace:
            new_nodes.extend(nodes_to_replace[idx])
        else:
            new_nodes.append(node)

    kept_inits = [init for init in graph.initializer if init.name not in removed_inits]
    kept_inits.extend(new_initializers)

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=kept_inits,
    )
    new_graph.doc_string = graph.doc_string

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    seen = set()
    del new_model.opset_import[:]
    for op in model.opset_import:
        key = (op.domain, op.version)
        if key not in seen:
            seen.add(key)
            new_model.opset_import.append(op)

    new_model.producer_name = "torq-w8a16-quant"
    new_model.doc_string = model.doc_string

    try:
        new_model = shape_inference.infer_shapes(new_model)
    except Exception as e:
        print(f"\n  Warning: post-quantization shape inference failed: {e}")

    onnx.checker.check_model(new_model)
    onnx.save(new_model, out_path)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  MatMul nodes quantized  : {stats['quantized']}")
    print(f"  Skipped (no const wt)   : {stats['skipped_no_const']}")
    print(f"  Skipped (non-2D wt)     : {stats['skipped_non_2d']}")
    if granularity in ("q4_0", "q4_k"):
        print(f"  Strategy                : {granularity.upper()} per-block (block_size={block_size})")
        sym = "symmetric" if granularity == "q4_0" else "asymmetric"
        print(f"  Bit width               : 4-bit  {sym}")
    else:
        print(f"  Granularity             : {granularity}"
              + (f"  (block_size={block_size})" if granularity == "per_block" else ""))
        print(f"  Bit width               : {bits}-bit  (W{bits}A16)")
    print(f"  Scheme                  : symmetric, zero_point=0")
    print(f"  Scale dtype             : {scale_dtype}")
    print(f"  Activation path         : unchanged (original float dtype)")
    print(f"  Output model            : {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="quantization for ONNX MatMul layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", nargs="?",
                        default="/home/meepat/meet/torq-compiler-dev-new/model.onnx",
                        help="Input .onnx model (default: model.onnx)")
    parser.add_argument("--out", default=None,
                        help="Output .onnx file (default: <model>_w8a16.onnx)")
    parser.add_argument("--granularity", choices=["per_channel", "per_tensor", "per_block", "q4_0", "q4_k"],
                        default="per_channel",
                        help="Quantization granularity (default: per_channel). "
                             "q4_0 = symmetric 4-bit, q4_k = asymmetric 4-bit (K-quant)")
    parser.add_argument("--block-size", type=int, default=32,
                        help="Block size for per_block mode, llama.cpp Q8_0 default=32")
    parser.add_argument("--scale-dtype", choices=["fp32", "fp16", "bf16"],
                        default="bf16",
                        help="Dtype for scale tensor (default: bf16)")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8,
                        help="Quantization bit width: 4 or 8 (default: 8)")
    args = parser.parse_args()

    if args.granularity in ("q4_0", "q4_k"):
        suffix = f"_{args.granularity}.onnx"
    else:
        suffix = f"_w{args.bits}a16.onnx"
    out_path = args.out or str(Path(args.model).stem) + suffix
    quantize_model(args.model, out_path, args.granularity, args.block_size,
                   args.scale_dtype, args.bits)


if __name__ == "__main__":
    main()


    
# """
# Weight-Only Quantization for ONNX MatMul layers (INT8 or INT4)
# ================================================================

# Supports --bits 8 (W8A16) and --bits 4 (W4A16):

#   INT8: W_q = clip(round(W / scale), -127, 127)     scale = max(|W|) / 127
#   INT4: W_q = clip(round(W / scale),   -7,   7)     scale = max(|W|) /   7

#   At runtime: DequantizeLinear(W_q, scale) -> W_float -> MatMul(activation, W)

# INT4 notes:
#   - Stored in int8 container (values restricted to [-7, 7]) for ONNX compat
#   - Actual 4-bit packing happens at deployment (NPU runtime / MLIR lowering)
#   - 8x compression vs FP32, 4x vs FP16, 2x vs INT8

# Usage:
#   python quantize_matmul_w8a16.py model.onnx --bits 8              # default W8A16
#   python quantize_matmul_w8a16.py model.onnx --bits 4              # W4A16
#   python quantize_matmul_w8a16.py model.onnx --bits 4 --out m4.onnx
#   python quantize_matmul_w8a16.py model.onnx --granularity per_block --block-size 32
# """

# import argparse
# import sys
# from pathlib import Path
# from collections import defaultdict

# import numpy as np
# import onnx
# from onnx import numpy_helper, helper, TensorProto, shape_inference


# # ── quantization helpers ──────────────────────────────────────────────────────

# def _qrange(bits: int) -> tuple:
#     """Return (qmin, qmax, max_positive) for symmetric quantization."""
#     if bits == 4:
#         return -7, 7, 7       # symmetric signed 4-bit
#     else:
#         return -127, 127, 127  # symmetric signed 8-bit


# def quant_per_channel(W: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Symmetric per-output-channel quantization (works for both 4-bit and 8-bit).
#     W shape: [K, N]  -> scale shape: [N]
#     Returns (W_q, scale, zero_point)
#     """
#     qmin, qmax, qp = _qrange(bits)
#     W = W.astype(np.float32)
#     scale = np.max(np.abs(W), axis=0) / float(qp)     # [N]
#     scale = np.where(scale == 0, 1e-8, scale)
#     W_q = np.clip(np.round(W / scale[np.newaxis, :]), qmin, qmax).astype(np.int8)
#     zp = np.zeros(scale.shape, dtype=np.int8)
#     return W_q, scale.astype(np.float32), zp


# def quant_per_tensor(W: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Symmetric per-tensor quantization.
#     Returns (W_q, scale scalar, zero_point scalar)
#     """
#     qmin, qmax, qp = _qrange(bits)
#     W = W.astype(np.float32)
#     scale = float(np.max(np.abs(W))) / float(qp)
#     scale = max(scale, 1e-8)
#     W_q = np.clip(np.round(W / scale), qmin, qmax).astype(np.int8)
#     return W_q, np.array(scale, dtype=np.float32), np.array(0, dtype=np.int8)


# def quant_per_block(W: np.ndarray, block_size: int = 32, bits: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     llama.cpp Q8_0 style: per-block symmetric.
#     W [K, N] is flattened, divided into blocks of `block_size` elements.
#     scale shape: [num_blocks]  (stored flat, requires custom dequant)
#     Because standard ONNX DequantizeLinear doesn't support per-block natively,
#     we reshape W so blocks map to a new axis, then use per-channel on that axis.
#     Falls back to per-channel if W cannot be evenly blocked.
#     """
#     qmin, qmax, qp = _qrange(bits)
#     W_fp = W.astype(np.float32)
#     orig_shape = W_fp.shape
#     flat = W_fp.flatten()
#     n = len(flat)
#     if n % block_size != 0:
#         print(f"    Warning: {n} elements not divisible by block_size={block_size}; "
#               f"falling back to per-channel.")
#         return quant_per_channel(W, bits)
#     num_blocks = n // block_size
#     blocks = flat.reshape(num_blocks, block_size)
#     scale  = np.max(np.abs(blocks), axis=1) / float(qp)
#     scale  = np.where(scale == 0, 1e-8, scale)
#     W_q = np.clip(
#         np.round(blocks / scale[:, np.newaxis]), qmin, qmax
#     ).astype(np.int8).reshape(orig_shape)
#     W_block = W_q.flatten().reshape(num_blocks, block_size)
#     zp = np.zeros(scale.shape, dtype=np.int8)
#     return W_block, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


# def quant_q4_0(W: np.ndarray, block_size: int = 32):
#     """
#     llama.cpp Q4_0: per-block 4-bit symmetric quantization.

#     Each block of `block_size` weights gets its own bf16 scale.
#       scale = max(|block|) / 7
#       W_q   = clip(round(W / scale), -8, 7)   — signed 4-bit range

#     Storage per block: 32 nibbles (16 bytes packed) + 1 bf16 scale (2 bytes) = 18 bytes
#     Effective compression: 32*4 / 18 = 7.1x vs fp32, ~3.6x vs bf16

#     Returns (W_q [num_blocks, block_size], scale [num_blocks], zp [num_blocks],
#              block_meta (num_blocks, block_size, orig_shape))
#     """
#     W_fp = W.astype(np.float32)
#     orig_shape = W_fp.shape
#     flat = W_fp.flatten()
#     n = len(flat)

#     # Pad to multiple of block_size
#     pad = (block_size - n % block_size) % block_size
#     if pad:
#         flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
#         n = len(flat)

#     num_blocks = n // block_size
#     blocks = flat.reshape(num_blocks, block_size)

#     # Q4_0: scale = amax / 7, range [-8, 7]
#     amax = np.max(np.abs(blocks), axis=1)
#     scale = amax / 7.0
#     scale = np.where(scale == 0, 1e-8, scale)

#     W_q = np.clip(np.round(blocks / scale[:, np.newaxis]), -8, 7).astype(np.int8)
#     zp = np.zeros(num_blocks, dtype=np.int8)

#     return W_q, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


# def quant_q4_k(W: np.ndarray, block_size: int = 32):
#     """
#     llama.cpp Q4_K-style: per-block 4-bit ASYMMETRIC quantization.

#     Key difference from Q4_0 (symmetric):
#       Q4_0: scale = max(|block|) / 7, range [-8, 7], zero_point = 0
#       Q4_K: scale = (max - min) / 15, range [0, 15], zero_point per block

#     Returns (W_q [num_blocks, block_size] uint8, scale [num_blocks] fp32,
#              zp [num_blocks] uint8, block_meta)
#     """
#     W_fp = W.astype(np.float32)
#     orig_shape = W_fp.shape
#     flat = W_fp.flatten()
#     n = len(flat)

#     pad = (block_size - n % block_size) % block_size
#     if pad:
#         flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
#         n = len(flat)

#     num_blocks = n // block_size
#     blocks = flat.reshape(num_blocks, block_size)

#     block_min = np.min(blocks, axis=1)
#     block_max = np.max(blocks, axis=1)

#     scale = (block_max - block_min) / 15.0
#     scale = np.where(scale == 0, 1e-8, scale)

#     zp = np.clip(np.round(-block_min / scale), 0, 15).astype(np.uint8)

#     W_q = np.clip(
#         np.round(blocks / scale[:, np.newaxis] + zp[:, np.newaxis].astype(np.float32)),
#         0, 15
#     ).astype(np.uint8)

#     return W_q, scale.astype(np.float32), zp, (num_blocks, block_size, orig_shape)


# # ── ONNX graph helpers ────────────────────────────────────────────────────────

# def make_unique_name(base: str, existing: set) -> str:
#     name, i = base, 0
#     while name in existing:
#         name = f"{base}_{i}"
#         i += 1
#     existing.add(name)
#     return name


# def activation_elem_type(node, vi_map: dict) -> int:
#     """Return the ONNX elem_type of the activation input (input[0]) of a MatMul node."""
#     inp = node.input[0]
#     if inp in vi_map:
#         return vi_map[inp].type.tensor_type.elem_type
#     return TensorProto.FLOAT


# # ── main quantization pass ────────────────────────────────────────────────────

# def quantize_model(model_path: str, out_path: str, granularity: str, block_size: int,
#                    scale_dtype: str = "bf16", bits: int = 8):
#     print(f"Loading: {model_path}")
#     model = onnx.load(model_path)
#     graph = model.graph

#     # Shape inference so we know activation dtypes
#     try:
#         model = shape_inference.infer_shapes(model)
#         graph = model.graph
#     except Exception as e:
#         print(f"  Warning: shape inference failed ({e})")

#     vi_map = {vi.name: vi for vi in graph.value_info}
#     for x in list(graph.input) + list(graph.output):
#         vi_map[x.name] = x

#     init_map = {init.name: init for init in graph.initializer}

#     # Collect names that are already used (to generate unique names)
#     used_names: set = {n.name for n in graph.node if n.name}
#     used_names.update(init_map.keys())
#     used_names.update(vi_map.keys())

#     # Accumulators for new nodes / initializers / removed initializers
#     new_initializers: list = []
#     nodes_to_replace: dict = {}   # node index -> list of replacement nodes
#     removed_inits: set = set()

#     stats = defaultdict(int)

#     if granularity in ("q4_0", "q4_k"):
#         print(f"\nQuantizing MatMul weights  ({granularity.upper()}, block_size={block_size}"
#               f", scale_dtype={scale_dtype})\n")
#     else:
#         bits_label = f"W{bits}A16"
#         print(f"\nQuantizing MatMul weights  ({bits_label}, granularity={granularity}"
#               + (f", block_size={block_size}" if granularity == "per_block" else "")
#               + f", scale_dtype={scale_dtype})\n")

#     for idx, node in enumerate(graph.node):
#         if node.op_type not in ("MatMul", "Gemm"):
#             continue

#         wt_name = node.input[1]
#         if wt_name not in init_map:
#             stats["skipped_no_const"] += 1
#             continue  # weight is not a constant initializer

#         # Handle bf16 weights: numpy has no bf16, so read raw bytes and convert
#         wt_proto = init_map[wt_name]
#         dims = list(wt_proto.dims)
#         if wt_proto.data_type == TensorProto.BFLOAT16:
#             raw = wt_proto.raw_data
#             n_elems = 1
#             for d in dims:
#                 n_elems *= d
#             if len(raw) == n_elems * 4:
#                 # stored as float32 raw bytes (e.g. from make_matmul_testcase)
#                 W_raw = np.frombuffer(raw, dtype=np.float32).reshape(dims)
#             else:
#                 # actual bf16: 2 bytes per elem, pad to f32 via uint16 trick
#                 u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
#                 W_raw = np.zeros(dims, dtype=np.float32)
#                 W_raw.view(np.uint32)[:] = u16.astype(np.uint32) << 16
#         else:
#             W_raw = numpy_helper.to_array(wt_proto)
#         orig_dtype = W_raw.dtype

#         # Must be 2-D for standard MatMul weight; handle batched later
#         if W_raw.ndim != 2:
#             print(f"  [SKIP] {node.name or wt_name}: weight ndim={W_raw.ndim} (need 2D)")
#             stats["skipped_non_2d"] += 1
#             continue

#         K, N = W_raw.shape
#         act_type = activation_elem_type(node, vi_map)
#         act_type_name = TensorProto.DataType.Name(act_type)

#         print(f"  MatMul  weight={wt_name}  shape=[{K},{N}]  "
#               f"orig_dtype={orig_dtype}  act={act_type_name}")

#         # ── quantize ─────────────────────────────────────────────────────────
#         block_meta = None
#         if granularity == "q4_k":
#             W_q, scale, zp, block_meta = quant_q4_k(W_raw, block_size)
#             dq_axis = 0
#             bits = 4
#         elif granularity == "q4_0":
#             W_q, scale, zp, block_meta = quant_q4_0(W_raw, block_size)
#             dq_axis = 0
#             bits = 4  # Q4_0 is always 4-bit
#         elif granularity == "per_channel":
#             W_q, scale, zp = quant_per_channel(W_raw, bits)
#             dq_axis = 1
#         elif granularity == "per_tensor":
#             W_q, scale, zp = quant_per_tensor(W_raw, bits)
#             dq_axis = None
#         else:  # per_block
#             result = quant_per_block(W_raw, block_size, bits)
#             if len(result) == 4:
#                 W_q, scale, zp, block_meta = result
#                 dq_axis = 0
#             else:
#                 W_q, scale, zp = result
#                 dq_axis = 1

#         # Compression stats
#         scale_elem_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[scale_dtype]
#         orig_bytes  = W_raw.size * W_raw.astype(np.float32).itemsize
#         if granularity in ("q4_0", "q4_k"):
#             n_blocks = W_q.shape[0]
#             zp_bytes = 1 if granularity == "q4_k" else 0
#             quant_bytes = n_blocks * (block_size // 2 + scale_elem_bytes + zp_bytes)
#         else:
#             wt_bytes = W_q.size if bits == 8 else W_q.size // 2
#             quant_bytes = wt_bytes + scale.size * scale_elem_bytes
#         ratio = orig_bytes / quant_bytes
#         if granularity == "q4_0":
#             range_str = "[-8,7] symmetric"
#         elif granularity == "q4_k":
#             range_str = "[0,15] asymmetric"
#         else:
#             qmin, qmax, _ = _qrange(bits)
#             range_str = f"[{qmin},{qmax}]"
#         print(f"    -> INT{bits}  range={range_str}  scale_shape={list(scale.shape)}  "
#               f"scale_dtype={scale_dtype}  "
#               f"size {orig_bytes/1024:.1f}KB -> {quant_bytes/1024:.1f}KB  "
#               f"compression={ratio:.1f}x")

#         # ── build new initializer names ───────────────────────────────────────
#         base = wt_name.replace("/", "_").replace(".", "_")
#         wt_int8_name  = make_unique_name(f"{base}_int8",  used_names)
#         scale_name    = make_unique_name(f"{base}_scale", used_names)
#         zp_name       = make_unique_name(f"{base}_zp",    used_names)
#         dq_out_name   = make_unique_name(f"{base}_dq",    used_names)

#         # ── add new initializers ──────────────────────────────────────────────
#         new_initializers.append(numpy_helper.from_array(W_q, name=wt_int8_name))

#         # Store scale in requested dtype
#         if scale_dtype == "bf16":
#             # Round to bf16 precision but store as fp32 (compiler can't lower bf16 scale)
#             scale_f32 = scale.astype(np.float32)
#             bf16_u16 = (scale_f32.view(np.uint32) >> 16).astype(np.uint16)
#             scale_bf16_as_f32 = np.zeros_like(scale_f32)
#             scale_bf16_as_f32.view(np.uint32)[:] = bf16_u16.astype(np.uint32) << 16
#             new_initializers.append(numpy_helper.from_array(
#                 scale_bf16_as_f32, name=scale_name))
#         elif scale_dtype == "fp16":
#             new_initializers.append(numpy_helper.from_array(
#                 scale.astype(np.float16), name=scale_name))
#         else:  # fp32
#             new_initializers.append(numpy_helper.from_array(scale, name=scale_name))

#         new_initializers.append(numpy_helper.from_array(zp,     name=zp_name))
#         removed_inits.add(wt_name)

#         # ── DequantizeLinear node ─────────────────────────────────────────────
#         dq_kwargs = {}
#         if dq_axis is not None:
#             dq_kwargs["axis"] = dq_axis
#         dq_node = helper.make_node(
#             "DequantizeLinear",
#             inputs=[wt_int8_name, scale_name, zp_name],
#             outputs=[dq_out_name],
#             name=make_unique_name(f"DQL_{base}", used_names),
#             **dq_kwargs,
#         )

#         # ── optional Reshape back to [K, N] for per-block mode ────────────────
#         reshape_nodes = []
#         matmul_weight_input = dq_out_name
#         if block_meta is not None:
#             num_blocks, bs, orig_shape = block_meta
#             target_shape_name = make_unique_name(f"{base}_target_shape", used_names)
#             reshape_out_name  = make_unique_name(f"{base}_reshaped",      used_names)
#             new_initializers.append(numpy_helper.from_array(
#                 np.array(list(orig_shape), dtype=np.int64), name=target_shape_name))
#             reshape_nodes.append(helper.make_node(
#                 "Reshape",
#                 inputs=[dq_out_name, target_shape_name],
#                 outputs=[reshape_out_name],
#                 name=make_unique_name(f"Reshape_{base}", used_names),
#             ))
#             matmul_weight_input = reshape_out_name

#         # ── Cast dequantized weight to activation dtype if needed ─────────────
#         # DequantizeLinear always outputs float32; cast to bf16/fp16 if needed
#         cast_nodes = []
#         mm_weight_input = matmul_weight_input
#         if act_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
#             cast_out_name = make_unique_name(f"{base}_cast", used_names)
#             cast_nodes.append(helper.make_node(
#                 "Cast",
#                 inputs=[matmul_weight_input],
#                 outputs=[cast_out_name],
#                 name=make_unique_name(f"Cast_{base}", used_names),
#                 to=act_type,
#             ))
#             mm_weight_input = cast_out_name

#         # ── rebuild the MatMul/Gemm node with new weight input ────────────────
#         new_inputs = list(node.input)
#         new_inputs[1] = mm_weight_input
#         new_matmul = helper.make_node(
#             node.op_type,
#             inputs=new_inputs,
#             outputs=list(node.output),
#             name=node.name or make_unique_name(f"MM_{base}", used_names),
#         )
#         # copy existing attributes (e.g. Gemm transB etc.)
#         for attr in node.attribute:
#             new_matmul.attribute.append(attr)

#         nodes_to_replace[idx] = [dq_node] + reshape_nodes + cast_nodes + [new_matmul]
#         stats["quantized"] += 1

#     # ── rebuild graph ─────────────────────────────────────────────────────────
#     new_nodes = []
#     for idx, node in enumerate(graph.node):
#         if idx in nodes_to_replace:
#             new_nodes.extend(nodes_to_replace[idx])
#         else:
#             new_nodes.append(node)

#     # Replace initializers
#     kept_inits = [init for init in graph.initializer if init.name not in removed_inits]
#     kept_inits.extend(new_initializers)

#     new_graph = helper.make_graph(
#         nodes=new_nodes,
#         name=graph.name,
#         inputs=list(graph.input),
#         outputs=list(graph.output),
#         initializer=kept_inits,
#     )
#     # copy doc_string and other graph meta
#     new_graph.doc_string = graph.doc_string

#     new_model = helper.make_model(new_graph)
#     new_model.ir_version = model.ir_version
#     new_model.opset_import.extend(model.opset_import)
#     # Remove duplicate default opset entries
#     seen = set()
#     del new_model.opset_import[:]
#     for op in model.opset_import:
#         key = (op.domain, op.version)
#         if key not in seen:
#             seen.add(key)
#             new_model.opset_import.append(op)

#     new_model.producer_name = "torq-w8a16-quant"
#     new_model.doc_string = model.doc_string

#     # Run shape inference on output model
#     try:
#         new_model = shape_inference.infer_shapes(new_model)
#     except Exception as e:
#         print(f"\n  Warning: post-quantization shape inference failed: {e}")

#     onnx.checker.check_model(new_model)
#     onnx.save(new_model, out_path)

#     # ── summary ───────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print("QUANTIZATION SUMMARY")
#     print(f"{'='*60}")
#     print(f"  MatMul nodes quantized  : {stats['quantized']}")
#     print(f"  Skipped (no const wt)   : {stats['skipped_no_const']}")
#     print(f"  Skipped (non-2D wt)     : {stats['skipped_non_2d']}")
#     if granularity in ("q4_0", "q4_k"):
#         print(f"  Strategy                : {granularity.upper()} per-block (block_size={block_size})")
#         sym = "symmetric" if granularity == "q4_0" else "asymmetric"
#         print(f"  Bit width               : 4-bit  {sym}")
#     else:
#         print(f"  Granularity             : {granularity}"
#               + (f"  (block_size={block_size})" if granularity == "per_block" else ""))
#         print(f"  Bit width               : {bits}-bit  (W{bits}A16)")
#     print(f"  Scheme                  : symmetric, zero_point=0")
#     print(f"  Scale dtype             : {scale_dtype}")
#     print(f"  Activation path         : unchanged (original float dtype)")
#     print(f"  Output model            : {out_path}")
#     print()


# def main():
#     parser = argparse.ArgumentParser(
#         description="quantization for ONNX MatMul layers",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument("model", nargs="?",
#                         default="/home/meepat/meet/torq-compiler-dev-new/model.onnx",
#                         help="Input .onnx model (default: model.onnx)")
#     parser.add_argument("--out", default=None,
#                         help="Output .onnx file (default: <model>_w8a16.onnx)")
#     parser.add_argument("--granularity", choices=["per_channel", "per_tensor", "per_block", "q4_0", "q4_k"],
#                         default="per_channel",
#                         help="Quantization granularity (default: per_channel). "
#                              "q4_0 = symmetric 4-bit, q4_k = asymmetric 4-bit (K-quant)")
#     parser.add_argument("--block-size", type=int, default=32,
#                         help="Block size for per_block mode, llama.cpp Q8_0 default=32")
#     parser.add_argument("--scale-dtype", choices=["fp32", "fp16", "bf16"],
#                         default="bf16",
#                         help="Dtype for scale tensor (default: bf16)")
#     parser.add_argument("--bits", type=int, choices=[4, 8], default=8,
#                         help="Quantization bit width: 4 or 8 (default: 8)")
#     args = parser.parse_args()

#     if args.granularity in ("q4_0", "q4_k"):
#         suffix = f"_{args.granularity}.onnx"
#     else:
#         suffix = f"_w{args.bits}a16.onnx"
#     out_path = args.out or str(Path(args.model).stem) + suffix
#     quantize_model(args.model, out_path, args.granularity, args.block_size,
#                    args.scale_dtype, args.bits)


# if __name__ == "__main__":
#     main()
