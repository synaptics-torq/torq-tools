import argparse
import os
import re
import subprocess
import sys

import numpy as np
import onnx
import torch
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper
from safetensors.torch import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))


"""
Convert compressed-tensors W4 GPTQ weights to an ONNX QDQ graph. The quantization
grouping axis is selected with --grouping (default: in).

--grouping in  (default) — in_features grouping (group_size=32), native block DQ.
    See load_infeat_qdq(). Matches gptq_int4_in_feature_blocksize_example.onnx:
      q[K,N] uint8, scale/zp [K//32,N]
      DequantizeLinear(axis=0, block_size=32) → Cast → MatMul   (no Reshape/Transpose)

--grouping out — out_features grouping (block_structure="32x1"), no block_size,
    no runtime Transpose. See load_outfeat_qdq():
      q[K*(N//32),32], scale[K*(N//32)]
      DequantizeLinear(axis=0) → Reshape[K,N] → Cast → MatMul

Both are numerically the compressed-tensors dequant (q - zp) * scale; only the
initializer layout / DQ form differ. Note K*(N//32) == N*(K//32) == N*K/32, so
the two forms reuse the same q4_k template skeleton — only values/attrs change.
"""


LINEAR_NAME_RE = re.compile(
    r"^/model/layers\.(?P<layer>\d+)/(?P<block>self_attn|mlp)/(?P<proj>[^/]+)/MatMul$"
)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def unpack_int4_from_int32(packed):
    packed = packed.to(torch.int32)
    parts = []
    for offset in range(8):
        parts.append(((packed >> (4 * offset)) & 0xF).to(torch.uint8))
    return torch.stack(parts, dim=-1).flatten(-2)


def unpack_int4_signed_from_int32(packed):
    return (unpack_int4_from_int32(packed).to(torch.int16) - 8).to(torch.int8)


def unpack_weight_zero_point(packed_zp, n_rows):
    """Unpack int4 zero-points packed 8-per-int32 along axis 0.

    packed_zp : [ceil(n_rows/8), n_cols]  int32
    returns   : [n_rows, n_cols]          uint8
    """
    packed_zp = packed_zp.to(torch.int32)
    n_cols = packed_zp.shape[1]
    zp = torch.empty((n_rows, n_cols), dtype=torch.int8)
    for offset in range(8):
        rows = torch.arange(offset, n_rows, 8)
        if rows.numel() == 0:
            continue
        zp[rows] = (
            ((packed_zp[: rows.numel(), :] >> (4 * offset)) & 0xF).to(torch.int16) - 8
        ).to(torch.int8)
    return zp


def zero_point_to_int4_domain(zero_point, expected_shape):
    """Return zero-points in the same signed int4 domain as unpacked weights."""
    if tuple(zero_point.shape) == tuple(expected_shape):
        zp = zero_point.to(torch.int8)
        if torch.any((zp < -8) | (zp > 7)):
            raise ValueError("zero_point values must be in signed int4 range [-8, 7]")
        return zp

    # Older pack-quantized checkpoints may pack zero-points along axis 0.
    return unpack_weight_zero_point(zero_point, int(expected_shape[0]))


def tensor_to_numpy(tensor, dtype=None):
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.numpy()


# ---------------------------------------------------------------------------
# ONNX graph helpers
# ---------------------------------------------------------------------------

def get_initializer_array(initializers, name):
    if name not in initializers:
        raise KeyError(f"Missing ONNX initializer: {name}")
    return numpy_helper.to_array(initializers[name])


def update_initializer(graph, name, array):
    replacement = numpy_helper.from_array(array, name=name)
    for idx, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            graph.initializer[idx].CopyFrom(replacement)
            return
    graph.initializer.append(replacement)


def get_node_attr_ints(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return tuple(int(value) for value in attr.ints)
    return default


def set_node_attr_int(node, name, value):
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return
    node.attribute.append(helper.make_attribute(name, int(value)))


def get_cast_to_dtype(cast, default=TensorProto.FLOAT):
    for attr in cast.attribute:
        if attr.name == "to":
            return int(attr.i)
    return default


def upsert_value_info_shape(graph, name, elem_type, shape):
    shape = [int(dim) for dim in shape]
    value_info = helper.make_tensor_value_info(name, elem_type, shape)
    for collection in (graph.value_info, graph.input, graph.output):
        for existing in collection:
            if existing.name == name:
                existing.CopyFrom(value_info)
                return
    graph.value_info.append(value_info)


def consumers_map(graph):
    consumers = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def only_consumer(consumers, value_name, op_type=None):
    nodes = consumers.get(value_name, [])
    if op_type is not None:
        nodes = [node for node in nodes if node.op_type == op_type]
    if len(nodes) != 1:
        return None
    return nodes[0]


def module_key_from_matmul_name(matmul_name):
    if matmul_name == "/lm_head/MatMul":
        return "lm_head"
    match = LINEAR_NAME_RE.match(matmul_name)
    if match is None:
        return None
    layer = match.group("layer")
    block = match.group("block")
    proj = match.group("proj")
    return f"model.layers.{layer}.{block}.{proj}"


def find_qdq_matmul_entries(graph, initializers):
    """Find DequantizeLinear → Reshape → (optional Transpose) → Cast → MatMul chains."""
    consumers = consumers_map(graph)
    entries = []

    for dq_node in graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue
        if len(dq_node.input) < 3 or len(dq_node.output) != 1:
            continue

        reshape = only_consumer(consumers, dq_node.output[0], "Reshape")
        if reshape is None or len(reshape.input) < 2 or len(reshape.output) != 1:
            continue

        transpose = None
        cast = only_consumer(consumers, reshape.output[0], "Cast")
        if cast is None:
            transpose = only_consumer(consumers, reshape.output[0], "Transpose")
            if transpose is None or len(transpose.output) != 1:
                continue
            perm = get_node_attr_ints(transpose, "perm", default=None)
            if perm != (1, 0):
                raise ValueError(f"{transpose.name}: expected perm=[1, 0], got {perm}")
            cast = only_consumer(consumers, transpose.output[0], "Cast")

        if cast is None or len(cast.output) != 1:
            continue

        matmul = only_consumer(consumers, cast.output[0], "MatMul")
        if matmul is None:
            continue

        module_key = module_key_from_matmul_name(matmul.name)
        if module_key is None:
            continue

        target_shape = get_initializer_array(initializers, reshape.input[1]).astype(np.int64)
        if target_shape.size != 2:
            raise ValueError(f"{reshape.name} target shape must be 2D, got {target_shape.tolist()}")

        entries.append(
            {
                "module_key": module_key,
                "dq_node": dq_node,
                "reshape": reshape,
                "transpose": transpose,
                "cast": cast,
                "target_shape": tuple(int(x) for x in target_shape.tolist()),
                "q_name": dq_node.input[0],
                "scale_name": dq_node.input[1],
                "zp_name": dq_node.input[2],
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Safetensors loader
# ---------------------------------------------------------------------------

def load_outfeat_qdq(safetensors_path, module_key):
    """Load out_features-grouped quantization and pre-process for ONNX.

    Expects scale shape [N//group_size, K]  (out_features grouping).

    Returns
    -------
    q_flat      : np.ndarray  [K*(N//gs), gs]   uint8
    scale_flat  : np.ndarray  [K*(N//gs)]        float32
    zp_flat     : np.ndarray  [K*(N//gs)]        uint8
    source_shape: (N, K)
    matmul_shape: (K, N)  ← Reshape target, no Transpose needed
    group_size  : int
    """
    prefix = f"{module_key}.weight"
    names = {
        "packed":     f"{prefix}_packed",
        "scale":      f"{prefix}_scale",
        "shape":      f"{prefix}_shape",
        "zero_point": f"{prefix}_zero_point",
    }

    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [n for n in names.values() if n not in keys]
        if missing:
            raise KeyError(f"Missing safetensors keys: {missing}")
        packed    = handle.get_tensor(names["packed"])
        scale     = handle.get_tensor(names["scale"]).float()
        shape     = handle.get_tensor(names["shape"]).to(torch.int64).tolist()
        packed_zp = handle.get_tensor(names["zero_point"])

    N = out_features = int(shape[0])
    K = in_features  = int(shape[1])

    # scale must be [N//gs, K] for out_features grouping
    if scale.shape[1] != K:
        raise ValueError(
            f"{module_key}: expected scale.shape[1]=={K} (in_features) for out_features "
            f"grouping, got scale.shape={tuple(scale.shape)}. "
            f"(If scale.shape[0]=={N}, this is in_features grouping, not out_features.)"
        )
    out_groups = int(scale.shape[0])          # N // group_size
    if N % out_groups != 0:
        raise ValueError(f"out_features={N} is not divisible by out_groups={out_groups}")
    group_size = N // out_groups              # e.g. 32

    expected_packed = (N, (K + 7) // 8)
    if tuple(packed.shape) != expected_packed:
        raise ValueError(
            f"{module_key}: unexpected packed shape {tuple(packed.shape)}, expected {expected_packed}"
        )

    # Unpack q in the same signed int4 domain used by compressed-tensors:
    # pack_to_int32 stores q + 8, and unpack_from_int32 subtracts 8.
    q = unpack_int4_signed_from_int32(packed)[:, :K]    # [N, K]

    # Unpack zp: packed [(out_groups+7)//8, K] → [out_groups, K]
    # Zero-points may be stored already-unpacked as signed int8 [N//gs, K],
    # or packed 8-per-int32 along axis 0.
    zp = zero_point_to_int4_domain(packed_zp, (out_groups, K))  # [N//gs, K]

    if tuple(zp.shape) != tuple(scale.shape):
        raise ValueError(
            f"{module_key}: zp shape {tuple(zp.shape)} != scale shape {tuple(scale.shape)}"
        )

    # ------------------------------------------------------------------
    # Offline pre-processing: arrange for DequantizeLinear(axis=0)
    # followed by Reshape → [K, N] (MatMul layout, no Transpose)
    #
    # q   [N, K]      → .T [K, N] → view [K, N//gs, gs] → view [K*(N//gs), gs]
    # s   [N//gs, K]  → .T [K, N//gs]                   → view [K*(N//gs)]
    #
    # DQ element at flat-group index (k*(N//gs) + g), position i:
    #   covers q[g*gs + i, k]  with scale[g, k]   ✓
    # After DQ Reshape [K*(N//gs), gs] → [K, N]:
    #   position [k, g*gs + i]  (already [K, N] = MatMul B layout)  ✓
    # ------------------------------------------------------------------

    q_t       = q.T.contiguous()                                  # [K, N]
    q_grouped = q_t.reshape(K, out_groups, group_size)            # [K, N//gs, gs]
    q_flat    = q_grouped.reshape(K * out_groups, group_size)     # [K*(N//gs), gs]

    scale_t    = scale.T.contiguous()                             # [K, N//gs]
    scale_flat = scale_t.reshape(K * out_groups)                  # [K*(N//gs)]

    zp_t    = zp.T.contiguous()                                   # [K, N//gs]
    zp_flat = zp_t.reshape(K * out_groups)                        # [K*(N//gs)]

    return (
        tensor_to_numpy(q_flat,    torch.int8),
        tensor_to_numpy(scale_flat, torch.float32),
        tensor_to_numpy(zp_flat,   torch.int8),
        (N, K),   # source_shape
        (K, N),   # matmul_shape  (Reshape target)
        group_size,
    )


def load_infeat_qdq(safetensors_path, module_key):
    """Load in_features-grouped quantization → native block_size QDQ form.

    Expects scale shape [N, K//group_size]  (in_features grouping, e.g. group_size=32).

    Produces the same layout as the reference example
    (onnx/gptq_int4_in_feature_blocksize_example.onnx):

      q     : [K, N]        uint8 (0..15)   ← MatMul B layout (weight.T)
      scale : [K//gs, N]    float32
      zp    : [K//gs, N]    uint8 (0..15)

    Output ONNX QDQ graph:
      DequantizeLinear(axis=0, block_size=gs) : q[K,N], scale[K//gs,N], zp[K//gs,N]
      Cast                                    : → bf16 / fp16
      MatMul                                  : B = [K, N] directly (no Reshape/Transpose)

    Note the unsigned domain: DequantizeLinear computes (q - zp) * scale with q, zp
    both uint8, which equals the signed (q-8) - (zp-8) used by out_features form —
    numerically identical dequant, only the +8 offset baked into both q and zp.
    """
    prefix = f"{module_key}.weight"
    names = {
        "packed":     f"{prefix}_packed",
        "scale":      f"{prefix}_scale",
        "shape":      f"{prefix}_shape",
        "zero_point": f"{prefix}_zero_point",
    }

    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [n for n in names.values() if n not in keys]
        if missing:
            raise KeyError(f"Missing safetensors keys: {missing}")
        packed    = handle.get_tensor(names["packed"])
        scale     = handle.get_tensor(names["scale"]).float()
        shape     = handle.get_tensor(names["shape"]).to(torch.int64).tolist()
        packed_zp = handle.get_tensor(names["zero_point"])

    N = out_features = int(shape[0])
    K = in_features  = int(shape[1])

    # scale must be [N, K//gs] for in_features grouping
    if scale.shape[0] != N:
        raise ValueError(
            f"{module_key}: expected scale.shape[0]=={N} (out_features) for in_features "
            f"grouping, got scale.shape={tuple(scale.shape)}. "
            f"(If scale.shape[1]=={K}, this is out_features grouping, not in_features.)"
        )
    in_groups = int(scale.shape[1])           # K // group_size
    if K % in_groups != 0:
        raise ValueError(f"in_features={K} is not divisible by in_groups={in_groups}")
    group_size = K // in_groups               # e.g. 32

    expected_packed = (N, (K + 7) // 8)
    if tuple(packed.shape) != expected_packed:
        raise ValueError(
            f"{module_key}: unexpected packed shape {tuple(packed.shape)}, expected {expected_packed}"
        )

    # Unpack q in the UNSIGNED int4 domain (0..15) — compressed-tensors stores q+8,
    # and the reference example keeps that unsigned value directly (uint8).
    q_uns = unpack_int4_from_int32(packed)[:, :K]          # [N, K] uint8 (0..15)

    # Unpack zp to signed int4 domain [N, K//gs], then shift to unsigned (+8) to match q.
    zp_sgn = zero_point_to_int4_domain(packed_zp, (N, in_groups))  # [N, K//gs] int8 (-8..7)
    if tuple(zp_sgn.shape) != tuple(scale.shape):
        raise ValueError(
            f"{module_key}: zp shape {tuple(zp_sgn.shape)} != scale shape {tuple(scale.shape)}"
        )
    zp_uns = (zp_sgn.to(torch.int16) + 8).to(torch.uint8)         # [N, K//gs] uint8 (0..15)

    # Transpose to MatMul B layout [K, N] / [K//gs, N] (baked offline into initializers).
    q_t     = q_uns.T.contiguous()                         # [K, N]
    scale_t = scale.T.contiguous()                         # [K//gs, N]
    zp_t    = zp_uns.T.contiguous()                        # [K//gs, N]

    return (
        tensor_to_numpy(q_t,     torch.uint8),
        tensor_to_numpy(scale_t, torch.float32),
        tensor_to_numpy(zp_t,    torch.uint8),
        (N, K),   # source_shape
        (K, N),   # matmul_shape
        group_size,
    )


# ---------------------------------------------------------------------------
# Graph rewrite helpers
# ---------------------------------------------------------------------------

def remove_nodes(graph, nodes_to_remove):
    remove_ids = {id(node) for node in nodes_to_remove if node is not None}
    if not remove_ids:
        return
    kept = [node for node in graph.node if id(node) not in remove_ids]
    del graph.node[:]
    graph.node.extend(kept)


def remove_stale_value_info(graph, value_names):
    if not value_names:
        return
    kept = [v for v in graph.value_info if v.name not in value_names]
    del graph.value_info[:]
    graph.value_info.extend(kept)


# ---------------------------------------------------------------------------
# Main convert
# ---------------------------------------------------------------------------

def ensure_template(onnx_template, base_onnx, block_size):
    """Ensure a template QDQ ONNX is available.

    If onnx_template already exists, use it as is; otherwise auto-generate it from
    base_onnx via quantize_matmul.py (q4_k, block-wise DequantizeLinear).
    The converter reuses only the template's DequantizeLinear→Reshape→Cast→MatMul
    skeleton and overwrites the q/scale/zp values with the out_feature version, so
    the template's actual quantized values do not matter (only the structure does).
    """
    if os.path.exists(onnx_template):
        return onnx_template

    if not base_onnx:
        raise FileNotFoundError(
            f"ONNX template not found: {onnx_template}\n"
            f"  → pass --base-onnx <model.onnx> to auto-generate it "
            f"(quantize_matmul.py --granularity q4_k), or prepare the template in advance."
        )
    if not os.path.exists(base_onnx):
        raise FileNotFoundError(f"--base-onnx not found: {base_onnx}")

    print(f"[template] {onnx_template} missing → auto-generating from {base_onnx} (q4_k, block-size={block_size})")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "quantize_matmul.py"), base_onnx,
         "--bits", "4", "--granularity", "q4_k", "--block-size", str(block_size),
         "--out", onnx_template],
        check=True,
    )
    return onnx_template


def convert(args):
    template    = ensure_template(args.onnx_template, args.base_onnx, args.template_block_size)
    model       = onnx.load(template)
    graph       = model.graph
    initializers = {init.name: init for init in graph.initializer}
    entries     = find_qdq_matmul_entries(graph, initializers)

    if not entries:
        raise RuntimeError("No DequantizeLinear -> Reshape -> Cast -> MatMul entries found")

    converted         = 0
    skipped           = 0
    nodes_to_remove   = []
    stale_values      = []

    for entry in entries:
        module_key = entry["module_key"]
        if args.skip_lm_head and module_key == "lm_head":
            skipped += 1
            continue

        if args.grouping == "in":
            q, scale, zp, source_shape, matmul_shape, group_size = load_infeat_qdq(
                args.safetensors, module_key
            )
        else:
            q, scale, zp, source_shape, matmul_shape, group_size = load_outfeat_qdq(
                args.safetensors, module_key
            )

        if args.expected_group_size is not None and group_size != args.expected_group_size:
            raise ValueError(
                f"{module_key}: group_size={group_size}, expected {args.expected_group_size}"
            )

        N, K = source_shape
        if entry["target_shape"] not in (source_shape, matmul_shape):
            raise ValueError(
                f"{module_key}: safetensors shape {source_shape} does not match "
                f"ONNX reshape target {entry['target_shape']}"
            )

        if args.grouping == "in":
            # ── in_features: native block DequantizeLinear ────────────────────
            # DequantizeLinear(axis=0, block_size=gs): q[K,N], scale[K//gs,N] → [K,N]
            # Collapse template chain to DQ → Cast → MatMul (drop Reshape + Transpose).
            set_node_attr_int(entry["dq_node"], "axis", 0)
            set_node_attr_int(entry["dq_node"], "block_size", group_size)

            # Wire Cast directly to DQ output; drop Reshape (and Transpose if present).
            entry["cast"].input[0] = entry["dq_node"].output[0]
            nodes_to_remove.append(entry["reshape"])
            stale_values.append(entry["reshape"].output[0])
            if entry["transpose"] is not None:
                nodes_to_remove.append(entry["transpose"])
                stale_values.append(entry["transpose"].output[0])

            # DQ output is now [K, N] (block dequant keeps input shape)
            upsert_value_info_shape(
                graph, entry["dq_node"].output[0], TensorProto.FLOAT, matmul_shape
            )
            upsert_value_info_shape(
                graph,
                entry["cast"].output[0],
                get_cast_to_dtype(entry["cast"]),
                matmul_shape,
            )
        else:
            # ── out_features: DequantizeLinear(axis=0) → Reshape([K,N]) → Cast ─
            set_node_attr_int(entry["dq_node"], "axis", 0)

            # Reshape target → [K, N]  (MatMul layout, no Transpose needed)
            update_initializer(
                graph,
                entry["reshape"].input[1],
                np.asarray(matmul_shape, dtype=np.int64),
            )

            # Remove Transpose — Reshape output is already [K, N]
            if entry["transpose"] is not None:
                entry["cast"].input[0] = entry["reshape"].output[0]
                nodes_to_remove.append(entry["transpose"])
                stale_values.append(entry["transpose"].output[0])
            # If template has no Transpose (Cast directly after Reshape), nothing extra needed.

            # Update value_info
            upsert_value_info_shape(
                graph, entry["reshape"].output[0], TensorProto.FLOAT, matmul_shape
            )
            upsert_value_info_shape(
                graph,
                entry["cast"].output[0],
                get_cast_to_dtype(entry["cast"]),
                matmul_shape,
            )

        # Replace initializer data
        update_initializer(graph, entry["q_name"],     q)
        update_initializer(graph, entry["scale_name"], scale)
        update_initializer(graph, entry["zp_name"],    zp)
        converted += 1

        print(
            f"{module_key}: gs={group_size} grouping={args.grouping} "
            f"q={q.shape} scale={scale.shape} → {matmul_shape}"
        )

    remove_nodes(graph, nodes_to_remove)
    remove_stale_value_info(graph, set(stale_values))

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"saved: {args.out}")
    print(f"converted={converted} skipped={skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert out_features-grouped W4 quantization (block_structure='32x1') "
            "to ONNX QDQ. "
            "Output graph: DequantizeLinear(axis=0) → Reshape([K,N]) → Cast → MatMul. "
            "No block_size attribute, no runtime Transpose node."
        )
    )
    parser.add_argument("--onnx-template",      required=True,
                        help="Path to the QDQ ONNX template. If absent, auto-generated from --base-onnx.")
    parser.add_argument("--base-onnx",           default=None,
                        help="Base ONNX to auto-generate the template from when missing (quantize_matmul q4_k).")
    parser.add_argument("--template-block-size", type=int, default=32,
                        help="block-size for quantize_matmul when auto-generating the template (default 32).")
    parser.add_argument("--safetensors",         required=True,  help="model.safetensors path")
    parser.add_argument("--out",                 required=True,  help="Output ONNX path")
    parser.add_argument(
        "--grouping", choices=["out", "in"], default="in",
        help="Quantization grouping axis. 'in' (default) = in_features (group_size 32), "
             "native block_size DQ form matching gptq_int4_in_feature_blocksize_example.onnx; "
             "'out' = out_features (block_structure '32x1').",
    )
    parser.add_argument(
        "--expected-group-size",
        type=int, default=None,
        help="Optionally assert that the source group_size matches this value.",
    )
    parser.add_argument(
        "--skip-lm-head",
        action="store_true",
        help="Do not replace /lm_head/MatMul quantized weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
