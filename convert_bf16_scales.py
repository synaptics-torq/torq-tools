#!/usr/bin/env python3
"""
Convert int4/int8 quantized models using bf16-precision scales.

Instead of the default approach (dequantize with fp32 scales, then cast all to bf16),
this script:
1. Casts quantization scales to bf16 (round-trip fp32 -> bf16 -> fp32)
2. Dequantizes weights using the bf16-rounded scales
3. Replaces the weight tensors in the existing static model
4. Converts the model to bf16 via the standard dtype converter

This produces models where the scale precision matches the final compute precision,
potentially giving different (possibly better) results than the default pipeline where
scales are in fp32 during dequantization but the weights are later truncated to bf16.
"""

import sys
import time
import shutil
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).parent / "src"))

BASE = Path("models/google/gemma-3-270m-it")


def float32_to_bf16_roundtrip(arr: np.ndarray) -> np.ndarray:
    """Round-trip fp32 values through bf16 (truncate lower 16 mantissa bits)."""
    arr_f32 = arr.astype(np.float32)
    as_int = arr_f32.view(np.uint32)
    bf16_int = as_int & np.uint32(0xFFFF0000)
    return bf16_int.view(np.float32)


def dequantize_matmulnbits_bf16_scales(
    W_q: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None,
    K: int,
    N: int,
    bits: int,
    block_size: int,
) -> np.ndarray:
    """
    Dequantize MatMulNBits weights using bf16-precision scales.
    
    Same as _dequantize_matmulnbits_weights but scales are first
    rounded to bf16 precision before the multiplication.
    """
    n_blocks = (K + block_size - 1) // block_size

    if bits == 4:
        low = (W_q & 0x0F).astype(np.int8)
        high = ((W_q >> 4) & 0x0F).astype(np.int8)
        unpacked = np.stack([low, high], axis=-1).reshape(N, n_blocks, block_size)
    elif bits == 8:
        unpacked = W_q.reshape(N, n_blocks, block_size)
    else:
        raise ValueError(f"Unsupported MatMulNBits bit width: {bits}")

    # Cast scales to bf16 precision BEFORE dequantization
    scales_f32 = np.asarray(scales, dtype=np.float32).reshape(N, n_blocks)
    scales_bf16 = float32_to_bf16_roundtrip(scales_f32)

    # Unpack zero points
    if zero_points is not None and zero_points.size > 0:
        if bits == 4:
            zp_low = (zero_points & 0x0F).astype(np.int8)
            zp_high = ((zero_points >> 4) & 0x0F).astype(np.int8)
            zp_unpacked = np.stack([zp_low, zp_high], axis=-1).reshape(N, n_blocks)
        else:
            zp_unpacked = zero_points.reshape(N, n_blocks)
    else:
        zp_unpacked = np.zeros((N, n_blocks), dtype=np.uint8)

    # Dequantize using bf16-precision scales
    W_float = (unpacked.astype(np.float32) - zp_unpacked[:, :, np.newaxis].astype(np.float32)) * scales_bf16[:, :, np.newaxis]

    W_float = W_float.reshape(N, -1)[:, :K]
    return W_float.T.astype(np.float32)


def download_source_if_needed(source_dir: Path, variant: str):
    """Download quantized source model from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download

    HF_REPO = "onnx-community/gemma-3-270m-it-ONNX"
    files_map = {
        "int4": ["model_q4.onnx", "model_q4.onnx_data"],
        "int8": ["model_quantized.onnx", "model_quantized.onnx_data"],
    }
    files = files_map.get(variant, [])
    if not files:
        return

    # Check if any expected file already exists
    if any((source_dir / f).exists() for f in files):
        return

    # Also check for model.onnx (some int8 dirs use this name)
    if (source_dir / "model.onnx").exists():
        return

    print(f"  Downloading {variant} source from {HF_REPO}...")
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        print(f"    {filename}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"onnx/{filename}",
            local_dir=str(source_dir),
            local_dir_use_symlinks=False,
        )
        downloaded = source_dir / "onnx" / filename
        if downloaded.exists():
            downloaded.rename(source_dir / filename)
    onnx_subdir = source_dir / "onnx"
    if onnx_subdir.exists() and not any(onnx_subdir.iterdir()):
        onnx_subdir.rmdir()


def dequantize_source_model(source_dir: Path, bits_expected: int) -> dict[str, np.ndarray]:
    """
    Load source ONNX model with MatMulNBits ops and dequantize all weights
    using bf16-precision scales.
    
    Returns dict mapping weight name -> dequantized fp32 array (K, N).
    """
    # Find source model
    source_path = source_dir / "model_q4.onnx"
    if not source_path.exists():
        source_path = source_dir / "model.onnx"
    if not source_path.exists():
        source_path = source_dir / "model_quantized.onnx"
    if not source_path.exists():
        raise FileNotFoundError(f"No source model found in {source_dir}")

    print(f"  Loading source: {source_path}")
    model = onnx.load(str(source_path))

    # Build initializer lookup by name
    init_lookup = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    # Find all MatMulNBits nodes
    matmul_nbits = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
    print(f"  Found {len(matmul_nbits)} MatMulNBits nodes")

    weights = {}
    for i, node in enumerate(matmul_nbits):
        # Extract attributes
        attrs = {a.name: a.i for a in node.attribute}
        K = attrs["K"]
        N = attrs["N"]
        bits = attrs.get("bits", 4)
        block_size = attrs.get("block_size", 32)

        # Get input tensors (inputs[1]=W_q, inputs[2]=scales, inputs[3]=zero_points)
        W_q = init_lookup[node.input[1]]
        scales = init_lookup[node.input[2]]
        zp = init_lookup.get(node.input[3]) if len(node.input) > 3 and node.input[3] else None

        W_float = dequantize_matmulnbits_bf16_scales(
            W_q, scales, zp, K, N, bits, block_size,
        )

        # Derive weight name matching the static model convention
        # Source node: /model/layers.0/attn/q_proj/MatMul_Q4 or /model/.../MatMul_Quant
        # Static weight: /model/layers.0/attn/q_proj/MatMul/weight_dequantized
        node_name = node.name
        # Strip _Q4/_Q8 or _Quant suffix
        if "_Q" in node_name and node_name.rsplit("_Q", 1)[1].isdigit():
            base_name = node_name.rsplit("_Q", 1)[0]
        elif node_name.endswith("_Quant"):
            base_name = node_name[:-6]  # Strip "_Quant"
        else:
            base_name = node_name
        weight_name = f"{base_name}/weight_dequantized"
        weights[weight_name] = W_float

        if (i + 1) % 20 == 0 or i == len(matmul_nbits) - 1:
            print(f"  Dequantized {i+1}/{len(matmul_nbits)} (bits={bits}, shape={W_float.shape})")

    return weights


def replace_weights_in_static_model(
    static_model_path: Path,
    new_weights: dict[str, np.ndarray],
    output_path: Path,
):
    """
    Replace weight initializers in existing static model with new weights.
    
    Name mapping:
      new_weights key: "/model/layers.0/attn/q_proj/MatMul/weight_dequantized"
      static model int4: "/model/layers.0/attn/q_proj/MatMul/weight_dequantized" (fp32)
      static model int8_converted: "/model/layers.0/attn/q_proj/MatMul/weight_dequantized_bf16" (bf16)
    """
    print(f"  Loading static model: {static_model_path}")
    model = onnx.load(str(static_model_path))

    # Build index: strip _bf16 suffix for matching
    init_map = {}
    for idx, init in enumerate(model.graph.initializer):
        name = init.name
        # Normalize: strip _bf16 suffix for matching
        match_name = name.replace("_bf16", "") if name.endswith("_bf16") else name
        init_map[match_name] = (idx, name)

    matched = 0
    for new_name, new_weight in new_weights.items():
        if new_name in init_map:
            idx, orig_name = init_map[new_name]
            init = model.graph.initializer[idx]
            existing_shape = tuple(init.dims)
            if existing_shape == new_weight.shape:
                new_tensor = numpy_helper.from_array(new_weight, name=orig_name)
                model.graph.initializer[idx].CopyFrom(new_tensor)
                matched += 1
            elif existing_shape == new_weight.T.shape:
                # Possible transpose mismatch — try transposing
                new_tensor = numpy_helper.from_array(new_weight.T.astype(np.float32), name=orig_name)
                model.graph.initializer[idx].CopyFrom(new_tensor)
                matched += 1
                print(f"    NOTE: Transposed {new_name} ({new_weight.shape} -> {new_weight.T.shape})")
            else:
                print(f"    WARNING: Shape mismatch for {new_name}: "
                      f"existing={existing_shape}, new={new_weight.shape}")
        else:
            print(f"    WARNING: No match for {new_name}")

    print(f"  Matched and replaced {matched}/{len(new_weights)} weights")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def convert_to_bf16(input_path: Path, output_path: Path):
    """Convert fp32 model to bf16 by casting all float32 initializers to bfloat16.
    Also converts int64 position_ids input to int32 (required by Torq compiler)."""
    print(f"  Converting to bf16: {input_path} -> {output_path}")
    model = onnx.load(str(input_path))

    bf16_dtype = onnx.TensorProto.BFLOAT16
    fp32_dtype = onnx.TensorProto.FLOAT
    int64_dtype = onnx.TensorProto.INT64
    int32_dtype = onnx.TensorProto.INT32

    # Convert all fp32 initializers to bf16
    converted_count = 0
    for init in model.graph.initializer:
        if init.data_type == fp32_dtype:
            arr = numpy_helper.to_array(init)
            # Convert to bf16 via uint16 representation
            arr_uint32 = arr.view(np.uint32)
            arr_bf16_uint16 = (arr_uint32 >> 16).astype(np.uint16)
            # Create new tensor with bf16 data
            new_init = onnx.TensorProto()
            new_init.name = init.name
            new_init.data_type = bf16_dtype
            new_init.dims.extend(init.dims)
            new_init.raw_data = arr_bf16_uint16.tobytes()
            init.CopyFrom(new_init)
            converted_count += 1

    # Update graph input/output types (only fp32 -> bf16, keep int types)
    # Also convert position_ids from int64 -> int32 (compiler requirement)
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.elem_type == fp32_dtype:
            vi.type.tensor_type.elem_type = bf16_dtype
        elif vi.type.tensor_type.elem_type == int64_dtype and "position" in vi.name:
            vi.type.tensor_type.elem_type = int32_dtype

    # Update value_info types
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == fp32_dtype:
            vi.type.tensor_type.elem_type = bf16_dtype

    # Convert position_ids-related int64 initializers to int32
    for init in model.graph.initializer:
        if init.data_type == int64_dtype:
            # Check if this is a small constant (cur_len, axes, etc.) that should be int32
            arr = numpy_helper.to_array(init)
            if arr.size <= 256 * 4:  # small constants (masks, indices)
                new_init = numpy_helper.from_array(arr.astype(np.int32), name=init.name)
                init.CopyFrom(new_init)

    print(f"  Converted {converted_count} initializers to bf16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"  Saved bf16 model: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def process_variant(source_type: str, output_name: str):
    """Process one variant (int4 or int8) with bf16 scales."""
    print(f"\n{'='*80}")
    print(f"  Variant: {source_type} -> {output_name}")
    print(f"{'='*80}")

    t0 = time.time()

    source_dir = BASE / "source" / source_type
    # Download source from HuggingFace if not present
    download_source_if_needed(source_dir, source_type)
    # Use the already-converted bf16 model as the base (has correct types, graph structure)
    converted_dir = BASE / "export" / "onnx" / (
        "converted" if source_type == "int4" else "int8_converted"
    ) / "static"
    output_dir = BASE / "export" / "onnx" / output_name / "static"

    # Step 1: Dequantize from source with bf16 scales
    print("\nStep 1: Dequantize with bf16 scales...")
    new_weights = dequantize_source_model(source_dir, bits_expected=4 if source_type == "int4" else 8)

    # Step 2: Replace weights in the already-converted bf16 model
    # This model already has correct int32 position_ids, bf16 types, etc.
    converted_model = converted_dir / "model.onnx"
    if not converted_model.exists():
        print(f"  ERROR: Converted model not found at {converted_model}")
        return

    print(f"\nStep 2: Replace weights in converted bf16 model...")
    print(f"  Source: {converted_model}")
    model = onnx.load(str(converted_model))

    # Build index: match by normalizing weight names
    # Converted model weights: "/model/layers.0/attn/q_proj/MatMul/weight_dequantized_bf16"
    # New weights key: "/model/layers.0/attn/q_proj/MatMul/weight_dequantized"
    init_map = {}
    for idx, init in enumerate(model.graph.initializer):
        name = init.name
        # Normalize: strip _bf16 suffix for matching
        match_name = name[:-5] if name.endswith("_bf16") else name
        init_map[match_name] = (idx, name, tuple(init.dims), init.data_type)

    matched = 0
    for new_name, new_weight in new_weights.items():
        if new_name in init_map:
            idx, orig_name, existing_shape, data_type = init_map[new_name]
            # The existing weight is bf16, we need to convert our fp32 weight to bf16
            if data_type == onnx.TensorProto.BFLOAT16:
                # Convert fp32 -> bf16 (matching existing format)
                arr_uint32 = new_weight.view(np.uint32)
                arr_bf16_uint16 = (arr_uint32 >> 16).astype(np.uint16)
                new_init = onnx.TensorProto()
                new_init.name = orig_name
                new_init.data_type = onnx.TensorProto.BFLOAT16
                new_init.dims.extend(existing_shape)
                new_init.raw_data = arr_bf16_uint16.tobytes()
                model.graph.initializer[idx].CopyFrom(new_init)
                matched += 1
            elif data_type == onnx.TensorProto.FLOAT:
                new_init = numpy_helper.from_array(new_weight, name=orig_name)
                model.graph.initializer[idx].CopyFrom(new_init)
                matched += 1
            else:
                print(f"    WARNING: Unexpected dtype {data_type} for {orig_name}")
        else:
            print(f"    WARNING: No match for {new_name}")

    print(f"  Matched and replaced {matched}/{len(new_weights)} weights")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"
    onnx.save(model, str(output_path))
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy embeddings
    emb_src = converted_dir / "token_embeddings.npy"
    if emb_src.exists():
        emb_dst = output_dir / "token_embeddings.npy"
        shutil.copy2(emb_src, emb_dst)
        print(f"  Copied embeddings: {emb_dst}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Convert quantized models using bf16-precision scales"
    )
    parser.add_argument(
        "--source-type", choices=["int4", "int8", "both"], default="both",
        help="Which source model to convert (default: both)"
    )
    args = parser.parse_args()

    sources = []
    if args.source_type in ("int4", "both"):
        sources.append(("int4", "int4_converted_bf16_scales"))
    if args.source_type in ("int8", "both"):
        sources.append(("int8", "int8_converted_bf16_scales"))

    for source_type, output_name in sources:
        process_variant(source_type, output_name)

    print(f"\n{'='*80}")
    print("DONE. Generated models:")
    base = BASE / "export" / "onnx"
    for _, output_name in sources:
        p = base / output_name / "static" / "model.onnx"
        if p.exists():
            print(f"  {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {p} -- NOT FOUND")


if __name__ == "__main__":
    main()
