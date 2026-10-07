
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

BF16 = TensorProto.BFLOAT16

TOKEN_ID_LUT_URL = (
    "https://huggingface.co/Synaptics/gemma-3-270-int4-it-torq/resolve/main/"
    "token_id_lut.npy"
)


def _initializers(graph):
    return {init.name: init for init in graph.initializer}


def find_lm_head_dql(graph):
    """Walk back from /lm_head/MatMul's weight input to its DequantizeLinear."""
    by_output = {out: node for node in graph.node for out in node.output}
    matmul = next((n for n in graph.node if n.name == "/lm_head/MatMul"), None)
    if matmul is None:
        raise RuntimeError("/lm_head/MatMul not found")
    name = matmul.input[1]
    while name in by_output and by_output[name].op_type != "DequantizeLinear":
        name = by_output[name].input[0]
    if name not in by_output:
        raise RuntimeError("lm_head DequantizeLinear not found")
    return by_output[name]


def resolve_token_id_lut(path: str | None, cache_dir: Path) -> np.ndarray:
    """Load the trim LUT, downloading the published one if no path is given."""
    if path:
        return np.load(path).astype(np.int64)
    cached = cache_dir / "token_id_lut.npy"
    if not cached.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[trim] downloading token_id_lut.npy -> {cached}")
        urllib.request.urlretrieve(TOKEN_ID_LUT_URL, cached)
    return np.load(cached).astype(np.int64)


def write_bf16_npy(path: str | Path, array_f32: np.ndarray) -> None:
    """Write a raw ``|V2`` npy, the layout the Torq runtime mmaps as bfloat16."""
    bits = np.ascontiguousarray(array_f32, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)  # round-to-nearest-even
    bf16 = (rounded >> 16).astype(np.uint16)
    header = "{'descr': '|V2', 'fortran_order': False, 'shape': %r, }" % (
        array_f32.shape,
    )
    header += " " * (16 - ((10 + len(header) + 1) % 16)) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1, 0]))
        f.write(len(header).to_bytes(2, "little"))
        f.write(header.encode("latin1"))
        f.write(bf16.tobytes(order="C"))


def extract_lm_head_embeddings(model, out_path: str | Path) -> tuple[int, int]:
    graph = model.graph
    inits = _initializers(graph)
    dql = find_lm_head_dql(graph)
    block = next((a.i for a in dql.attribute if a.name == "block_size"), 0)

    q = numpy_helper.to_array(inits[dql.input[0]]).astype(np.float32)   # [K, N]
    scale = numpy_helper.to_array(inits[dql.input[1]]).astype(np.float32)
    zp = numpy_helper.to_array(inits[dql.input[2]]).astype(np.float32)

    k = q.shape[0]
    if block:
        rep = np.repeat(np.arange(scale.shape[0]), block)[:k]
        scale, zp = scale[rep], zp[rep]
    emb = np.ascontiguousarray(((q - zp) * scale).T)                   # [N, K]
    write_bf16_npy(out_path, emb)
    return emb.shape


def cast_dql_scale_to_bf16(model) -> tuple[int, int]:
    graph = model.graph
    inits = _initializers(graph)
    dql_outputs = {n.output[0] for n in graph.node
                   if n.op_type == "DequantizeLinear"}

    n_scale = 0
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        init = inits.get(node.input[1])
        if init is not None and init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(ml_dtypes.bfloat16)
            init.CopyFrom(numpy_helper.from_array(arr, node.input[1]))
            n_scale += 1

    # Keep the node objects alive: protobuf hands out temporary wrappers whose
    # id() would otherwise be reused and delete an unrelated node.
    rewire, removed = {}, []
    for node in graph.node:
        if node.op_type != "Cast" or node.input[0] not in dql_outputs:
            continue
        if next((a.i for a in node.attribute if a.name == "to"), None) == BF16:
            rewire[node.output[0]] = node.input[0]
            removed.append(node)

    removed_ids = {id(n) for n in removed}
    kept = []
    for node in graph.node:
        if id(node) in removed_ids:
            continue
        for i, name in enumerate(node.input):
            if name in rewire:
                node.input[i] = rewire[name]
        kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)

    # shape_inference does not follow the scale dtype, so force the DQL outputs
    # to bf16 — otherwise the importer keeps them f32 and refuses the fusion.
    model.CopyFrom(
        onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    )
    dql_out_names = {n.output[0] for n in model.graph.node
                     if n.op_type == "DequantizeLinear"}
    for vi in model.graph.value_info:
        tt = vi.type.tensor_type
        if vi.name in dql_out_names and tt.elem_type == TensorProto.FLOAT:
            tt.elem_type = BF16
    return n_scale, len(removed)


def pack_dql_to_int4(model, signed: bool = True) -> tuple[int, int, int]:
    """Signed mode re-centres weight and zero_point by -8, so (x-8)-(zp-8) == x-zp."""
    graph = model.graph
    inits = _initializers(graph)

    targets = set()
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        targets.add(node.input[0])
        if len(node.input) > 2 and node.input[2]:
            targets.add(node.input[2])

    packed, before, after = 0, 0, 0
    for name in sorted(targets):
        init = inits.get(name)
        if init is None or init.data_type != TensorProto.UINT8:
            continue
        arr = numpy_helper.to_array(init)
        if int(arr.max()) > 15:
            print(f"[pack] skip {name}: max={int(arr.max())} exceeds 4-bit range")
            continue
        before += len(init.raw_data)
        packed_arr = ((arr.astype(np.int16) - 8).astype(ml_dtypes.int4) if signed
                      else arr.astype(ml_dtypes.uint4))
        new = numpy_helper.from_array(packed_arr, name)
        after += len(new.raw_data)
        init.data_type = new.data_type
        init.raw_data = new.raw_data
        del init.int32_data[:]
        del init.int64_data[:]
        packed += 1

    for opset in model.opset_import:  # 4-bit + block_size DQL needs opset >= 21
        if opset.domain == "" and opset.version < 21:
            opset.version = 21
    return packed, before, after


def trim_lm_head_vocab(model, lut: np.ndarray) -> tuple[int, int]:
    graph = model.graph
    inits = _initializers(graph)
    dql = find_lm_head_dql(graph)

    # vocab is the N axis while the block quantisation is on axis 0, so a plain
    # column select is enough.
    orig_n = int(numpy_helper.to_array(inits[dql.input[0]]).shape[1])
    for idx in (0, 1, 2):
        name = dql.input[idx]
        sliced = np.ascontiguousarray(numpy_helper.to_array(inits[name])[:, lut])
        inits[name].CopyFrom(numpy_helper.from_array(sliced, name))

    n_trim = int(len(lut))
    for collection in (graph.output, graph.value_info):
        for vi in collection:
            dims = vi.type.tensor_type.shape.dim
            if dims and dims[-1].dim_value == orig_n:
                dims[-1].dim_value = n_trim
    return orig_n, n_trim


def remove_gqa_expand(model) -> tuple[int, int, int]:
    graph = model.graph
    expands = [n for n in graph.node if n.op_type == "Expand"]
    shape_inputs, rewired = set(), 0
    for exp in expands:
        data_in, out = exp.input[0], exp.output[0]
        if len(exp.input) > 1:
            shape_inputs.add(exp.input[1])
        for consumer in graph.node:
            for i, name in enumerate(consumer.input):
                if name == out:
                    consumer.input[i] = data_in
                    rewired += 1

    expand_ids = {id(n) for n in expands}
    kept = [n for n in graph.node if id(n) not in expand_ids]
    del graph.node[:]
    graph.node.extend(kept)

    still_used = {inp for n in graph.node for inp in n.input}
    kept_inits = [i for i in graph.initializer
                  if i.name not in shape_inputs or i.name in still_used]
    removed_inits = len(graph.initializer) - len(kept_inits)
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    del graph.value_info[:]  # force re-inference on load
    return len(expands), rewired, removed_inits


def drop_unreferenced_initializers(model) -> int:
    graph = model.graph
    used = {inp for n in graph.node for inp in n.input}
    used.update(vi.name for vi in graph.output)
    kept = [i for i in graph.initializer if i.name in used]
    removed = len(graph.initializer) - len(kept)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return removed


def apply_torq_deploy_postprocess(model, out_onnx: str | Path,
                                  token_id_lut: str | None = None,
                                  signed_int4: bool = True):
    out_onnx = Path(out_onnx)
    emb_path = out_onnx.with_name("token_embeddings.npy")

    shape = extract_lm_head_embeddings(model, emb_path)
    print(f"[emb ] lm_head dequantised -> {shape} bf16 -> {emb_path}")

    n_scale, n_cast = cast_dql_scale_to_bf16(model)
    print(f"[cast] scale f32->bf16: {n_scale} DQL, removed Cast(bf16): {n_cast}")

    packed, before, after = pack_dql_to_int4(model, signed=signed_int4)
    print(f"[pack] {packed} tensors -> {'INT4' if signed_int4 else 'UINT4'}: "
          f"{before / 2**20:.1f}MB -> {after / 2**20:.1f}MB")

    lut = resolve_token_id_lut(token_id_lut, out_onnx.parent)
    orig_n, n_trim = trim_lm_head_vocab(model, lut)
    print(f"[trim] lm_head vocab {orig_n} -> {n_trim}")

    n_exp, rewired, n_shape_init = remove_gqa_expand(model)
    print(f"[gqa ] removed Expand: {n_exp}, rewired consumers: {rewired}, "
          f"removed shape initializers: {n_shape_init}")

    print(f"[dead] removed unreferenced initializers: "
          f"{drop_unreferenced_initializers(model)}")
    return emb_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx_in", help="QDQ ONNX from safetensors_to_onnx_qdq.py")
    parser.add_argument("--out", required=True,
                        help="deployment ONNX; token_embeddings.npy is written beside it")
    parser.add_argument("--token-id-lut", default=None,
                        help="vocab trim LUT (.npy); when omitted the published "
                             "one is downloaded next to --out and cached")
    parser.add_argument("--unsigned-int4", action="store_true",
                        help="pack as UINT4 instead of the default signed INT4")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.onnx_in)
    emb_path = apply_torq_deploy_postprocess(
        model, args.out, args.token_id_lut, not args.unsigned_int4)
    onnx.save(model, args.out)
    print(f"[save] {args.out}\n[save] {emb_path}")


if __name__ == "__main__":
    main()
