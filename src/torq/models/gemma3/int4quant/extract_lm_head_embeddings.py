import argparse
from pathlib import Path
import struct

import numpy as np


def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def parse_fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, pos = read_varint(buf, pos)
            yield field, wire_type, value
        elif wire_type == 1:
            value = buf[pos:pos + 8]
            pos += 8
            yield field, wire_type, value
        elif wire_type == 2:
            size, pos = read_varint(buf, pos)
            value = buf[pos:pos + size]
            pos += size
            yield field, wire_type, value
        elif wire_type == 5:
            value = buf[pos:pos + 4]
            pos += 4
            yield field, wire_type, value
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def get_graph(model_buf):
    for field, wire_type, value in parse_fields(model_buf):
        if field == 7 and wire_type == 2:
            return value
    raise ValueError("ONNX graph field was not found")


def parse_tensor(tensor_buf):
    dims = []
    dtype = None
    name = None
    raw = None
    int64_data = []
    float_data = []

    for field, wire_type, value in parse_fields(tensor_buf):
        if field == 1:
            if wire_type == 0:
                dims.append(value)
            elif wire_type == 2:
                pos = 0
                while pos < len(value):
                    item, pos = read_varint(value, pos)
                    dims.append(item)
        elif field == 2:
            dtype = value
        elif field == 4:
            if wire_type == 5:
                float_data.append(struct.unpack("<f", value)[0])
            elif wire_type == 2:
                float_data.extend(struct.unpack("<" + "f" * (len(value) // 4), value))
        elif field == 7:
            if wire_type == 0:
                int64_data.append(value)
            elif wire_type == 2:
                pos = 0
                while pos < len(value):
                    item, pos = read_varint(value, pos)
                    int64_data.append(item)
        elif field == 8:
            name = value.decode("utf-8")
        elif field == 9:
            raw = value

    return {
        "name": name,
        "dtype": dtype,
        "dims": tuple(dims),
        "raw": raw,
        "int64_data": int64_data,
        "float_data": float_data,
    }


def parse_node(node_buf):
    inputs = []
    outputs = []
    name = None
    op_type = None
    for field, wire_type, value in parse_fields(node_buf):
        if field == 1:
            inputs.append(value.decode("utf-8"))
        elif field == 2:
            outputs.append(value.decode("utf-8"))
        elif field == 3:
            name = value.decode("utf-8")
        elif field == 4:
            op_type = value.decode("utf-8")
    return {"inputs": inputs, "outputs": outputs, "name": name, "op_type": op_type}


def load_graph_parts(path):
    graph = get_graph(Path(path).read_bytes())
    tensors = {}
    nodes = []
    for field, wire_type, value in parse_fields(graph):
        if field == 1 and wire_type == 2:
            nodes.append(parse_node(value))
        elif field == 5 and wire_type == 2:
            tensor = parse_tensor(value)
            if tensor["name"]:
                tensors[tensor["name"]] = tensor
    return nodes, tensors


def tensor_array(tensor):
    dtype = tensor["dtype"]
    dims = tensor["dims"]
    raw = tensor["raw"]

    if dtype == 1:
        if raw is not None:
            return np.frombuffer(raw, dtype=np.float32).reshape(dims)
        return np.asarray(tensor["float_data"], dtype=np.float32).reshape(dims)
    if dtype == 3:
        return np.frombuffer(raw, dtype=np.int8).reshape(dims)
    if dtype == 7:
        if raw is not None:
            return np.frombuffer(raw, dtype=np.int64).reshape(dims)
        return np.asarray(tensor["int64_data"], dtype=np.int64).reshape(dims)

    raise ValueError(f"Unsupported TensorProto dtype {dtype} for {tensor['name']}")


def find_lm_head_dql(nodes):
    by_output = {output: node for node in nodes for output in node["outputs"]}
    lm_head = next(
        (
            node for node in nodes
            if node["op_type"] == "MatMul" and node["name"] and "lm_head" in node["name"]
        ),
        None,
    )
    if lm_head is None:
        raise ValueError("Could not find lm_head MatMul node")

    weight = lm_head["inputs"][1]
    cast = by_output.get(weight)
    reshape = by_output.get(cast["inputs"][0] if cast and cast["op_type"] == "Cast" else weight)
    if reshape is None or reshape["op_type"] != "Reshape":
        raise ValueError("Could not trace lm_head weight back to Reshape")

    dql = by_output.get(reshape["inputs"][0])
    if dql is None or dql["op_type"] != "DequantizeLinear":
        raise ValueError("Could not trace lm_head weight back to DequantizeLinear")

    return dql, reshape


def write_bf16_npy(path, array):
    bits = array.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    bf16 = (rounded >> 16).astype(np.uint16)

    shape = bf16.shape
    header = {
        "descr": "|V2",
        "fortran_order": False,
        "shape": shape,
    }
    header_text = str(header)
    header_bytes = (header_text + " " * (16 - ((10 + len(header_text) + 1) % 16)) + "\n").encode("latin1")

    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        f.write(bf16.tobytes(order="C"))


def extract_lm_head_embeddings(onnx_path, output_path, dtype):
    nodes, tensors = load_graph_parts(onnx_path)
    dql, reshape = find_lm_head_dql(nodes)

    q = tensor_array(tensors[dql["inputs"][0]])
    scale = tensor_array(tensors[dql["inputs"][1]])
    zp = tensor_array(tensors[dql["inputs"][2]])
    target_shape = tensor_array(tensors[reshape["inputs"][1]]).astype(np.int64)

    dequant = (q.astype(np.float32) - zp.reshape(-1, 1).astype(np.float32)) * scale.reshape(-1, 1)
    lm_head = dequant.reshape(tuple(target_shape.tolist()))
    embeddings = np.ascontiguousarray(lm_head.T)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "fp32":
        np.save(output_path, embeddings.astype(np.float32, copy=False))
    elif dtype == "bf16":
        write_bf16_npy(output_path, embeddings.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported output dtype: {dtype}")

    print(f"saved {output_path} shape={embeddings.shape} dtype={dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract tied token embeddings from a quantized lm_head inside an ONNX model."
    )
    parser.add_argument("--onnx", default="gptq_int4_out_feature.onnx", help="Input ONNX model containing quantized lm_head")
    parser.add_argument(
        "-o",
        "--output",
        default="gptq_int4_out_feature.token_embeddings.npy",
        help="Output token_embeddings.npy path",
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    extract_lm_head_embeddings(args.onnx, args.output, args.dtype)


if __name__ == "__main__":
    main()
