#!/usr/bin/env python3
"""Report tensors with unknown/symbolic dimensions in an ONNX model."""
import sys
import onnx
from onnx import shape_inference, TensorProto

path = sys.argv[1] if len(sys.argv) > 1 else "end2end.onnx"
model = onnx.load(path)
DT = {v: k for k, v in TensorProto.DataType.items()}

try:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
except Exception as e:
    print(f"[warn] strict inference failed: {e}\n")
    inferred = shape_inference.infer_shapes(model, strict_mode=False)


def dims(vi):
    """Return (list_of_dim_reprs, is_unknown, no_shape_at_all)."""
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return [], True, True
    out, unk = [], False
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            out.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
            unk = True
        else:
            out.append("?")
            unk = True
    return out, unk, False


def fmt(vi):
    ds, unk, noshape = dims(vi)
    dt = DT.get(vi.type.tensor_type.elem_type, "?")
    if noshape:
        return f"{dt}[<no shape>]", True
    return f"{dt}[{','.join(ds)}]", unk


# producer map: tensor name -> node
producer = {}
for n in inferred.graph.node:
    for o in n.output:
        producer[o] = n

print("=== graph inputs ===")
for vi in inferred.graph.input:
    s, unk = fmt(vi)
    print(f"  {'UNK ' if unk else '    '}{vi.name}: {s}")

print("\n=== graph outputs ===")
for vi in inferred.graph.output:
    s, unk = fmt(vi)
    print(f"  {'UNK ' if unk else '    '}{vi.name}: {s}")

print("\n=== intermediate tensors with unknown dims ===")
symbols = {}
count = 0
first = None
for vi in inferred.graph.value_info:
    s, unk = fmt(vi)
    if not unk:
        continue
    count += 1
    node = producer.get(vi.name)
    op = node.op_type if node else "?"
    nm = node.name if node else "?"
    if first is None:
        first = (vi.name, op, nm)
    print(f"  [{op:20s}] {vi.name}: {s}")
    ds, _, _ = dims(vi)
    for d in ds:
        if not d.isdigit():
            symbols.setdefault(d, []).append(vi.name)

print(f"\n=== summary ===")
total = len(inferred.graph.value_info)
print(f"  intermediate tensors : {total}")
print(f"  with unknown dims    : {count}")
if first:
    print(f"  first unknown        : {first[0]}  <- [{first[1]}] {first[2]}")
print(f"\n  distinct symbols ({len(symbols)}):")
for sym, users in sorted(symbols.items(), key=lambda kv: -len(kv[1])):
    print(f"    {sym:24s} {len(users):4d} tensors   (first: {users[0]})")
