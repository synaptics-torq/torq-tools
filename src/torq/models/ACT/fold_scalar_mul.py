#!/usr/bin/env python3
"""Fold a scalar `Mul(·, s)` back into the upstream matmul weight + bias.

The attention Q-scale (×1/√d = 0.125) appears as a scalar `Mul` a few shape-ops downstream of the
Q-projection: `MatMul(Wq) → Reshape → Add(bq) → Reshape → Transpose → Mul(0.125)`. Since `s` is a
scalar it commutes through the reshapes/transposes, so `(X@Wq + bq)·s = X@(Wq·s) + (bq·s)` — fold `s`
into the projection weight and bias and drop the Mul. `0.125 = 2⁻³` is exact in bf16 → lossless.

Walks back from each scalar `Mul` through single-consumer shape-only ops (Reshape/Transpose/Squeeze/
Unsqueeze/Identity), scales the bias of an intervening `Add` and the weight of the feeding
`MatMul`/`Gemm`, then removes the `Mul`. Single-consumer guard ensures no other use is affected.

Usage:  python fold_scalar_mul.py IN.onnx OUT.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = None

SHAPE_OPS = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}


def _np(t):
    if t.data_type == TensorProto.BFLOAT16:
        raw = t.raw_data or np.array(t.int32_data, np.uint16).tobytes()
        return np.frombuffer(raw, BF16).reshape(list(t.dims)).astype(np.float32)
    return numpy_helper.to_array(t).astype(np.float32)


def _set(t, arr):
    if t.data_type == TensorProto.BFLOAT16:
        t.raw_data = arr.astype(BF16).tobytes(); t.ClearField("int32_data")
    else:
        t.raw_data = arr.astype(helper.tensor_dtype_to_np_dtype(t.data_type)).tobytes()


def fold(model):
    g = model.graph
    all_nodes = list(g.node)
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    single = lambda t: len(cons.get(t, [])) == 1

    drop = []
    folded = 0
    for mul in [n for n in g.node if n.op_type == "Mul"]:
        sc = [i for i in mul.input if i in inits and _np(inits[i]).size == 1]
        act = [i for i in mul.input if i not in inits]
        if len(sc) != 1 or len(act) != 1:
            continue
        s = float(_np(inits[sc[0]]).reshape(-1)[0])
        # walk back through single-consumer shape-only ops, allowing one bias Add
        cur = act[0]
        bias_t = None
        ok = False
        for _ in range(8):
            if cur not in prod or not single(cur):
                break
            n = prod[cur]
            if n.op_type in SHAPE_OPS:
                cur = n.input[0]
            elif n.op_type == "Add" and any(i in inits for i in n.input) and bias_t is None:
                bias_t = next(i for i in n.input if i in inits)
                cur = next(i for i in n.input if i not in inits)
            elif n.op_type == "MatMul" and n.input[1] in inits:
                _set(inits[n.input[1]], _np(inits[n.input[1]]) * s); ok = True; break
            elif n.op_type == "Gemm" and n.input[1] in inits:
                _set(inits[n.input[1]], _np(inits[n.input[1]]) * s)
                if len(n.input) > 2 and n.input[2] in inits:
                    _set(inits[n.input[2]], _np(inits[n.input[2]]) * s)
                ok = True; break
            else:
                break
        if not ok:
            continue
        if bias_t is not None:
            _set(inits[bias_t], _np(inits[bias_t]) * s)
        # drop the Mul: rewire its consumers to its activation input
        for u in cons.get(mul.output[0], []):
            u.input[:] = [act[0] if x == mul.output[0] else x for x in u.input]
        for o in g.output:
            if o.name == mul.output[0]:
                o.name = act[0]
        drop.append(id(mul))
        folded += 1

    kept = [n for n in all_nodes if id(n) not in drop]
    g.ClearField("node"); g.node.extend(kept)
    print(f"folded {folded} scalar Mul into upstream matmul weight/bias")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python fold_scalar_mul.py IN.onnx OUT.onnx"); sys.exit(1)
    onnx.save(fold(onnx.load(sys.argv[1])), sys.argv[2])
    print("wrote", sys.argv[2])
