#!/usr/bin/env python3
"""Constant-fold an ONNX graph: precompute every all-constant subgraph into initializers.

The transformer pieces carry large constant islands the exporter left un-evaluated — sinusoidal
positional embeddings (`Sin/Cos/Unsqueeze/Concat` → `stack_*`), and the per-layer QKV weight prep
(`Split`/`Transpose`/`Gemm` of constant weights). These depend only on initializers, so they can be
computed once offline and replaced by a single constant each, deleting dozens of runtime ops/dispatches.

Done in numpy (ORT can't run bf16 on CPU): read initializers upcast to fp32, evaluate the constant
island, store each fold-boundary tensor back as an initializer (bf16 for float, int64 for integer),
then drop the dead nodes and prune now-unused initializers.

Usage:  python const_fold.py IN.onnx OUT.onnx
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


def _to_np(t):
    if t.data_type == TensorProto.BFLOAT16:
        raw = t.raw_data if t.raw_data else np.array(t.int32_data, np.uint16).tobytes()
        return np.frombuffer(raw, BF16).reshape(list(t.dims)).astype(np.float32)
    return numpy_helper.to_array(t).astype(np.float32) if t.data_type in (
        TensorProto.FLOAT, TensorProto.FLOAT16) else numpy_helper.to_array(t)


def _attr(n, k, d=None):
    for a in n.attribute:
        if a.name == k:
            if a.type == a.INTS: return list(a.ints)
            if a.type == a.INT: return a.i
            if a.type == a.FLOAT: return a.f
            if a.type == a.TENSOR: return _to_np(a.t)
    return d


def fold(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    ginp = {i.name for i in g.input}
    prod = {o: n for n in g.node for o in n.output}

    # constant closure
    const = set(inits)
    ch = True
    while ch:
        ch = False
        for n in g.node:
            if all((i in const or i == "") for i in n.input) and not all(o in const for o in n.output):
                const.update(n.output); ch = True

    cache = {nm: _to_np(t) for nm, t in inits.items()}

    def ev(name):
        if name in cache: return cache[name]
        n = prod[name]; t = n.op_type
        ins = [ev(i) if i != "" else None for i in n.input]
        if t == "Constant": r = _attr(n, "value")
        elif t == "Identity": r = ins[0]
        elif t == "Cast": r = ins[0]
        elif t == "Transpose": r = np.transpose(ins[0], _attr(n, "perm"))
        elif t == "Reshape": r = ins[0].reshape([int(x) if x != 0 else ins[0].shape[i] for i, x in enumerate(ins[1].astype(int))])
        elif t == "Unsqueeze":
            ax = (ins[1].astype(int).tolist() if len(ins) > 1 and ins[1] is not None else _attr(n, "axes"))
            r = ins[0]
            for a in sorted(ax): r = np.expand_dims(r, a)
        elif t == "Squeeze":
            ax = (ins[1].astype(int).tolist() if len(ins) > 1 and ins[1] is not None else _attr(n, "axes", None))
            r = np.squeeze(ins[0], tuple(ax) if ax else None)
        elif t == "Concat": r = np.concatenate(ins, axis=_attr(n, "axis"))
        elif t == "Sin": r = np.sin(ins[0])
        elif t == "Cos": r = np.cos(ins[0])
        elif t == "Relu": r = np.maximum(ins[0], 0)
        elif t == "Sqrt": r = np.sqrt(ins[0])
        elif t == "Pow": r = np.power(ins[0], ins[1])
        elif t == "Exp": r = np.exp(ins[0])
        elif t == "Reciprocal": r = 1.0 / ins[0]
        elif t == "Erf":
            import math
            r = np.vectorize(math.erf)(ins[0]).astype(np.float32)
        elif t == "Tanh": r = np.tanh(ins[0])
        elif t == "Softmax":
            ax = _attr(n, "axis", -1); x = ins[0] - ins[0].max(ax, keepdims=True)
            e = np.exp(x); r = e / e.sum(ax, keepdims=True)
        elif t == "ReduceMean":
            ax = (ins[1].astype(int).tolist() if len(ins) > 1 and ins[1] is not None else _attr(n, "axes"))
            r = ins[0].mean(tuple(ax), keepdims=bool(_attr(n, "keepdims", 1)))
        elif t == "Pad":
            pads = ins[1].astype(int); cval = float(ins[2]) if len(ins) > 2 and ins[2] is not None else 0.0
            nd = ins[0].ndim; r = np.pad(ins[0], [(pads[i], pads[i + nd]) for i in range(nd)], constant_values=cval)
        elif t in ("Add", "Sub", "Mul", "Div"):
            r = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[t](ins[0], ins[1])
        elif t == "MatMul": r = np.matmul(ins[0], ins[1])
        elif t == "Gemm":
            A, B = ins[0], ins[1]
            if _attr(n, "transA", 0): A = A.T
            if _attr(n, "transB", 0): B = B.T
            r = _attr(n, "alpha", 1.0) * (A @ B)
            if len(ins) > 2 and ins[2] is not None: r = r + _attr(n, "beta", 1.0) * ins[2]
        elif t == "Slice":
            data = ins[0]; starts = ins[1].astype(int); ends = ins[2].astype(int)
            axes = ins[3].astype(int) if len(ins) > 3 and ins[3] is not None else np.arange(len(starts))
            steps = ins[4].astype(int) if len(ins) > 4 and ins[4] is not None else np.ones_like(starts)
            sl = [slice(None)] * data.ndim
            for ax, st, en, sp in zip(axes, starts, ends, steps): sl[ax] = slice(st, en, sp)
            r = data[tuple(sl)]
        elif t == "Split":
            ax = _attr(n, "axis", 0)
            parts = (np.split(ins[0], np.cumsum(ins[1].astype(int))[:-1], ax) if len(ins) > 1 and ins[1] is not None
                     else np.split(ins[0], len(n.output), ax))
            for o, p in zip(n.output, parts): cache[o] = p
            return cache[name]
        else:
            raise RuntimeError(f"const_fold: unhandled op {t}")
        cache[name] = r
        return r

    cons = {}
    for n in g.node:
        for i in n.input: cons.setdefault(i, []).append(n)
    # boundary = constant tensor consumed by a NON-constant node (or a graph output)
    gout = {o.name for o in g.output}
    boundary = [t for t in const if t not in inits and
                (t in gout or any(not all(o in const for o in u.output) for u in cons.get(t, [])))]

    new_inits = []
    for t in boundary:
        v = ev(t)
        if np.issubdtype(np.asarray(v).dtype, np.integer):
            new_inits.append(numpy_helper.from_array(np.asarray(v).astype(np.int64), name=t))
        else:
            new_inits.append(numpy_helper.from_array(np.asarray(v).astype(BF16), name=t))

    # drop all constant-island nodes (their outputs are now initializers or dead)
    kept = [n for n in g.node if not (all((i in const or i == "") for i in n.input))]
    g.ClearField("node"); g.node.extend(kept)
    g.initializer.extend(new_inits)

    # prune unused initializers
    used = {i for n in g.node for i in n.input}
    keep_init = [i for i in g.initializer if i.name in used]
    n_pruned = len(g.initializer) - len(keep_init)
    g.ClearField("initializer"); g.initializer.extend(keep_init)
    print(f"folded {len(const) - len(inits)} constant tensors -> {len(boundary)} initializers; "
          f"nodes {len(prod) and len(kept)} kept; pruned {n_pruned} dead initializers")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python const_fold.py IN.onnx OUT.onnx"); sys.exit(1)
    m = onnx.load(sys.argv[1])
    onnx.save(fold(m), sys.argv[2])
    print("wrote", sys.argv[2])
