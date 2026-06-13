#!/usr/bin/env python3
"""Surgery — decompose onnx.LayerNormalization into NSS-tileable primitives ("lnnss").

Adapted from work-dev/maxpool_tiling_repro/decompose_layernorm_for_nss.py.

WHY: the NSS executor cannot tile a last-axis-reduction linalg.generic (the
mean/variance reduce that LayerNormalization lowers to) -- it bails with
"result is not accessed using a permuted projection". Re-expressing each
reduction as a MatMul against a constant ones-vector routes it through the
NSS-supported contraction-tiling path instead. The normalize step is a single
broadcast Div (a standalone Reciprocal lowers to an unsupported divf on NSS).

For X[..., N] with scale S[N], bias B[N], epsilon eps (biased variance, == ONNX):
    sum   = MatMul(X,   ones[N,1]) ; mean = sum  * (1/N)
    xc    = X - mean
    sumsq = MatMul(xc*xc, ones[N,1]) ; var = sumsq * (1/N)
    std   = Sqrt(var + eps)
    Y     = Div(xc, std) * S + B

BONUS: the two ones-matmuls are themselves [seq,1,N]@[N,1] shaped, so running
wrap_matmuls.py afterward (or first) keeps them on the fast 2D path too.

Usage:  python decompose_layernorm_nss.py IN.onnx OUT.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = None


def _const(name, arr, dtype):
    if dtype == TensorProto.BFLOAT16:
        return numpy_helper.from_array(arr.astype(BF16), name=name)
    return numpy_helper.from_array(arr.astype(helper.tensor_dtype_to_np_dtype(dtype)), name=name)


def decompose(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)
    g = model.graph
    vi = {v.name: v.type.tensor_type for v in g.value_info}
    for c in (g.input, g.output):
        for t in c:
            vi[t.name] = t.type.tensor_type
    inits = {i.name: i for i in g.initializer}

    ln = [n for n in g.node if n.op_type == "LayerNormalization"]
    if not ln:
        print("no LayerNormalization nodes")
        return model

    keep = [n for n in g.node if n.op_type != "LayerNormalization"]
    new, added = [], []
    # shared zero constant for the distinct-tensor square trick (see below)
    dt0 = inits[ln[0].input[1]].data_type if ln[0].input[1] in inits else TensorProto.BFLOAT16
    added.append(_const("ln_nss_zero", np.zeros((1, 1, 1), np.float32), dt0))
    i = 0
    for n in ln:
        X, S = n.input[0], n.input[1]
        Bname = n.input[2] if len(n.input) > 2 else None
        Y = n.output[0]
        dt = inits[S].data_type if S in inits else TensorProto.BFLOAT16
        N = list(inits[S].dims)[-1]
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        p = f"ln{i}_"
        # reduce-by-mean as MatMul against a constant (1/N)-vector: folds the (1/N) scale
        # straight into the matmul weight, so there is NO separate `Mul` after each reduce.
        # 1/N is exact in bf16 when N is a power of two (1/512 = 2^-9). mean/var come out
        # of the MatMul directly: mean path = MatMul; variance path = MatMul -> Add(eps).
        onesN = f"{p}onesN"
        added.append(_const(onesN, np.full((N, 1), 1.0 / N, np.float32), dt))
        added.append(_const(f"{p}eps", np.full((1, 1, 1), eps, np.float32), dt))
        new += [
            helper.make_node("MatMul", [X, onesN], [f"{p}mean"]),
            helper.make_node("Sub", [X, f"{p}mean"], [f"{p}xc"]),
            # square as Mul of two DISTINCT tensors (xc, xc+0). A self-multiply Mul(xc,xc)
            # is mishandled on the NSS in bf16 (-> ~3.4e38). xc2 = xc + 0 forces a proper
            # binary elementwise multiply. (Matches the working lnnssfix surgery.)
            helper.make_node("Add", [f"{p}xc", "ln_nss_zero"], [f"{p}xc2"]),
            helper.make_node("Mul", [f"{p}xc", f"{p}xc2"], [f"{p}sq"]),
            helper.make_node("MatMul", [f"{p}sq", onesN], [f"{p}var"]),
            helper.make_node("Add", [f"{p}var", f"{p}eps"], [f"{p}vare"]),
            helper.make_node("Sqrt", [f"{p}vare"], [f"{p}std"]),
            helper.make_node("Div", [f"{p}xc", f"{p}std"], [f"{p}norm"]),
            helper.make_node("Mul", [f"{p}norm", S], [f"{p}scaled"] if Bname else [Y]),
        ]
        if Bname:
            new.append(helper.make_node("Add", [f"{p}scaled", Bname], [Y]))
        i += 1

    g.initializer.extend(added)
    # topo-sort (ones-matmuls feed later ops)
    avail = {x.name for x in g.initializer} | {x.name for x in g.input}
    nodes = keep + new
    ordered, pend = [], list(nodes)
    while pend:
        prog = False
        for nd in list(pend):
            if all(x in avail or x == "" for x in nd.input):
                ordered.append(nd); avail.update(nd.output); pend.remove(nd); prog = True
        if not prog:
            raise RuntimeError("topo stuck")
    g.ClearField("node"); g.node.extend(ordered)
    print(f"decomposed {len(ln)} LayerNorm -> matmul-by-ones")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python decompose_layernorm_nss.py IN.onnx OUT.onnx"); sys.exit(1)
    onnx.save(decompose(onnx.load(sys.argv[1])), sys.argv[2])
    print("wrote", sys.argv[2])
