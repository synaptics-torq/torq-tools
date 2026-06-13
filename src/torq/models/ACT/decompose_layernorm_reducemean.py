#!/usr/bin/env python3
"""EXPERIMENT variant of decompose_layernorm_nss.py.

Same decomposition, but the mean/variance reductions use onnx.ReduceMean instead
of MatMul-against-a-ones-vector. This is the form lnnss was created to AVOID -- the
NSS reportedly cannot tile the last-axis reduce that ReduceMean lowers to
("result is not accessed using a permuted projection"). This script lets us
empirically re-check that on the current torq.

    mean  = ReduceMean(X, axes=[-1])
    xc    = X - mean
    var   = ReduceMean(xc*xc, axes=[-1])
    std   = Sqrt(var + eps)
    Y     = Div(xc, std) * S + B

Usage:  python decompose_layernorm_reducemean.py IN.onnx OUT.onnx
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
    inits = {i.name: i for i in g.initializer}
    ln = [n for n in g.node if n.op_type == "LayerNormalization"]
    if not ln:
        print("no LayerNormalization nodes")
        return model

    # opset >= 18 takes ReduceMean axes as an input tensor (not an attribute)
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 22)
    axes_as_input = opset >= 18
    added = []
    if axes_as_input:
        added.append(numpy_helper.from_array(np.array([-1], np.int64), name="ln_rm_axes"))
    # distinct-tensor square: Mul(xc, xc+0), NOT Mul(xc,xc) -- self-multiply into a reduce
    # miscompiles on the NSS in bf16 (see gh-issues/self-multiply-into-reduce-miscompile/).
    dt0 = inits[ln[0].input[1]].data_type if ln[0].input[1] in inits else TensorProto.BFLOAT16
    added.append(_const("ln_rm_zero", np.zeros((1, 1, 1), np.float32), dt0))

    def reduce_mean(inp, out):
        if axes_as_input:
            return helper.make_node("ReduceMean", [inp, "ln_rm_axes"], [out], keepdims=1)
        return helper.make_node("ReduceMean", [inp], [out], axes=[-1], keepdims=1)

    keep = [n for n in g.node if n.op_type != "LayerNormalization"]
    new = []
    for i, n in enumerate(ln):
        X, S = n.input[0], n.input[1]
        Bname = n.input[2] if len(n.input) > 2 else None
        Y = n.output[0]
        dt = inits[S].data_type if S in inits else TensorProto.BFLOAT16
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        p = f"lnrm{i}_"
        added.append(_const(f"{p}eps", np.full((1, 1, 1), eps, np.float32), dt))
        new += [
            reduce_mean(X, f"{p}mean"),
            helper.make_node("Sub", [X, f"{p}mean"], [f"{p}xc"]),
            helper.make_node("Add", [f"{p}xc", "ln_rm_zero"], [f"{p}xc2"]),
            helper.make_node("Mul", [f"{p}xc", f"{p}xc2"], [f"{p}sq"]),
            reduce_mean(f"{p}sq", f"{p}var"),
            helper.make_node("Add", [f"{p}var", f"{p}eps"], [f"{p}vare"]),
            helper.make_node("Sqrt", [f"{p}vare"], [f"{p}std"]),
            helper.make_node("Div", [f"{p}xc", f"{p}std"], [f"{p}norm"]),
            helper.make_node("Mul", [f"{p}norm", S], [f"{p}scaled"] if Bname else [Y]),
        ]
        if Bname:
            new.append(helper.make_node("Add", [f"{p}scaled", Bname], [Y]))

    g.initializer.extend(added)
    avail = {x.name for x in g.initializer} | {x.name for x in g.input}
    ordered, pend = [], keep + new
    while pend:
        prog = False
        for nd in list(pend):
            if all(x in avail or x == "" for x in nd.input):
                ordered.append(nd); avail.update(nd.output); pend.remove(nd); prog = True
        if not prog:
            raise RuntimeError("topo stuck")
    g.ClearField("node"); g.node.extend(ordered)
    print(f"decomposed {len(ln)} LayerNorm -> ReduceMean form")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python decompose_layernorm_reducemean.py IN.onnx OUT.onnx"); sys.exit(1)
    onnx.save(decompose(onnx.load(sys.argv[1])), sys.argv[2])
    print("wrote", sys.argv[2])
