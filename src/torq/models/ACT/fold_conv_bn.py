#!/usr/bin/env python3
"""Fold `Conv -> Mul(per-channel) -> Add(per-channel)` into a single Conv.

The ResNet backbone exports BatchNorm (in eval mode) as a bias-less Conv followed by a
per-output-channel Mul (scale) and Add (shift). These are algebraically a single conv:

    y = Add(Mul(Conv(x, W), s), b) = Conv(x, W')  with  W'[oc] = W[oc]*s[oc],  bias[oc] = b[oc]
    (if the conv already had a bias c:  bias[oc] = c[oc]*s[oc] + b[oc])

Folding removes 2 ops + a dispatch per conv (~40 nodes on ResNet-18) and is ~16% faster on
the NSS. Only folds when the Mul/Add second operands are per-channel CONSTANTS (so residual
Adds of two activations are left alone). Fold math is done in fp32 then cast back to the conv
weight's dtype (bf16), to minimize rounding.

Usage:  python fold_conv_bn.py IN.onnx OUT.onnx
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


def _to_f32(t):
    if t.data_type == TensorProto.BFLOAT16:
        raw = t.raw_data if t.raw_data else np.array(t.int32_data, np.uint16).tobytes()
        return np.frombuffer(raw, BF16).reshape(list(t.dims)).astype(np.float32)
    return numpy_helper.to_array(t).astype(np.float32)


def _from_f32(name, arr, dtype):
    if dtype == TensorProto.BFLOAT16:
        return numpy_helper.from_array(arr.astype(BF16), name=name)
    return numpy_helper.from_array(arr.astype(helper.tensor_dtype_to_np_dtype(dtype)), name=name)


def fold(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    all_nodes = list(g.node)
    inits = {i.name: i for i in g.initializer}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def only_consumer(t):
        c = cons.get(t, [])
        return c[0] if len(c) == 1 else None

    def per_channel_const(node, C):
        """Return (const_input_name, values[C]) if exactly one input is a const broadcastable to per-channel."""
        for a in node.input:
            if a in inits:
                v = _to_f32(inits[a]).reshape(-1)
                if v.size in (C, 1):
                    return a, (np.full(C, v[0]) if v.size == 1 else v)
        return None, None

    drop = set()
    new_inits = []
    folded = 0
    for conv in [n for n in g.node if n.op_type == "Conv"]:
        co = conv.output[0]
        mul = only_consumer(co)
        if mul is None or mul.op_type != "Mul":
            continue
        W = _to_f32(inits[conv.input[1]])      # [Cout,Cin,kh,kw]
        Cout = W.shape[0]
        s_name, s = per_channel_const(mul, Cout)
        if s is None:
            continue
        add = only_consumer(mul.output[0])
        if add is None or add.op_type != "Add":
            continue
        b_name, b = per_channel_const(add, Cout)
        if b is None:
            continue

        dt = inits[conv.input[1]].data_type
        Wf = W * s.reshape(Cout, 1, 1, 1)
        c = _to_f32(inits[conv.input[2]]).reshape(-1) if len(conv.input) > 2 else np.zeros(Cout, np.float32)
        bias = c * s + b

        wname, bname = f"{conv.name or co}_Wfold", f"{conv.name or co}_bfold"
        new_inits += [_from_f32(wname, Wf, dt), _from_f32(bname, bias, dt)]
        conv.input[1] = wname
        if len(conv.input) > 2:
            conv.input[2] = bname
        else:
            conv.input.append(bname)
        conv.output[0] = add.output[0]   # take over the Add's output name
        drop.update((id(mul), id(add)))
        folded += 1

    # rebuild node list preserving order, dropping the folded Mul/Add (conv nodes already mutated in place)
    kept = [n for n in all_nodes if id(n) not in drop]
    g.ClearField("node"); g.node.extend(kept)
    g.initializer.extend(new_inits)
    print(f"folded {folded} Conv->Mul->Add into Conv")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python fold_conv_bn.py IN.onnx OUT.onnx"); sys.exit(1)
    m = onnx.load(sys.argv[1])
    onnx.save(fold(m), sys.argv[2])
    print("wrote", sys.argv[2])
