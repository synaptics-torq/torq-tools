#!/usr/bin/env python3
"""Surgery — wrap batched-with-size-1 matmuls into plain 2D matmuls.

This is the single most important transformer surgery. It covers BOTH the
historical "ffnwrap" (the 10 FFN matmuls) AND "projwrap" (the ~42 attention
Q/K/V/output projections) in one pass.

WHY: a matmul of shape [seq, 1, K] @ [K, N] (a constant weight, with a spurious
size-1 batch axis from the [seq, 1, d_model] residual stream) lowers on the NSS
to a [seq, K, N] *broadcast* materialization — catastrophically slow / huge.
Stripping the size-1 axis to a plain [seq, K] @ [K, N] removes the broadcast and
lets torq pick its canonical matmul kernel.

    Reshape [seq,1,K] -> [seq,K]   ;   MatMul [seq,K] @ [K,N]   ;   Reshape -> [seq,1,N]

We target exactly: MatMul whose input0 is rank-3 with middle dim == 1 and whose
weight (input1) is a rank-2 initializer [K, N]. This deliberately EXCLUDES the
attention score matmuls (Q@K^T, scores@V): those are activation x activation
(weight is not an initializer) and middle dim != 1, so they are left untouched
(they stay as bf16 batched matmuls, which compile fine).

Usage:  python wrap_matmuls.py IN.onnx OUT.onnx
"""
import sys
import onnx
from onnx import helper, shape_inference


def wrap(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in g.value_info}
    for c in (g.input, g.output):
        for t in c:
            vi[t.name] = [d.dim_value for d in t.type.tensor_type.shape.dim]
    for nm, t in inits.items():
        vi[nm] = list(t.dims)

    # Target: input0 rank-3 with a spurious middle dim == 1, weight rank-2.
    # The weight need NOT be an initializer -- the attention projection weights are
    # computed constants (Transpose(Split(...))). The broadcast pathology is about
    # input0's size-1 axis, not the weight's origin. The middle==1 check excludes the
    # attention SCORE matmuls (input0 [8,seq,64], middle != 1), which we leave as bf16.
    targets = []
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        a, w = n.input
        ash, wsh = vi.get(a, []), vi.get(w, [])
        if len(wsh) == 2 and len(ash) == 3 and ash[1] == 1:
            targets.append(n)

    new_nodes = [n for n in g.node if n not in targets]
    ctr = 0
    for n in targets:
        a, w = n.input
        out = n.output[0]
        B, _, K = vi[a]
        N = vi[w][1]
        p = f"wrp{ctr}_"
        new_nodes += [
            helper.make_node("Reshape", [a, f"{p}s0"], [f"{p}r0"]),
            helper.make_node("MatMul", [f"{p}r0", w], [f"{p}mm"]),
            helper.make_node("Reshape", [f"{p}mm", f"{p}s1"], [out]),
        ]
        g.initializer.append(helper.make_tensor(f"{p}s0", onnx.TensorProto.INT64, [2], [B, K]))
        g.initializer.append(helper.make_tensor(f"{p}s1", onnx.TensorProto.INT64, [3], [B, 1, N]))
        ctr += 1

    # topo-sort
    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    ordered, pend = [], list(new_nodes)
    while pend:
        prog = False
        for nd in list(pend):
            if all(x in avail or x == "" for x in nd.input):
                ordered.append(nd)
                avail.update(nd.output)
                pend.remove(nd)
                prog = True
        if not prog:
            raise RuntimeError("topo sort stuck (cycle?)")
    g.ClearField("node")
    g.node.extend(ordered)
    print(f"wrapped {ctr} matmuls -> 2D")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python wrap_matmuls.py IN.onnx OUT.onnx")
        sys.exit(1)
    onnx.save(wrap(onnx.load(sys.argv[1])), sys.argv[2])
    print("wrote", sys.argv[2])
