# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Graph edits that make PP-OCRv6-tiny compilable: freeze dynamic shapes, cascade the
detector's wide nearest-Resizes into 2x steps, and split its ConvTranspose ops along H.
The two detector rewrites are numerically exact (see issue #2236 in torq-compiler-dev)."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _shapes(graph):
    out = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if all(dims): out[v.name] = dims
    return out


def freeze(model: onnx.ModelProto, shape: list[int]) -> onnx.ModelProto:
    """Pin the input to ``shape`` and constant-fold the Resize size computations."""
    import onnxoptimizer
    from onnxsim import simplify

    for d, v in zip(model.graph.input[0].type.tensor_type.shape.dim, shape): d.dim_value = v
    for o in model.graph.output:
        for d in o.type.tensor_type.shape.dim:
            d.ClearField("dim_value"); d.dim_param = "?"
    # Constant-fold only (pins the Resize sizes); onnxsim's optimizer folds GELU's 0.5
    # multiply into the next conv, a form torq mis-tiles at wide widths. The whitelist
    # reproduces the reference exports: bias fusion and cleanup, nothing else.
    model, ok = simplify(model, overwrite_input_shapes={model.graph.input[0].name: shape}, perform_optimization=False)
    assert ok, "onnx-simplifier failed"
    model = onnxoptimizer.optimize(model, ["eliminate_identity", "eliminate_deadend", "fuse_add_bias_into_conv"])
    return onnx.shape_inference.infer_shapes(model)


def cascade_resizes(model: onnx.ModelProto) -> int:
    """Rewrite scale>2 nearest Resizes as chains of 2x resizes (floor(y/2^k) == floor(y/s))."""
    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    shapes = _shapes(graph)
    new_nodes, n = [], 0
    for node in graph.node:
        scales = next((init[x] for x in node.input if x in init and init[x].size == 4), None)
        if node.op_type != "Resize" or scales is None or float(scales[2]) <= 2.0 or scales[2] != scales[3]:
            new_nodes.append(node); continue
        factor, in_shape, cur = int(scales[2]), shapes.get(node.input[0]), node.input[0]
        steps = factor.bit_length() - 1
        assert 1 << steps == factor, f"scale {factor} is not a power of two"
        scales_name = next(x for x in node.input if x in init and init[x].size == 4)
        two = _const(graph, f"{node.name}_scales2", np.array([1, 1, 2, 2], np.float32))
        for step in range(steps):
            out = node.output[0] if step == steps - 1 else f"{node.name}_step{step}"
            inputs = [cur if i == 0 else (two if x == scales_name else x) for i, x in enumerate(node.input)]
            new = helper.make_node("Resize", inputs, [out], name=f"{node.name}_x2_{step}")
            new.attribute.extend(node.attribute)
            new_nodes.append(new)
            if step < steps - 1 and in_shape:
                h, w = in_shape[2] << (step + 1), in_shape[3] << (step + 1)
                graph.value_info.append(helper.make_tensor_value_info(out, graph.input[0].type.tensor_type.elem_type, [in_shape[0], in_shape[1], h, w]))
            cur = out
        n += 1
    del graph.node[:]; graph.node.extend(new_nodes)
    return n


def _const(graph, name, arr):
    graph.initializer.append(numpy_helper.from_array(arr, name))
    return name


def split_convtranspose(model: onnx.ModelProto, budget: int = 150_000) -> int:
    """Split k2x2/s2x2/pad0 ConvTransposes along H (non-overlapping, hence exact) so each
    slice's input and output stay under ``budget`` bytes."""
    graph = model.graph
    shapes, init = _shapes(graph), {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    elem = graph.input[0].type.tensor_type.elem_type
    new_nodes, n_split = [], 0
    for node in graph.node:
        if node.op_type != "ConvTranspose" or len(node.input) < 2:
            new_nodes.append(node); continue
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in node.attribute}
        in_shape, w = shapes.get(node.input[0]), init.get(node.input[1])
        if (attrs.get("kernel_shape") != [2, 2] or attrs.get("strides") != [2, 2]
                or attrs.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0] or in_shape is None or w is None):
            new_nodes.append(node); continue
        n, c_in, h, width = in_shape
        c_out = w.shape[1]
        fits = lambda hs: max(n * c_in * hs * width, n * c_out * 4 * hs * width) * 2 <= budget
        chunks = next((c for c in range(1, h + 1) if h % c == 0 and fits(h // c)), None)
        if chunks in (None, 1):
            new_nodes.append(node); continue
        hs, outs = h // chunks, []
        axes = _const(graph, f"{node.name}_axes", np.array([2], np.int64))
        for k in range(chunks):
            s = _const(graph, f"{node.name}_s{k}", np.array([k * hs], np.int64))
            e = _const(graph, f"{node.name}_e{k}", np.array([(k + 1) * hs], np.int64))
            sl, ct = f"{node.name}_in{k}", f"{node.name}_out{k}"
            new_nodes.append(helper.make_node("Slice", [node.input[0], s, e, axes], [sl], name=f"{node.name}_sl{k}"))
            graph.value_info.append(helper.make_tensor_value_info(sl, elem, [n, c_in, hs, width]))
            new = helper.make_node("ConvTranspose", [sl] + list(node.input[1:]), [ct], name=f"{node.name}_ct{k}")
            new.attribute.extend(node.attribute)
            new_nodes.append(new)
            graph.value_info.append(helper.make_tensor_value_info(ct, elem, [n, c_out, 2 * hs, 2 * width]))
            outs.append(ct)
        new_nodes.append(helper.make_node("Concat", outs, [node.output[0]], axis=2, name=f"{node.name}_cat"))
        n_split += 1
    del graph.node[:]; graph.node.extend(new_nodes)
    return n_split


def fold_bn(model: onnx.ModelProto) -> int:
    """Fold each BatchNormalization into its producing Conv (exact, fp32)."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    arr = lambda n: numpy_helper.to_array(init[n])
    by_out = {n.output[0]: n for n in g.node}
    n_fold = 0
    for bn in [n for n in g.node if n.op_type == "BatchNormalization"]:
        conv = by_out.get(bn.input[0])
        while conv is not None and conv.op_type in ("Squeeze", "Unsqueeze", "Reshape"): conv = by_out.get(conv.input[0])
        if conv is None or conv.op_type != "Conv" or conv.input[1] not in init: continue
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        a = arr(bn.input[1]) / np.sqrt(arr(bn.input[4]) + eps)
        b = arr(bn.input[2]) - arr(bn.input[3]) * a
        w = arr(conv.input[1])
        init[conv.input[1]].CopyFrom(numpy_helper.from_array((w * a.reshape([-1] + [1] * (w.ndim - 1))).astype(w.dtype), conv.input[1]))
        if len(conv.input) > 2 and conv.input[2] in init:
            init[conv.input[2]].CopyFrom(numpy_helper.from_array((arr(conv.input[2]) * a + b).astype(w.dtype), conv.input[2]))
        else:
            conv.input.append(_const(g, conv.name.replace(".", "_") + "_foldbias", b.astype(w.dtype)))
        for n in g.node: n.input[:] = [bn.input[0] if x == bn.output[0] else x for x in n.input]
        g.node.remove(bn)
        n_fold += 1
    return n_fold


def decompose_avgpool(model: onnx.ModelProto) -> int:
    """Replace AvgPool k[H,2]/s[H,2] with ReduceMean(H) + pairwise ReduceMean(W) reshapes."""
    g = model.graph
    shapes = _shapes(g)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    n_dec = 0
    for ap in [n for n in g.node if n.op_type == "AveragePool"]:
        attrs = {a.name: list(a.ints) for a in ap.attribute if a.ints}
        s = shapes.get(ap.input[0])
        if s is None or attrs.get("kernel_shape") != [s[2], 2] or attrs.get("strides") != [s[2], 2]: continue
        n, c, h, t = s
        p = ap.name.replace(".", "_")
        rm = lambda inp, ax, out: (helper.make_node("ReduceMean", [inp, _const(g, out + "_ax", np.array([ax], np.int64))], [out], keepdims=1)
                                   if opset >= 18 else helper.make_node("ReduceMean", [inp], [out], axes=[ax], keepdims=1))
        s1 = _const(g, p + "_s1", np.array([n, c, t // 2, 2], np.int64))
        s2 = _const(g, p + "_s2", np.array([n, c, 1, t // 2], np.int64))
        ns = [rm(ap.input[0], 2, p + "_h"), helper.make_node("Reshape", [p + "_h", s1], [p + "_r1"]),
              rm(p + "_r1", 3, p + "_w"), helper.make_node("Reshape", [p + "_w", s2], [ap.output[0]])]
        i = list(g.node).index(ap)
        g.node.remove(ap)
        for k, nn in enumerate(ns): g.node.insert(i + k, nn)
        n_dec += 1
    return n_dec


def split_anisotropic_dw(model: onnx.ModelProto) -> int:
    """Split stride-[2,1] depthwise convs into two isotropic [2,2] convs over even/odd
    width phases (zero-tap-padded kernels, symmetric pads) + interleave. torq mis-handles
    anisotropic depthwise stride; the phase pair is numerically exact."""
    g = model.graph
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    shapes = _shapes(g)
    init = {i.name: i for i in g.initializer}
    n_split = 0
    for c in [n for n in g.node if n.op_type == "Conv"]:
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in c.attribute}
        if attrs.get("strides") != [2, 1] or c.input[1] not in init: continue
        w = numpy_helper.to_array(init[c.input[1]])
        C, _, kh, kw = w.shape
        if attrs.get("group") != C: continue
        out_shape, pfx = shapes[c.output[0]], c.name.replace(".", "_")
        we, wo = (np.zeros((C, 1, kh, kw + 1), w.dtype) for _ in range(2))
        we[:, :, :, :kw], wo[:, :, :, 1:] = w, w
        bias = list(c.input[2:3])
        convs = [helper.make_node("Conv", [c.input[0], _const(g, f"{pfx}_w{t}", arr)] + bias, [f"{pfx}_{t}"], group=C, kernel_shape=[kh, kw + 1], strides=[2, 2], pads=[1, 1, 1, 1]) for t, arr in (("even", we), ("odd", wo))]
        unsq = lambda i, o: (helper.make_node("Unsqueeze", [i, _const(g, o + "_ax", np.array([4], np.int64))], [o]) if opset >= 13 else helper.make_node("Unsqueeze", [i], [o], axes=[4]))
        ns = convs + [unsq(f"{pfx}_even", f"{pfx}_e5"), unsq(f"{pfx}_odd", f"{pfx}_o5"),
                      helper.make_node("Concat", [f"{pfx}_e5", f"{pfx}_o5"], [f"{pfx}_il"], axis=4),
                      helper.make_node("Reshape", [f"{pfx}_il", _const(g, f"{pfx}_rs", np.array(out_shape, np.int64))], [c.output[0]])]
        i = list(g.node).index(c)
        g.node.remove(c)
        for k, nn in enumerate(ns): g.node.insert(i + k, nn)
        n_split += 1
    return n_split


def decompose_hardsigmoid(model: onnx.ModelProto) -> int:
    """Rewrite HardSigmoid as Clip(x*alpha + beta, 0, 1): identical arithmetic,
    but Mul/Add/Clip all lower to NSS while HardSigmoid falls back to host."""
    g = model.graph
    n_rw = 0
    for node in [n for n in g.node if n.op_type == "HardSigmoid"]:
        alpha = next((a.f for a in node.attribute if a.name == "alpha"), 0.2)
        beta = next((a.f for a in node.attribute if a.name == "beta"), 0.5)
        pfx = (node.name or node.output[0]).replace(".", "_")
        cst = lambda nm, v: _const(g, nm, np.array(v, np.float32))
        ns = [helper.make_node("Mul", [node.input[0], cst(f"{pfx}_a", alpha)], [f"{pfx}_m"]),
              helper.make_node("Add", [f"{pfx}_m", cst(f"{pfx}_b", beta)], [f"{pfx}_ab"]),
              helper.make_node("Clip", [f"{pfx}_ab", cst(f"{pfx}_lo", 0.0), cst(f"{pfx}_hi", 1.0)], [node.output[0]])]
        i = list(g.node).index(node)
        g.node.remove(node)
        for k, nn in enumerate(ns): g.node.insert(i + k, nn)
        n_rw += 1
    return n_rw


def global_reduce_to_gap(model: onnx.ModelProto) -> int:
    """Rewrite full-spatial ReduceMean (axes [2,3], keepdims=1, 4-D) as
    GlobalAveragePool: same semantics, but the dedicated pooling kernel has no
    NSS width limit, while the reduction generic falls off NSS at wide widths
    and cannot be tiled on small-LRAM targets."""
    g = model.graph
    shapes = _shapes(onnx.shape_inference.infer_shapes(model).graph)
    init = {i.name: i for i in g.initializer}
    n_rw = 0
    for node in [n for n in g.node if n.op_type == "ReduceMean"]:
        kd = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
        s_in = shapes.get(node.input[0])
        if kd != 1 or s_in is None or len(s_in) != 4: continue
        if len(node.input) > 1:
            if node.input[1] not in init: continue
            axes = sorted(a % 4 for a in numpy_helper.to_array(init[node.input[1]]).tolist())
        else:
            axes = sorted(a % 4 for a in next((list(a.ints) for a in node.attribute if a.name == "axes"), []))
        if axes != [2, 3]: continue
        gap = helper.make_node("GlobalAveragePool", [node.input[0]], [node.output[0]])
        i = list(g.node).index(node)
        g.node.remove(node)
        g.node.insert(i, gap)
        n_rw += 1
    return n_rw


def gemm_to_matmul(model: onnx.ModelProto) -> int:
    """Rewrite plain Gemm (alpha=beta=1, no transpose) as MatMul + Add, matching the
    reference exports; the 2-D Gemm form trips the NDL builder at wide widths."""
    g = model.graph
    new_nodes, n_conv = [], 0
    for node in g.node:
        attrs = {a.name: (a.i or a.f) for a in node.attribute}
        if node.op_type != "Gemm" or attrs.get("transA") or attrs.get("transB") or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1:
            new_nodes.append(node); continue
        mm = node.name + "_mm"
        new_nodes.append(helper.make_node("MatMul", [node.input[0], node.input[1]], [mm], name=mm))
        new_nodes.append(helper.make_node("Add", [mm, node.input[2]], [node.output[0]], name=node.name + "_add"))
        n_conv += 1
    del g.node[:]; g.node.extend(new_nodes)
    return n_conv
