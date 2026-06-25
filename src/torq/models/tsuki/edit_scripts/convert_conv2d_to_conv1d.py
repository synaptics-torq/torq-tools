#!/usr/bin/env python3
"""Rewrite Conv2D nodes back into Conv1D — the inverse of convert_conv1d_to_conv2d.py.

The forward script turns a Conv1D into one of two shapes:

  length-as-height:  Unsqueeze(axis=3) -> Conv2D(kernel [k,1]) -> Squeeze(axis=3)
  height-2 (default): Unsqueeze(axis=2) -> Concat(axis=2) -> Conv2D(kernel [2,k]) -> Squeeze(axis=2)

This script finds that ``Unsqueeze [-> Concat] -> Conv2D -> Squeeze`` group, replaces
it with a single Conv1D on the original 3D tensors, and drops the inserted
Unsqueeze/Concat/Squeeze (and the zero-height helper) nodes.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper


def get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return [int(v) for v in attr.ints]
    return None


def get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def node_axes(node, inits):
    """Axes of an Unsqueeze/Squeeze, whether an attribute (opset<13) or input."""
    axes = get_attr_ints(node, "axes")
    if axes is not None:
        return axes
    if len(node.input) >= 2 and node.input[1] in inits:
        return [int(v) for v in numpy_helper.to_array(inits[node.input[1]]).reshape(-1)]
    return None


def take0_initializer(tensor, axis, new_name):
    """Return a copy of `tensor` indexed at 0 along `axis` (drops that dim)."""
    dims = list(tensor.dims)
    if tensor.data_type == onnx.TensorProto.BFLOAT16:
        u16 = np.frombuffer(tensor.raw_data, dtype=np.uint16).reshape(dims)
        sliced = np.take(u16, 0, axis=axis)
        out = onnx.TensorProto()
        out.name = new_name
        out.data_type = onnx.TensorProto.BFLOAT16
        out.dims.extend(sliced.shape)
        out.raw_data = sliced.tobytes()
        return out
    arr = numpy_helper.to_array(tensor)
    return numpy_helper.from_array(np.take(arr, 0, axis=axis).copy(), name=new_name)


def prune_unused(graph):
    """Drop initializers and value_info not referenced by any remaining node."""
    referenced = {name for node in graph.node for name in list(node.input) + list(node.output) if name}
    referenced |= {v.name for v in list(graph.input) + list(graph.output)}

    kept_init = [init for init in graph.initializer if init.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)

    kept_vi = [vi for vi in graph.value_info if vi.name in referenced]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)


def invert_conv(graph, conv, conv_index, producer, consumers, inits):
    """Try to invert one Conv2D. Returns (conv1d_node, indices_to_drop) or None.

    Threads through optional Cast nodes (the bf16-island variant inserts a Cast
    between Unsqueeze and Conv, between Conv and Squeeze, and on the weight).
    """
    if conv.op_type != "Conv":
        return None
    drop = {conv_index}

    # --- input side: optional Cast, then Unsqueeze, or Concat(axis=2) <- Unsqueeze ---
    src = conv.input[0]
    sp = producer.get(src)
    if sp is not None and sp[1].op_type == "Cast":
        drop.add(sp[0])
        src = sp[1].input[0]
    pnode_i = producer.get(src)
    if pnode_i is None:
        return None
    pidx, pnode = pnode_i

    if pnode.op_type == "Unsqueeze":
        axes = node_axes(pnode, inits)
        if axes != [3] and axes != [2]:
            return None
        inserted_axis = axes[0]
        x_name = pnode.input[0]
        drop.add(pidx)
    elif pnode.op_type == "Concat" and get_attr_int(pnode, "axis") == 2:
        up_i = producer.get(pnode.input[0])
        if up_i is None or up_i[1].op_type != "Unsqueeze":
            return None
        inserted_axis = 2
        x_name = up_i[1].input[0]
        drop.add(pidx)
        drop.add(up_i[0])
    else:
        return None

    # --- output side: optional Cast, then a single Squeeze ---
    cur_out = conv.output[0]
    outs = consumers.get(cur_out, [])
    if len(outs) != 1:
        return None
    nidx, nn = outs[0]
    if nn.op_type == "Cast":
        drop.add(nidx)
        outs2 = consumers.get(nn.output[0], [])
        if len(outs2) != 1:
            return None
        nidx, nn = outs2[0]
    if nn.op_type != "Squeeze":
        return None
    if node_axes(nn, inits) != [inserted_axis]:
        return None
    y_name = nn.output[0]
    drop.add(nidx)

    # real 1D spatial position: inserted axis 2 -> width(idx 1); axis 3 -> height(idx 0)
    real = 3 - inserted_axis

    # --- weight back to 3D (skip optional Cast) ---
    weight_name = conv.input[1]
    wp = producer.get(weight_name)
    if wp is not None and wp[1].op_type == "Cast":
        drop.add(wp[0])
        weight_name = wp[1].input[0]
    if weight_name in inits:  # constant 4D weight -> 3D initializer
        new_w = take0_initializer(inits[weight_name], inserted_axis, f"{y_name}_weight_3d")
        graph.initializer.append(new_w)
        weight_1d = new_w.name
    else:  # dynamic weight: trace back through the inserted Unsqueeze/Concat
        wp_i = producer.get(weight_name)
        if wp_i is None:
            return None
        widx, wp = wp_i
        if wp.op_type == "Unsqueeze":
            weight_1d = wp.input[0]
            drop.add(widx)
        elif wp.op_type == "Concat" and get_attr_int(wp, "axis") == 2:
            wup_i = producer.get(wp.input[0])  # weight_4d_h1 = Unsqueeze(orig_w)
            wmul_i = producer.get(wp.input[1])  # zero-height Mul helper
            if wup_i is None or wup_i[1].op_type != "Unsqueeze":
                return None
            weight_1d = wup_i[1].input[0]
            drop.add(widx)
            drop.add(wup_i[0])
            if wmul_i is not None:
                drop.add(wmul_i[0])
        else:
            return None

    # --- build the Conv1D attributes (pick the real spatial slot) ---
    kernel = get_attr_ints(conv, "kernel_shape")
    strides = get_attr_ints(conv, "strides") or [1, 1]
    dilations = get_attr_ints(conv, "dilations") or [1, 1]
    pads = get_attr_ints(conv, "pads") or [0, 0, 0, 0]
    group = get_attr_int(conv, "group", 1)

    attrs = {
        "strides": [strides[real]],
        "dilations": [dilations[real]],
        "pads": [pads[real], pads[real + 2]],
        "group": group,
    }
    if kernel is not None:
        attrs["kernel_shape"] = [kernel[real]]

    inputs = [x_name, weight_1d]
    if len(conv.input) >= 3 and conv.input[2]:
        inputs.append(conv.input[2])

    conv1d = helper.make_node("Conv", inputs, [y_name], name=conv.name, **attrs)
    return conv1d, drop


def convert(model):
    graph = model.graph
    inits = initializer_map(graph)
    producer = {}
    consumers = {}
    for i, node in enumerate(graph.node):
        for o in node.output:
            if o:
                producer[o] = (i, node)
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append((i, node))

    drop_all = set()
    replace = {}
    for i, node in enumerate(graph.node):
        result = invert_conv(graph, node, i, producer, consumers, inits)
        if result is None:
            continue
        conv1d, drop = result
        replace[i] = conv1d
        drop_all |= drop

    new_nodes = []
    for i, node in enumerate(graph.node):
        if i in replace:
            new_nodes.append(replace[i])
        elif i in drop_all:
            continue
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    prune_unused(graph)
    return len(replace)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                        help="Run onnx.checker on the output (default: on).")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    n = convert(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        checker.check_model(model)
        print("ONNX checker passed")
    print(f"Conv2D -> Conv1D conversions: {n}")
    print(f"Wrote model: {args.output}")


if __name__ == "__main__":
    main()
