#!/usr/bin/env python3
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs


def const_values(graph):
    out = {}
    for t in graph.tensors().values():
        if isinstance(t, gs.Constant):
            out[t.name] = np.array(t.values)
    return out


def get_const_array(tensor, const_map):
    if tensor is None:
        return None
    if isinstance(tensor, gs.Constant):
        return np.array(tensor.values)
    return const_map.get(tensor.name, None)


def get_tensor_shape_as_np(tensor):
    if tensor is None or tensor.shape is None:
        return None
    shape = []
    for d in tensor.shape:
        if d is None or isinstance(d, str):
            return None
        shape.append(int(d))
    return np.asarray(shape, dtype=np.int64)


def normalize_scalar_or_array(x):
    arr = np.array(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def eval_slice(data, starts, ends, axes=None, steps=None):
    data = np.array(data)
    rank = data.ndim

    if axes is None:
        axes = np.arange(len(starts), dtype=np.int64)
    if steps is None:
        steps = np.ones(len(starts), dtype=np.int64)

    slices = [slice(None)] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = int(a)
        slices[a] = slice(int(s), int(e), int(st))
    return data[tuple(slices)]


def eval_node_output(node, values):
    op = node.op
    ins = node.inputs

    def inp(i, default=None):
        if i >= len(ins):
            return default
        t = ins[i]
        if isinstance(t, gs.Constant):
            return np.array(t.values)
        return values.get(t.name, default)

    if op == "Shape":
        base = node.inputs[0]
        return get_tensor_shape_as_np(base)

    elif op == "Slice":
        data = inp(0)
        starts = inp(1)
        ends = inp(2)
        axes = inp(3, None)
        steps = inp(4, None)
        if data is None or starts is None or ends is None:
            return None
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        ends = np.asarray(ends, dtype=np.int64).reshape(-1)
        if axes is not None:
            axes = np.asarray(axes, dtype=np.int64).reshape(-1)
        if steps is not None:
            steps = np.asarray(steps, dtype=np.int64).reshape(-1)
        return eval_slice(data, starts, ends, axes, steps)

    elif op == "Gather":
        data = inp(0)
        indices = inp(1)
        axis = 0
        for attr in node.attrs:
            pass
        axis = int(node.attrs.get("axis", 0))
        if data is None or indices is None:
            return None
        return np.take(data, np.asarray(indices, dtype=np.int64), axis=axis)

    elif op == "Unsqueeze":
        data = inp(0)
        axes = inp(1, None)
        if data is None:
            return None

        if axes is None:
            axes = node.attrs.get("axes", None)
        if axes is None:
            return None

        axes = sorted(int(a) for a in np.asarray(axes).reshape(-1))
        out = np.array(data)
        for a in axes:
            out = np.expand_dims(out, axis=a)
        return out

    elif op == "Squeeze":
        data = inp(0)
        axes = inp(1, None)
        if data is None:
            return None

        if axes is None:
            axes = node.attrs.get("axes", None)

        if axes is None:
            return np.squeeze(data)

        axes = tuple(int(a) for a in np.asarray(axes).reshape(-1))
        return np.squeeze(data, axis=axes)

    elif op == "Concat":
        arrs = []
        axis = int(node.attrs.get("axis", 0))
        for t in node.inputs:
            if isinstance(t, gs.Constant):
                v = np.array(t.values)
            else:
                v = values.get(t.name, None)
            if v is None:
                return None
            arrs.append(normalize_scalar_or_array(v))
        return np.concatenate(arrs, axis=axis)

    elif op == "Cast":
        data = inp(0)
        if data is None:
            return None
        to = node.attrs.get("to", None)
        if to is None:
            return None

        # Enough for shape subgraphs in practice
        onnx_to_np = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            4: np.uint16,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            9: np.bool_,
            10: np.float16,
            11: np.float64,
        }
        dtype = onnx_to_np.get(int(to), None)
        if dtype is None:
            return None
        return np.asarray(data, dtype=dtype)

    elif op == "Mul":
        a = inp(0)
        b = inp(1)
        if a is None or b is None:
            return None
        return np.asarray(a) * np.asarray(b)

    elif op == "Add":
        a = inp(0)
        b = inp(1)
        if a is None or b is None:
            return None
        return np.asarray(a) + np.asarray(b)

    elif op == "Sub":
        a = inp(0)
        b = inp(1)
        if a is None or b is None:
            return None
        return np.asarray(a) - np.asarray(b)

    elif op == "Div":
        a = inp(0)
        b = inp(1)
        if a is None or b is None:
            return None
        return np.asarray(a) // np.asarray(b)

    return None


def fold_reshape_shape_subgraphs(graph):
    changed = 0
    const_map = const_values(graph)

    for node in graph.nodes:
        if node.op != "Reshape" or len(node.inputs) < 2:
            continue

        data_in = node.inputs[0]
        shape_in = node.inputs[1]

        # already constant
        if isinstance(shape_in, gs.Constant):
            continue

        shape_producer = shape_in.inputs[0] if getattr(shape_in, "inputs", None) else None
        if shape_producer is None:
            continue

        values = dict(const_map)

        # Seed any Shape nodes in the backward cone by using known tensor.shape.
        # Simpler approach: iteratively evaluate producers reachable from shape_in.
        progress = True
        while progress:
            progress = False
            for n in graph.nodes:
                for out in n.outputs:
                    if out.name in values:
                        continue
                out_val = eval_node_output(n, values)
                if out_val is None:
                    continue
                for out in n.outputs:
                    if out.name not in values:
                        values[out.name] = np.asarray(out_val)
                        progress = True

        resolved = values.get(shape_in.name, None)
        if resolved is None:
            continue

        resolved = np.asarray(resolved).astype(np.int64).reshape(-1)

        # sanity checks
        if np.any(resolved < -1):
            continue

        const_name = shape_in.name + "_folded"
        new_const = gs.Constant(name=const_name, values=resolved)
        node.inputs[1] = new_const
        changed += 1
        print(f"[fold] Reshape shape folded for node: {node.name or '<unnamed>'} -> {resolved.tolist()}")

    if changed:
        graph.cleanup().toposort()
    return changed


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.onnx output.onnx")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    graph = gs.import_onnx(model)

    changed = fold_reshape_shape_subgraphs(graph)
    print(f"Total folded reshape-shape subgraphs: {changed}")

    out_model = gs.export_onnx(graph)
    onnx.save(out_model, outp)


if __name__ == "__main__":
    main()