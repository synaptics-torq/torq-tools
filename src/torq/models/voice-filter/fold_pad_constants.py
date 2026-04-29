#!/usr/bin/env python3
import argparse
import copy
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto


# -----------------------------
# Basic graph utilities
# -----------------------------

def parse_name_shape(spec: str) -> Tuple[str, List[int]]:
    # Example: in_frame_mag=1,2,1,256
    name, dims = spec.split("=")
    shape = [int(x) for x in dims.split(",") if x.strip() != ""]
    return name, shape


def tensorproto_dtype_to_numpy(dtype: int):
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.FLOAT16: np.float16,
        TensorProto.BFLOAT16: np.float32,   # approximate
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.INT16: np.int16,
        TensorProto.INT8: np.int8,
        TensorProto.UINT64: np.uint64,
        TensorProto.UINT32: np.uint32,
        TensorProto.UINT16: np.uint16,
        TensorProto.UINT8: np.uint8,
        TensorProto.BOOL: np.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def get_attr(node: onnx.NodeProto, name: str):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def get_attr_int(node: onnx.NodeProto, name: str, default=None):
    a = get_attr(node, name)
    return a.i if a is not None else default


def get_attr_ints(node: onnx.NodeProto, name: str, default=None):
    a = get_attr(node, name)
    return list(a.ints) if a is not None else default


def get_attr_tensor(node: onnx.NodeProto, name: str):
    a = get_attr(node, name)
    if a is None:
        return None
    return numpy_helper.to_array(a.t)


def add_initializer(graph: onnx.GraphProto, name: str, array: np.ndarray):
    init = numpy_helper.from_array(array, name=name)
    graph.initializer.append(init)
    return name


def replace_value_info_shape(value_info, shape: List[int]):
    tensor_type = value_info.type.tensor_type
    for i, dim in enumerate(shape):
        d = tensor_type.shape.dim[i]
        d.ClearField("dim_param")
        d.dim_value = int(dim)


def set_graph_value_shape(graph: onnx.GraphProto, name: str, shape: List[int]) -> bool:
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name == name:
            # Resize dims if needed
            while len(vi.type.tensor_type.shape.dim) < len(shape):
                vi.type.tensor_type.shape.dim.add()
            replace_value_info_shape(vi, shape)
            return True
    return False


def apply_fixed_shapes(model: onnx.ModelProto,
                       input_shapes: Dict[str, List[int]],
                       output_shapes: Dict[str, List[int]]) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    graph = model.graph

    for inp in graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            tensor_type = inp.type.tensor_type
            while len(tensor_type.shape.dim) < len(shape):
                tensor_type.shape.dim.add()
            replace_value_info_shape(inp, shape)

    for out in graph.output:
        if out.name in output_shapes:
            shape = output_shapes[out.name]
            tensor_type = out.type.tensor_type
            while len(tensor_type.shape.dim) < len(shape):
                tensor_type.shape.dim.add()
            replace_value_info_shape(out, shape)

    return model


def build_initializer_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in graph.initializer}


def build_producer_map(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    prod = {}
    for node in graph.node:
        for out in node.output:
            if out:
                prod[out] = node
    return prod


def extract_shape_from_value_info(vi) -> Optional[List[int]]:
    if not vi.type.HasField("tensor_type"):
        return None
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            return None
    return dims


def build_known_shapes(graph: onnx.GraphProto) -> Dict[str, List[int]]:
    known = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = extract_shape_from_value_info(vi)
        if shape is not None:
            known[vi.name] = shape
    for init in graph.initializer:
        known[init.name] = list(numpy_helper.to_array(init).shape)
    return known


# -----------------------------
# Partial evaluator
# -----------------------------

class EvalContext:
    def __init__(self, graph: onnx.GraphProto):
        self.graph = graph
        self.initializers = build_initializer_map(graph)
        self.producers = build_producer_map(graph)
        self.known_shapes = build_known_shapes(graph)
        self.cache: Dict[str, np.ndarray] = {}

    def eval_value(self, name: str) -> Optional[np.ndarray]:
        if name == "":
            return None
        if name in self.cache:
            return self.cache[name]
        if name in self.initializers:
            self.cache[name] = self.initializers[name]
            return self.cache[name]
        if name not in self.producers:
            return None

        node = self.producers[name]
        vals = self.eval_node(node)
        if vals is None:
            return None

        for out_name, arr in zip(node.output, vals):
            if out_name:
                self.cache[out_name] = arr

        return self.cache.get(name)

    def eval_shape_of_tensor(self, tensor_name: str) -> Optional[np.ndarray]:
        shape = self.known_shapes.get(tensor_name)
        if shape is None:
            return None
        return np.asarray(shape, dtype=np.int64)

    def eval_node(self, node: onnx.NodeProto) -> Optional[List[np.ndarray]]:
        op = node.op_type

        # Constant
        if op == "Constant":
            value = get_attr_tensor(node, "value")
            if value is None:
                return None
            return [np.asarray(value)]

        # Identity
        if op == "Identity":
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            return [x]

        # Shape
        if op == "Shape":
            s = self.eval_shape_of_tensor(node.input[0])
            if s is None:
                return None

            start = get_attr_int(node, "start", None)
            end = get_attr_int(node, "end", None)
            if start is not None or end is not None:
                rank = s.shape[0]
                st = 0 if start is None else start
                ed = rank if end is None else end
                if st < 0:
                    st += rank
                if ed < 0:
                    ed += rank
                s = s[st:ed]
            return [s.astype(np.int64)]

        # ConstantOfShape
        if op == "ConstantOfShape":
            shape_arr = self.eval_value(node.input[0])
            if shape_arr is None:
                return None
            shape_arr = np.asarray(shape_arr).astype(np.int64).reshape(-1)
            shape_tuple = tuple(int(x) for x in shape_arr.tolist())

            value_attr = get_attr(node, "value")
            if value_attr is not None:
                fill_value = numpy_helper.to_array(value_attr.t).reshape(-1)[0]
                dtype = numpy_helper.to_array(value_attr.t).dtype
            else:
                fill_value = 0.0
                dtype = np.float32

            return [np.full(shape_tuple, fill_value, dtype=dtype)]

        # Cast
        if op == "Cast":
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            to_dtype = get_attr_int(node, "to")
            return [x.astype(tensorproto_dtype_to_numpy(to_dtype))]

        # Reshape
        if op == "Reshape":
            x = self.eval_value(node.input[0])
            shape = self.eval_value(node.input[1])
            if x is None or shape is None:
                return None
            shape = np.asarray(shape).astype(np.int64).reshape(-1).tolist()
            try:
                y = np.reshape(x, shape)
            except Exception:
                return None
            return [y]

        # Concat
        if op == "Concat":
            axis = get_attr_int(node, "axis")
            xs = []
            for inp in node.input:
                x = self.eval_value(inp)
                if x is None:
                    return None
                xs.append(np.asarray(x))
            try:
                y = np.concatenate(xs, axis=axis)
            except Exception:
                return None
            return [y]

        # Unsqueeze
        if op == "Unsqueeze":
            x = self.eval_value(node.input[0])
            if x is None:
                return None

            if len(node.input) >= 2 and node.input[1]:
                axes = self.eval_value(node.input[1])
                if axes is None:
                    return None
                axes = [int(a) for a in np.asarray(axes).reshape(-1).tolist()]
            else:
                axes = get_attr_ints(node, "axes", [])
            y = x
            # sort to apply consistently
            for ax in sorted(axes):
                y = np.expand_dims(y, axis=ax)
            return [y]

        # Squeeze
        if op == "Squeeze":
            x = self.eval_value(node.input[0])
            if x is None:
                return None

            if len(node.input) >= 2 and node.input[1]:
                axes = self.eval_value(node.input[1])
                if axes is None:
                    return None
                axes = [int(a) for a in np.asarray(axes).reshape(-1).tolist()]
            else:
                axes = get_attr_ints(node, "axes", None)

            try:
                if axes is None:
                    y = np.squeeze(x)
                else:
                    y = x
                    for ax in sorted(axes, reverse=True):
                        y = np.squeeze(y, axis=ax)
            except Exception:
                return None
            return [y]

        # Transpose
        if op == "Transpose":
            x = self.eval_value(node.input[0])
            if x is None:
                return None
            perm = get_attr_ints(node, "perm", None)
            return [np.transpose(x, axes=perm)]

        # Gather
        if op == "Gather":
            data = self.eval_value(node.input[0])
            indices = self.eval_value(node.input[1])
            if data is None or indices is None:
                return None
            axis = get_attr_int(node, "axis", 0)
            try:
                y = np.take(data, indices.astype(np.int64), axis=axis)
            except Exception:
                return None
            return [y]

        # Slice
        if op == "Slice":
            data = self.eval_value(node.input[0])
            starts = self.eval_value(node.input[1])
            ends = self.eval_value(node.input[2])
            if data is None or starts is None or ends is None:
                return None

            if len(node.input) >= 4 and node.input[3]:
                axes = self.eval_value(node.input[3])
                if axes is None:
                    return None
                axes = axes.astype(np.int64).reshape(-1).tolist()
            else:
                axes = list(range(len(np.asarray(starts).reshape(-1))))

            if len(node.input) >= 5 and node.input[4]:
                steps = self.eval_value(node.input[4])
                if steps is None:
                    return None
                steps = steps.astype(np.int64).reshape(-1).tolist()
            else:
                steps = [1] * len(axes)

            starts = np.asarray(starts).astype(np.int64).reshape(-1).tolist()
            ends = np.asarray(ends).astype(np.int64).reshape(-1).tolist()

            slc = [slice(None)] * data.ndim
            for ax, st, ed, step in zip(axes, starts, ends, steps):
                slc[int(ax)] = slice(int(st), int(ed), int(step))

            try:
                y = data[tuple(slc)]
            except Exception:
                return None
            return [y]

        # Expand
        if op == "Expand":
            x = self.eval_value(node.input[0])
            shape = self.eval_value(node.input[1])
            if x is None or shape is None:
                return None
            try:
                y = np.broadcast_to(x, tuple(int(v) for v in np.asarray(shape).reshape(-1).tolist()))
            except Exception:
                return None
            return [np.asarray(y)]

        # If node not supported
        return None


# -----------------------------
# Graph rewriting
# -----------------------------

def unique_name(base: str, existing: Set[str]) -> str:
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def all_value_names(graph: onnx.GraphProto) -> Set[str]:
    names = set()
    for init in graph.initializer:
        names.add(init.name)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(vi.name)
    for n in graph.node:
        names.update([x for x in n.input if x])
        names.update([x for x in n.output if x])
    return names


def replace_pad_inputs_with_constants(model: onnx.ModelProto, verbose: bool = True) -> Tuple[onnx.ModelProto, int]:
    model = copy.deepcopy(model)
    graph = model.graph
    ctx = EvalContext(graph)
    existing_names = all_value_names(graph)

    num_replaced = 0

    for node in graph.node:
        if node.op_type != "Pad":
            continue

        # input[1] = pads
        if len(node.input) >= 2 and node.input[1]:
            pads_name = node.input[1]
            pads_val = ctx.eval_value(pads_name)
            if pads_val is not None:
                pads_val = np.asarray(pads_val).astype(np.int64).reshape(-1)
                new_name = unique_name(f"{node.name or 'pad'}_pads_const", existing_names)
                existing_names.add(new_name)
                add_initializer(graph, new_name, pads_val)
                node.input[1] = new_name
                num_replaced += 1
                if verbose:
                    print(f"[Pad fold] {node.name or '(unnamed)'}: pads <- {pads_val.tolist()}")

        # input[2] = constant_value (optional)
        if len(node.input) >= 3 and node.input[2]:
            cv_name = node.input[2]
            cv_val = ctx.eval_value(cv_name)
            if cv_val is not None:
                cv_val = np.asarray(cv_val)
                if cv_val.size == 1:
                    new_name = unique_name(f"{node.name or 'pad'}_value_const", existing_names)
                    existing_names.add(new_name)
                    add_initializer(graph, new_name, cv_val.reshape(()))
                    node.input[2] = new_name
                    if verbose:
                        print(f"[Pad fold] {node.name or '(unnamed)'}: constant_value <- {cv_val.reshape(-1).tolist()}")

    return model, num_replaced


def remove_dead_nodes_and_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    graph = model.graph

    # Build producer map
    prod = build_producer_map(graph)

    # Start from graph outputs
    needed_values = set(out.name for out in graph.output if out.name)
    needed_nodes = set()
    queue = deque(needed_values)

    while queue:
        val = queue.popleft()
        if val in prod:
            node = prod[val]
            node_id = id(node)
            if node_id in needed_nodes:
                continue
            needed_nodes.add(node_id)
            for inp in node.input:
                if inp:
                    queue.append(inp)

    # Keep only needed nodes
    new_nodes = [n for n in graph.node if id(n) in needed_nodes]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Recompute used value names
    used_values = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                used_values.add(inp)
    for out in graph.output:
        used_values.add(out.name)

    # Keep only used initializers
    keep_inits = [init for init in graph.initializer if init.name in used_values]
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)

    return model


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Fold ConstantOfShape/shape-only subgraphs into Pad constants.")
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        help="Fixed input shape, e.g. in_frame_mag=1,2,1,256",
    )
    parser.add_argument(
        "--output-shape",
        action="append",
        default=[],
        help="Optional fixed output shape, e.g. vad=1,1 or hidden_state=1,16,64",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Do not remove dead nodes/initializers after folding",
    )
    args = parser.parse_args()

    input_shapes = dict(parse_name_shape(x) for x in args.input_shape)
    output_shapes = dict(parse_name_shape(x) for x in args.output_shape)

    model = onnx.load(args.input_model)

    # Apply known static shapes
    model = apply_fixed_shapes(model, input_shapes, output_shapes)

    # Run shape inference first
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference before folding failed: {e}")

    # Fold Pad inputs
    model, num_replaced = replace_pad_inputs_with_constants(model, verbose=True)

    # Cleanup dead branches
    if not args.skip_cleanup:
        model = remove_dead_nodes_and_initializers(model)

    # Run shape inference again
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference after folding failed: {e}")

    onnx.save(model, args.output_model)
    print(f"\nSaved: {args.output_model}")
    print(f"Pad inputs folded: {num_replaced}")


if __name__ == "__main__":
    main()