import copy
from collections import deque

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto, shape_inference


MODEL_PATH = "model_epoch_0015_12.2733_folded.onnx"
OUTPUT_PATH = "model_epoch_0015_12.2733_folded_static.onnx"

# Fixed deployment inputs
FEEDS = {
    "in_frame_mag": np.zeros((1, 2, 1, 256), dtype=np.float32),
    "embedding": np.zeros((1, 256), dtype=np.float32),
    "input_state": np.zeros((1, 16, 64), dtype=np.float32),
}

# Ops allowed in shape-only backward slicing
SHAPE_OPS = {
    "Shape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Equal",
    "Where",
    "ConstantOfShape",
    "Mul",
    "Div",
    "Add",
    "Sub",
    "Cast",
    "Reshape",
    "Slice",
    "Expand",
    "Transpose",
    "Constant",
    "Identity",
}


def build_maps(model):
    producer = {}
    consumers = {}
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}

    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)

    return producer, consumers, initializers, graph_inputs


def node_attr_ints(node, attr_name):
    for attr in node.attribute:
        if attr.name == attr_name:
            return list(attr.ints)
    return None


def get_target_shape_input_indices(node):
    # Which inputs are shape/control tensors that we should slice from
    if node.op_type == "Expand" and len(node.input) >= 2:
        return [1]
    if node.op_type == "Reshape" and len(node.input) >= 2:
        return [1]
    if node.op_type == "Slice":
        # optional if you also want to staticize Slice params
        return list(range(1, len(node.input)))
    return []


def collect_backward_shape_subgraph(model, seed_tensor_names, shape_ops):
    """
    Walk backward from seed tensors through only shape-related producers.
    Returns:
      kept_tensors: tensors in the backward slice
      kept_nodes: nodes in the backward slice
    """
    producer, _, initializers, graph_inputs = build_maps(model)

    kept_tensors = set()
    kept_nodes = set()

    q = deque(seed_tensor_names)

    while q:
        tensor_name = q.popleft()
        if not tensor_name or tensor_name in kept_tensors:
            continue
        kept_tensors.add(tensor_name)

        if tensor_name in initializers or tensor_name in graph_inputs:
            continue

        p = producer.get(tensor_name)
        if p is None:
            continue
        if p.op_type not in shape_ops:
            continue

        kept_nodes.add(p)
        for inp in p.input:
            if inp:
                q.append(inp)

    return kept_tensors, kept_nodes


def expose_intermediate_outputs_with_identity(model, tensor_names, dtype=TensorProto.INT64):
    """
    Create a probe model that exposes the given intermediate tensors as graph outputs via Identity.
    """
    probe = copy.deepcopy(model)
    del probe.graph.output[:]

    probe_output_names = []
    for i, tname in enumerate(tensor_names):
        out_name = f"__probe_out_{i}"
        probe_output_names.append(out_name)
        probe.graph.node.append(
            helper.make_node(
                "Identity",
                inputs=[tname],
                outputs=[out_name],
                name=f"ProbeIdentity_{i}",
            )
        )
        probe.graph.output.append(
            helper.make_tensor_value_info(out_name, dtype, None)
        )

    return probe, probe_output_names


def evaluate_tensors(model, tensor_names, feeds):
    probe, probe_output_names = expose_intermediate_outputs_with_identity(model, tensor_names)
    probe = shape_inference.infer_shapes(probe)
    probe_path = "/tmp/probe_shape_slice.onnx"
    onnx.save(probe, probe_path)

    sess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
    vals = sess.run(probe_output_names, feeds)
    return dict(zip(tensor_names, vals))


def replace_consumers_with_frozen_initializers(model, frozen_map):
    """
    frozen_map: old_tensor_name -> np.ndarray
    Adds new initializers with suffix __frozen and rewires consumers.
    Returns rename_map: old_name -> new_name
    """
    rename_map = {}

    for old_name, value in frozen_map.items():
        new_name = old_name + "__frozen"
        rename_map[old_name] = new_name
        model.graph.initializer.append(numpy_helper.from_array(value, name=new_name))

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]

    for out in model.graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]

    return rename_map


def prune_dead_nodes(model):
    """
    Reverse reachability from graph outputs.
    """
    needed_tensors = {o.name for o in model.graph.output}
    needed_nodes = []

    changed = True
    while changed:
        changed = False
        for node in reversed(model.graph.node):
            if any(out in needed_tensors for out in node.output):
                if node not in needed_nodes:
                    needed_nodes.append(node)
                    for inp in node.input:
                        if inp and inp not in needed_tensors:
                            needed_tensors.add(inp)
                            changed = True

    needed_nodes = list(reversed(needed_nodes))
    del model.graph.node[:]
    model.graph.node.extend(needed_nodes)

    used_names = set()
    used_names |= {i.name for i in model.graph.input}
    used_names |= {o.name for o in model.graph.output}
    used_names |= {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        used_names.update([x for x in node.input if x])
        used_names.update([x for x in node.output if x])

    kept_vi = [vi for vi in model.graph.value_info if vi.name in used_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def find_shape_seed_tensors(model):
    """
    Find all shape/control tensor inputs to Expand and Reshape.
    """
    seeds = []
    for node in model.graph.node:
        idxs = get_target_shape_input_indices(node)
        for idx in idxs:
            if idx < len(node.input):
                seeds.append(node.input[idx])
    return seeds


def describe_seeds(model):
    producer, _, initializers, graph_inputs = build_maps(model)

    print("Shape/control seeds:")
    for node in model.graph.node:
        idxs = get_target_shape_input_indices(node)
        if not idxs:
            continue
        print(f"\n{node.name} ({node.op_type})")
        for idx in idxs:
            if idx >= len(node.input):
                continue
            t = node.input[idx]
            print(f"  input[{idx}] = {t}")
            if t in initializers:
                print("    producer: initializer")
            elif t in graph_inputs:
                print("    producer: graph input")
            elif t in producer:
                p = producer[t]
                print(f"    producer: {p.name} ({p.op_type})")
            else:
                print("    producer: <none>")


def main():
    model = onnx.load(MODEL_PATH)
    model = shape_inference.infer_shapes(model)

    describe_seeds(model)

    seed_tensors = find_shape_seed_tensors(model)
    print("\nCollected seed tensors:")
    for s in seed_tensors:
        print(" ", s)

    # Only keep seeds whose producer is shape-logic, not already plain initializers.
    producer, _, initializers, graph_inputs = build_maps(model)
    slice_seeds = []
    for s in seed_tensors:
        if s in initializers or s in graph_inputs:
            continue
        p = producer.get(s)
        if p is not None and p.op_type in SHAPE_OPS:
            slice_seeds.append(s)

    # Deduplicate while preserving order
    seen = set()
    slice_seeds = [x for x in slice_seeds if not (x in seen or seen.add(x))]

    print("\nSeeds to freeze:")
    for s in slice_seeds:
        p = producer.get(s)
        print(f"  {s} <- {p.name} ({p.op_type})")

    # Evaluate the final shape/control tensors once
    frozen_values = evaluate_tensors(model, slice_seeds, FEEDS)

    print("\nEvaluated frozen values:")
    for name, val in frozen_values.items():
        print(f"  {name}: shape={val.shape}, dtype={val.dtype}, value={val}")

    # Replace consumer uses with new constant initializers
    rename_map = replace_consumers_with_frozen_initializers(model, frozen_values)

    print("\nRewired tensors:")
    for old_name, new_name in rename_map.items():
        print(f"  {old_name} -> {new_name}")

    # Prune dead nodes/subgraphs
    prune_dead_nodes(model)

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()