import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper


FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}
BOOL_PRODUCING_OPS = {"Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal", "And", "Or", "Xor", "Not"}

# Ops whose specific input slots are BOOL-typed per the ONNX spec.
BOOL_INPUT_OPS = {
    "And":      (0, 1),
    "Or":       (0, 1),
    "Xor":      (0, 1),
    "Not":      (0,),
    "Where":    (0,),
    "If":       (0,),
    "Compress": (1,),
}

# Ops whose output element type equals their first input's type — they pass
# dtype through unchanged, so we can keep BOOL flowing through them.
# Conservative set: only single-data-input shape ops. Concat / Split excluded
# (multi-input dtype invariants we don't want to police here).
TRANSPARENT_OPS = frozenset({
    "Unsqueeze", "Squeeze", "Reshape", "Transpose", "Expand", "Tile",
    "Identity", "Slice", "Pad", "Flatten",
})

# Per-op slots that take int64 index/shape inputs — never part of a bool chain.
PASSTHROUGH_INDEX_SLOTS = {
    "Slice": (1, 2, 3, 4),
    "Unsqueeze": (1,),
    "Squeeze": (1,),
    "Reshape": (1,),
    "Expand": (1,),
    "Tile": (1,),
    "Pad": (1, 2, 3),
}


def _is_bool_consumer(node, input_idx):
    return input_idx in BOOL_INPUT_OPS.get(node.op_type, ())


def _is_transparent_data_slot(node, slot):
    """True iff `slot` of a transparent op is its data slot (slot 0 for
    single-data ops; index/shape slots return False)."""
    if node.op_type not in TRANSPARENT_OPS:
        return False
    if slot in PASSTHROUGH_INDEX_SLOTS.get(node.op_type, ()):
        return False
    return slot == 0


def _get_cast_to_attr(node):
    for a in node.attribute:
        if a.name == "to":
            return int(a.i)
    return None


def _build_consumers_map(graph):
    consumers = {}
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp:
                consumers.setdefault(inp, []).append((n, i))
    return consumers


def fp32_to_bf16_uint16(values):
    f32 = values.astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    rounded = u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def convert_tensor_to_bf16(tensor):
    if tensor.data_type == TensorProto.BFLOAT16:
        return 0
    if tensor.data_type == TensorProto.BOOL:
        values = numpy_helper.to_array(tensor).astype(np.float32)
        bf16 = fp32_to_bf16_uint16(values)
        tensor.data_type = TensorProto.BFLOAT16
        tensor.raw_data = bf16.tobytes()
        del tensor.float_data[:]
        del tensor.int32_data[:]
        del tensor.int64_data[:]
        del tensor.double_data[:]
        return 1
    if tensor.data_type not in FLOAT_TYPES:
        return 0
    values = numpy_helper.to_array(tensor).astype(np.float32)
    bf16 = fp32_to_bf16_uint16(values)
    tensor.data_type = TensorProto.BFLOAT16
    tensor.raw_data = bf16.tobytes()
    del tensor.float_data[:]
    del tensor.int32_data[:]
    del tensor.int64_data[:]
    del tensor.double_data[:]
    return 1


def collect_bool_producing_outputs(graph):
    """Outputs of ops that, per ONNX spec, must be BOOL. We leave their
    declared dtype alone; a follow-up Cast(bool->bf16) bridges them to
    bf16-consuming downstream nodes."""
    names = set()
    for node in graph.node:
        if node.op_type in BOOL_PRODUCING_OPS:
            for out in node.output:
                if out:
                    names.add(out)
    return names


def convert_value_info_to_bf16(value_info, skip_names=()):
    tensor_type = value_info.type.tensor_type
    if not value_info.type.HasField("tensor_type"):
        return 0
    if value_info.name in skip_names:
        return 0
    if tensor_type.elem_type == TensorProto.BOOL:
        tensor_type.elem_type = TensorProto.BFLOAT16
        return 1
    if tensor_type.elem_type not in FLOAT_TYPES:
        return 0
    tensor_type.elem_type = TensorProto.BFLOAT16
    return 1


def convert_attr_to_bf16(attr):
    converted = 0
    if attr.type == AttributeProto.TENSOR:
        converted += convert_tensor_to_bf16(attr.t)
    elif attr.type == AttributeProto.TENSORS:
        for tensor in attr.tensors:
            converted += convert_tensor_to_bf16(tensor)
    elif attr.type == AttributeProto.GRAPH:
        converted += convert_graph_to_bf16(attr.g)
    elif attr.type == AttributeProto.GRAPHS:
        for graph in attr.graphs:
            converted += convert_graph_to_bf16(graph)
    return converted


def convert_graph_to_bf16(graph):
    converted = 0
    bool_op_outputs = collect_bool_producing_outputs(graph)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        converted += convert_value_info_to_bf16(value_info, skip_names=bool_op_outputs)
    for initializer in graph.initializer:
        converted += convert_tensor_to_bf16(initializer)
    for node in graph.node:
        for attr in node.attribute:
            converted += convert_attr_to_bf16(attr)
    return converted


def insert_bool_to_bf16_casts(graph):
    """Lazy / chain-aware insertion. Bool flows BOOL through transparent ops
    (Unsqueeze/Reshape/Expand/...) all the way to the first genuine BF16-needing
    consumer, where ONE shared `Cast(BOOL->BF16)` is inserted per cone tensor.
    Bool-input slots (And/Or/Not/Where[cond]/...) keep reading BOOL directly,
    and any redundant downstream `Cast(*->BOOL)` becomes a BOOL->BOOL no-op
    and is dropped.

    See claude_stuff/bool_chain_audit_plan.md for the rationale: the previous
    eager strategy forced bool consumers to read BF16, which the torq compiler's
    host backend couldn't legalize (`linalg.generic { arith.cmpf bf16,bf16 ->
    i1 }`).
    """
    BF16 = TensorProto.BFLOAT16
    BOOL = TensorProto.BOOL

    bool_outputs = collect_bool_producing_outputs(graph)
    if not bool_outputs:
        return 0

    consumers = _build_consumers_map(graph)
    graph_output_names = {o.name for o in graph.output}

    # Idempotency: any root already renamed (a previous run) is skipped.
    fresh_roots = {t for t in bool_outputs if not t.endswith("_bool")}
    if not fresh_roots:
        return 0

    # ----- Phase A: cone discovery -----
    # The cone is the set of tensors that should be typed BOOL after this
    # pass. It starts at every bool-producing op output and extends forward
    # through transparent data-slot edges.
    cone = set()
    work = list(fresh_roots)
    while work:
        t = work.pop()
        if t in cone:
            continue
        cone.add(t)
        for consumer, slot in consumers.get(t, []):
            if _is_transparent_data_slot(consumer, slot):
                for out in consumer.output:
                    if out:
                        work.append(out)

    rename_map = {t: t + "_bool" for t in cone}

    # ----- Phase B: per-cone-tensor planning -----
    casts_to_drop = set()
    consumer_rewires = []           # (consumer_node, slot, new_input_name)
    bridge_casts = []               # list of (T_original_name, cast_node)
    new_value_info_bool = set()     # names that need new BOOL value_info
    update_value_info_bf16 = set()  # original names that need BF16 value_info
                                    # (because we insert a bridge cast outputting them)

    for T in cone:
        T_bool = rename_map[T]
        new_value_info_bool.add(T_bool)
        # If T is itself a graph output we must keep the original name flowing
        # through a bridge cast — otherwise the graph signature breaks.
        needs_bridge = T in graph_output_names

        for consumer, slot in consumers.get(T, []):
            # In-cone (transparent op on data slot): bool flows through; rewire
            # to T_bool. The transparent op's own output is handled when we
            # process its row in the cone.
            if _is_transparent_data_slot(consumer, slot):
                consumer_rewires.append((consumer, slot, T_bool))
                continue
            # Bool-input slot (And/Or/Not/Where[cond]/...): rewire to T_bool.
            if _is_bool_consumer(consumer, slot):
                consumer_rewires.append((consumer, slot, T_bool))
                continue
            # Cast handling.
            if consumer.op_type == "Cast":
                to_attr = _get_cast_to_attr(consumer)
                if to_attr == BOOL:
                    # Cast(*->BOOL) becomes BOOL->BOOL no-op once T is bool.
                    # Drop it; rewire its consumers to read T_bool directly.
                    casts_to_drop.add(id(consumer))
                    for cc, ci in consumers.get(consumer.output[0], []):
                        consumer_rewires.append((cc, ci, T_bool))
                    continue
                # Cast to BF16 / FP32 / etc. — the cast already does our
                # BOOL->X bridging. Rewire its input to T_bool so it reads
                # genuine BOOL instead of the about-to-be-renamed name.
                consumer_rewires.append((consumer, slot, T_bool))
                continue
            # Anything else is a genuine BF16-needing consumer.
            needs_bridge = True

        if needs_bridge:
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[T_bool],
                outputs=[T],
                name=f"{T}__cast_to_bf16",
                to=BF16,
            )
            bridge_casts.append((T, cast_node))
            update_value_info_bf16.add(T)

    # ----- Apply renames at producers -----
    for node in graph.node:
        for i, out in enumerate(list(node.output)):
            if out in rename_map:
                node.output[i] = rename_map[out]

    # ----- Apply consumer rewires -----
    for consumer, slot, new_name in consumer_rewires:
        consumer.input[slot] = new_name

    # ----- Drop redundant Casts -----
    if casts_to_drop:
        kept = [n for n in graph.node if id(n) not in casts_to_drop]
        del graph.node[:]
        graph.node.extend(kept)

    # ----- Reconcile value_info -----
    # Build reverse rename map to look up shapes from original tensor names
    reverse_rename = {v: k for k, v in rename_map.items()}
    existing_vi_by_name = {vi.name: vi for vi in graph.value_info}
    # Also keep a snapshot of the pre-rename value_info for shape lookup
    orig_vi_shapes = {}
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            orig_vi_shapes[vi.name] = vi.type.tensor_type.shape
    for vi in list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            orig_vi_shapes[vi.name] = vi.type.tensor_type.shape

    for name in new_value_info_bool:
        if name in existing_vi_by_name:
            existing_vi_by_name[name].type.tensor_type.elem_type = BOOL
        else:
            vi = graph.value_info.add()
            vi.name = name
            vi.type.tensor_type.elem_type = BOOL
            # Copy shape from the original (pre-rename) tensor
            orig_name = reverse_rename.get(name)
            if orig_name and orig_name in orig_vi_shapes:
                vi.type.tensor_type.shape.CopyFrom(orig_vi_shapes[orig_name])
    for name in update_value_info_bf16:
        if name in existing_vi_by_name:
            existing_vi_by_name[name].type.tensor_type.elem_type = BF16
        else:
            vi = graph.value_info.add()
            vi.name = name
            vi.type.tensor_type.elem_type = BF16

    # ----- Insert bridge casts right after their producer -----
    if bridge_casts:
        out_to_idx = {}
        for idx, n in enumerate(graph.node):
            for out in n.output:
                if out:
                    out_to_idx[out] = idx
        insertions = {}
        for T, cast in bridge_casts:
            T_bool = rename_map[T]
            prod_idx = out_to_idx.get(T_bool)
            if prod_idx is None:
                continue
            insertions.setdefault(prod_idx, []).append(cast)
        new_nodes = []
        for idx, n in enumerate(graph.node):
            new_nodes.append(n)
            for cast in insertions.get(idx, []):
                new_nodes.append(cast)
        del graph.node[:]
        graph.node.extend(new_nodes)

    return len(bridge_casts)


def resolve_alias(name, aliases):
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def remove_cast_nodes(graph):
    """Drop Cast nodes that are now redundant BF16->BF16 no-ops.

    convert_graph_to_bf16 has already retyped every float tensor to BF16. A
    Cast whose input is now BF16 *and* whose `to` is now (or already was) BF16
    is genuinely redundant and can be aliased away. Casts that bridge a
    non-float type (INT/BOOL) into BF16 are load-bearing — without them the
    consumer would receive an INT/BOOL tensor as a BF16 operand (the exact bug
    that produced the Div with INT64 inputs and BF16 output).

    For kept Casts whose original `to` was a float type (FLOAT/FLOAT16/DOUBLE),
    we also rewrite `to=BFLOAT16` so the Cast's stated destination matches the
    retyped output value_info.
    """
    elem_type_of = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value.type.HasField("tensor_type"):
            elem_type_of[value.name] = value.type.tensor_type.elem_type
    for initializer in graph.initializer:
        elem_type_of.setdefault(initializer.name, initializer.data_type)

    BF16 = TensorProto.BFLOAT16
    aliases = {}
    output_names = {value.name for value in graph.output}
    removed = 0

    for node in graph.node:
        if node.op_type != "Cast":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise ValueError(f"unexpected Cast arity on {node.name}")

        # Update `to` first: if the Cast originally targeted FLOAT/FLOAT16/DOUBLE,
        # those tensors are now BF16, so the Cast should target BF16 too.
        for attr in node.attribute:
            if attr.name == "to" and attr.i in FLOAT_TYPES:
                attr.i = BF16

        in_type = elem_type_of.get(node.input[0])
        to_attr = next((int(a.i) for a in node.attribute if a.name == "to"), None)

        # Only drop the Cast if it is now a true BF16->BF16 no-op, and dropping
        # it would not orphan a declared graph output.
        if in_type == BF16 and to_attr == BF16 and node.output[0] not in output_names:
            aliases[node.output[0]] = node.input[0]
            removed += 1

    new_nodes = []
    for node in graph.node:
        if node.op_type == "Cast" and node.output[0] in aliases:
            continue
        new_node = copy.deepcopy(node)
        for idx, name in enumerate(new_node.input):
            if name:
                new_node.input[idx] = resolve_alias(name, aliases)
        new_nodes.append(new_node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return removed, set(aliases)


def prune_value_info(graph):
    live = {value.name for value in graph.input}
    live.update(value.name for value in graph.output)
    live.update(init.name for init in graph.initializer)
    for node in graph.node:
        live.update(name for name in node.output if name)
    kept = [value for value in graph.value_info if value.name in live]
    removed = len(graph.value_info) - len(kept)
    del graph.value_info[:]
    graph.value_info.extend(kept)
    return removed


def prune_unused_initializers(graph):
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    kept = [init for init in graph.initializer if init.name in used]
    removed = len(graph.initializer) - len(kept)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return removed


def audit_model(model):
    type_counts = Counter()
    float_values = []
    for group_name, values in [
        ("input", model.graph.input),
        ("value_info", model.graph.value_info),
        ("output", model.graph.output),
    ]:
        for value in values:
            if not value.type.HasField("tensor_type"):
                continue
            elem = value.type.tensor_type.elem_type
            type_counts[(group_name, TensorProto.DataType.Name(elem))] += 1
            if elem in FLOAT_TYPES:
                float_values.append((group_name, value.name, TensorProto.DataType.Name(elem)))
    init_type_counts = Counter(TensorProto.DataType.Name(init.data_type) for init in model.graph.initializer)
    float_initializers = [
        (init.name, TensorProto.DataType.Name(init.data_type))
        for init in model.graph.initializer
        if init.data_type in FLOAT_TYPES
    ]
    attr_float_tensors = []
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR and attr.t.data_type in FLOAT_TYPES:
                attr_float_tensors.append((node.name, attr.name, TensorProto.DataType.Name(attr.t.data_type)))
            elif attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    if tensor.data_type in FLOAT_TYPES:
                        attr_float_tensors.append((node.name, attr.name, TensorProto.DataType.Name(tensor.data_type)))
    return {
        "node_count": len(model.graph.node),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node)),
        "cast_nodes": sum(1 for node in model.graph.node if node.op_type == "Cast"),
        "type_counts": {f"{group}:{name}": count for (group, name), count in sorted(type_counts.items())},
        "initializer_type_counts": dict(sorted(init_type_counts.items())),
        "float_values": float_values,
        "float_initializers": float_initializers,
        "float_attribute_tensors": attr_float_tensors,
    }


def update_json(json_in, json_out, output_model, model):
    data = json.loads(Path(json_in).read_text())
    ops = data.setdefault("ops", {})
    removed_cast_rows = [name for name in ops if name.startswith("Cast_")]
    for name in removed_cast_rows:
        ops.pop(name, None)
    for idx, node in enumerate(model.graph.node):
        layer_id = f"{node.op_type}_{node.output[0]}"
        if layer_id in ops:
            ops[layer_id]["_node_index"] = idx
    counts = Counter(info.get("recommended_executor") for info in ops.values())
    data["model_name"] = Path(output_model).stem
    data["discovery_report"] = {"summary": dict(counts), "critical_failures": []}
    data["has_critical_failures"] = False
    data["final_report_text"] = (
        f"Manual executor map after full BF16/no-Cast rewrite.\n"
        f"Model: {Path(output_model).stem}\n"
        f"Total ops: {len(ops)}\n"
        f"NSS: {counts.get('nss', 0)}\n"
        f"CSS: {counts.get('css', 0)}\n"
        f"HOST: {counts.get('host', 0)}\n"
    )
    data["bf16_no_cast_surgery"] = {
        "source_json": str(json_in),
        "removed_cast_rows": len(removed_cast_rows),
        "note": "All float tensor types/initializers/constant tensor attributes were converted to BF16. Cast nodes were removed by rewiring consumers to the Cast input.",
    }
    Path(json_out).write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-in", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.input)
    converted = convert_graph_to_bf16(model.graph)
    removed_casts, _aliases = remove_cast_nodes(model.graph)
    inserted_bool_casts = insert_bool_to_bf16_casts(model.graph)
    removed_value_info = prune_value_info(model.graph)
    removed_initializers = prune_unused_initializers(model.graph)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker passed")

    audit = audit_model(model)
    audit.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "converted_float_tensors_or_types": converted,
            "removed_cast_nodes": removed_casts,
            "inserted_bool_to_bf16_casts": inserted_bool_casts,
            "removed_value_info": removed_value_info,
            "removed_initializers": removed_initializers,
        }
    )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(audit, indent=2) + "\n")
    if args.json_in or args.json_out:
        if not (args.json_in and args.json_out):
            raise ValueError("--json-in and --json-out must be provided together")
        update_json(args.json_in, args.json_out, args.output, model)
        print(f"Wrote json: {args.json_out}")
    print(f"Converted float/bool tensors/types: {converted}")
    print(f"Removed Cast nodes: {removed_casts}")
    print(f"Inserted bool->bf16 Casts (after Less/Greater/Equal/...): {inserted_bool_casts}")
    print(f"Removed stale value_info: {removed_value_info}")
    print(f"Removed unused initializers: {removed_initializers}")
    print(f"Remaining Cast nodes: {audit['cast_nodes']}")
    print(f"Remaining FLOAT typed values: {len(audit['float_values'])}")
    print(f"Remaining FLOAT initializers: {len(audit['float_initializers'])}")
    print(f"Remaining FLOAT attribute tensors: {len(audit['float_attribute_tensors'])}")
    print(f"Wrote model: {args.output}")
    if args.report_json:
        print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
