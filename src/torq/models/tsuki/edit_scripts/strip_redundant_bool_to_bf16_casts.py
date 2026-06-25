#!/usr/bin/env python3
"""Strip `Cast(BOOL→BF16)` nodes that sit between two bool-only ops.

`make_full_bf16_no_casts.py` unconditionally inserts a `Cast(BOOL→BF16)` after
every bool-producing op (Less / Greater / And / Or / Not / ...) so downstream
BF16-typed consumers can read them. But when the *next* op is itself a
bool-consuming op (And, Or, Not, Xor, Where[cond], etc.), that cast is
useless — and worse, it forces the bool consumer to read BF16, which the torq
compiler's host backend can't legalize for some shapes:

    error: failed to legalize operation 'linalg.generic' that was explicitly
           marked illegal (encountered while running the pipeline to checking
           if a tile fits in memory)

This pass:
  1. Finds every Cast whose input is BOOL and `to=BFLOAT16`.
  2. For each consumer of the cast output that needs bool on a specific port,
     re-routes that port to the *original* BOOL tensor (the cast's input).
  3. If all consumers were bool-consuming and the cast output isn't a graph
     output, drops the Cast.

Other consumers (genuine BF16 ones, or the graph output) are left alone — so
this is safe to run on any model that's been through the bool→bf16 pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


BF16 = TensorProto.BFLOAT16
BOOL = TensorProto.BOOL

# Ops whose specific input slots are BOOL-typed per the ONNX spec.
BOOL_INPUT_OPS = {
    "And":      (0, 1),
    "Or":       (0, 1),
    "Xor":      (0, 1),
    "Not":      (0,),
    "Where":    (0,),   # only condition (input 0); X/Y can be any type
    "If":       (0,),
    "Compress": (1,),
}

# Ops whose output element type is exactly their first input's type — they
# pass dtype through unchanged, so we can retype the whole tensor chain by
# just updating value_info, no structural changes needed.
TRANSPARENT_OPS = frozenset({
    "Unsqueeze", "Squeeze", "Reshape", "Transpose", "Expand", "Tile",
    "Identity", "Slice", "Concat", "Pad", "Flatten", "Split",
    "GatherElements", "Gather", "GatherND",
})


def get_attr_int(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def build_types(graph):
    types = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        types.setdefault(init.name, init.data_type)
    return types


def is_bool_consumer(node, input_idx):
    slots = BOOL_INPUT_OPS.get(node.op_type, ())
    return input_idx in slots


def _build_consumers(graph):
    consumers = {}
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp:
                consumers.setdefault(inp, []).append((n, i))
    return consumers


def _try_retype_cone(cast_node, graph, types, consumers, graph_outputs):
    """Try to retype every tensor reachable from `cast_node.output` (via
    transparent ops) back to BOOL. Returns
        (success, cone_tensors, redundant_cast_ids, downstream_rewires)
    or (False, None, None, None) if any leaf is unsafe (graph output, mixed
    consumers, non-transparent / non-bool consumer reachable).

    A consumer is "safe" if it is:
      - a bool-only consumer port (And/Or/Not/Where[cond]/…)
      - a transparent shape/layout op (Unsqueeze/Reshape/Expand/…)
      - a `Cast` whose `to=BOOL` (becomes BOOL→BOOL no-op after retype; we
        mark it for removal and rewire its consumers to the cone tensor)
    """
    cone_tensors: set[str] = set()
    transparent_ids: set[int] = set()
    redundant_cast_ids: set[int] = set()
    downstream_rewires: list[tuple] = []  # (consumer, slot, tensor_name)

    work = [cast_node.output[0]]
    while work:
        t = work.pop()
        if t in cone_tensors:
            continue
        cone_tensors.add(t)
        if t in graph_outputs:
            return False, None, None, None
        for consumer, slot in consumers.get(t, []):
            # Leaf: a bool-only op port reads our cone tensor (now BOOL).
            if is_bool_consumer(consumer, slot):
                continue
            # Leaf: a Cast(*→BOOL) becomes a BOOL→BOOL no-op after we retype
            # this tensor to BOOL. Drop the Cast and re-route its consumers
            # to read the cone tensor directly.
            if consumer.op_type == "Cast":
                to_attr = get_attr_int(consumer, "to")
                if to_attr == BOOL:
                    redundant_cast_ids.add(id(consumer))
                    for cc, ci in consumers.get(consumer.output[0], []):
                        downstream_rewires.append((cc, ci, t))
                    continue
                # Cast to anything else: would change the input dtype the Cast
                # was originally getting. Conservative — bail.
                return False, None, None, None
            # Walk through transparent ops; their outputs join the cone.
            if consumer.op_type in TRANSPARENT_OPS:
                # Gather*: the indices port must stay int — don't traverse.
                if consumer.op_type in ("Gather", "GatherElements", "GatherND") and slot == 1:
                    return False, None, None, None
                transparent_ids.add(id(consumer))
                for out in consumer.output:
                    if out:
                        work.append(out)
                continue
            # Genuine BF16 consumer (Mul, Add, Conv, MatMul, …) — bail.
            return False, None, None, None
    return True, cone_tensors, redundant_cast_ids, downstream_rewires


def _update_value_info_dtype(graph, name, new_dtype):
    """Find name in graph.value_info / output and set its tensor elem_type."""
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name == name and vi.type.HasField("tensor_type"):
            vi.type.tensor_type.elem_type = new_dtype
            return


def strip(model):
    graph = model.graph
    types = build_types(graph)
    graph_outputs = {o.name for o in graph.output}
    consumers = _build_consumers(graph)

    casts_dropped = 0
    casts_dropped_names = []
    rewires = 0
    cones_retyped = 0
    cone_tensors_retyped = 0
    nodes_to_remove = set()

    # Pass 1: SMART CONE RETYPING. For each Cast(BOOL→BF16), try to retype
    # the entire downstream cone (through Unsqueeze/Reshape/Expand/...) back
    # to BOOL when every leaf is a bool-only consumer. This subsumes the
    # naive "1-hop" strip and also handles cases like
    #   Cast(BOOL→BF16) → Unsqueeze → Expand → Or
    # by retyping the Unsqueeze/Expand outputs to BOOL and dropping the Cast.
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        if id(node) in nodes_to_remove:
            continue
        in_name = node.input[0]
        out_name = node.output[0]
        if types.get(in_name) != BOOL:
            continue
        to_attr = get_attr_int(node, "to")
        if to_attr != BF16:
            continue

        ok, cone, redundant_cast_ids, downstream_rewires = _try_retype_cone(
            node, graph, types, consumers, graph_outputs)
        if not ok:
            continue

        # Retype all cone tensors to BOOL.
        for t in cone:
            _update_value_info_dtype(graph, t, BOOL)
            types[t] = BOOL
        cone_tensors_retyped += len(cone)
        cones_retyped += 1

        # Re-route consumers of the head Cast (originally BOOL→BF16) to read
        # cast.input (the original BOOL tensor) directly. Their dtype now
        # matches because the cone tensors are typed BOOL.
        for consumer, slot in consumers.get(out_name, []):
            consumer.input[slot] = in_name
            rewires += 1
        nodes_to_remove.add(id(node))
        casts_dropped += 1
        casts_dropped_names.append(node.name or "<unnamed>")

        # Drop any leaf Cast(*→BOOL) ops that became BOOL→BOOL no-ops, and
        # rewire their consumers to read the cone tensor directly.
        for cast_id in redundant_cast_ids:
            nodes_to_remove.add(cast_id)
            casts_dropped += 1
        for consumer, slot, t in downstream_rewires:
            # If the rewire target is the head Cast's output (which we just
            # dropped — has no producer anymore), redirect to the head Cast's
            # input (the original BOOL tensor).
            if t == out_name:
                t = in_name
            consumer.input[slot] = t
            rewires += 1

    if nodes_to_remove:
        kept = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(kept)

    # Second pass: any remaining bool-consumer ports whose input is BF16 (e.g.,
    # because there's a transparent op like Unsqueeze/Reshape between the
    # Cast(BOOL→BF16) and the bool consumer) need an explicit Cast(BF16→BOOL)
    # inserted right before the consumer. This matches what bf16_to_fp32.py's
    # _fix_bool_consumers does for the FP32-sanitizer side.
    types = build_types(graph)  # rebuild — first pass may have changed types
    used_names = (set(types.keys())
                  | {n.name for n in graph.node if n.name}
                  | {init.name for init in graph.initializer})

    inserted_casts = []
    casts_inserted = 0
    for node_idx, node in enumerate(graph.node):
        slots = BOOL_INPUT_OPS.get(node.op_type, ())
        for slot in slots:
            if slot >= len(node.input):
                continue
            in_name = node.input[slot]
            if not in_name:
                continue
            in_type = types.get(in_name)
            if in_type == BOOL:
                continue  # already bool; nothing to do
            base = f"{in_name}__to_bool_{node.name or node_idx}_{slot}"
            cast_out_name = base
            suffix = 0
            while cast_out_name in used_names:
                suffix += 1
                cast_out_name = f"{base}_{suffix}"
            used_names.add(cast_out_name)

            cast_node = helper.make_node(
                "Cast", [in_name], [cast_out_name],
                name=f"strip_to_bool_cast_{node_idx}_{slot}",
                to=BOOL,
            )
            inserted_casts.append((node_idx, cast_node))
            node.input[slot] = cast_out_name
            types[cast_out_name] = BOOL
            casts_inserted += 1

    if inserted_casts:
        by_target: dict[int, list] = {}
        for tgt, cast in inserted_casts:
            by_target.setdefault(tgt, []).append(cast)
        new_nodes = []
        for i, n in enumerate(graph.node):
            new_nodes.extend(by_target.get(i, []))
            new_nodes.append(n)
        del graph.node[:]
        graph.node.extend(new_nodes)

    return {
        "casts_dropped": casts_dropped,
        "casts_dropped_names": casts_dropped_names,
        "rewires": rewires,
        "casts_inserted": casts_inserted,
        "cones_retyped": cones_retyped,
        "cone_tensors_retyped": cone_tensors_retyped,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    model = onnx.load(str(args.input))
    report = strip(model)

    print(f"Bool cones retyped         : {report['cones_retyped']}")
    print(f"  tensors retyped to BOOL  : {report['cone_tensors_retyped']}")
    print(f"Rewired bool consumers     : {report['rewires']}")
    print(f"Redundant Cast→BF16 dropped: {report['casts_dropped']}")
    print(f"Cast(BF16→BOOL) inserted   : {report.get('casts_inserted', 0)}")
    for nm in report["casts_dropped_names"][:25]:
        print(f"  dropped: {nm}")
    if len(report["casts_dropped_names"]) > 25:
        print(f"  ... ({len(report['casts_dropped_names']) - 25} more)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
