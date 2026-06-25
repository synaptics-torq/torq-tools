#!/usr/bin/env python3
"""Split tsuki part-A at the bool-mask boundary.

Produces three artefacts from a single source ONNX:

  <output-dir>/
      part_a1_pre_mask.onnx              # stage 1: text -> durations + features
      part_a_mask_compute.py             # stage 2: hand-written numpy masks
      part_a_mask_compute_constants.npz  # constants the numpy stage needs
      part_a2_post_mask.onnx             # stage 3: features + masks -> outputs
      split_manifest.json                # I/O schema for orchestration

Algorithm summary (see claude_stuff/length_prediction_split_audit.md for the
full audit):

  1. Find the bool cone: BFS from every bool-producing op output through
     transparent shape ops (Unsqueeze/Squeeze/Reshape/Transpose/Expand/...).
  2. Find boundary ops: non-cone ops (Cast / Where) that read a bool-cone
     tensor on a bool-required slot. These produce the 14 boundary outputs.
  3. Combine cone + boundary as the 'numpy_cone' to move out of the ONNX.
  4. Stage 3 = forward reachability from boundary outputs (excluding numpy_cone).
  5. Stage 1 = everything else (excluding numpy_cone).
  6. Build stage-1 and stage-3 ONNXs via _build_model_from_node_subset.
  7. Generate stage-2 numpy by walking the numpy_cone topologically and
     emitting one numpy expression per op.

Works on either the PRE-surgery or POST-surgery model: the detector keys
on op_type + Cast 'to=BOOL' attribute, not on tensor names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Op categorization (mirrors scripts/audit_bool_chains.py)
# ---------------------------------------------------------------------------

BOOL_PRODUCING_OPS = frozenset({
    "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal",
    "And", "Or", "Xor", "Not", "IsNaN", "IsInf",
})

BOOL_INPUT_OPS: dict[str, tuple[int, ...]] = {
    "And":      (0, 1),
    "Or":       (0, 1),
    "Xor":      (0, 1),
    "Not":      (0,),
    "Where":    (0,),
    "If":       (0,),
    "Compress": (1,),
}

# Single-data-input shape ops we extend the cone through. Concat/Split excluded
# (multi-input dtype invariants we don't want to police).
TRANSPARENT_OPS = frozenset({
    "Unsqueeze", "Squeeze", "Reshape", "Transpose", "Expand", "Tile",
    "Identity", "Slice", "Pad", "Flatten",
})

# Per-op slots that take int64 index/shape inputs (never bool chains).
PASSTHROUGH_INDEX_SLOTS = {
    "Slice":     (1, 2, 3, 4),
    "Unsqueeze": (1,),
    "Squeeze":   (1,),
    "Reshape":   (1,),
    "Expand":    (1,),
    "Tile":      (1,),
    "Pad":       (1, 2, 3),
}

BOOL = TensorProto.BOOL

BRIDGE_ELIGIBLE_OPS = frozenset({
    "Mul", "Add", "Sub", "Div", "Neg", "Abs", "Ceil", "Floor", "Round", "Clip",
    "ArgMax", "ReduceSum", "ReduceMean", "Min", "Max", "CumSum",
    "Cast", "Unsqueeze", "Squeeze", "Reshape", "Slice", "Concat",
    "Gather", "GatherElements", "GatherND", "Shape", "Range", "ConstantOfShape",
    "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal",
    "And", "Or", "Not", "Where",
    "Pad", "Flatten", "Identity", "Transpose", "Expand", "Tile",
    "Sqrt", "Tanh", "Sigmoid", "Relu", "LeakyRelu",
    "Pow", "Reciprocal", "Exp", "Log",
})

HEAVY_COMPUTE_OPS = frozenset({
    "Conv", "ConvTranspose", "MatMul", "Gemm",
    "Softmax", "Gelu",
    "LayerNormalization", "BatchNormalization", "InstanceNormalization",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_attr_int(node, name):
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return None


def get_attr_ints(node, name):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints)
    return None


def is_transparent_data_slot(node, slot):
    if node.op_type not in TRANSPARENT_OPS:
        return False
    if slot in PASSTHROUGH_INDEX_SLOTS.get(node.op_type, ()):
        return False
    return slot == 0


def build_consumers_map(graph):
    consumers = defaultdict(list)
    for i, n in enumerate(graph.node):
        for slot, inp in enumerate(n.input):
            if inp:
                consumers[inp].append((i, slot))
    return consumers


def build_producer_map(graph):
    producer = {}
    for i, n in enumerate(graph.node):
        for o in n.output:
            if o:
                producer[o] = i
    return producer


def py_name(tensor_name: str) -> str:
    """Sanitize ONNX tensor name to a valid Python identifier."""
    safe = tensor_name.replace(".", "_").replace("-", "_").replace("/", "_")
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


# ---------------------------------------------------------------------------
# Fast subgraph builder (replaces the O(N^3) helper in torq.testing.onnx)
# ---------------------------------------------------------------------------

def build_subgraph(model, node_indices_set, graph_name, *, log_prefix=""):
    """Like torq.testing.onnx._build_model_from_node_subset but uses sets +
    a precomputed consumer map so it runs in O(N) on a 7500-node graph
    rather than O(N^3).

    Args:
        model: full ONNX model
        node_indices_set: set of integer node indices to include in the subgraph
        graph_name: name for the new graph
        log_prefix: optional prefix for diagnostic prints

    Returns the new ONNX model (without running shape_inference / checker).
    """
    import copy as _copy

    graph = model.graph
    all_nodes = list(graph.node)
    nodes_subset = [all_nodes[i] for i in sorted(node_indices_set)]

    # produced / used by the subset
    produced = set()
    used = set()
    for n in nodes_subset:
        for o in n.output:
            if o:
                produced.add(o)
        for inp in n.input:
            if inp:
                used.add(inp)

    external_inputs = used - produced

    # external_outputs: a produced tensor is "external" if any OUTSIDE node consumes it,
    # OR if it's a declared model output.
    orig_output_names = {o.name for o in graph.output}
    consumers_outside_index_by_tensor = defaultdict(set)
    for i, n in enumerate(all_nodes):
        if i in node_indices_set:
            continue
        for inp in n.input:
            if inp:
                consumers_outside_index_by_tensor[inp].add(i)
    external_outputs = {
        name for name in produced
        if name in orig_output_names or consumers_outside_index_by_tensor.get(name)
    }

    orig_inputs = {vi.name: vi for vi in graph.input}
    orig_value_info = {vi.name: vi for vi in graph.value_info}
    orig_outputs = {vi.name: vi for vi in graph.output}
    orig_initializers = {init.name: init for init in graph.initializer}

    def _value_info_from_initializer(init):
        try:
            arr = numpy_helper.to_array(init)
            shape = list(arr.shape)
            return helper.make_tensor_value_info(init.name, init.data_type, shape)
        except Exception:
            return helper.make_tensor_value_info(init.name, TensorProto.FLOAT, [])

    new_inputs = []
    new_outputs = []
    new_initializers = []
    added_initializer_names = set()

    for name in sorted(external_inputs):
        if name in orig_inputs:
            new_inputs.append(_copy.deepcopy(orig_inputs[name]))
        elif name in orig_initializers:
            init = orig_initializers[name]
            new_initializers.append(_copy.deepcopy(init))
            added_initializer_names.add(name)
            new_inputs.append(_value_info_from_initializer(init))
        elif name in orig_value_info:
            new_inputs.append(_copy.deepcopy(orig_value_info[name]))
        elif name in orig_outputs:
            new_inputs.append(_copy.deepcopy(orig_outputs[name]))
        else:
            new_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    # All initializers consumed (whether external_input or fully internal) get copied.
    for name in sorted(used):
        if name in orig_initializers and name not in added_initializer_names:
            new_initializers.append(_copy.deepcopy(orig_initializers[name]))
            added_initializer_names.add(name)

    for name in sorted(external_outputs):
        if name in orig_outputs:
            new_outputs.append(_copy.deepcopy(orig_outputs[name]))
        elif name in orig_value_info:
            new_outputs.append(_copy.deepcopy(orig_value_info[name]))
        elif name in orig_initializers:
            new_outputs.append(_value_info_from_initializer(orig_initializers[name]))
        else:
            new_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    new_value_info = []
    for name in sorted(produced | used):
        if name in orig_value_info:
            new_value_info.append(_copy.deepcopy(orig_value_info[name]))

    nodes_copy = [_copy.deepcopy(n) for n in nodes_subset]

    new_graph = helper.make_graph(
        nodes_copy, graph_name,
        inputs=new_inputs, outputs=new_outputs, initializer=new_initializers,
    )
    if new_value_info:
        new_graph.value_info.extend(new_value_info)

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    return new_model


# ---------------------------------------------------------------------------
# Cone discovery
# ---------------------------------------------------------------------------

def discover_bool_cone(graph):
    """Return (cone_op_indices, boundary_op_indices, bool_cone_tensors).

    - cone_op_indices: bool-producing ops + their forward-extended transparent ops.
    - boundary_op_indices: non-cone Cast/Where that read a bool-cone tensor on a
      bool-input slot (these are also pulled into the numpy stage).
    - bool_cone_tensors: set of tensor names known to be BOOL after extension.
    """
    consumers = build_consumers_map(graph)

    # Step 1: bool-producing op outputs + Cast(*->BOOL) outputs are seed bool tensors.
    bool_tensors: set[str] = set()
    cone_indices: set[int] = set()
    for i, n in enumerate(graph.node):
        if n.op_type in BOOL_PRODUCING_OPS:
            cone_indices.add(i)
            for o in n.output:
                if o:
                    bool_tensors.add(o)
        elif n.op_type == "Cast" and get_attr_int(n, "to") == BOOL:
            cone_indices.add(i)
            for o in n.output:
                if o:
                    bool_tensors.add(o)

    # Step 2: extend cone through transparent ops on the data slot.
    work = list(bool_tensors)
    while work:
        t = work.pop()
        for ci, slot in consumers.get(t, []):
            cn = graph.node[ci]
            if not is_transparent_data_slot(cn, slot):
                continue
            if ci in cone_indices:
                continue
            cone_indices.add(ci)
            for out in cn.output:
                if out and out not in bool_tensors:
                    bool_tensors.add(out)
                    work.append(out)

    # Step 3: boundary ops = non-cone consumers of bool tensors on bool slots.
    # These include any Cast (regardless of `to`) and any Where reading the cond slot.
    boundary_indices: set[int] = set()
    for t in bool_tensors:
        for ci, slot in consumers.get(t, []):
            if ci in cone_indices:
                continue
            cn = graph.node[ci]
            if cn.op_type == "Cast":
                # Cast of a bool tensor is a boundary (convert bool to numeric).
                boundary_indices.add(ci)
            elif cn.op_type in BOOL_INPUT_OPS and slot in BOOL_INPUT_OPS[cn.op_type]:
                # Where/And/Or/Not/etc. on the bool-required slot.
                # And/Or/Not should already be in cone_indices; this catches Where.
                boundary_indices.add(ci)

    return cone_indices, boundary_indices, bool_tensors


def expand_cone_with_bridges(graph, cone_indices, producer_map, *, log_prefix=""):
    """Backward BFS from cone op inputs through bridge-eligible ops.

    Pulls upstream ops into the cone when they are light-weight (arithmetic,
    shape manipulation, reductions) and stops at heavy compute ops (Conv,
    MatMul, etc.) whose outputs become stage-1 → numpy boundary tensors.
    """
    expanded = set(cone_indices)
    initializer_names = {init.name for init in graph.initializer}
    g_input_names = {gi.name for gi in graph.input}

    work = []
    for ci in cone_indices:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is not None and prod not in expanded:
                work.append(prod)

    while work:
        ci = work.pop()
        if ci in expanded:
            continue
        node = graph.node[ci]
        if node.op_type in HEAVY_COMPUTE_OPS:
            continue
        if node.op_type not in BRIDGE_ELIGIBLE_OPS:
            continue
        expanded.add(ci)
        for inp in node.input:
            if not inp:
                continue
            if inp in initializer_names or inp in g_input_names:
                continue
            prod = producer_map.get(inp)
            if prod is not None and prod not in expanded:
                work.append(prod)

    n_added = len(expanded) - len(cone_indices)
    if n_added:
        print(f"{log_prefix} bridge expansion: +{n_added} ops pulled into cone")
    return expanded


def _expand_exclusive_bridges(graph, cone_indices, producer_map, consumers_map,
                              *, log_prefix=""):
    """Pull upstream ops into the cone ONLY if all their consumers are already
    in the cone. This prevents pulling in ops shared with the main model
    compute graph, which would create circular dependencies."""
    expanded = set(cone_indices)
    initializer_names = {init.name for init in graph.initializer}
    g_input_names = {gi.name for gi in graph.input}

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        candidates = set()
        for ci in expanded:
            for inp in graph.node[ci].input:
                if not inp:
                    continue
                if inp in initializer_names or inp in g_input_names:
                    continue
                prod = producer_map.get(inp)
                if prod is None or prod in expanded:
                    continue
                node = graph.node[prod]
                if node.op_type in HEAVY_COMPUTE_OPS:
                    continue
                if node.op_type not in BRIDGE_ELIGIBLE_OPS:
                    continue
                # Check: ALL consumers of this op's outputs must be in expanded
                all_consumers_in_cone = True
                for out in node.output:
                    if not out:
                        continue
                    for cci, _ in consumers_map.get(out, []):
                        if cci not in expanded:
                            all_consumers_in_cone = False
                            break
                    if not all_consumers_in_cone:
                        break
                if all_consumers_in_cone:
                    candidates.add(prod)
        if candidates:
            expanded |= candidates
            changed = True
            print(f"{log_prefix} exclusive bridge iter {iteration}: "
                  f"+{len(candidates)} ops (all consumers in cone)")

    n_added = len(expanded) - len(cone_indices)
    if n_added:
        print(f"{log_prefix} exclusive bridge total: +{n_added} ops")
    return expanded


def _resolve_between_cone_ops(graph, cone_indices, producer_map, consumers_map,
                              *, log_prefix=""):
    """Pull in non-cone ops that sit between two cone ops on the output side.

    These are ops that:
    1. Consume a tensor produced by a cone op (i.e., they read a boundary output)
    2. Produce a tensor consumed by another cone op

    Without pulling them in, they'd end up in stage_3 (downstream of a boundary
    output) while their output is needed by numpy_post → circular dependency.

    Only pulls in bridge-eligible ops (no Conv/MatMul/etc).
    """
    expanded = set(cone_indices)
    changed = True
    total_added = 0
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        boundary_outputs = discover_boundary_outputs(graph, expanded)
        added_this_iter = 0
        for bo in boundary_outputs:
            for ci, _ in consumers_map.get(bo, []):
                if ci in expanded:
                    continue
                node = graph.node[ci]
                if node.op_type in HEAVY_COMPUTE_OPS:
                    continue
                if node.op_type not in BRIDGE_ELIGIBLE_OPS:
                    continue
                # Check: does any of this op's outputs feed back into the cone?
                feeds_cone = False
                for out in node.output:
                    if not out:
                        continue
                    for cci, _ in consumers_map.get(out, []):
                        if cci in expanded:
                            feeds_cone = True
                            break
                    if feeds_cone:
                        break
                if feeds_cone:
                    expanded.add(ci)
                    changed = True
                    added_this_iter += 1
        if added_this_iter:
            total_added += added_this_iter
            print(f"{log_prefix} between-cone iter {iteration}: "
                  f"+{added_this_iter} ops bridging boundary→cone")

    if total_added:
        print(f"{log_prefix} between-cone total: +{total_added} ops")
    return expanded


# Op types we treat as "trivial" for the trivial-origin classifier — they
# don't introduce dependency on the bulk of the model. Used by Option C.
_TRIVIAL_ORIGIN_OPS = frozenset({
    "Unsqueeze", "Squeeze", "Reshape", "Slice", "Cast", "Identity",
    "Transpose", "Expand", "Tile", "Pad", "Flatten", "Concat",
    "Gather", "GatherElements", "GatherND", "Shape", "Range",
    "Constant", "ConstantOfShape",
})




def classify_early_late(graph, numpy_cone_indices, producer_map):
    """Partition the cone into 'early' (computable from graph_inputs +
    initializers alone, via trivial shape/index ops) and 'late' (depends
    on some stage_1 native compute).

    Topologically propagates lateness through cone-internal dependencies:
    a cone op X is early iff ALL its cone inputs are early AND all its
    non-cone inputs trace back through trivial ops to graph_inputs/inits.
    """
    g_inputs = {gi.name for gi in graph.input}
    inits = {init.name for init in graph.initializer}

    memo: dict[str, bool] = {}

    def trivial_origin(tname: str) -> bool:
        if tname in memo:
            return memo[tname]
        if tname in g_inputs or tname in inits:
            memo[tname] = True
            return True
        prod = producer_map.get(tname)
        if prod is None:
            memo[tname] = False
            return False
        n = graph.node[prod]
        if n.op_type not in _TRIVIAL_ORIGIN_OPS:
            memo[tname] = False
            return False
        memo[tname] = True  # provisional to break cycles
        ok = all(trivial_origin(i) for i in n.input if i)
        memo[tname] = ok
        return ok

    early: set[int] = set()
    late: set[int] = set()
    for ci in sorted(numpy_cone_indices):  # topological order
        n = graph.node[ci]
        is_early = True
        for inp in n.input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod in numpy_cone_indices:
                if prod in late:
                    is_early = False
                    break
                continue
            if not trivial_origin(inp):
                is_early = False
                break
        (early if is_early else late).add(ci)
    return early, late


def expand_cone_to_closure(graph, numpy_cone_indices, log_prefix=""):
    """Iteratively expand numpy_cone to include 'bridging' ops between
    early-cone outputs and late-cone inputs.

    A 'bridging' op is one that (a) is forward-reachable from some cone
    output AND (b) backward-reachable from some cone input, while not
    itself being in the cone. These ops would otherwise sit in stage-1 yet
    consume tensors produced by numpy, creating a circular dependency.

    Iterates until no bridging ops remain.
    """
    producer_map = build_producer_map(graph)
    consumers = build_consumers_map(graph)
    numpy_cone_indices = set(numpy_cone_indices)
    iteration = 0
    while True:
        iteration += 1
        # Forward closure from cone outputs (excluding cone)
        forward: set[int] = set()
        seed_outs: set[str] = set()
        for ci in numpy_cone_indices:
            for o in graph.node[ci].output:
                if o:
                    seed_outs.add(o)
        work = list(seed_outs)
        visited: set[str] = set()
        while work:
            t = work.pop()
            if t in visited:
                continue
            visited.add(t)
            for ci, _slot in consumers.get(t, []):
                if ci in numpy_cone_indices or ci in forward:
                    continue
                forward.add(ci)
                for out in graph.node[ci].output:
                    if out:
                        work.append(out)

        # Backward closure from cone inputs (excluding cone)
        seed_ins: set[str] = set()
        for ci in numpy_cone_indices:
            for inp in graph.node[ci].input:
                if inp:
                    seed_ins.add(inp)
        backward: set[int] = set()
        work = list(seed_ins)
        visited_t: set[str] = set()
        while work:
            t = work.pop()
            if t in visited_t:
                continue
            visited_t.add(t)
            prod = producer_map.get(t)
            if prod is None or prod in numpy_cone_indices or prod in backward:
                continue
            backward.add(prod)
            for inp in graph.node[prod].input:
                if inp:
                    work.append(inp)

        bridging = (forward & backward) - numpy_cone_indices
        if not bridging:
            print(f"{log_prefix} cone closure stable after {iteration} iteration(s)")
            return numpy_cone_indices
        print(f"{log_prefix} closure iter {iteration}: +{len(bridging)} bridging ops")
        numpy_cone_indices |= bridging


def discover_boundary_outputs(graph, numpy_cone_indices):
    """Find tensors produced by numpy_cone ops that are consumed by non-cone
    ops (or are model outputs). These become the new stage-3 graph inputs."""
    boundary_outputs: set[str] = set()
    graph_output_names = {o.name for o in graph.output}
    consumers = build_consumers_map(graph)
    for ci in numpy_cone_indices:
        for o in graph.node[ci].output:
            if not o:
                continue
            if o in graph_output_names:
                boundary_outputs.add(o)
                continue
            for cci, _slot in consumers.get(o, []):
                if cci not in numpy_cone_indices:
                    boundary_outputs.add(o)
                    break
    return boundary_outputs


def discover_stage3_forward(graph, numpy_cone_indices, boundary_outputs):
    """Stage 3 = forward closure from boundary outputs (excluding numpy_cone).

    Identifies the 'synthesis' ops downstream of the numpy boundary. Any op
    whose output is forward-reachable from a boundary output (through
    non-cone ops) ends up in stage_3.
    """
    consumers = build_consumers_map(graph)
    stage_3: set[int] = set()
    visited: set[str] = set()
    work = list(boundary_outputs)
    while work:
        t = work.pop()
        if t in visited:
            continue
        visited.add(t)
        for ci, _slot in consumers.get(t, []):
            if ci in numpy_cone_indices or ci in stage_3:
                continue
            stage_3.add(ci)
            for out in graph.node[ci].output:
                if out:
                    work.append(out)
    return stage_3


def discover_stage1(graph, numpy_cone_indices, producer_map):
    """Stage 1 = backwards-closure from numpy_cone inputs through non-cone ops.

    Every op whose output flows (directly or transitively) into the numpy
    stage. This guarantees the numpy stage has all the inputs it needs.
    """
    seeds: set[int] = set()
    for ci in numpy_cone_indices:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is not None and prod not in numpy_cone_indices:
                seeds.add(prod)

    stage_1_indices: set[int] = set()
    work = list(seeds)
    while work:
        ci = work.pop()
        if ci in stage_1_indices:
            continue
        stage_1_indices.add(ci)
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is not None and prod not in numpy_cone_indices and prod not in stage_1_indices:
                work.append(prod)
    return stage_1_indices


def topo_sort(node_indices, graph):
    """Return node_indices in topological order matching the graph's natural order."""
    indices_set = set(node_indices)
    return [i for i in range(len(graph.node)) if i in indices_set]


# ---------------------------------------------------------------------------
# Numpy code generation
# ---------------------------------------------------------------------------

# Map ONNX TensorProto dtypes to numpy dtype strings used in generated code.
_ONNX_TO_NP = {
    TensorProto.FLOAT:    "np.float32",
    TensorProto.DOUBLE:   "np.float64",
    TensorProto.FLOAT16:  "np.float16",
    TensorProto.BFLOAT16: "np.float32",  # numpy has no bf16; cast through f32
    TensorProto.INT8:     "np.int8",
    TensorProto.INT16:    "np.int16",
    TensorProto.INT32:    "np.int32",
    TensorProto.INT64:    "np.int64",
    TensorProto.UINT8:    "np.uint8",
    TensorProto.UINT16:   "np.uint16",
    TensorProto.UINT32:   "np.uint32",
    TensorProto.UINT64:   "np.uint64",
    TensorProto.BOOL:     "np.bool_",
}


def onnx_to_numpy_dtype_str(onnx_dtype: int) -> str:
    return _ONNX_TO_NP.get(onnx_dtype, "np.float32")


def emit_op_expr(node, input_vars: dict[str, str]) -> str:
    """Emit a Python expression that computes this node's output from its
    inputs. `input_vars` maps each ONNX input name to the local Python
    variable name holding its value.
    """
    ins = [input_vars[i] for i in node.input if i]
    op = node.op_type

    if op == "Less":           return f"np.less({ins[0]}, {ins[1]})"
    if op == "LessOrEqual":    return f"np.less_equal({ins[0]}, {ins[1]})"
    if op == "Greater":        return f"np.greater({ins[0]}, {ins[1]})"
    if op == "GreaterOrEqual": return f"np.greater_equal({ins[0]}, {ins[1]})"
    if op == "Equal":          return f"np.equal({ins[0]}, {ins[1]})"
    if op == "And":            return f"np.logical_and({ins[0]}, {ins[1]})"
    if op == "Or":             return f"np.logical_or({ins[0]}, {ins[1]})"
    if op == "Xor":            return f"np.logical_xor({ins[0]}, {ins[1]})"
    if op == "Not":            return f"np.logical_not({ins[0]})"
    if op == "IsNaN":          return f"np.isnan({ins[0]})"
    if op == "IsInf":          return f"np.isinf({ins[0]})"

    if op == "Cast":
        to = get_attr_int(node, "to")
        np_dtype = onnx_to_numpy_dtype_str(to)
        return f"{ins[0]}.astype({np_dtype})"

    if op == "Where":
        # np.where preserves the dtype of x; ensure broadcasting works.
        return f"np.where({ins[0]}, {ins[1]}, {ins[2]})"

    if op == "Unsqueeze":
        # ONNX-13+: axes is input[1]. Pre-13: attribute. We support both.
        if len(ins) >= 2:
            return f"np.expand_dims({ins[0]}, axis=tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)))"
        axes = get_attr_ints(node, "axes") or []
        return f"np.expand_dims({ins[0]}, axis={tuple(axes)!r})"

    if op == "Squeeze":
        if len(ins) >= 2:
            return f"np.squeeze({ins[0]}, axis=tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)))"
        axes = get_attr_ints(node, "axes")
        if axes:
            return f"np.squeeze({ins[0]}, axis={tuple(axes)!r})"
        return f"np.squeeze({ins[0]})"

    if op == "Reshape":
        return f"np.asarray({ins[0]}).reshape([int(x) for x in np.asarray({ins[1]}).reshape(-1)])"

    if op == "Expand":
        return f"np.broadcast_to({ins[0]}, tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)))"

    if op == "Transpose":
        perm = get_attr_ints(node, "perm")
        if perm is None:
            return f"np.transpose({ins[0]})"
        return f"np.transpose({ins[0]}, axes={tuple(perm)!r})"

    if op == "Tile":
        return f"np.tile({ins[0]}, tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)))"

    if op == "Identity":
        return f"np.asarray({ins[0]})"

    # --- Bridge ops (arithmetic, reductions, slicing) ---

    if op == "Mul":       return f"np.multiply({ins[0]}, {ins[1]})"
    if op == "Add":       return f"np.add({ins[0]}, {ins[1]})"
    if op == "Sub":       return f"np.subtract({ins[0]}, {ins[1]})"
    if op == "Div":       return f"np.divide({ins[0]}, {ins[1]})"
    if op == "Neg":       return f"np.negative({ins[0]})"
    if op == "Abs":       return f"np.abs({ins[0]})"
    if op == "Ceil":      return f"np.ceil({ins[0]})"
    if op == "Floor":     return f"np.floor({ins[0]})"

    if op == "Round":
        return f"np.round({ins[0]})"

    if op == "Sqrt":       return f"np.sqrt({ins[0]})"
    if op == "Tanh":       return f"np.tanh({ins[0]})"
    if op == "Sigmoid":    return f"(1.0 / (1.0 + np.exp(-{ins[0]})))"
    if op == "Relu":       return f"np.maximum({ins[0]}, 0)"
    if op == "LeakyRelu":
        alpha = 0.01
        for a in node.attribute:
            if a.name == "alpha":
                alpha = a.f
        return f"np.where({ins[0]} >= 0, {ins[0]}, {alpha} * {ins[0]})"
    if op == "Pow":        return f"np.power({ins[0]}, {ins[1]})"
    if op == "Reciprocal": return f"np.reciprocal({ins[0]}.astype(np.float64)).astype({ins[0]}.dtype)"
    if op == "Exp":        return f"np.exp({ins[0]})"
    if op == "Log":        return f"np.log({ins[0]})"

    if op == "Clip":
        if len(ins) == 3:
            return f"np.clip({ins[0]}, {ins[1]}, {ins[2]})"
        if len(ins) == 2:
            return f"np.clip({ins[0]}, {ins[1]}, None)"
        return f"np.clip({ins[0]}, None, None)"

    if op == "ArgMax":
        axis = get_attr_int(node, "axis") or 0
        keepdims = get_attr_int(node, "keepdims")
        keepdims = True if keepdims is None else bool(keepdims)
        if keepdims:
            return f"np.expand_dims(np.argmax({ins[0]}, axis={axis}), axis={axis})"
        return f"np.argmax({ins[0]}, axis={axis})"

    if op == "ReduceSum":
        keepdims = get_attr_int(node, "keepdims")
        keepdims = True if keepdims is None else bool(keepdims)
        axes = get_attr_ints(node, "axes")
        if axes is not None:
            return f"np.sum({ins[0]}, axis={tuple(axes)!r}, keepdims={keepdims})"
        if len(ins) >= 2:
            return f"np.sum({ins[0]}, axis=tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)), keepdims={keepdims})"
        return f"np.sum({ins[0]}, keepdims={keepdims})"

    if op == "ReduceMean":
        keepdims = get_attr_int(node, "keepdims")
        keepdims = True if keepdims is None else bool(keepdims)
        axes = get_attr_ints(node, "axes")
        if axes is not None:
            return f"np.mean({ins[0]}, axis={tuple(axes)!r}, keepdims={keepdims})"
        if len(ins) >= 2:
            return f"np.mean({ins[0]}, axis=tuple(int(x) for x in np.asarray({ins[1]}).reshape(-1)), keepdims={keepdims})"
        return f"np.mean({ins[0]}, keepdims={keepdims})"

    if op == "Min":
        if len(ins) == 1:
            return f"np.min({ins[0]})"
        return f"np.minimum({ins[0]}, {ins[1]})"

    if op == "Max":
        if len(ins) == 1:
            return f"np.max({ins[0]})"
        return f"np.maximum({ins[0]}, {ins[1]})"

    if op == "CumSum":
        axis_expr = f"int(np.asarray({ins[1]}).flat[0])" if len(ins) >= 2 else "0"
        return f"np.cumsum({ins[0]}, axis={axis_expr})"

    if op == "Concat":
        axis = get_attr_int(node, "axis") or 0
        return f"np.concatenate([{', '.join(ins)}], axis={axis})"

    if op == "Slice":
        # ONNX Slice: data, starts, ends[, axes[, steps]]
        data = ins[0]
        starts = ins[1] if len(ins) > 1 else "np.array([0])"
        ends = ins[2] if len(ins) > 2 else f"np.array([{data}.shape[0]])"
        axes_var = ins[3] if len(ins) > 3 else None
        steps_var = ins[4] if len(ins) > 4 else None
        return (
            f"_onnx_slice({data}, {starts}, {ends}, "
            f"{axes_var if axes_var else 'None'}, "
            f"{steps_var if steps_var else 'None'})"
        )

    if op == "Gather":
        axis = get_attr_int(node, "axis") or 0
        return f"np.take({ins[0]}, np.asarray({ins[1]}).astype(np.intp), axis={axis})"

    if op == "GatherElements":
        axis = get_attr_int(node, "axis") or 0
        return f"np.take_along_axis({ins[0]}, np.asarray({ins[1]}).astype(np.intp), axis={axis})"

    if op == "GatherND":
        batch_dims = get_attr_int(node, "batch_dims") or 0
        return f"_onnx_gather_nd({ins[0]}, {ins[1]}, batch_dims={batch_dims})"

    if op == "Shape":
        return f"np.array({ins[0]}.shape, dtype=np.int64)"

    if op == "Range":
        return f"np.arange({ins[0]}.flat[0], {ins[1]}.flat[0], {ins[2]}.flat[0])"

    if op == "ConstantOfShape":
        val = None
        for a in node.attribute:
            if a.name == "value":
                val = numpy_helper.to_array(a.t).flat[0]
        if val is None:
            val = 0.0
        return f"np.full(tuple(int(x) for x in np.asarray({ins[0]}).reshape(-1)), {val!r})"

    if op == "Pad":
        if len(ins) >= 3:
            return f"np.pad({ins[0]}, list(zip(np.asarray({ins[1]}).reshape(-1)[:len({ins[0]}.shape)].tolist(), np.asarray({ins[1]}).reshape(-1)[len({ins[0]}.shape):].tolist())), constant_values={ins[2]}.flat[0])"
        return f"np.pad({ins[0]}, list(zip(np.asarray({ins[1]}).reshape(-1)[:len({ins[0]}.shape)].tolist(), np.asarray({ins[1]}).reshape(-1)[len({ins[0]}.shape):].tolist())))"

    if op == "Flatten":
        axis = get_attr_int(node, "axis")
        if axis is None:
            axis = 1
        return f"np.asarray({ins[0]}).reshape(({ins[0]}.shape[:{axis}] if {axis} > 0 else (1,)) + (-1,))"

    raise NotImplementedError(
        f"Numpy translation not implemented for op_type {op!r} "
        f"(node {node.name!r}). Add it to emit_op_expr().")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def collect_constants_for_numpy(graph, numpy_cone_indices, stage_1_outputs_names):
    """Collect initializers that the numpy_cone reads but that aren't stage-1
    activations or numpy_cone outputs. These get baked into the .npz."""
    consumed_by_cone: set[str] = set()
    for ci in numpy_cone_indices:
        for inp in graph.node[ci].input:
            if inp:
                consumed_by_cone.add(inp)
    initializer_names = {init.name for init in graph.initializer}
    constants_needed = (consumed_by_cone & initializer_names) - set(stage_1_outputs_names)
    return [init for init in graph.initializer if init.name in constants_needed]


def split(model, output_dir: Path, *, log_prefix: str = "[Split]"):
    graph = model.graph

    # --- Phase 1: discover the bool cone ---
    cone_indices, boundary_indices, _bool_tensors = discover_bool_cone(graph)
    full_cone = cone_indices | boundary_indices
    print(f"{log_prefix} bool cone ops: {len(cone_indices)}")
    print(f"{log_prefix} boundary Cast/Where ops: {len(boundary_indices)}")
    print(f"{log_prefix} cone+boundary total: {len(full_cone)}")

    # --- Phase 2: use the cone as-is, no aggressive expansion ---
    # The mask computation in this model is interleaved with Gemm/Softmax
    # (alignment, duration prediction). Expanding the cone pulls in these
    # heavy-compute ops and creates unresolvable circular dependencies.
    # Instead, we use the bool cone directly and take all non-bool/non-init
    # inputs as numpy inputs from stage_1.
    producer_map = build_producer_map(graph)

    # --- Phase 3: classify early vs late; early stays in stage_1 ---
    # Chain A (early ops) produce tensors consumed by stage_1 (text encoder,
    # duration predictor). They must stay in stage_1 and compile there.
    # Only late ops (chains B-F + existing cone) move to numpy_post.
    early_cone, late_cone = classify_early_late(graph, full_cone, producer_map)
    numpy_pre_cone = set()  # early stays in stage_1
    numpy_post_cone = late_cone
    all_numpy = numpy_post_cone
    print(f"{log_prefix} early cone (stays in stage_1): {len(early_cone)}")
    print(f"{log_prefix} numpy_post cone:               {len(numpy_post_cone)}")

    # --- Phase 4: partition the rest into stage_1 and stage_3 ---
    post_boundary_outputs = discover_boundary_outputs(graph, numpy_post_cone)
    print(f"{log_prefix} numpy_post boundary outputs: {len(post_boundary_outputs)}")

    stage_3_indices = discover_stage3_forward(graph, all_numpy, post_boundary_outputs)
    stage_1_indices = set(range(len(graph.node))) - all_numpy - stage_3_indices
    print(f"{log_prefix} stage_1 ops (ONNX): {len(stage_1_indices)}")
    print(f"{log_prefix} stage_3 ops (ONNX): {len(stage_3_indices)}")
    total = len(stage_1_indices) + len(stage_3_indices) + len(all_numpy)
    assert total == len(graph.node), \
        f"node partition is not exhaustive: {total} vs {len(graph.node)}"

    # Circular dependency check: no numpy_post op should read from a stage_3 op.
    # First find direct circular deps, then cascade.
    initial_circular = []
    for ci in numpy_post_cone:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is not None and prod in stage_3_indices:
                prod_n = graph.node[prod]
                initial_circular.append((ci, graph.node[ci], inp, prod, prod_n))

    if initial_circular:
        print(f"{log_prefix} INITIAL circular deps ({len(initial_circular)}):")
        for ci, n, inp_name, prod_ci, prod_n in initial_circular:
            print(f"{log_prefix}   cone idx={ci} {n.op_type} {n.name!r} "
                  f"← {inp_name!r} ← stage_3 idx={prod_ci} {prod_n.op_type} {prod_n.name!r}")

    # Cascade: removing an op may cause its cone consumers to also need removal.
    removed_from_cone: set[int] = set()
    removed_outputs: set[str] = set()
    changed = True
    while changed:
        changed = False
        for ci in list(numpy_post_cone):
            if ci in removed_from_cone:
                continue
            needs_removal = False
            for inp in graph.node[ci].input:
                if not inp:
                    continue
                prod = producer_map.get(inp)
                if prod is not None and prod in stage_3_indices:
                    needs_removal = True
                    break
                if inp in removed_outputs:
                    needs_removal = True
                    break
            if needs_removal:
                removed_from_cone.add(ci)
                for o in graph.node[ci].output:
                    if o:
                        removed_outputs.add(o)
                changed = True

    if removed_from_cone:
        print(f"{log_prefix} cascaded removal: {len(removed_from_cone)} numpy_post ops → stay in model")
        numpy_post_cone = numpy_post_cone - removed_from_cone
        all_numpy = numpy_pre_cone | numpy_post_cone
        post_boundary_outputs = discover_boundary_outputs(graph, numpy_post_cone)
        stage_3_indices = discover_stage3_forward(graph, all_numpy, post_boundary_outputs)
        stage_1_indices = set(range(len(graph.node))) - all_numpy - stage_3_indices
        print(f"{log_prefix} after fix: numpy_post={len(numpy_post_cone)} "
              f"stage_1={len(stage_1_indices)} stage_3={len(stage_3_indices)}")

    # --- Phase 5: figure out what crosses each boundary ---
    g_input_names = {gi.name for gi in graph.input}
    initializer_names = {init.name for init in graph.initializer}

    # numpy_pre outputs → stage_1 inputs
    pre_boundary_outputs = discover_boundary_outputs(graph, numpy_pre_cone)
    print(f"{log_prefix} numpy_pre boundary outputs: {len(pre_boundary_outputs)}")

    # numpy_pre inputs: graph inputs + initializers only (by definition of early)
    pre_input_names = set()
    for ci in numpy_pre_cone:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is None and inp in g_input_names:
                pre_input_names.add(inp)

    # numpy_post inputs from stage_1 and from numpy_pre
    numpy_post_external_inputs: set[str] = set()
    stage_1_outputs_needed: set[str] = set()
    numpy_post_inputs_from_pre: set[str] = set()
    for ci in numpy_post_cone:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is None:
                continue
            if prod in stage_1_indices:
                numpy_post_external_inputs.add(inp)
                stage_1_outputs_needed.add(inp)
            elif prod in numpy_pre_cone:
                numpy_post_external_inputs.add(inp)
                numpy_post_inputs_from_pre.add(inp)

    if numpy_post_inputs_from_pre:
        print(f"{log_prefix} numpy_post reads {len(numpy_post_inputs_from_pre)} "
              f"tensor(s) from numpy_pre (will be chained)")

    # numpy_post inputs from graph inputs (e.g., text_lengths used by both pre and post)
    post_graph_inputs = _graph_input_names(graph, numpy_post_cone, producer_map)

    # stage_3 inputs from stage_1
    for ci in stage_3_indices:
        for inp in graph.node[ci].input:
            if not inp:
                continue
            prod = producer_map.get(inp)
            if prod is None:
                continue
            if prod in stage_1_indices:
                stage_1_outputs_needed.add(inp)

    # stage_1 also needs numpy_pre outputs as inputs
    # (build_subgraph handles this via external_inputs detection)

    print(f"{log_prefix} numpy_post external inputs: {len(numpy_post_external_inputs)}")
    print(f"{log_prefix} stage_1 outputs to expose: {len(stage_1_outputs_needed)}")

    # --- Phase 6: build the ONNX subgraphs ---
    type_map = _build_type_map(graph)

    print(f"{log_prefix} building stage_1 subgraph...")
    stage_1_model = build_subgraph(
        model, stage_1_indices,
        graph_name="part_a1_pre_mask",
        log_prefix=f"{log_prefix}[stage1]",
    )
    existing_s1_outputs = {o.name for o in stage_1_model.graph.output}
    for tname in sorted(stage_1_outputs_needed - existing_s1_outputs):
        stage_1_model.graph.output.append(_make_value_info(tname, type_map))

    print(f"{log_prefix} building stage_3 subgraph...")
    stage_3_model = build_subgraph(
        model, stage_3_indices,
        graph_name="part_a2_post_mask",
        log_prefix=f"{log_prefix}[stage3]",
    )

    # --- Phase 7: generate numpy code ---
    all_numpy_outputs = set(stage_1_outputs_needed)
    constants_for_npz = collect_constants_for_numpy(
        graph, all_numpy, all_numpy_outputs)
    print(f"{log_prefix} constants baked into .npz: {len(constants_for_npz)}")

    post_input_names = sorted(numpy_post_external_inputs | post_graph_inputs)
    post_output_names = sorted(post_boundary_outputs)

    function_specs = []
    if numpy_pre_cone:
        function_specs.append({
            "name": "compute_masks_early",
            "doc": f"Pre-stage_1 mask computation ({len(numpy_pre_cone)} ops).",
            "cone_indices": numpy_pre_cone,
            "input_names": sorted(pre_input_names),
            "output_names": sorted(pre_boundary_outputs),
        })
    if numpy_post_cone:
        function_specs.append({
            "name": "compute_masks_late",
            "doc": f"Post-stage_1 mask computation ({len(numpy_post_cone)} ops).",
            "cone_indices": numpy_post_cone,
            "input_names": post_input_names,
            "output_names": post_output_names,
        })

    numpy_code = generate_numpy_module(
        graph, function_specs, constants_for_npz=constants_for_npz)

    # --- Phase 8: write artefacts ---
    output_dir.mkdir(parents=True, exist_ok=True)
    part_a1_path  = output_dir / "part_a1_pre_mask.onnx"
    part_a2_path  = output_dir / "part_a2_post_mask.onnx"
    numpy_path    = output_dir / "part_a_mask_compute.py"
    npz_path      = output_dir / "part_a_mask_compute_constants.npz"
    manifest_path = output_dir / "split_manifest.json"

    onnx.save(stage_1_model, str(part_a1_path))
    onnx.save(stage_3_model, str(part_a2_path))
    numpy_path.write_text(numpy_code)

    npz_arrays = {init.name: numpy_helper.to_array(init)
                  for init in constants_for_npz}
    np.savez(npz_path, **npz_arrays)

    # Manifest with 4-stage pipeline info
    manifest = {
        "source_model_node_count": len(graph.node),
        "numpy_total_op_count": len(all_numpy),
    }
    if numpy_pre_cone:
        manifest["stage_1_numpy_pre"] = {
            "function": "compute_masks_early",
            "op_count": len(numpy_pre_cone),
            "inputs": [_tensor_schema(n, type_map) for n in sorted(pre_input_names)],
            "outputs": [_tensor_schema(n, type_map) for n in sorted(pre_boundary_outputs)],
        }
    manifest["stage_2_onnx"] = {
        "path": part_a1_path.name,
        "op_count": len(stage_1_indices),
        "graph_inputs": [io.name for io in stage_1_model.graph.input],
        "graph_outputs": [io.name for io in stage_1_model.graph.output],
    }
    manifest["stage_3_numpy_post"] = {
        "function": "compute_masks_late",
        "op_count": len(numpy_post_cone),
        "inputs": [_tensor_schema(n, type_map) for n in post_input_names],
        "outputs": [_tensor_schema(n, type_map) for n in post_output_names],
    }
    manifest["stage_4_onnx"] = {
        "path": part_a2_path.name,
        "op_count": len(stage_3_indices),
        "graph_inputs": [io.name for io in stage_3_model.graph.input],
        "graph_outputs": [io.name for io in stage_3_model.graph.output],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Validate
    for path, m in [(part_a1_path, stage_1_model), (part_a2_path, stage_3_model)]:
        try:
            onnx.checker.check_model(m)
            print(f"{log_prefix} onnx.checker OK: {path.name}")
        except Exception as e:
            print(f"{log_prefix} onnx.checker WARN on {path.name}: {e}")

    print(f"{log_prefix} wrote {part_a1_path}")
    print(f"{log_prefix} wrote {numpy_path}")
    print(f"{log_prefix} wrote {npz_path}")
    print(f"{log_prefix} wrote {part_a2_path}")
    print(f"{log_prefix} wrote {manifest_path}")
    print(f"{log_prefix} pipeline: ", end="")
    if numpy_pre_cone:
        print(f"numpy_pre({len(numpy_pre_cone)}) → ", end="")
    print(f"stage_1({len(stage_1_indices)}) → "
          f"numpy_post({len(numpy_post_cone)}) → "
          f"stage_3({len(stage_3_indices)})")


# ---------------------------------------------------------------------------
# Helpers for value_info / schemas
# ---------------------------------------------------------------------------

def _build_type_map(graph):
    types = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
            types[vi.name] = (vi.type.tensor_type.elem_type, dims)
    for init in graph.initializer:
        types.setdefault(init.name, (init.data_type, list(init.dims)))
    return types


def _make_value_info(name, type_map):
    et, dims = type_map.get(name, (TensorProto.FLOAT, []))
    return helper.make_tensor_value_info(name, et, dims)


def _tensor_schema(name, type_map):
    et, dims = type_map.get(name, (TensorProto.FLOAT, []))
    return {
        "name": name,
        "dtype": TensorProto.DataType.Name(et) if isinstance(et, int) else str(et),
        "shape": dims,
    }


def _graph_input_names(graph, numpy_cone, producer_map):
    """Return tensor names that are graph inputs (or unknown producers) and
    consumed by numpy_cone — these come from the original model's graph inputs
    (the numpy stage runs in Python and receives them directly)."""
    g_inputs = {gi.name for gi in graph.input}
    result = set()
    for ci in numpy_cone:
        for inp in graph.node[ci].input:
            if inp and inp in g_inputs and producer_map.get(inp) is None:
                result.add(inp)
    return result


# ---------------------------------------------------------------------------
# Numpy module generation
# ---------------------------------------------------------------------------

def generate_numpy_module(graph, function_specs, *, constants_for_npz):
    """Generate part_a_mask_compute.py with one or two numpy functions.

    function_specs: list of dicts, each with keys:
        name:         function name (e.g., "compute_masks_early")
        doc:          one-line docstring
        cone_indices: set of node indices for this function
        input_names:  list of input tensor names
        output_names: list of output tensor names
    """
    total_ops = sum(len(fs["cone_indices"]) for fs in function_specs)
    constant_names = {init.name for init in constants_for_npz}

    lines: list[str] = []
    A = lines.append

    A('"""Auto-generated mask computation for the part-A split.\n')
    A("Generated by scripts/split_part_a_at_mask_boundary.py.")
    A("DO NOT EDIT BY HAND — re-run the splitter to regenerate.\n")
    A(f"This file implements {total_ops} ONNX ops in numpy across "
      f"{len(function_specs)} function(s).")
    A('"""')
    A("")
    A("from __future__ import annotations")
    A("")
    A("from pathlib import Path")
    A("")
    A("import numpy as np")
    A("")
    A("_CONSTANTS = np.load(Path(__file__).with_name('part_a_mask_compute_constants.npz'))")
    A("")
    A("")
    A("def _onnx_slice(data, starts, ends, axes=None, steps=None):")
    A("    starts = np.asarray(starts).reshape(-1).tolist()")
    A("    ends = np.asarray(ends).reshape(-1).tolist()")
    A("    ndim = data.ndim")
    A("    if axes is None:")
    A("        axes = list(range(len(starts)))")
    A("    else:")
    A("        axes = np.asarray(axes).reshape(-1).tolist()")
    A("    if steps is None:")
    A("        steps = [1] * len(starts)")
    A("    else:")
    A("        steps = np.asarray(steps).reshape(-1).tolist()")
    A("    slices = [slice(None)] * ndim")
    A("    for i in range(len(axes)):")
    A("        ax = int(axes[i])")
    A("        s = int(starts[i])")
    A("        e = int(ends[i])")
    A("        st = int(steps[i])")
    A("        slices[ax] = slice(s, e, st)")
    A("    return data[tuple(slices)]")
    A("")
    A("")
    A("def _onnx_gather_nd(data, indices, batch_dims=0):")
    A("    indices = np.asarray(indices)")
    A("    if batch_dims == 0:")
    A("        idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))")
    A("        return data[idx_tuple]")
    A("    result_slices = []")
    A("    for b in range(data.shape[0]):")
    A("        idx = indices[b]")
    A("        idx_tuple = tuple(idx[..., i] for i in range(idx.shape[-1]))")
    A("        result_slices.append(data[b][idx_tuple])")
    A("    return np.stack(result_slices)")
    A("")

    for fs in function_specs:
        func_name = fs["name"]
        doc = fs["doc"]
        cone_indices = fs["cone_indices"]
        input_names = fs["input_names"]
        output_names = fs["output_names"]

        var_for = {}
        for n in input_names:
            var_for[n] = py_name(n)
        for n in constant_names:
            var_for[n] = py_name(n)

        sorted_inputs = sorted(input_names)
        A("")
        A(f"def {func_name}(")
        A("    *,")
        for n in sorted_inputs:
            A(f"    {py_name(n)},")
        A(") -> dict:")
        A(f'    """{doc}"""')

        # Constants used by this function
        fn_constants = set()
        for ci in cone_indices:
            for inp in graph.node[ci].input:
                if inp and inp in constant_names:
                    fn_constants.add(inp)
        if fn_constants:
            A("    # Bake-in constants")
            for cn in sorted(fn_constants):
                A(f"    {py_name(cn)} = _CONSTANTS[{cn!r}]")
            A("")

        topo_indices = topo_sort(cone_indices, graph)
        A(f"    # === {len(cone_indices)} ops ===")
        for ci in topo_indices:
            n = graph.node[ci]
            input_vars = {}
            for inp in n.input:
                if not inp:
                    continue
                if inp not in var_for:
                    raise RuntimeError(
                        f"Cone op {n.name!r} ({n.op_type}) at idx {ci} reads "
                        f"tensor {inp!r} which is neither an input nor a known "
                        f"constant. Splitter bug?")
                input_vars[inp] = var_for[inp]
            for o in n.output:
                if not o:
                    continue
                var_for[o] = py_name(o)
            expr = emit_op_expr(n, input_vars)
            primary_out = n.output[0]
            comment = f"  # {n.op_type} {n.name!r}".rstrip()
            A(f"    {py_name(primary_out)} = {expr}{comment}")

        A("")
        A("    return {")
        for o in sorted(output_names):
            A(f"        {o!r}: {py_name(o)},")
        A("    }")
        A("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Source ONNX model (pre- or post-surgery).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write the split artefacts into.")
    args = p.parse_args()

    print(f"[Split] loading {args.input}")
    model = onnx.load(str(args.input))
    print(f"[Split] graph has {len(model.graph.node)} nodes")
    split(model, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
