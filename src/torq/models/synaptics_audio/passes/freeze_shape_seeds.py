# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Freeze the shape-only producers of ``Reshape`` / ``Expand`` / ``Slice`` controls.

After :class:`ApplyFixedShapes` makes graph I/O static, the dynamic shape
arithmetic that feeds ``Reshape``'s ``shape`` input (and ``Expand``'s shape,
``Slice``'s starts/ends/axes/steps) is fully determined. This pass:

1. Identifies those control inputs ("seeds") whose producer is a shape op
   (``Shape``, ``Gather``, ``Concat``, ``Cast``, ``Squeeze``, ``Unsqueeze``,
   ``Mul``, ``Div``, ``Add``, ``Sub``, ``Equal``, ``Where``, ``ConstantOfShape``,
   ``Reshape``, ``Slice``, ``Expand``, ``Transpose``, ``Constant``, ``Identity``).
2. Builds a tiny *probe* model that exposes those seed tensors as outputs and
   evaluates them under onnxruntime CPU with all-zeros feeds matching
   :attr:`PassContext.input_shapes` (the actual values don't matter; only
   shape arithmetic does).
3. Replaces every consumer's reference with a new constant initializer
   (``__frozen`` suffix) carrying the evaluated value.
4. Reverse-reachability prunes the now-orphan shape-computation nodes.

If :attr:`PassContext.input_shapes` is empty the pass is a no-op (no way to
build feeds).
"""

from __future__ import annotations

import copy
import logging
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto

from .._ort import make_cpu_session
from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.freeze_shape_seeds")


_SHAPE_OPS: frozenset[str] = frozenset(
    {
        "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat",
        "Equal", "Where", "ConstantOfShape",
        "Mul", "Div", "Add", "Sub",
        "Cast", "Reshape", "Slice", "Expand", "Transpose",
        "Constant", "Identity",
    }
)


def _build_maps(model: onnx.ModelProto):
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}
    return producer, initializers, graph_inputs


def _shape_control_input_indices(node: onnx.NodeProto) -> list[int]:
    if node.op_type in ("Expand", "Reshape") and len(node.input) >= 2:
        return [1]
    if node.op_type == "Slice":
        return list(range(1, len(node.input)))
    return []


def _find_shape_seed_tensors(model: onnx.ModelProto) -> list[str]:
    seeds: list[str] = []
    for node in model.graph.node:
        for idx in _shape_control_input_indices(node):
            if idx < len(node.input):
                seeds.append(node.input[idx])
    return seeds


def _evaluate_tensors(
    model: onnx.ModelProto,
    tensor_names: list[str],
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the model with the named tensors exposed via Identity probes."""
    probe = copy.deepcopy(model)
    del probe.graph.output[:]

    out_names: list[str] = []
    for i, tname in enumerate(tensor_names):
        out_name = f"__probe_out_{i}"
        out_names.append(out_name)
        probe.graph.node.append(
            helper.make_node("Identity", inputs=[tname], outputs=[out_name],
                             name=f"_ProbeIdentity_{i}")
        )
        probe.graph.output.append(
            helper.make_tensor_value_info(out_name, TensorProto.INT64, None)
        )

    probe = shape_inference.infer_shapes(probe)
    with tempfile.TemporaryDirectory(prefix="freeze_shape_seeds_") as tmp:
        probe_path = Path(tmp) / "probe.onnx"
        onnx.save(probe, str(probe_path))
        sess = make_cpu_session(str(probe_path))
        vals = sess.run(out_names, feeds)
    return dict(zip(tensor_names, vals))


def _replace_consumers_with_frozen_initializers(
    model: onnx.ModelProto, frozen: dict[str, np.ndarray]
) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for old, value in frozen.items():
        new = f"{old}__frozen"
        rename_map[old] = new
        model.graph.initializer.append(numpy_helper.from_array(value, name=new))
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]
    for out in model.graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]
    return rename_map


def _prune_dead_nodes(model: onnx.ModelProto) -> None:
    needed = {o.name for o in model.graph.output}
    kept: list[onnx.NodeProto] = []
    changed = True
    while changed:
        changed = False
        for node in reversed(model.graph.node):
            if any(o in needed for o in node.output) and node not in kept:
                kept.append(node)
                for inp in node.input:
                    if inp and inp not in needed:
                        needed.add(inp)
                        changed = True
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)

    used = (
        {i.name for i in model.graph.input}
        | {o.name for o in model.graph.output}
        | {init.name for init in model.graph.initializer}
    )
    for n in model.graph.node:
        used.update(x for x in n.input if x)
        used.update(x for x in n.output if x)
    keep_vi = [vi for vi in model.graph.value_info if vi.name in used]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)


class FreezeShapeSeeds:
    """Pass: constant-fold shape-arithmetic subgraphs feeding Reshape/Expand/Slice."""

    name = "freeze_shape_seeds"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        if not ctx.input_shapes:
            logger.debug("no input_shapes in PassContext; skipping")
            return model

        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.debug("pre-pass shape_inference skipped: %s", exc)

        seeds = _find_shape_seed_tensors(model)
        if not seeds:
            return model

        producer, initializers, graph_inputs = _build_maps(model)
        seen: set[str] = set()
        slice_seeds: list[str] = []
        for s in seeds:
            if not s or s in seen or s in initializers or s in graph_inputs:
                continue
            seen.add(s)
            p = producer.get(s)
            if p is not None and p.op_type in _SHAPE_OPS:
                slice_seeds.append(s)
        if not slice_seeds:
            return model

        feeds = {
            name: np.zeros(tuple(shape), dtype=np.float32)
            for name, shape in ctx.input_shapes.items()
        }

        for s in slice_seeds:
            p = producer.get(s)
            logger.info(
                "freeze seed %s <- %s (%s)",
                s, p.name if p else "?", p.op_type if p else "?",
            )

        out = copy.deepcopy(model)
        frozen = _evaluate_tensors(model, slice_seeds, feeds)
        _replace_consumers_with_frozen_initializers(out, frozen)
        _prune_dead_nodes(out)

        try:
            out = shape_inference.infer_shapes(out)
        except Exception as exc:
            logger.debug("post-pass shape_inference skipped: %s", exc)
        logger.info("froze %d shape seed(s)", len(frozen))
        return out
