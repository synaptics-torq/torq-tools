# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Piper (VITS) graph surgery: split where the output length becomes known, pin windows."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import shape_inference

# Total frame count (duration Ceil+ReduceSum) and the vocoder latent z: everything
# needed to produce these is partA, the rest (the HiFi-GAN vocoder) is partB.
CUTS = ("/ReduceSum_output_0", "/Mul_7_output_0")


def _ancestors(graph, tensors):
    producer = {o: i for i, n in enumerate(graph.node) for o in n.output}
    skip = {i.name for i in graph.initializer} | {i.name for i in graph.input}
    needed, stack = set(), list(tensors)
    while stack:
        name = stack.pop()
        i = producer.get(name)
        if name in skip or i is None or i in needed:
            continue
        needed.add(i)
        stack.extend(graph.node[i].input)
    return needed


def split(model_path: Path, part_a: Path, part_b: Path) -> list[str]:
    """Extract partA/partB around CUTS; return the boundary tensors partB consumes."""
    graph = onnx.load(model_path).graph
    a_ids = _ancestors(graph, CUTS)
    a_out = {o for i in a_ids for o in graph.node[i].output}
    b_in = {i for k, n in enumerate(graph.node) if k not in a_ids for i in n.input}
    boundary = sorted((a_out & b_in) - {i.name for i in graph.initializer})
    b_inputs = boundary + [i.name for i in graph.input if i.name in b_in]
    onnx.utils.extract_model(str(model_path), str(part_a), [i.name for i in graph.input],
                             sorted(set(boundary) | set(CUTS)))
    onnx.utils.extract_model(str(model_path), str(part_b), b_inputs,
                             [o.name for o in graph.output])
    return boundary


def freeze(model_path: Path, latent: str, frames: int) -> onnx.ModelProto:
    """Pin the latent input to [1, C, frames] and every other free dim to 1."""
    model = onnx.load(model_path)
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if inp.name == latent:
            dims[0].dim_value, dims[2].dim_value = 1, frames
        else:
            for d in dims:
                if not d.HasField("dim_value"):
                    d.dim_value = 1
    del model.graph.value_info[:]
    for out in model.graph.output:
        out.type.tensor_type.ClearField("shape")
    model = shape_inference.infer_shapes(model, strict_mode=True)
    model.graph.name = "main"  # becomes the vmfb entry point; the runtime invokes @main
    return model
