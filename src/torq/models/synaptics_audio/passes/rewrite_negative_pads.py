# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Rewrite ONNX ``Pad`` ops with negative (crop) paddings into ``Pad(positive)+Slice``.

Negative ONNX Pad entries crop tensor edges; many stacks (IREE/Torch-MLIR,
isolated Pad tests) handle ``Slice`` more reliably than ``Pad`` with negative
paddings. This pass requires fully static ranked shapes on each Pad's data
input (typically true after :class:`FinalizeTorqReady` / symbolic shape
propagation, or after :class:`ApplyFixedShapes` + standard shape inference).

Only ``mode == constant`` Pads are rewritten. Pads that are already non-negative
or whose data input has dynamic rank are left unchanged.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .base import PassContext

logger = logging.getLogger("synaptics-audio.passes.rewrite_negative_pads")


def _get_initializer_array(graph: onnx.GraphProto, name: str) -> np.ndarray | None:
    if not name:
        return None
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _static_tensor_shape(graph: onnx.GraphProto, name: str) -> list[int] | None:
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name != name:
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None
        dims: list[int] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                return None
        return dims
    for init in graph.initializer:
        if init.name == name:
            return list(numpy_helper.to_array(init).shape)
    return None


def _pad_mode(node: onnx.NodeProto) -> str:
    for a in node.attribute:
        if a.name == "mode":
            raw = a.s
            if isinstance(raw, bytes):
                return raw.decode("ascii")
            return str(raw)
    return "constant"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 0
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _rewrite(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    work = copy.deepcopy(model)
    graph = work.graph
    used_names = {i.name for i in graph.initializer}
    used_names.update(v.name for v in graph.input)
    used_names.update(v.name for v in graph.output)
    used_names.update(v.name for v in graph.value_info)
    for n in graph.node:
        used_names.update(x for x in n.input if x)
        used_names.update(x for x in n.output if x)

    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0

    for node in graph.node:
        if node.op_type != "Pad" or _pad_mode(node) != "constant":
            new_nodes.append(node)
            continue
        if len(node.input) < 2 or not node.input[1]:
            new_nodes.append(node)
            continue

        data_in = node.input[0]
        pads_arr = _get_initializer_array(graph, node.input[1])
        if pads_arr is None:
            new_nodes.append(node)
            continue

        pads_flat = pads_arr.astype(np.int64).reshape(-1)
        if pads_flat.size % 2 != 0:
            new_nodes.append(node)
            continue
        if np.all(pads_flat >= 0):
            new_nodes.append(node)
            continue

        rank = pads_flat.size // 2
        shape_in = _static_tensor_shape(graph, data_in)
        if shape_in is None or len(shape_in) != rank:
            logger.debug(
                "skip Pad %s: need static rank-%d shape for %s (got %s)",
                node.name, rank, data_in, shape_in,
            )
            new_nodes.append(node)
            continue

        pos_begin = np.maximum(pads_flat[:rank], 0)
        neg_begin = np.minimum(pads_flat[:rank], 0)
        pos_end = np.maximum(pads_flat[rank:], 0)
        neg_end = np.minimum(pads_flat[rank:], 0)

        inter_shape = [
            int(shape_in[i] + pos_begin[i] + pos_end[i]) for i in range(rank)
        ]
        slice_starts = (-neg_begin).astype(np.int64)
        slice_ends = (np.asarray(inter_shape, dtype=np.int64) + neg_end).astype(
            np.int64
        )

        valid = all(
            0 <= slice_starts[i] <= slice_ends[i] <= inter_shape[i]
            for i in range(rank)
        )
        if not valid:
            logger.debug(
                "skip Pad %s: invalid slice range starts=%s ends=%s inter=%s",
                node.name, slice_starts.tolist(), slice_ends.tolist(), inter_shape,
            )
            new_nodes.append(node)
            continue

        out_name = node.output[0]
        needs_pos_pad = bool(np.any(pos_begin > 0) or np.any(pos_end > 0))

        mid_tensor = data_in
        if needs_pos_pad:
            pads_pos = np.concatenate([pos_begin, pos_end]).astype(np.int64)
            pads_pos_name = _unique_name(f"{out_name}_pos_pads", used_names)
            graph.initializer.append(
                numpy_helper.from_array(pads_pos, name=pads_pos_name)
            )

            pad_inputs = [data_in, pads_pos_name]
            if len(node.input) >= 3 and node.input[2]:
                pad_inputs.append(node.input[2])
            mid_tensor = _unique_name(f"{out_name}_pos_pad_tmp", used_names)
            new_nodes.append(
                helper.make_node(
                    "Pad",
                    inputs=pad_inputs,
                    outputs=[mid_tensor],
                    name=_unique_name(
                        f"{node.name or out_name}_pos_only", used_names
                    ),
                    mode="constant",
                )
            )

        starts_name = _unique_name(f"{out_name}_slice_starts", used_names)
        ends_name = _unique_name(f"{out_name}_slice_ends", used_names)
        graph.initializer.append(numpy_helper.from_array(slice_starts, name=starts_name))
        graph.initializer.append(numpy_helper.from_array(slice_ends, name=ends_name))

        new_nodes.append(
            helper.make_node(
                "Slice",
                inputs=[mid_tensor, starts_name, ends_name],
                outputs=[out_name],
                name=_unique_name(
                    f"{node.name or out_name}_crop_slice", used_names
                ),
            )
        )
        rewritten += 1
        logger.info(
            "rewrote Pad %s -> %s + Slice (inter_shape=%s)",
            node.name,
            "Pad+" if needs_pos_pad else "(no pos Pad)",
            inter_shape,
        )

    del graph.node[:]
    graph.node.extend(new_nodes)
    return work, rewritten


class RewriteNegativePads:
    """Pass: rewrite constant-mode Pad with negative paddings to ``Pad(positive)+Slice``."""

    name = "rewrite_negative_pads"

    def __call__(
        self, model: onnx.ModelProto, ctx: PassContext
    ) -> onnx.ModelProto:
        out_model, _ = _rewrite(model)
        return out_model
