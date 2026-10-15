# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Graph surgery for RTMO tiny.

:func:`strip_postprocess` cuts the model at the eight dense head convolutions,
dropping the dynamic-shaped mmdeploy decode/NMS/DCC tail (runs host-side
instead). :func:`resize_input` re-targets to a new square input size by
rewriting the only two size-tied constants: the ``neck.pos_enc_*`` positional
encoding and the AIFI unflatten ``Reshape`` target. Output order is grouped by
branch, level ascending (stride 16 then 32), mirroring mmpose head forward.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ._pos_enc import build_2d_sincos_position_embedding

logger = logging.getLogger("rtmo-surgery")

# (weight token, output stem, channels); order defines graph-output order.
_HEAD_BRANCHES = (("out_cls", "cls_scores", 1), ("out_bbox", "bbox_preds", 4), ("out_kpt_vis", "kpt_vis", 17), ("out_pose", "pose_feats", 192))
_STRIDES = (16, 32)
_POS_ENC_PREFIX = "neck.pos_enc_"


def _find_head_conv_outputs(graph: gs.Graph) -> dict[tuple[str, int], gs.Variable]:
    """Map ``(weight_token, level)`` -> head conv output, via ``head.head_module.out_*.<level>.weight``."""
    found = {}
    for node in graph.nodes:
        if node.op != "Conv":
            continue
        for inp in node.inputs:
            if inp.name.startswith("head.head_module.out_"):
                parts = inp.name.split(".")
                found[(parts[2], int(parts[3]))] = node.outputs[0]
    return found


def strip_postprocess(graph: gs.Graph) -> gs.Graph:
    """Set the eight head conv outputs as graph outputs and drop the tail."""
    conv_outs = _find_head_conv_outputs(graph)
    new_outputs = []
    for token, stem, _ch in _HEAD_BRANCHES:
        for level, stride in enumerate(_STRIDES):
            if (token, level) not in conv_outs:
                raise RuntimeError(f"head conv for {token} level {level} not found")
            tensor = conv_outs[(token, level)]
            tensor.name, tensor.dtype = f"{stem}_s{stride}", np.float32
            new_outputs.append(tensor)
    graph.outputs = new_outputs
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    logger.info("Stripped post-processing; kept %d nodes", len(graph.nodes))
    return graph


def _rewrite_pos_enc(graph: gs.Graph, s32: int) -> int:
    """Regenerate every ``neck.pos_enc_*`` constant for an ``s32 x s32`` grid."""
    n = 0
    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Constant) and tensor.name.startswith(_POS_ENC_PREFIX):
            tensor.values = build_2d_sincos_position_embedding(s32, s32, int(tensor.values.shape[-1]))
            n += 1
    return n


def _rewrite_encoder_unflatten(graph: gs.Graph, old_s32: int, new_s32: int) -> int:
    """Rewrite the AIFI unflatten Reshape target ``[-1, C, old, old]``."""
    n = 0
    for tensor in graph.tensors().values():
        if not isinstance(tensor, gs.Constant):
            continue
        v = np.asarray(tensor.values).ravel()
        if v.shape == (4,) and int(v[0]) == -1 and int(v[2]) == old_s32 and int(v[3]) == old_s32:
            tensor.values = np.array([-1, int(v[1]), new_s32, new_s32], dtype=v.dtype)
            n += 1
    return n


def resize_input(graph: gs.Graph, input_size: int, batch: int = 1) -> gs.Graph:
    """Re-target to ``input_size``; call after :func:`strip_postprocess`."""
    if input_size % 32 != 0:
        raise ValueError(f"input_size must be divisible by 32 (stride-32 level), got {input_size}")
    inp = graph.inputs[0]
    old_h = inp.shape[2]
    if not isinstance(old_h, int):
        raise RuntimeError(f"expected a static input height, got {inp.shape!r}")
    old_s32, new_s32 = old_h // 32, input_size // 32
    inp.shape, inp.dtype = [batch, 3, input_size, input_size], np.float32

    if _rewrite_pos_enc(graph, new_s32) == 0:
        raise RuntimeError("no neck.pos_enc_* constant found to rewrite")
    if _rewrite_encoder_unflatten(graph, old_s32, new_s32) == 0:
        raise RuntimeError(f"encoder unflatten Reshape target [-1,C,{old_s32},{old_s32}] not found")

    # Static output shapes for the compiler.
    idx = 0
    for _token, _stem, ch in _HEAD_BRANCHES:
        for stride in _STRIDES:
            out = graph.outputs[idx]
            out.shape, out.dtype = [batch, ch, input_size // stride, input_size // stride], np.float32
            idx += 1
    graph.cleanup().toposort()
    return graph


def build_stripped_model(model: onnx.ModelProto, input_size: int = 320, batch: int = 1) -> onnx.ModelProto:
    """Strip post-processing and re-target to ``input_size``; return the ONNX model."""
    graph = gs.import_onnx(model)
    strip_postprocess(graph)
    resize_input(graph, input_size=input_size, batch=batch)
    out = gs.export_onnx(graph)
    out.ir_version = model.ir_version
    # Stale source value_info (sized for the original input) trips shape inference; recompute downstream.
    del out.graph.value_info[:]
    return out
