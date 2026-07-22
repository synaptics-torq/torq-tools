# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Graph surgery for the RTMO tiny pose model.

Two independent transforms live here:

1. :func:`strip_postprocess` — cut the model at the eight dense head
   convolutions (``out_cls`` / ``out_bbox`` / ``out_kpt_vis`` / ``out_pose`` for
   each of the two feature levels) and expose them as the graph outputs. This
   removes the mmdeploy decode + NMS tail (``TopK`` / ``NonMaxSuppression`` /
   ``NonZero`` / ``Range`` / dynamic ``Reshape`` … and the DCC pose decoder,
   which only runs on post-NMS instances) — all of which is dynamic-shaped and
   not NPU-friendly. The decode / DCC / NMS is expected to run host-side.

2. :func:`resize_input` — re-target the (otherwise fully convolutional)
   backbone + neck + head to a new square input size. Only two constants in the
   kept subgraph are tied to the input size: the transformer positional
   encoding ``neck.pos_enc_0`` and the encoder "unflatten" ``Reshape`` target
   ``[-1, 256, s32, s32]``. Both are rewritten for the new stride-32 grid.

The eight head outputs are named/ordered grouped by branch, level ascending
(stride 16 then stride 32), mirroring mmpose's ``head_module.forward`` return.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from ._pos_enc import build_2d_sincos_position_embedding

logger = logging.getLogger("rtmo-surgery")

# Weight-token (``head.head_module.out_<TOKEN>.<level>.weight``) -> output name
# stem and channel count. Order here defines the graph-output order.
_HEAD_BRANCHES: tuple[tuple[str, str, int], ...] = (
    ("out_cls", "cls_scores", 1),
    ("out_bbox", "bbox_preds", 4),
    ("out_kpt_vis", "kpt_vis", 17),
    ("out_pose", "pose_feats", 192),
)
_STRIDES: tuple[int, ...] = (16, 32)  # feature level 0, 1
_EMBED_DIM = 256  # transformer (neck AIFI) embedding dim
_POS_ENC_PREFIX = "neck.pos_enc_"


def _find_head_conv_outputs(
    graph: gs.Graph,
) -> dict[tuple[str, int], gs.Variable]:
    """Map ``(weight_token, level)`` -> the head conv's output tensor."""
    found: dict[tuple[str, int], gs.Variable] = {}
    for node in graph.nodes:
        if node.op != "Conv":
            continue
        for inp in node.inputs:
            name = inp.name
            if not name.startswith("head.head_module.out_"):
                continue
            # e.g. head.head_module.out_kpt_vis.0.weight
            parts = name.split(".")
            token, level = parts[2], int(parts[3])
            found[(token, level)] = node.outputs[0]
    return found


def strip_postprocess(graph: gs.Graph) -> gs.Graph:
    """Set the eight head conv outputs as the graph outputs and drop the tail."""
    conv_outs = _find_head_conv_outputs(graph)

    new_outputs: list[gs.Variable] = []
    for token, stem, _ch in _HEAD_BRANCHES:
        for level, stride in enumerate(_STRIDES):
            key = (token, level)
            if key not in conv_outs:
                raise RuntimeError(f"head conv for {token} level {level} not found")
            tensor = conv_outs[key]
            tensor.name = f"{stem}_s{stride}"
            tensor.dtype = np.float32
            new_outputs.append(tensor)

    graph.outputs = new_outputs
    # Drop the now-unreachable decode/NMS/DCC nodes and their initializers.
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    logger.info("Stripped post-processing; kept %d nodes", len(graph.nodes))
    return graph


def _rewrite_pos_enc(graph: gs.Graph, s32: int) -> int:
    """Regenerate every ``neck.pos_enc_*`` constant for an ``s32 x s32`` grid."""
    n = 0
    for tensor in graph.tensors().values():
        if not (isinstance(tensor, gs.Constant) and tensor.name.startswith(_POS_ENC_PREFIX)):
            continue
        embed_dim = int(tensor.values.shape[-1])
        tensor.values = build_2d_sincos_position_embedding(s32, s32, embed_dim)
        logger.info("Rewrote %s -> %s", tensor.name, tensor.values.shape)
        n += 1
    return n


def _rewrite_encoder_unflatten(graph: gs.Graph, old_s32: int, new_s32: int) -> int:
    """Rewrite the AIFI unflatten Reshape target ``[-1, C, old, old]``."""
    n = 0
    for tensor in graph.tensors().values():
        if not isinstance(tensor, gs.Constant):
            continue
        v = np.asarray(tensor.values).ravel()
        if (
            v.shape == (4,)
            and int(v[0]) == -1
            and int(v[2]) == old_s32
            and int(v[3]) == old_s32
        ):
            tensor.values = np.array([-1, int(v[1]), new_s32, new_s32], dtype=v.dtype)
            logger.info("Rewrote encoder unflatten Reshape target -> %s", tensor.values.tolist())
            n += 1
    return n


def resize_input(graph: gs.Graph, input_size: int, batch: int = 1) -> gs.Graph:
    """Re-target the backbone+neck+head subgraph to ``input_size x input_size``.

    Must be called *after* :func:`strip_postprocess` (the post-processing tail
    carries input-size-specific anchor priors that would otherwise be stale).
    """
    if input_size % 32 != 0:
        raise ValueError(f"input_size must be divisible by 32 (stride-32 level), got {input_size}")

    inp = graph.inputs[0]
    old_h = inp.shape[2]
    if not isinstance(old_h, int):
        raise RuntimeError(f"expected a static input height, got {inp.shape!r}")
    old_s32 = old_h // 32
    new_s32 = input_size // 32

    inp.shape = [batch, 3, input_size, input_size]
    inp.dtype = np.float32

    n_pos = _rewrite_pos_enc(graph, new_s32)
    if n_pos == 0:
        raise RuntimeError("no neck.pos_enc_* constant found to rewrite")
    n_rs = _rewrite_encoder_unflatten(graph, old_s32, new_s32)
    if n_rs == 0:
        raise RuntimeError(
            f"encoder unflatten Reshape target [-1,C,{old_s32},{old_s32}] not found"
        )

    # Give the outputs concrete static shapes for the compiler.
    idx = 0
    for _token, stem, ch in _HEAD_BRANCHES:
        for stride in _STRIDES:
            side = input_size // stride
            out = graph.outputs[idx]
            out.shape = [batch, ch, side, side]
            out.dtype = np.float32
            idx += 1

    graph.cleanup().toposort()
    return graph


def build_stripped_model(
    model: onnx.ModelProto,
    input_size: int = 320,
    batch: int = 1,
) -> onnx.ModelProto:
    """Strip post-processing and re-target to ``input_size``; return the ONNX model."""
    graph = gs.import_onnx(model)
    strip_postprocess(graph)
    resize_input(graph, input_size=input_size, batch=batch)
    out = gs.export_onnx(graph)
    out.ir_version = model.ir_version
    # Drop value_info carried over from the source graph: after resizing, those
    # stale shapes (sized for the original input) contradict the real ones and
    # trip ORT/compiler shape inference. Let it be recomputed downstream.
    del out.graph.value_info[:]
    return out
