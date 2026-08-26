# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Graph surgery for YOLO26 detection.

:func:`strip_postprocess` cuts the model at the six one2one (NMS-free) head
convolutions, dropping the fixed-k TopK / GatherElements / xyxy-decode tail
that Ultralytics bakes into the ONNX export (data-dependent-shape-free, but
still not NPU-friendly: sort/gather ops with no TOSA analogue). Output order
is level-ascending (stride 8, 16, 32), box branch then cls branch per level.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import onnx_graphsurgeon as gs

logger = logging.getLogger("yolo26-surgery")

# Ultralytics' Detect.one2one_cv{2,3} are Sequential(ConvBnAct, ConvBnAct, Conv2d);
# only the final plain nn.Conv2d (no ".conv." infix -> no fused BN/act) is the
# raw box/cls output for that level, regardless of model scale (n/s/m/l/x).
_HEAD_WEIGHT_RE = re.compile(r"\.one2one_cv([23])\.(\d+)\.\d+\.weight$")
_BRANCH_STEM = {"2": "box", "3": "cls"}
_STRIDES = (8, 16, 32)


def _find_head_conv_outputs(graph: gs.Graph) -> dict[tuple[str, int], gs.Variable]:
    """Map ``(branch, level)`` -> final one2one_cv{2,3} conv output, via weight name."""
    found = {}
    for node in graph.nodes:
        if node.op != "Conv":
            continue
        for inp in node.inputs:
            m = _HEAD_WEIGHT_RE.search(inp.name)
            if m and "conv.weight" not in inp.name:
                branch, level = m.group(1), int(m.group(2))
                found[(branch, level)] = node.outputs[0]
    return found


def strip_postprocess(graph: gs.Graph) -> gs.Graph:
    """Set the six one2one head conv outputs as graph outputs and drop the decode/TopK tail."""
    conv_outs = _find_head_conv_outputs(graph)
    expected = {(b, lvl) for b in _BRANCH_STEM for lvl in range(len(_STRIDES))}
    missing = expected - conv_outs.keys()
    if missing:
        raise RuntimeError(f"one2one head conv(s) not found for (branch, level): {sorted(missing)}")

    new_outputs = []
    for level, stride in enumerate(_STRIDES):
        for branch, stem in _BRANCH_STEM.items():
            tensor = conv_outs[(branch, level)]
            tensor.name, tensor.dtype = f"{stem}_s{stride}", np.float32
            new_outputs.append(tensor)
    graph.outputs = new_outputs
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    logger.info("Stripped post-processing; kept %d nodes", len(graph.nodes))
    return graph
