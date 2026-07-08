# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import numpy as np
import onnx_graphsurgeon as gs


def _static_int_shape(var) -> list[int] | None:
    shape = getattr(var, "shape", None)
    if shape is None:
        return None
    out: list[int] = []
    for dim in shape:
        try:
            val = int(dim)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        out.append(val)
    return out

def _const_array(t) -> np.ndarray | None:
    if isinstance(t, gs.Constant):
        return np.asarray(t.values)
    producers = getattr(t, "inputs", None) or []
    if len(producers) != 1 or producers[0].op != "Constant":
        return None
    value = producers[0].attrs.get("value")
    if value is None or not hasattr(value, "values"):
        return None
    return np.asarray(value.values)

def _const_int_list(t) -> list[int] | None:
    arr = _const_array(t)
    if arr is None:
        return None
    return [int(v) for v in arr.reshape(-1)]

def _consumers_in_graph(graph: gs.Graph, name: str) -> list[gs.Node]:
    return [
        n for n in graph.nodes
        if any(getattr(inp, "name", None) == name for inp in n.inputs)
    ]
