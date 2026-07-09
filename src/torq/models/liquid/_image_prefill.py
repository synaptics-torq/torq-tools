# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""One-shot image-prefill decoder (``decoder_image``) build + layer split.

LFM2-VL feeds an image to the SigLIP encoder, which emits 64 image tokens.
Instead of pushing those through the decoder one token at a time (64 serial NPU
calls, dominating TTFT), the image-prefill decoder runs the whole 16-layer
stack once over ``[1, 64, 1024]`` and returns only the per-layer conv/KV caches
(lm_head dropped — prefill produces no token).

Build (§3b of image_prefill.md), on the custom-op-replaced *dynamic* decoder:
fix ``token_embedding`` to ``[1, 64, 1024]``, bake the incoming caches to empty
constants (image is first in the sequence), drop ``logits`` so the lm_head is
pruned, then apply the same Conv1D->MatMul + shape-prop chip rewrites the decode
decoder uses. Building in **3D** (keeping the leading batch dim) is required —
the rank-2 variant is numerically broken on the NPU (see image_prefill.md §3c).

The full 16-layer model is too large to co-fit in NPU SRAM, so it is cut at
layer boundaries into N parts, each ``[1,64,1024] -> [1,64,1024]`` chained by
the layer-boundary hidden ``/model/layers.{k}/Add_2/output_0`` and emitting the
caches for the layers it owns.
"""
import re

import numpy as np
import onnx
import onnx_graphsurgeon as gs

S_IMAGE = 64
NLAYERS = 16

# Layer ranges per part count (image_prefill.md §3d).
PART_LAYER_RANGES = {
    2: [(0, 7), (8, 15)],
    3: [(0, 5), (6, 10), (11, 15)],
    5: [(0, 2), (3, 5), (6, 8), (9, 11), (12, 15)],
}


def build_image_decoder(dynamic_decoder, replace_conv1d, propagate_shapes):
    """Build the static cache-only image decoder from the custom-op-replaced
    *dynamic* decoder.

    ``replace_conv1d`` is ``LiquidModelExporter._replace_conv1d_with_matmul``
    (``model -> (model, n)``) and ``propagate_shapes`` is the exporter's
    ``_propagate_static_shapes`` (``model -> model``) — passed in so this module
    stays free of exporter imports.
    """
    g = gs.import_onnx(dynamic_decoder)
    keep = []
    for inp in list(g.inputs):
        n = inp.name
        if n in ("inputs_embeds", "token_embedding"):
            inp.shape = [1, S_IMAGE, 1024]
            keep.append(inp)
        elif n == "attention_mask":
            c = gs.Constant(n + "_const", np.ones((1, S_IMAGE), np.int64))
            for cons in list(inp.outputs):
                cons.inputs = [c if x is inp else x for x in cons.inputs]
        elif n.startswith("past_conv"):
            c = gs.Constant(n + "_const", np.zeros((1, 1024, 3), np.float32))
            for cons in list(inp.outputs):
                cons.inputs = [c if x is inp else x for x in cons.inputs]
        elif "key" in n or "value" in n:
            c = gs.Constant(n + "_const", np.zeros((1, 8, 0, 64), np.float32))
            for cons in list(inp.outputs):
                cons.inputs = [c if x is inp else x for x in cons.inputs]
        else:
            keep.append(inp)
    g.inputs = keep
    g.cleanup().toposort()

    # Replace Conv1D->MatMul while the graph is still intact (logits present).
    # Dropping logits first prunes final_norm and makes the rewrite skip the
    # last conv layer, leaving a stray Split that breaks the layer-15 part.
    m = onnx.shape_inference.infer_shapes(gs.export_onnx(g))
    m.ir_version = 10
    m = propagate_shapes(m)
    m, _ = replace_conv1d(m)
    m = propagate_shapes(m)

    # Make it cache-only: drop logits so the lm_head becomes dead + pruned. Keep
    # the last layer's boundary hidden as an output so the final layer's gating
    # (and thus its conv Split) stays alive — otherwise the last part would lose
    # a Split output and fail shape inference.
    g2 = gs.import_onnx(m)
    g2.outputs = [o for o in g2.outputs if o.name != "logits"]
    # Keep the last decoder layer alive so its conv Split stays 3-output (else
    # the last part loses a Split slice). Prefer the highest-index layer-boundary
    # Add_2; fall back to the final-norm hidden (the lm_head's activation input).
    tmap = g2.tensors()
    boundaries = sorted(
        (t for name, t in tmap.items()
         if re.fullmatch(r"/model/layers\.\d+/Add_2/output_0", name)),
        key=lambda t: int(re.search(r"layers\.(\d+)", t.name).group(1)))
    keep_alive = boundaries[-1] if boundaries else None
    if keep_alive is None:
        logits_t = next((o for o in gs.import_onnx(m).outputs if o.name == "logits"), None)
        if logits_t is not None and logits_t.inputs:
            keep_alive = next((tmap.get(i.name) for i in logits_t.inputs[0].inputs
                               if getattr(i, "name", None) in tmap
                               and not isinstance(tmap.get(i.name), gs.Constant)), None)
    if keep_alive is not None and keep_alive not in g2.outputs:
        g2.outputs.append(keep_alive)
    g2.cleanup().toposort()
    m = onnx.shape_inference.infer_shapes(gs.export_onnx(g2))
    m.ir_version = 10
    return m


def _layer_of(name):
    mm = re.search(r"present_conv\.(\d+)|present\.(\d+)\.(key|value)", name)
    return int(mm.group(1) or mm.group(2)) if mm else None


def split_image_decoder(full_model, nparts):
    """Split the full image decoder into ``nparts`` layer-boundary parts.

    Returns ``[(label, ModelProto), ...]`` with labels ``A, B, ...``. Each part
    takes the hidden entering its first layer (``token_embedding`` for part A,
    else the previous layer boundary) and outputs the next boundary hidden
    (unless it is the last part) plus the caches for the layers it owns.

    Uses onnx_graphsurgeon (not ``onnx.utils.extract_model``): the last layer's
    hidden path is dead in the cache-only decoder, and extract_model prunes a
    Split output there without updating its ``split`` count, which then fails
    shape inference. gs ``cleanup()`` keeps every Split output (unused ones
    become harmless dangling variables), so the parts stay well-formed.
    """
    if nparts not in PART_LAYER_RANGES:
        raise ValueError(f"unsupported image-decoder part count {nparts} "
                         f"(supported: {sorted(PART_LAYER_RANGES)})")
    ranges = PART_LAYER_RANGES[nparts]
    all_out = [o.name for o in full_model.graph.output]

    parts = []
    for i, (a, b) in enumerate(ranges):
        g = gs.import_onnx(full_model)
        tmap = g.tensors()
        in_name = "token_embedding" if a == 0 else f"/model/layers.{a - 1}/Add_2/output_0"
        in_t = tmap[in_name]
        in_t.inputs.clear()          # detach producer -> pure graph input
        in_t.dtype = np.float32
        in_t.shape = [1, S_IMAGE, 1024]

        # caches for the owned layers (in the full-decoder order). Intermediate
        # parts also output the boundary hidden last — the runner reads outs[-1]
        # as the next hidden. The LAST part emits caches only (the runner treats
        # every output of the final part as a cache). remove_unused_node_outputs
        # is off so the last layer's conv Split keeps all 3 outputs (its dead
        # gating slices dangle harmlessly) rather than being trimmed to an
        # invalid split-count / -outputs mismatch.
        caches = [o for o in all_out if (_layer_of(o) is not None and a <= _layer_of(o) <= b)]
        if b < NLAYERS - 1:
            out_names = caches + [f"/model/layers.{b}/Add_2/output_0"]
        else:
            out_names = caches
        g.inputs = [in_t]
        g.outputs = [tmap[o] for o in out_names]
        g.cleanup().toposort()

        # In the last part the final layer's gating dies, leaving its conv Split
        # with dead (dangling) output slices. A Split's output count must equal
        # its `split` sizes, but the bf16 converter trims the dangling outputs
        # and then shape inference fails. So replace any Split that has a dead
        # output with a Slice per *live* output (correct byte range from the
        # `/output_N` position) and drop the Split — the dead slices are simply
        # never computed.
        graph_out = {t.name for t in g.outputs}
        for node in list(g.nodes):
            if node.op != "Split":
                continue
            sizes = None
            if len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
                sizes = [int(v) for v in node.inputs[1].values]
            elif "split" in node.attrs:
                sizes = [int(v) for v in node.attrs["split"]]
            if not sizes:
                continue
            live = [o for o in node.outputs if o.outputs or o.name in graph_out]
            if len(live) == len(node.outputs) == len(sizes):
                continue  # well-formed and fully used
            data = node.inputs[0]
            axis = int(node.attrs.get("axis", 0))
            off = [0]
            for s in sizes:
                off.append(off[-1] + s)
            for o in live:
                mm = re.search(r"/output_(\d+)$", o.name)
                k = int(mm.group(1)) if mm else 0
                sl = gs.Variable(o.name + "_sl", dtype=o.dtype or np.float32)
                g.layer(
                    op="Slice", name=o.name + "/slice",
                    inputs=[data,
                            gs.Constant(o.name + "_st", np.array([off[k]], np.int64)),
                            gs.Constant(o.name + "_en", np.array([off[k + 1]], np.int64)),
                            gs.Constant(o.name + "_ax", np.array([axis], np.int64))],
                    outputs=[sl],
                )
                for c in g.nodes:
                    c.inputs = [sl if x is o else x for x in c.inputs]
            node.outputs.clear()
            node.inputs.clear()
        g.cleanup().toposort()
        parts.append((chr(ord("A") + i), gs.export_onnx(g)))
    return parts
