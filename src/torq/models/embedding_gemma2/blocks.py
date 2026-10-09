# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Cut layer ranges out of the static EmbeddingGemma-2 graphs (compiler bisection).

A layer's input is the residual-stream tensor normalised by its ``input_layernorm``.
``extract_layers(model, kind, first, last)`` returns a graph with that tensor as input
(plus whatever else the range needs: ``attention_bias`` and, for text, ``inputs_embeds``
which feeds the per-layer-embedding projection) and the residual stream after ``last``
(or the graph output for the final layer) as output.

    python -m torq.models.embedding_gemma2.blocks <static.onnx> {text,vision} FIRST LAST -o out.onnx
"""

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs

LAYER_COUNT = {"text": 24, "vision": 16, "audio": 12}
NORM_WEIGHT = {
    "text": "model.layers.{}.input_layernorm.weight",
    "vision": "vision_tower.encoder.layers.{}.input_layernorm.weight",
    "audio": "audio_tower.layers.{}.feed_forward1.pre_layer_norm.weight",
}


def _residual_into(graph: gs.Graph, kind: str, layer: int) -> gs.Variable:
    """The residual tensor that layer ``layer``'s input_layernorm normalises."""
    wname = NORM_WEIGHT[kind].format(layer)
    for node in graph.nodes:
        if node.op == "Mul" and any(getattr(i, "name", "") == wname for i in node.inputs):
            x = next(i for i in node.inputs if getattr(i, "name", "") != wname)
            inner = x.inputs[0]  # Mul(x, 1/rms)
            if inner.op != "Mul":
                break
            for cand in inner.inputs:
                if cand.inputs and cand.inputs[0].op not in ("Div",):
                    return cand
                if not cand.inputs:
                    return cand
    raise KeyError(f"cannot locate the input of {kind} layer {layer}")


def extract_layers(model: onnx.ModelProto, kind: str, first: int, last: int) -> onnx.ModelProto:
    graph = gs.import_onnx(model)
    n = LAYER_COUNT[kind]
    # last == -1 selects the front end only: graph input up to the input of layer 0
    if not (0 <= first <= last < n) and not (first == 0 and last == -1):
        raise ValueError(f"layer range {first}..{last} outside 0..{n - 1}")
    out = graph.outputs[0] if last == n - 1 else _residual_into(graph, kind, last + 1)
    inp = graph.inputs[0] if first == 0 else _residual_into(graph, kind, first)
    shape, dtype = inp.shape, inp.dtype
    new_in = gs.Variable(f"layer{first}_input", dtype=dtype, shape=shape)
    for consumer in list(inp.outputs):
        consumer.inputs = [new_in if i is inp else i for i in consumer.inputs]
    keep = [i for i in graph.inputs if i is not inp]
    graph.inputs = [new_in] + keep
    graph.outputs = [out]
    graph.cleanup().toposort()
    used = {t.name for node in graph.nodes for t in node.inputs}
    graph.inputs = [i for i in graph.inputs if i.name in used]
    graph.cleanup().toposort()
    m = gs.export_onnx(graph)
    m.graph.name = "main"
    # Remember which full-graph tensor feeds the block, for reference generation.
    onnx.helper.set_model_props(m, {"block_source_input": inp.name, "block_input": new_in.name})
    return onnx.shape_inference.infer_shapes(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("kind", choices=list(LAYER_COUNT))
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    m = extract_layers(onnx.load(args.model), args.kind, args.first, args.last)
    onnx.save(m, args.output)
    print(args.output, [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
          "->", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
          len(m.graph.node), "nodes")


if __name__ == "__main__":
    main()
