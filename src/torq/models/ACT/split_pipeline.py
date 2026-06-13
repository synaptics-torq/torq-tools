#!/usr/bin/env python3
"""Stage — split the surgered ONNX into hardware-runnable pieces (onnx.utils.extract_model).

Two splits, both by tensor name:

1. HYBRID split at `permute_1` ([15,20,1,512]) — the ResNet backbone output:
     backbone   : [image, state] -> permute_1     (conv-heavy; torq's strength)
     transformer: [permute_1, state] -> action     (attention; the part we surgered)

2. TRANSFORMER 2-PIECE split (avoids the 714 MB single module OOMing on the 2 GB board).
   The full bf16 transformer is ~4 encoder layers (156 MB each) + decoder (60 MB). We cut
   the encoder in half at the inter-layer residual tensor `layer_norm_3`:
     piece_A: [permute_1, state] -> layer_norm_3        (encoder L1+L2,   ~327 MB)
     piece_B: layer_norm_3       -> action              (encoder L3+L4 + decoder, ~387 MB)
   Run sequentially A -> (layer_norm_3 tensor) -> B. The decoder's query stream is a
   constant (ACT action-slot embeddings), so piece_B needs only layer_norm_3 as input.

Tensor names (permute_1, layer_norm_3, state, action) come from this model's export; if
they differ, inspect with work-dev/lerobot/split_model.py to find the boundaries.

Usage:
  python split_pipeline.py MODEL.onnx --hybrid            # -> backbone.onnx, transformer.onnx
  python split_pipeline.py transformer.onnx --two-piece   # -> piece_A.onnx, piece_B.onnx
"""
import argparse, onnx
from onnx.utils import extract_model
from collections import Counter

SPLIT_TENSOR = "permute_1"      # backbone <-> transformer boundary
MID_TENSOR = "layer_norm_3"     # encoder L2|L3 residual boundary


def _summ(path):
    m = onnx.load(path)
    c = Counter(n.op_type for n in m.graph.node)
    io = lambda lst: [(t.name, [d.dim_value for d in t.type.tensor_type.shape.dim]) for t in lst]
    print(f"  {path}: {len(m.graph.node)} nodes, {c['Softmax']} softmax | "
          f"in={io(m.graph.input)} out={io(m.graph.output)}")


def _rename_main(path):
    m = onnx.load(path); m.graph.name = "main_graph"; onnx.save(m, path)  # torq func name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--two-piece", action="store_true")
    a = ap.parse_args()

    if a.hybrid:
        extract_model(a.model, "backbone.onnx", ["image_side"], [SPLIT_TENSOR])
        extract_model(a.model, "transformer.onnx", [SPLIT_TENSOR, "state"], ["action"])
        for f in ("backbone.onnx", "transformer.onnx"):
            _rename_main(f); _summ(f)
    if a.two_piece:
        extract_model(a.model, "piece_A.onnx", [SPLIT_TENSOR, "state"], [MID_TENSOR])
        extract_model(a.model, "piece_B.onnx", [MID_TENSOR], ["action"])
        for f in ("piece_A.onnx", "piece_B.onnx"):
            _rename_main(f); _summ(f)


if __name__ == "__main__":
    main()
