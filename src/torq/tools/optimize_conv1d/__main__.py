# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from torq.models.tsuki._graph import TsukiOnnxGraphEditor
from ..common import maybe_load_onnx_model


def ensure_compatible_ops(input_path: str) -> onnx.ModelProto:
    model = maybe_load_onnx_model(input_path)
    graph = gs.import_onnx(model)
    graph_editor = TsukiOnnxGraphEditor(graph)
    graph_editor.convert_1x1_conv1d_to_gemm()
    return gs.export_onnx(graph)

def main():
    parser = argparse.ArgumentParser(description="Decompose Instance/Layer Normalization into component parts")
    parser.add_argument("-i", "--input", help="Input ONNX model path", required=True)
    parser.add_argument("-o", "--output", help="Output ONNX model path", required=True)
    args = parser.parse_args()
    model = ensure_compatible_ops(args.input)
    onnx.save(model, args.output)

if __name__ == "__main__":
    main()