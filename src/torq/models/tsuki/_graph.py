# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os

import onnx
import onnx_graphsurgeon as gs

from ...graph_edit import (
    DimMatchType,
    FixedDimMapping,
    OnnxGraphEditor,
)
from ...graph_edit.edits import *
from ...graph_edit.tsuki_edits import *

class TsukiOnnxGraphEditor(OnnxGraphEditor, NormalizationPatches, MiscTsukiPatches):
    def __init__(
        self,
        graph: gs.Graph,
    ):
        super().__init__(
            graph,
            "model",
        )