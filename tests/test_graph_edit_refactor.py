# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import dataclasses
import inspect
import unittest

import numpy as np
import onnx
import onnx_graphsurgeon as gs


class GraphEditImportSurfaceTests(unittest.TestCase):
    def test_package_reexports_and_direct_submodule_imports_match(self):
        from torq.graph_edit.edits import (
            CommonGraphEditsMixin,
            CombineKVCacheMixin,
            DecomposeStridedConv1D,
            ReplaceInt64FloatCast,
            ReplacePadWithConcat,
        )
        from torq.graph_edit.edits.arithmetic import (
            ReplaceInt64FloatCast as ArithmeticReplaceInt64FloatCast,
        )
        from torq.graph_edit.edits.conv import (
            DecomposeStridedConv1D as ConvDecomposeStridedConv1D,
        )
        from torq.graph_edit.edits.padding import (
            ReplacePadWithConcat as PaddingReplacePadWithConcat,
        )
        from torq.graph_edit.edits.transformer import (
            CombineKVCacheMixin as TransformerCombineKVCacheMixin,
        )
        from torq.graph_edit.edits.mixins import (
            CommonGraphEditsMixin as MixinsCommonGraphEditsMixin,
        )

        self.assertIs(ReplaceInt64FloatCast, ArithmeticReplaceInt64FloatCast)
        self.assertIs(ReplacePadWithConcat, PaddingReplacePadWithConcat)
        self.assertIs(DecomposeStridedConv1D, ConvDecomposeStridedConv1D)
        self.assertIs(CombineKVCacheMixin, TransformerCombineKVCacheMixin)
        self.assertIs(CommonGraphEditsMixin, MixinsCommonGraphEditsMixin)

    def test_reexported_graph_edit_classes_remain_dataclasses(self):
        import torq.graph_edit.edits as edits
        from torq.graph_edit import OnnxGraphEdit

        missing = []
        for name in edits.__all__:
            obj = getattr(edits, name)
            if inspect.isclass(obj) and issubclass(obj, OnnxGraphEdit):
                if not dataclasses.is_dataclass(obj):
                    missing.append(name)
        self.assertEqual(missing, [])

    def test_model_graph_editors_import_with_explicit_mixins(self):
        from torq.models.gemma3._graph import Gemma3OnnxGraphEditor
        from torq.models.moonshine._graph import MoonshineOnnxGraphEditor
        from torq.models.smollm2._graph import SmolLM2OnnxGraphEditor
        from torq.models.synaptics_audio.prepare import _SynapticsAudioGraphEditor

        for cls in (
            Gemma3OnnxGraphEditor,
            MoonshineOnnxGraphEditor,
            SmolLM2OnnxGraphEditor,
            _SynapticsAudioGraphEditor,
        ):
            self.assertTrue(hasattr(cls, "eliminate_transposes"))


class PromotedGraphEditTests(unittest.TestCase):
    def test_replace_int64_float_cast_rewires_consumers_to_lut_path(self):
        from torq.graph_edit.edits.arithmetic import ReplaceInt64FloatCast

        idx = gs.Variable("idx", dtype=np.dtype(np.int64), shape=[1, 1])
        cast_out = gs.Variable(
            "idx_float", dtype=onnx.TensorProto.FLOAT, shape=[1, 1]
        )
        add_out = gs.Variable(
            "add_out", dtype=onnx.TensorProto.FLOAT, shape=[1, 1]
        )
        cast = gs.Node(
            op="Cast",
            name="cast_idx",
            inputs=[idx],
            outputs=[cast_out],
            attrs={"to": onnx.TensorProto.FLOAT},
        )
        add = gs.Node(
            op="Add",
            name="use_cast",
            inputs=[cast_out, gs.Constant("one", np.array([[1]], dtype=np.float32))],
            outputs=[add_out],
        )
        graph = gs.Graph(nodes=[cast, add], inputs=[idx], outputs=[add_out])

        edit = ReplaceInt64FloatCast(graph, "test", max_int=8)
        self.assertTrue(edit.match(cast))
        edit.transform(cast)

        self.assertEqual(cast.outputs, [])
        self.assertIsNot(add.inputs[0], cast_out)
        self.assertEqual(add.inputs[0].name, "idx_float_value_batched")
        self.assertIn("Gather", {node.op for node in graph.nodes})

    def test_replace_pad_with_concat_replaces_constant_pad_node(self):
        from torq.graph_edit.edits.padding import ReplacePadWithConcat

        data = gs.Variable("data", dtype=np.float32, shape=[1, 2])
        out = gs.Variable("padded", dtype=np.float32, shape=[1, 4])
        pad = gs.Node(
            op="Pad",
            name="pad",
            inputs=[
                data,
                gs.Constant("pads", np.array([0, 1, 0, 1], dtype=np.int64)),
                gs.Constant("value", np.array(0, dtype=np.float32)),
            ],
            outputs=[out],
            attrs={"mode": "constant"},
        )
        graph = gs.Graph(nodes=[pad], inputs=[data], outputs=[out])

        edit = ReplacePadWithConcat(graph, "test")
        self.assertTrue(edit.match(pad))
        edit.transform(pad)

        self.assertEqual(pad.outputs, [])
        concat_nodes = [node for node in graph.nodes if node.op == "Concat"]
        self.assertEqual(len(concat_nodes), 1)
        self.assertIs(concat_nodes[0].outputs[0], out)
        self.assertEqual(concat_nodes[0].attrs["axis"], 1)

    def test_decompose_strided_conv1d_replaces_conv_graph_output(self):
        from torq.graph_edit.edits.conv import DecomposeStridedConv1D

        data = gs.Variable("data", dtype=np.float32, shape=[1, 1, 5])
        weight = gs.Constant("weight", np.ones((2, 1, 3), dtype=np.float32))
        out = gs.Variable("conv_out", dtype=np.float32, shape=[1, 2, 2])
        conv = gs.Node(
            op="Conv",
            name="conv",
            inputs=[data, weight],
            outputs=[out],
            attrs={"kernel_shape": [3], "strides": [2], "group": 1, "pads": [0, 0]},
        )
        graph = gs.Graph(nodes=[conv], inputs=[data], outputs=[out])

        edit = DecomposeStridedConv1D(graph, "test")
        self.assertTrue(edit.match(conv))
        edit.transform(conv)

        self.assertEqual(conv.outputs, [])
        self.assertIn("MatMul", {node.op for node in graph.nodes})
        self.assertIn("Slice", {node.op for node in graph.nodes})
        self.assertEqual(graph.outputs[0].name, "conv_out")


if __name__ == "__main__":
    unittest.main()
