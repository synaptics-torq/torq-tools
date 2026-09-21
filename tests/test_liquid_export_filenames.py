# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import json
import tempfile
import unittest
from pathlib import Path

from torq.models.liquid.export import LiquidModelExporter
from torq.models.liquid.export_vl import DECODER, DECODER_PREFILL, LiquidVLModelExporter


_CONFIG = {
    "hidden_size": 1024,
    "vocab_size": 65536,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "conv_dim": 1024,
    "conv_L_cache": 3,
    "num_hidden_layers": 16,
    "layer_types": ["conv", "full_attention"],
    "bos_token_id": 1,
    "eos_token_id": 7,
}


def _write_config(directory: Path) -> Path:
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text(json.dumps(_CONFIG))
    return source_dir


class LiquidExportFilenameTests(unittest.TestCase):
    def test_text_export_uses_model_filename(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/onnx/fp32/static/model.onnx"),
        )

    def test_batch_prefill_uses_distinct_prefill_filename(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component("model_prefill"),
            Path("export/onnx/fp32/static/model_prefill.onnx"),
        )

    def test_vl_batch_prefill_uses_distinct_decoder_filename(self):
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        exporter._export_dir = Path("export/onnx/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component(DECODER),
            Path("export/onnx/fp32/static/decoder_model_merged.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component(DECODER_PREFILL),
            Path("export/onnx/fp32/static/decoder_model_merged_prefill.onnx"),
        )


class LiquidBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidModelExporter:
        source_dir = _write_config(tmp_path)
        return LiquidModelExporter(onnx_source_dir=source_dir, **kwargs)

    def test_batch_prefill_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._create_exporter(self.tmp, batch_prefill=0)

    def test_batch_prefill_cannot_exceed_max_gen_tokens(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._create_exporter(self.tmp, batch_prefill=9, max_gen_tokens=8)

    def test_batch_prefill_requires_static_models(self):
        with self.assertRaisesRegex(ValueError, "static LFM exports"):
            self._create_exporter(self.tmp, batch_prefill=8, static_models=False)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)

    def test_batch_prefill_block_present_only_when_enabled(self):
        enabled = self._create_exporter(self.tmp, batch_prefill=8)
        disabled = self._create_exporter(self.tmp)

        self.assertIn("model.patch (batch prefill)", enabled.graph_edit_blocks())
        self.assertNotIn("model.patch (batch prefill)", disabled.graph_edit_blocks())


class LiquidVLBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidVLModelExporter:
        source_dir = _write_config(tmp_path)
        return LiquidVLModelExporter(onnx_source_dir=source_dir, **kwargs)

    def test_batch_prefill_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._create_exporter(self.tmp, batch_prefill=0)

    def test_batch_prefill_cannot_exceed_max_gen_tokens(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._create_exporter(self.tmp, batch_prefill=9, max_gen_tokens=8)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)


class LiquidSplitLMHeadTests(unittest.TestCase):
    HIDDEN = 8
    VOCAB = 16

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        # Tiny liquid-shaped decoder: x -> Add(bias) -> normed,
        # normed -> MatMul(lm_head_W) -> logits, plus a kv passthrough output
        # to verify non-logits outputs survive the split.
        weight = np.arange(cls.HIDDEN * cls.VOCAB, dtype=np.float32).reshape(cls.HIDDEN, cls.VOCAB)
        bias = np.ones(cls.HIDDEN, dtype=np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("Add", ["x", "norm_bias"], ["normed"], name="/model/final_norm/Add"),
                helper.make_node("MatMul", ["normed", "lm_head_W"], ["logits"], name="/model/lm_head/MatMul"),
                helper.make_node("Identity", ["kv_in"], ["kv_out"], name="/model/kv/Identity"),
            ],
            "main",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, cls.HIDDEN]),
                helper.make_tensor_value_info("kv_in", TensorProto.FLOAT, [1, 1, cls.HIDDEN]),
            ],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, cls.VOCAB]),
                helper.make_tensor_value_info("kv_out", TensorProto.FLOAT, [1, 1, cls.HIDDEN]),
            ],
            [numpy_helper.from_array(weight, "lm_head_W"), numpy_helper.from_array(bias, "norm_bias")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        # value_info must carry the boundary tensor's dtype, as the real
        # static export does after _propagate_static_shapes.
        cls._model = onnx.shape_inference.infer_shapes(
            model, check_type=False, strict_mode=False, data_prop=True
        )

    def _write_model(self, directory: Path) -> Path:
        import onnx

        onnx.save(self._model, str(directory / "model.onnx"))
        return directory / "model.onnx"

    def _make_exporter(self, model_path: Path) -> LiquidModelExporter:
        import logging
        from onnx import TensorProto

        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._logger = logging.getLogger("split-lm-head-test")
        exporter._hidden_size = self.HIDDEN
        exporter._vocab_size = self.VOCAB
        exporter._onnx_export_dtype = TensorProto.FLOAT
        exporter._split_weights = False
        exporter._export_paths = {"model": model_path}
        return exporter

    def test_split_lm_head_requires_static_models(self):
        with tempfile.TemporaryDirectory() as td:
            source_dir = _write_config(Path(td))
            with self.assertRaisesRegex(ValueError, "static LFM exports"):
                LiquidModelExporter(onnx_source_dir=source_dir, static_models=False, split_lm_head=True)

    def test_setup_dirs_keeps_topologies_separate(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._models_dir = Path("models/liquid-2p5-350m")
        exporter._onnx_source_dir = Path("models/liquid-2p5-350m/source/onnx/fp32")
        exporter._model_dtype = "fp32"
        exporter._static_models = True
        exporter._convert_dtypes = True

        exporter._split_lm_head = False
        _, unified_export_dir, unified_quant_dir, unified_convert_dir, unified_torq_dir = exporter._setup_dirs()
        exporter._split_lm_head = True
        _, split_export_dir, split_quant_dir, split_convert_dir, split_torq_dir = exporter._setup_dirs()

        for a, b in (
            (unified_export_dir, split_export_dir),
            (unified_quant_dir, split_quant_dir),
            (unified_convert_dir, split_convert_dir),
            (unified_torq_dir, split_torq_dir),
        ):
            self.assertNotEqual(a, b)
        self.assertIn("unified", unified_export_dir.parts)
        self.assertIn("split_lm_head", split_export_dir.parts)
        self.assertIn("split_lm_head", split_torq_dir.parts)

    def test_make_lm_head_split_is_fp32_stage_independent(self):
        import onnx
        import numpy as np
        from onnx import TensorProto

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # The split rewrites model.onnx in place (the fused decoder is
            # not kept), so keep a reference copy for the recomposition check.
            ref_path = td / "fused_ref.onnx"
            onnx.save(self._model, str(ref_path))
            model_path = self._write_model(td)
            exporter = self._make_exporter(model_path)
            exporter.make_lm_head_split(model_path)

            # The head is registered as a first-class component next to the
            # model (not into a conversion-specific dir).
            self.assertEqual(exporter._export_paths["model"], model_path)
            self.assertEqual(exporter._export_paths["lm_head"], model_path.parent / "lm_head.onnx")

            body = onnx.load(str(model_path))
            lm_head = onnx.load(str(exporter._export_paths["lm_head"]))
            onnx.checker.check_model(body, full_check=True)
            onnx.checker.check_model(lm_head, full_check=True)

            # model.onnx is now the body: lm_head node dropped, hidden state
            # exposed as `last_hidden_states` first output (gemma3 naming),
            # dtype inferred from the (fp32) model, not assumed.
            body_nodes = {n.name for n in body.graph.node}
            self.assertNotIn("/model/lm_head/MatMul", body_nodes)
            body_outputs = [o.name for o in body.graph.output]
            self.assertEqual(body_outputs, ["last_hidden_states", "kv_out"])
            self.assertEqual(
                body.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT
            )
            self.assertEqual([i.name for i in body.graph.input], ["x", "kv_in"])

            # lm_head: standalone last_hidden_states -> logits.
            self.assertEqual(len(lm_head.graph.node), 1)
            self.assertEqual(len(lm_head.graph.initializer), 1)
            self.assertEqual([i.name for i in lm_head.graph.input], ["last_hidden_states"])
            self.assertEqual(
                lm_head.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT
            )
            self.assertEqual([o.name for o in lm_head.graph.output], ["logits"])

            # body + lm_head recompose to the (former) fused model's logits.
            import onnxruntime as ort

            rng = np.random.default_rng(0)
            x = rng.standard_normal([1, 1, self.HIDDEN]).astype(np.float32)
            kv = rng.standard_normal([1, 1, self.HIDDEN]).astype(np.float32)
            sess_body = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            hidden = dict(
                zip([o.name for o in sess_body.get_outputs()], sess_body.run(None, {"x": x, "kv_in": kv}))
            )
            sess_lm = ort.InferenceSession(
                str(exporter._export_paths["lm_head"]), providers=["CPUExecutionProvider"]
            )
            logits = sess_lm.run(None, {"last_hidden_states": hidden["last_hidden_states"]})[0]
            sess_ref = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
            ref_logits = sess_ref.run(None, {"x": x, "kv_in": kv})[0]
            self.assertLess(float(np.max(np.abs(logits - ref_logits))), 1e-6)


if __name__ == "__main__":
    unittest.main()
