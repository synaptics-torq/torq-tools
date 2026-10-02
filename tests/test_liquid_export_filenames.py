# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from torq.models.liquid.export import LiquidModelExporter
from torq.models.liquid.export_vl import (
    DECODER,
    DECODER_PREFILL,
    LiquidVLModelExporter,
    VISION,
)


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


def _build_tiny_liquid_decoder(hidden: int = 8, vocab: int = 16):
    """Tiny liquid-shaped decoder: x -> Add(bias) -> normed,
    normed -> MatMul(lm_head_W) -> logits, plus a kv passthrough output
    to verify non-logits outputs survive the split."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.arange(hidden * vocab, dtype=np.float32).reshape(hidden, vocab)
    bias = np.ones(hidden, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "norm_bias"], ["normed"], name="/model/final_norm/Add"),
            helper.make_node("MatMul", ["normed", "lm_head_W"], ["logits"], name="/model/lm_head/MatMul"),
            helper.make_node("Identity", ["kv_in"], ["kv_out"], name="/model/kv/Identity"),
        ],
        "main",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, hidden]),
            helper.make_tensor_value_info("kv_in", TensorProto.FLOAT, [1, 1, hidden]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab]),
            helper.make_tensor_value_info("kv_out", TensorProto.FLOAT, [1, 1, hidden]),
        ],
        [numpy_helper.from_array(weight, "lm_head_W"), numpy_helper.from_array(bias, "norm_bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    # value_info must carry the boundary tensor's dtype, as the real
    # static export does after _propagate_static_shapes.
    return onnx.shape_inference.infer_shapes(
        model, check_type=False, strict_mode=False, data_prop=True
    )


class LiquidExportFilenameTests(unittest.TestCase):
    def test_text_export_uses_model_filename(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/fp32/static/model.onnx"),
        )

    def test_split_lm_head_uses_transformer_filenames(self):
        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._export_dir = Path("export/fp32/static")
        exporter._split_lm_head = True

        self.assertEqual(
            exporter._export_path_for_component("model"),
            Path("export/fp32/static/transformer.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component("model_prefill"),
            Path("export/fp32/static/transformer_prefill.onnx"),
        )

    def test_vl_batch_prefill_uses_distinct_decoder_filename(self):
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        exporter._export_dir = Path("export/fp32/static")

        self.assertEqual(
            exporter._export_path_for_component(DECODER),
            Path("export/fp32/static/decoder_model_merged.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component(DECODER_PREFILL),
            Path("export/fp32/static/decoder_model_merged_prefill.onnx"),
        )

    def test_vl_split_lm_head_uses_transformer_filenames(self):
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        exporter._export_dir = Path("export/split_lm_head/fp32/static")
        exporter._split_lm_head = True

        self.assertEqual(
            exporter._export_path_for_component(DECODER),
            Path("export/split_lm_head/fp32/static/transformer.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component(DECODER_PREFILL),
            Path("export/split_lm_head/fp32/static/transformer_prefill.onnx"),
        )
        self.assertEqual(
            exporter._export_path_for_component(VISION),
            Path("export/split_lm_head/fp32/static/vision_encoder.onnx"),
        )


class LiquidBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidModelExporter:
        source_dir = _write_config(tmp_path)
        kwargs.setdefault("split_lm_head", True)
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

    def test_batch_prefill_requires_split_lm_head(self):
        with self.assertRaisesRegex(ValueError, "requires `--split-lm-head`"):
            self._create_exporter(self.tmp, batch_prefill=8, split_lm_head=False)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)

    def test_batch_prefill_block_present_only_when_enabled(self):
        enabled = self._create_exporter(self.tmp, batch_prefill=8)
        disabled = self._create_exporter(self.tmp)

        self.assertIn("model.patch (batch prefill)", enabled.graph_edit_blocks())
        self.assertNotIn("model.patch (batch prefill)", disabled.graph_edit_blocks())

    def test_static_export_stages_runtime_assets(self):
        source_dir = _write_config(self.tmp)
        (source_dir / "tokenizer.json").write_text("tokenizer")
        export_dir = self.tmp / "export"
        model_path = export_dir / "model.onnx"

        exporter = LiquidModelExporter.__new__(LiquidModelExporter)
        exporter._onnx_dir = source_dir
        exporter._split_lm_head = False
        exporter._simulate_bf16 = False
        exporter._patch_static_model = Mock()

        exporter.apply_post_static_patches(model_path, "model")

        self.assertEqual((export_dir / "config.json").read_text(), json.dumps(_CONFIG))
        self.assertEqual((export_dir / "tokenizer.json").read_text(), "tokenizer")


class LiquidVLBatchPrefillValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_exporter(self, tmp_path: Path, **kwargs) -> LiquidVLModelExporter:
        source_dir = _write_config(tmp_path)
        kwargs.setdefault("split_lm_head", True)
        return LiquidVLModelExporter(onnx_source_dir=source_dir, **kwargs)

    def test_batch_prefill_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._create_exporter(self.tmp, batch_prefill=0)

    def test_batch_prefill_cannot_exceed_max_gen_tokens(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._create_exporter(self.tmp, batch_prefill=9, max_gen_tokens=8)

    def test_batch_prefill_requires_split_lm_head(self):
        with self.assertRaisesRegex(ValueError, "requires `--split-lm-head`"):
            self._create_exporter(self.tmp, batch_prefill=8, split_lm_head=False)

    def test_batch_prefill_is_retained_when_valid(self):
        exporter = self._create_exporter(self.tmp, batch_prefill=8, max_gen_tokens=16)

        self.assertEqual(exporter._batch_prefill, 8)


class LiquidSplitLMHeadTests(unittest.TestCase):
    HIDDEN = 8
    VOCAB = 16

    @classmethod
    def setUpClass(cls):
        cls._model = _build_tiny_liquid_decoder(cls.HIDDEN, cls.VOCAB)

    def _write_model(self, directory: Path, filename: str = "model.onnx") -> Path:
        import onnx

        onnx.save(self._model, str(directory / filename))
        return directory / filename

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
            # The split rewrites transformer.onnx in place (the fused decoder is
            # not kept), so keep a reference copy for the recomposition check.
            ref_path = td / "fused_ref.onnx"
            onnx.save(self._model, str(ref_path))
            model_path = self._write_model(td, "transformer.onnx")
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

            # transformer.onnx is now the body: lm_head node dropped, hidden state
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

    def test_make_lm_head_split_write_lm_head_false(self):
        import onnx

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_path = self._write_model(td, "transformer_prefill.onnx")
            exporter = self._make_exporter(model_path)
            exporter.make_lm_head_split(model_path, write_lm_head=False)

            # The body is split, but no lm_head.onnx is written or registered
            # (the prefill body reuses the decode model's head).
            self.assertFalse((td / "lm_head.onnx").exists())
            self.assertNotIn("lm_head", exporter._export_paths)
            body = onnx.load(str(model_path))
            self.assertEqual(
                [o.name for o in body.graph.output], ["last_hidden_states", "kv_out"]
            )
            self.assertNotIn("/model/lm_head/MatMul", {n.name for n in body.graph.node})

    def test_make_lm_head_split_keeps_source_last_token_slice_in_body(self):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_path = td / "transformer_prefill.onnx"
            model = _build_tiny_liquid_decoder()
            model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 64
            model.graph.node.insert(1, helper.make_node(
                "Slice", ["normed", "starts", "ends", "axes"], ["last_normed"],
                name="/lm_head/num_logits_to_keep/Slice",
            ))
            model.graph.node[2].input[0] = "last_normed"
            model.graph.initializer.extend([
                numpy_helper.from_array(np.array([-1], dtype=np.int64), "starts"),
                numpy_helper.from_array(np.array([64], dtype=np.int64), "ends"),
                numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            ])
            del model.graph.value_info[:]
            model = onnx.shape_inference.infer_shapes(model)
            onnx.save(model, str(model_path))

            exporter = self._make_exporter(model_path)
            exporter.make_lm_head_split(model_path)

            body = onnx.load(str(model_path))
            head = onnx.load(str(td / "lm_head.onnx"))
            onnx.checker.check_model(body, full_check=True)
            onnx.checker.check_model(head, full_check=True)
            self.assertIn("/lm_head/num_logits_to_keep/Slice", {n.name for n in body.graph.node})
            self.assertEqual([n.op_type for n in head.graph.node], ["MatMul"])
            self.assertEqual(
                [d.dim_value for d in body.graph.output[0].type.tensor_type.shape.dim],
                [1, 1, self.HIDDEN],
            )

    def test_apply_post_static_patches_splits_decode_and_prefill_bodies(self):
        import logging
        import onnx
        from onnx import TensorProto

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source_dir = _write_config(td)
            (source_dir / "tokenizer.json").write_text("tokenizer")
            model_path = td / "transformer.onnx"
            prefill_path = td / "transformer_prefill.onnx"
            onnx.save(_build_tiny_liquid_decoder(), str(model_path))
            onnx.save(_build_tiny_liquid_decoder(), str(prefill_path))

            exporter = LiquidModelExporter.__new__(LiquidModelExporter)
            exporter._logger = logging.getLogger("liquid-apply-patches-test")
            exporter._hidden_size = self.HIDDEN
            exporter._vocab_size = self.VOCAB
            exporter._onnx_export_dtype = TensorProto.FLOAT
            exporter._split_weights = False
            exporter._split_lm_head = True
            exporter._simulate_bf16 = False
            exporter._onnx_dir = source_dir
            exporter._export_paths = {"model": model_path, "model_prefill": prefill_path}
            exporter._patch_static_model = Mock()

            exporter.apply_post_static_patches(model_path, "model")
            exporter.apply_post_static_patches(prefill_path, "model_prefill")

            # Both bodies are split; a single lm_head.onnx (derived from the
            # decode model) is shared.
            for body_path in (model_path, prefill_path):
                body = onnx.load(str(body_path))
                self.assertEqual(body.graph.output[0].name, "last_hidden_states")
                self.assertNotIn(
                    "/model/lm_head/MatMul", {n.name for n in body.graph.node}
                )
            self.assertEqual(exporter._export_paths["lm_head"], td / "lm_head.onnx")


class LiquidVLSplitLMHeadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _vl_source_dir(self) -> Path:
        source_dir = _write_config(self.tmp)
        (source_dir / "decoder_model_merged.onnx").write_bytes(b"dummy")
        return source_dir

    def _vl_exporter(self, **attrs) -> LiquidVLModelExporter:
        import logging
        from onnx import TensorProto

        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        defaults = dict(
            _logger=logging.getLogger("vl-split-test"),
            _hidden_size=8,
            _vocab_size=16,
            _onnx_export_dtype=TensorProto.FLOAT,
            _split_weights=False,
            _split_lm_head=True,
            _simulate_bf16=False,
        )
        defaults.update(attrs)
        for name, value in defaults.items():
            setattr(exporter, name, value)
        return exporter

    def test_split_lm_head_requires_static_models(self):
        source_dir = self._vl_source_dir()
        with self.assertRaisesRegex(ValueError, "static LFM exports"):
            LiquidVLModelExporter(
                onnx_source_dir=source_dir, static_models=False, split_lm_head=True
            )

    def test_setup_dirs_keeps_topologies_separate(self):
        source_dir = self._vl_source_dir()
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        exporter._models_dir = self.tmp
        exporter._onnx_source_dir = source_dir
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

    def test_apply_post_static_patches_splits_decoder(self):
        import logging
        import onnx
        from onnx import TensorProto

        export_dir = self.tmp / "export" / "split_lm_head" / "onnx" / "fp32" / "static"
        export_dir.mkdir(parents=True)
        model_path = export_dir / "transformer.onnx"
        onnx.save(_build_tiny_liquid_decoder(), str(model_path))

        exporter = self._vl_exporter(
            _export_paths={DECODER: model_path},
            _patch_static_model=Mock(),
            _reorder_decoder_inputs=Mock(),
            _stage_runtime_assets=Mock(),
        )

        exporter.apply_post_static_patches(model_path, DECODER)

        body = onnx.load(str(model_path))
        body_outputs = [o.name for o in body.graph.output]
        self.assertEqual(body_outputs[0], "last_hidden_states")
        self.assertNotIn("/model/lm_head/MatMul", {n.name for n in body.graph.node})

        self.assertIn("lm_head", exporter._export_paths)
        self.assertEqual(exporter._export_paths["lm_head"], export_dir / "lm_head.onnx")
        lm_head = onnx.load(str(exporter._export_paths["lm_head"]))
        self.assertEqual([i.name for i in lm_head.graph.input], ["last_hidden_states"])
        self.assertEqual([o.name for o in lm_head.graph.output], ["logits"])

        # The prefill component is split too (gemma3-style); it reuses the
        # lm_head.onnx derived from the decode model instead of re-writing it.
        prefill_path = export_dir / "transformer_prefill.onnx"
        onnx.save(_build_tiny_liquid_decoder(), str(prefill_path))
        lm_head_before = (export_dir / "lm_head.onnx").read_bytes()
        exporter._export_paths = {
            DECODER: model_path,
            DECODER_PREFILL: prefill_path,
        }
        exporter._patch_static_model = Mock()
        exporter._reorder_decoder_inputs = Mock()
        exporter._stage_runtime_assets = Mock()
        exporter.apply_post_static_patches(prefill_path, DECODER_PREFILL)
        prefill = onnx.load(str(prefill_path))
        self.assertEqual(prefill.graph.output[0].name, "last_hidden_states")
        self.assertNotIn("/model/lm_head/MatMul", {n.name for n in prefill.graph.node})
        self.assertEqual((export_dir / "lm_head.onnx").read_bytes(), lm_head_before)

    def test_apply_post_static_patches_ignores_vision(self):
        model_path = self.tmp / "vision_encoder.onnx"
        exporter = self._vl_exporter(_export_paths={})
        exporter._patch_static_model = Mock()
        exporter._reorder_decoder_inputs = Mock()
        exporter._stage_runtime_assets = Mock()

        exporter.apply_post_static_patches(model_path, VISION)

        exporter._patch_static_model.assert_not_called()
        exporter._reorder_decoder_inputs.assert_not_called()
        exporter._stage_runtime_assets.assert_not_called()


class LiquidVLRuntimeAssetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _vl_exporter(self, **attrs) -> LiquidVLModelExporter:
        import logging

        source_dir = _write_config(self.tmp)
        (source_dir / "tokenizer.json").write_text("vl-tokenizer")
        exporter = LiquidVLModelExporter.__new__(LiquidVLModelExporter)
        defaults = dict(
            _logger=logging.getLogger("vl-assets-test"),
            _onnx_dir=source_dir,
            _models_dir=self.tmp,
            _config_dict=_CONFIG,
            _conv_L_cache=_CONFIG.get("conv_L_cache", 3),
            _split_lm_head=False,
            _simulate_bf16=False,
        )
        defaults.update(attrs)
        for name, value in defaults.items():
            setattr(exporter, name, value)
        return exporter

    def test_stage_runtime_assets_writes_flat_config_and_copies_tokenizer(self):
        exporter = self._vl_exporter()
        dst = self.tmp / "export" / "unified" / "onnx" / "fp32" / "static"

        exporter._stage_runtime_assets(dst)

        self.assertEqual(json.loads((dst / "config.json").read_text()), _CONFIG)
        self.assertEqual((dst / "tokenizer.json").read_text(), "vl-tokenizer")
        # Staging is idempotent: an existing config.json is not clobbered.
        (dst / "config.json").write_text("{\"sentinel\": true}")
        exporter._stage_runtime_assets(dst)
        self.assertEqual(json.loads((dst / "config.json").read_text()), {"sentinel": True})

    def test_static_export_stages_runtime_assets(self):
        exporter = self._vl_exporter(_split_lm_head=False)
        export_dir = self.tmp / "export"
        model_path = export_dir / "decoder_model_merged.onnx"
        export_dir.mkdir(parents=True, exist_ok=True)
        exporter._patch_static_model = Mock()
        exporter._reorder_decoder_inputs = Mock()

        exporter.apply_post_static_patches(model_path, DECODER)

        self.assertEqual(json.loads((export_dir / "config.json").read_text()), _CONFIG)
        self.assertEqual((export_dir / "tokenizer.json").read_text(), "vl-tokenizer")

    def test_dynamic_quantization_updates_decoder_path_and_stages_assets(self):
        import onnx

        export_dir = self.tmp / "export" / "unified" / "onnx" / "fp32" / "static"
        export_dir.mkdir(parents=True)
        decoder_path = export_dir / f"{DECODER}.onnx"
        onnx.save(_build_tiny_liquid_decoder(), str(decoder_path))

        exporter = self._vl_exporter(
            _dynamic_quantize=True,
            _prepared=True,
            _export_dir=export_dir,
            _quantize_dir=self.tmp / "export" / "unified" / "onnx" / "quantized" / "static",
            _export_paths={DECODER: decoder_path},
        )
        exporter._stage_runtime_assets(export_dir)

        exporter.dynamic_quantize_models(skip_preprocess=True)

        quantized_path = exporter._quantize_dir / decoder_path.name
        self.assertEqual(exporter._export_paths[DECODER], quantized_path)
        self.assertTrue(quantized_path.exists())
        self.assertEqual(json.loads((exporter._quantize_dir / "config.json").read_text()), _CONFIG)
        self.assertEqual((exporter._quantize_dir / "tokenizer.json").read_text(), "vl-tokenizer")


if __name__ == "__main__":
    unittest.main()
