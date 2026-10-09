# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

"""Liquid VL exporter behavior, under tests/unit so CI collects it (the
repo-root tests/test_liquid_export_filenames.py is outside CI's testpaths).
"""

import json
import tempfile
import unittest
from pathlib import Path

from torq.models.liquid.export_vl import DECODER, LiquidVLModelExporter, VISION


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


class LiquidVLExporterTests(unittest.TestCase):
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
            _logger=logging.getLogger("vl-exporter-test"),
            _onnx_dir=source_dir,
            _models_dir=self.tmp,
            _config_dict=_CONFIG,
            _conv_L_cache=_CONFIG.get("conv_L_cache", 3),
            _split_lm_head=False,
            _simulate_bf16=False,
            _compile_vision=False,
            _vision_res=None,
        )
        defaults.update(attrs)
        for name, value in defaults.items():
            setattr(exporter, name, value)
        return exporter

    def test_dynamic_quantization_skips_vision_by_default(self):
        """E-6: the vision encoder is a CPU/ORT component; without
        --compile-vision / --vision-res it is never compiled, and ORT's
        quantizer pre-processing crashes on its dynamic shapes — so the
        exporter must skip it from dynamic quantization by default."""
        import onnx

        export_dir = self.tmp / "export" / "unified" / "onnx" / "fp32" / "static"
        export_dir.mkdir(parents=True)
        decoder_path = export_dir / f"{DECODER}.onnx"
        vision_path = export_dir / f"{VISION}.onnx"
        onnx.save(_build_tiny_liquid_decoder(), str(decoder_path))
        onnx.save(_build_tiny_liquid_decoder(), str(vision_path))

        quantize_dir = self.tmp / "export" / "unified" / "onnx" / "quantized" / "static"
        exporter = self._vl_exporter(
            _dynamic_quantize=True,
            _prepared=True,
            _export_dir=export_dir,
            _quantize_dir=quantize_dir,
            _export_paths={DECODER: decoder_path, VISION: vision_path},
            _compile_vision=False,
            _vision_res=None,
        )

        exporter.dynamic_quantize_models(skip_preprocess=True)

        # The decoder is quantized; the vision encoder is copied through
        # unquantized (the skip path keeps the original bytes).
        self.assertEqual(exporter._export_paths[DECODER], quantize_dir / decoder_path.name)
        self.assertTrue((quantize_dir / decoder_path.name).exists())
        self.assertEqual(
            (quantize_dir / vision_path.name).read_bytes(),
            vision_path.read_bytes(),
        )

    def test_dynamic_quantization_keeps_vision_when_compiled(self):
        """With --compile-vision (or --vision-res) the vision encoder is a
        chip component, so it must not be auto-skipped."""
        import onnx

        export_dir = self.tmp / "export" / "unified" / "onnx" / "fp32" / "static"
        export_dir.mkdir(parents=True)
        vision_path = export_dir / f"{VISION}.onnx"
        onnx.save(_build_tiny_liquid_decoder(), str(vision_path))

        quantize_dir = self.tmp / "export" / "unified" / "onnx" / "quantized" / "static"
        for attrs in (
            dict(_compile_vision=True, _vision_res=None),
            dict(_compile_vision=False, _vision_res=256),
        ):
            with self.subTest(**attrs):
                exporter = self._vl_exporter(
                    _dynamic_quantize=True,
                    _prepared=True,
                    _export_dir=export_dir,
                    _quantize_dir=quantize_dir,
                    _export_paths={VISION: vision_path},
                    **attrs,
                )
                # If the auto-skip still applied, the quantize step would
                # copy the original bytes through instead of quantizing.
                exporter.dynamic_quantize_models(skip_preprocess=True)
                self.assertNotEqual(
                    (quantize_dir / vision_path.name).read_bytes(),
                    vision_path.read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
