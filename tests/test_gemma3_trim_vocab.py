import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

from torq.models.gemma3.export import Gemma3ModelExporter
from torq.models.gemma3._trim_vocab import (
    build_trimmed_vocab_spec,
    load_json,
    rewrite_config_json,
    rewrite_tokenizer_json,
    trim_embedding_rows,
    trim_logits_projection,
)


class Gemma3TrimVocabTests(unittest.TestCase):
    def _build_test_assets(self):
        temp_dir = tempfile.TemporaryDirectory()
        tokenizer_path = Path(temp_dir.name) / "tokenizer.json"

        vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
            "<unk>": 3,
            "Hello": 4,
            "¿": 5,
            "?": 6,
            "こんにちは": 7,
            "<0x41>": 8,
        }
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A",
            pair="<bos> $A <bos> $B",
            special_tokens=[("<bos>", 2)],
        )
        tokenizer.add_special_tokens(["<image_soft_token>"])
        tokenizer.save(str(tokenizer_path))

        tokenizer_json = load_json(tokenizer_path)
        extra_token_id = next(
            entry["id"]
            for entry in tokenizer_json["added_tokens"]
            if entry["content"] == "<image_soft_token>"
        )
        config_json = {
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "image_token_index": extra_token_id,
            "vocab_size": len(vocab),
        }
        return temp_dir, tokenizer_path, tokenizer_json, config_json, extra_token_id

    def test_build_trimmed_vocab_spec_keeps_groups_specials_and_extra(self):
        temp_dir, tokenizer_path, tokenizer_json, config_json, extra_token_id = self._build_test_assets()
        self.addCleanup(temp_dir.cleanup)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        spec = build_trimmed_vocab_spec(
            tokenizer=tokenizer,
            tokenizer_json=tokenizer_json,
            config_json=config_json,
            selected_groups=["latin", "punct"],
            byte_fallback=True,
        )

        self.assertEqual(spec.model_vocab_size, 9)
        self.assertEqual(spec.extra_token_ids, (extra_token_id,))
        self.assertIn(4, spec.kept_model_ids)
        self.assertIn(5, spec.kept_model_ids)
        self.assertIn(6, spec.kept_model_ids)
        self.assertIn(8, spec.kept_model_ids)
        self.assertNotIn(7, spec.kept_model_ids)
        self.assertEqual(spec.trimmed_vocab_size, len(spec.kept_model_ids) + 1)

    def test_rewrite_tokenizer_and_config_produces_loadable_dense_bundle(self):
        temp_dir, tokenizer_path, tokenizer_json, config_json, _ = self._build_test_assets()
        self.addCleanup(temp_dir.cleanup)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        spec = build_trimmed_vocab_spec(
            tokenizer=tokenizer,
            tokenizer_json=tokenizer_json,
            config_json=config_json,
            selected_groups=["latin", "punct"],
            byte_fallback=True,
        )

        trimmed_tokenizer_json = rewrite_tokenizer_json(tokenizer_json, spec)
        trimmed_config_json = rewrite_config_json(config_json, spec)
        dense_model_ids = sorted(trimmed_tokenizer_json["model"]["vocab"].values())

        self.assertEqual(dense_model_ids, list(range(len(dense_model_ids))))
        self.assertEqual(trimmed_config_json["vocab_size"], spec.trimmed_vocab_size)
        self.assertEqual(
            trimmed_tokenizer_json["post_processor"]["special_tokens"]["<bos>"]["ids"],
            [trimmed_config_json["bos_token_id"]],
        )

        trimmed_path = Path(temp_dir.name) / "trimmed_tokenizer.json"
        with open(trimmed_path, "w") as f:
            import json

            json.dump(trimmed_tokenizer_json, f)
        trimmed_tokenizer = Tokenizer.from_file(str(trimmed_path))
        encoded = trimmed_tokenizer.encode("こんにちは")

        self.assertTrue(encoded.ids)
        self.assertTrue(all(token_id < trimmed_config_json["vocab_size"] for token_id in encoded.ids))

    def test_trim_embedding_rows_and_logits_projection(self):
        temp_dir, tokenizer_path, tokenizer_json, config_json, _ = self._build_test_assets()
        self.addCleanup(temp_dir.cleanup)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        spec = build_trimmed_vocab_spec(
            tokenizer=tokenizer,
            tokenizer_json=tokenizer_json,
            config_json=config_json,
            selected_groups=["latin", "punct"],
            byte_fallback=True,
        )

        embeddings = np.arange(spec.model_vocab_size * 3, dtype=np.float32).reshape(spec.model_vocab_size, 3)
        logits = np.arange(2 * spec.model_vocab_size, dtype=np.float32).reshape(2, spec.model_vocab_size)

        trimmed_embeddings = trim_embedding_rows(embeddings, spec)
        trimmed_logits = trim_logits_projection(logits, spec)

        self.assertEqual(trimmed_embeddings.shape, (spec.trimmed_vocab_size, 3))
        self.assertEqual(trimmed_logits.shape, (2, spec.trimmed_vocab_size))
        self.assertTrue(np.array_equal(trimmed_embeddings[-2], embeddings[8]))
        self.assertTrue(np.array_equal(trimmed_embeddings[-1], np.zeros(3, dtype=np.float32)))
        self.assertTrue(np.array_equal(trimmed_logits[:, -2], logits[:, 8]))
        self.assertTrue(np.array_equal(trimmed_logits[:, -1], np.zeros(2, dtype=np.float32)))

    def test_update_logits_metadata_rewrites_output_shape(self):
        logits_info = onnx.helper.make_tensor_value_info(
            "logits",
            onnx.TensorProto.FLOAT,
            [1, 1, 262144],
        )
        aux_info = onnx.helper.make_tensor_value_info(
            "hidden",
            onnx.TensorProto.FLOAT,
            [1, 1, 16],
        )
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=[],
                name="trim-test",
                inputs=[],
                outputs=[logits_info],
                initializer=[],
                value_info=[onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 1, 262144]), aux_info],
            )
        )

        Gemma3ModelExporter._update_logits_metadata(model, 123)

        output_dims = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        value_info_dims = [
            dim.dim_value
            for value in model.graph.value_info
            if value.name == "logits"
            for dim in value.type.tensor_type.shape.dim
        ]
        self.assertEqual(output_dims, [1, 1, 123])
        self.assertEqual(value_info_dims, [1, 1, 123])


if __name__ == "__main__":
    unittest.main()
