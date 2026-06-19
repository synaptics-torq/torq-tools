import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

from torq.models.gemma3._trim_vocab import (
    build_trimmed_vocab_spec,
    load_json,
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
            "¢": 9,
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

        self.assertEqual(spec.model_vocab_size, 10)
        self.assertEqual(spec.extra_token_ids, (extra_token_id,))
        # Latin token
        self.assertIn(4, spec.kept_model_ids)
        # Punct tokens
        self.assertIn(5, spec.kept_model_ids)
        self.assertIn(6, spec.kept_model_ids)
        self.assertIn(9, spec.kept_model_ids)
        # Byte fallback token
        self.assertIn(8, spec.kept_model_ids)
        # Non-latin, non-punct token excluded
        self.assertNotIn(7, spec.kept_model_ids)
        # trimmed_vocab_size = kept_model_ids + extra_token_ids
        self.assertEqual(spec.trimmed_vocab_size, len(spec.kept_model_ids) + len(spec.extra_token_ids))

    def test_kept_model_ids_preserves_original_ids(self):
        """Verify that kept_model_ids are original (not remapped) token IDs."""
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

        # All kept IDs should be valid original vocab indices
        for token_id in spec.kept_model_ids:
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, spec.model_vocab_size)

        # IDs should be sorted (for deterministic weight slicing)
        self.assertEqual(spec.kept_model_ids, tuple(sorted(spec.kept_model_ids)))

    def test_spec_with_no_config_still_keeps_specials(self):
        """build_trimmed_vocab_spec works without config_json (keeps added_tokens)."""
        temp_dir, tokenizer_path, tokenizer_json, _, _ = self._build_test_assets()
        self.addCleanup(temp_dir.cleanup)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        spec = build_trimmed_vocab_spec(
            tokenizer=tokenizer,
            tokenizer_json=tokenizer_json,
            config_json=None,
            selected_groups=["latin"],
            byte_fallback=False,
        )

        # Special tokens from added_tokens should still be kept
        self.assertIn(0, spec.kept_model_ids)  # <pad>
        self.assertIn(1, spec.kept_model_ids)  # <eos>
        self.assertIn(2, spec.kept_model_ids)  # <bos>
        self.assertIn(3, spec.kept_model_ids)  # <unk>


if __name__ == "__main__":
    unittest.main()
