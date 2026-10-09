# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""ONNX Runtime reference for EmbeddingGemma-2 (text + image + video).

Runs the *unmodified* dynamic onnx-community fp32 graphs (``onnx/model.onnx`` and
``onnx/vision_encoder.onnx``) exactly as transformers.js does: the vision encoder
consumes padded patches + positions, and the text model merges the soft tokens at the
``<|image|>`` / ``<|video|>`` placeholders and returns the pooled, normalised embedding.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import preprocess as pp


class OrtEmbeddingGemma2:
    def __init__(self, source_dir: str | Path, threads: int = 0, vision: bool = True, audio: bool = False,
                 variant: str = ""):
        """``variant`` selects an onnx-community weight variant by file suffix, e.g. "q4"."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        source_dir = Path(source_dir)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        sfx = f"_{variant}" if variant else ""

        def session(name):
            return ort.InferenceSession(str(source_dir / "onnx" / f"{name}{sfx}.onnx"), so, providers=prov)

        self.text = session("model")
        self.vision = session("vision_encoder") if vision else None
        self.audio = session("audio_encoder") if audio else None
        self.tokenizer = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
        self._empty = np.zeros((0, 512), dtype=np.float32)

    def vision_features(self, v: pp.VisionInput, pad_to: int | None = None) -> np.ndarray:
        """Soft tokens [n, 512] for one image/frame. ``pad_to`` reproduces the HF padding
        of patches to ``max_soft_tokens * 9`` (positions -1)."""
        patches, positions = v.patches, v.positions
        if pad_to and pad_to > patches.shape[0]:
            extra = pad_to - patches.shape[0]
            patches = np.concatenate([patches, np.zeros((extra, patches.shape[1]), np.float32)])
            positions = np.concatenate([positions, -np.ones((extra, 2), np.int64)])
        out = self.vision.run(None, {"pixel_values": patches[None].astype(np.float32),
                                     "pixel_position_ids": positions[None].astype(np.int64)})[0]
        return out.astype(np.float32)

    def audio_tokens(self, mel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Audio soft tokens [n, 512] from log-mel features [T, 128] and frame mask [T]."""
        return self.audio.run(None, {"input_features": mel[None].astype(np.float32),
                                     "input_features_mask": mask[None].astype(bool)})[0].astype(np.float32)

    def run_text_model(self, ids: list[int], image_features=None, video_features=None, audio_features=None):
        ids_arr = np.asarray(ids, dtype=np.int64)[None]
        feeds = {
            "input_ids": ids_arr,
            "attention_mask": np.ones_like(ids_arr),
            "image_features": self._empty if image_features is None else image_features,
            "video_features": self._empty if video_features is None else video_features,
            "audio_features": self._empty if audio_features is None else audio_features,
        }
        last_hidden_state, sentence_embedding = self.text.run(None, feeds)
        return last_hidden_state, sentence_embedding[0]

    def encode_text(self, text: str, prompt_name: str | None = None) -> np.ndarray:
        return self.run_text_model(pp.text_ids(self.tokenizer, text, prompt_name))[1]

    def encode_image(self, image: Image.Image, mode: str = "square", max_soft_tokens: int = 70,
                     hf_padding: bool = True) -> np.ndarray:
        v = pp.preprocess_image(image, mode, max_soft_tokens)
        feats = self.vision_features(v, max_soft_tokens * pp.POOL_KERNEL**2 if hf_padding else None)
        return self.run_text_model(pp.image_ids(feats.shape[0]), image_features=feats)[1]

    def encode_frames(self, frames: list[Image.Image], mode: str = "square", max_soft_tokens: int = 70,
                      hf_padding: bool = True) -> np.ndarray:
        feats = []
        for f in frames:
            v = pp.preprocess_image(f, mode, max_soft_tokens)
            feats.append(self.vision_features(v, max_soft_tokens * pp.POOL_KERNEL**2 if hf_padding else None))
        n = feats[0].shape[0]
        ids = pp.video_ids(len(feats), n)
        return self.run_text_model(ids, video_features=np.concatenate(feats))[1]
