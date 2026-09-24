# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from . import LABEL_FILES, DEFAULT_MAX_SEQ_LEN


logger = logging.getLogger(__name__)


@dataclass
class WashingBERTResult:
    """Structured prediction result from WashingBERT."""
    intent: str
    intent_confidence: float
    type1_labels: list[str]
    type1_scores: list[float]
    type2_labels: list[str]
    type2_scores: list[float]

    def __repr__(self) -> str:
        parts = [f"Intent: {self.intent} ({self.intent_confidence:.3f})"]
        if self.type1_labels:
            t1 = ", ".join(
                f"{l}({s:.3f})" for l, s in zip(self.type1_labels, self.type1_scores)
            )
            parts.append(f"Type1: [{t1}]")
        if self.type2_labels:
            t2 = ", ".join(
                f"{l}({s:.3f})" for l, s in zip(self.type2_labels, self.type2_scores)
            )
            parts.append(f"Type2: [{t2}]")
        return " | ".join(parts)


@dataclass
class LabelMap:
    """Maps numeric label indices to human-readable names.

    Auto-discovers labels from JSON files co-located with the model:
      - intent_classes.json
      - types_classes.json      (Type1 / cloth type)
      - sec_types_classes.json  (Type2 / secondary attributes)
    """
    intents: list[str] = field(default_factory=list)
    type1: list[str] = field(default_factory=list)
    type2: list[str] = field(default_factory=list)

    @classmethod
    def from_dir(cls, model_dir: str | os.PathLike) -> "LabelMap":
        """Load label maps from JSON files alongside the ONNX model."""
        model_dir = Path(model_dir)
        if model_dir.is_file():
            model_dir = model_dir.parent

        labels: dict[str, list[str]] = {}
        for key, filename in LABEL_FILES.items():
            path = model_dir / filename
            if path.exists():
                with open(path) as f:
                    labels[key] = json.load(f)
                logger.info("Loaded %d %s labels from '%s'", len(labels[key]), key, path)
            else:
                labels[key] = []
                logger.debug("No %s label file found at '%s'", key, path)
        return cls(
            intents=labels.get("intents", []),
            type1=labels.get("type1", []),
            type2=labels.get("type2", []),
        )

    def intent_name(self, idx: int) -> str:
        return self.intents[idx] if idx < len(self.intents) else f"intent_{idx}"

    def type1_name(self, idx: int) -> str:
        return self.type1[idx] if idx < len(self.type1) else f"type1_{idx}"

    def type2_name(self, idx: int) -> str:
        return self.type2[idx] if idx < len(self.type2) else f"type2_{idx}"


class WashingBERTRunner:
    """Run WashingBERT inference via ONNX Runtime or IREE."""

    MULTI_LABEL_THRESHOLD: Final[float] = 0.5

    def __init__(
        self,
        session,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        label_map: LabelMap | None = None,
        threads: int | None = None,
    ):
        self._session = session
        self._max_seq_len = max_seq_len
        self._label_map = label_map or LabelMap()
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]
        self._infer_times: list[float] = []

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        label_map: LabelMap | None = None,
        threads: int | None = None,
    ) -> "WashingBERTRunner":
        import onnxruntime as ort

        model_path = Path(model_path)
        if label_map is None:
            label_map = LabelMap.from_dir(model_path)

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = threads
        session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        logger.info("Loaded ONNX model from '%s'", model_path)
        return cls(session, max_seq_len=max_seq_len, label_map=label_map, threads=threads)

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return sum(self._infer_times) / len(self._infer_times) if self._infer_times else 0.0

    def _pad_inputs(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad or truncate inputs to max_seq_len."""
        seq_len = input_ids.shape[-1]
        if seq_len > self._max_seq_len:
            input_ids = input_ids[:, : self._max_seq_len]
            attention_mask = attention_mask[:, : self._max_seq_len]
        elif seq_len < self._max_seq_len:
            pad_len = self._max_seq_len - seq_len
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)
        return input_ids, attention_mask

    def run_raw(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Run inference and return raw output arrays."""
        input_ids, attention_mask = self._pad_inputs(input_ids, attention_mask)

        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids.astype(np.int64)
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask.astype(np.int64)
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, feed)
        self._infer_times.append(time.perf_counter() - t0)
        return outputs

    def run(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> WashingBERTResult:
        """Run inference and return structured result with label names."""
        outputs = self.run_raw(input_ids, attention_mask)

        intent_logits = outputs[0]
        intent_idx = int(np.argmax(intent_logits, axis=-1).item())
        intent_probs = _softmax(intent_logits[0])

        type1_labels: list[str] = []
        type1_scores: list[float] = []
        if len(outputs) > 1:
            type1_probs = _sigmoid(outputs[1][0])
            for idx in np.where(type1_probs > self.MULTI_LABEL_THRESHOLD)[0]:
                type1_labels.append(self._label_map.type1_name(int(idx)))
                type1_scores.append(float(type1_probs[idx]))

        type2_labels: list[str] = []
        type2_scores: list[float] = []
        if len(outputs) > 2:
            type2_probs = _sigmoid(outputs[2][0])
            for idx in np.where(type2_probs > self.MULTI_LABEL_THRESHOLD)[0]:
                type2_labels.append(self._label_map.type2_name(int(idx)))
                type2_scores.append(float(type2_probs[idx]))

        return WashingBERTResult(
            intent=self._label_map.intent_name(intent_idx),
            intent_confidence=float(intent_probs[intent_idx]),
            type1_labels=type1_labels,
            type1_scores=type1_scores,
            type2_labels=type2_labels,
            type2_scores=type2_scores,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
