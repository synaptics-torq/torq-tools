# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from __future__ import annotations

import copy
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tokenizers import Tokenizer

COMMON_ES_EN_PUNCT = {
    ".",
    ",",
    "!",
    "?",
    ":",
    ";",
    "'",
    '"',
    "-",
    "—",
    "–",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "¡",
    "¿",
    "…",
    "«",
    "»",
    "‹",
    "›",
    "`",
    "´",
    "/",
    "\\",
    "|",
    "@",
    "#",
    "&",
    "*",
    "%",
    "_",
}
BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
TRIM_GROUP_CHOICES = ("latin", "punct", "other")


@dataclass(frozen=True)
class TrimmedVocabSpec:
    selected_groups: tuple[str, ...]
    byte_fallback: bool
    model_vocab_size: int
    kept_model_ids: tuple[int, ...]
    extra_token_ids: tuple[int, ...]
    byte_token_ids: tuple[int, ...]
    old_to_new: dict[int, int]
    new_to_old: tuple[int, ...]

    @property
    def trimmed_vocab_size(self) -> int:
        return len(self.new_to_old)

    @property
    def kept_token_ids(self) -> tuple[int, ...]:
        return self.new_to_old

    @property
    def kept_model_id_set(self) -> set[int]:
        return set(self.kept_model_ids)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _is_latin_char(ch: str) -> bool:
    if not ch:
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "LATIN" in name


def _is_common_punctuation(ch: str) -> bool:
    if ch in COMMON_ES_EN_PUNCT:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_punctuation_only_text(text: str) -> bool:
    if not text:
        return False
    return all(ch.isspace() or _is_common_punctuation(ch) for ch in text)


def build_token_groups(tokenizer: Tokenizer, vocab_size: int) -> dict[str, list[int]]:
    groups = {"latin": [], "punct": [], "other": [], "byte": []}
    for token_id in range(vocab_size):
        raw_token = tokenizer.id_to_token(token_id) or ""
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if BYTE_TOKEN_RE.match(raw_token):
            groups["byte"].append(token_id)
        if any(_is_latin_char(ch) for ch in decoded):
            groups["latin"].append(token_id)
        elif _is_punctuation_only_text(decoded):
            groups["punct"].append(token_id)
        else:
            groups["other"].append(token_id)
    return groups


def _collect_required_token_ids(config_json: dict[str, Any], tokenizer_json: dict[str, Any]) -> set[int]:
    required_ids: set[int] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and key.endswith(("_token_id", "_token_index"))
                ):
                    required_ids.add(value)
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(config_json)
    for token_info in tokenizer_json.get("post_processor", {}).get("special_tokens", {}).values():
        for token_id in token_info.get("ids", []):
            if isinstance(token_id, int) and not isinstance(token_id, bool):
                required_ids.add(token_id)
    return required_ids


def build_trimmed_vocab_spec(
    tokenizer: Tokenizer,
    tokenizer_json: dict[str, Any],
    config_json: dict[str, Any],
    selected_groups: Iterable[str],
    *,
    byte_fallback: bool,
) -> TrimmedVocabSpec:
    selected_groups = tuple(dict.fromkeys(selected_groups))
    invalid_groups = sorted(set(selected_groups) - set(TRIM_GROUP_CHOICES))
    if invalid_groups:
        raise ValueError(
            f"Invalid trim-vocab groups {invalid_groups}; expected subset of {TRIM_GROUP_CHOICES}"
        )

    model_vocab = tokenizer_json["model"]["vocab"]
    model_vocab_size = len(model_vocab)
    base_groups = build_token_groups(tokenizer, model_vocab_size)

    kept_model_ids: set[int] = set()
    for group in selected_groups:
        kept_model_ids.update(base_groups[group])
    if byte_fallback:
        kept_model_ids.update(base_groups["byte"])

    added_token_ids = {
        int(entry["id"])
        for entry in tokenizer_json.get("added_tokens", [])
        if isinstance(entry.get("id"), int) and not isinstance(entry.get("id"), bool)
    }
    kept_special_ids = added_token_ids | _collect_required_token_ids(config_json, tokenizer_json)
    kept_model_ids.update(token_id for token_id in kept_special_ids if 0 <= token_id < model_vocab_size)
    kept_model_ids_sorted = tuple(sorted(kept_model_ids))
    extra_token_ids = tuple(sorted(token_id for token_id in kept_special_ids if token_id >= model_vocab_size))

    old_to_new: dict[int, int] = {
        token_id: new_id for new_id, token_id in enumerate((*kept_model_ids_sorted, *extra_token_ids))
    }

    return TrimmedVocabSpec(
        selected_groups=selected_groups,
        byte_fallback=byte_fallback,
        model_vocab_size=model_vocab_size,
        kept_model_ids=kept_model_ids_sorted,
        extra_token_ids=extra_token_ids,
        byte_token_ids=tuple(base_groups["byte"]),
        old_to_new=old_to_new,
        new_to_old=tuple(old_to_new.keys()),
    )


def rewrite_tokenizer_json(
    tokenizer_json: dict[str, Any],
    spec: TrimmedVocabSpec,
) -> dict[str, Any]:
    old_to_new = spec.old_to_new
    kept_model_id_set = spec.kept_model_id_set
    trimmed = {
        key: copy.deepcopy(value)
        for key, value in tokenizer_json.items()
        if key not in {"model", "added_tokens", "post_processor"}
    }
    old_model = tokenizer_json["model"]
    old_vocab = old_model["vocab"]
    new_vocab = {
        token: old_to_new[token_id]
        for token, token_id in old_vocab.items()
        if token_id in kept_model_id_set
    }
    model = {
        key: copy.deepcopy(value)
        for key, value in old_model.items()
        if key not in {"vocab", "merges"}
    }
    model["vocab"] = new_vocab

    kept_tokens = set(new_vocab)
    filtered_merges = []
    for merge in old_model.get("merges", []):
        if isinstance(merge, str):
            parts = merge.split()
        else:
            parts = list(merge)
        if len(parts) != 2:
            continue
        left, right = parts
        if left in kept_tokens and right in kept_tokens and f"{left}{right}" in kept_tokens:
            filtered_merges.append(merge)
    model["merges"] = filtered_merges
    trimmed["model"] = model

    remapped_added_tokens = []
    for entry in tokenizer_json.get("added_tokens", []):
        token_id = entry.get("id")
        if token_id not in old_to_new:
            continue
        new_entry = copy.deepcopy(entry)
        new_entry["id"] = old_to_new[token_id]
        remapped_added_tokens.append(new_entry)
    trimmed["added_tokens"] = sorted(remapped_added_tokens, key=lambda entry: entry["id"])

    post_processor = copy.deepcopy(tokenizer_json.get("post_processor"))
    if isinstance(post_processor, dict):
        for token_info in post_processor.get("special_tokens", {}).values():
            ids = token_info.get("ids")
            if isinstance(ids, list):
                token_info["ids"] = [old_to_new[token_id] for token_id in ids]
    trimmed["post_processor"] = post_processor

    return trimmed


def rewrite_config_json(config_json: dict[str, Any], spec: TrimmedVocabSpec) -> dict[str, Any]:
    trimmed = copy.deepcopy(config_json)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            remapped = {}
            for key, value in obj.items():
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and key.endswith(("_token_id", "_token_index"))
                    and value in spec.old_to_new
                ):
                    remapped[key] = spec.old_to_new[value]
                else:
                    remapped[key] = _walk(value)
            return remapped
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    trimmed = _walk(trimmed)
    trimmed["vocab_size"] = spec.trimmed_vocab_size
    trimmed["torq_trimmed_vocab"] = {
        "enabled": True,
        "selected_groups": list(spec.selected_groups),
        "byte_fallback": spec.byte_fallback,
        "original_vocab_size": spec.model_vocab_size,
        "trimmed_vocab_size": spec.trimmed_vocab_size,
    }
    return trimmed


def trim_embedding_rows(embeddings: np.ndarray, spec: TrimmedVocabSpec) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] != spec.model_vocab_size:
        raise ValueError(
            f"Expected embedding table shape ({spec.model_vocab_size}, hidden), got {embeddings.shape}"
        )
    trimmed = np.take(embeddings, spec.kept_model_ids, axis=0)
    if spec.extra_token_ids:
        extra = np.zeros((len(spec.extra_token_ids), embeddings.shape[1]), dtype=embeddings.dtype)
        trimmed = np.concatenate([trimmed, extra], axis=0)
    return trimmed


def trim_logits_projection(weight: np.ndarray, spec: TrimmedVocabSpec) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D logits projection weight, got {weight.shape}")
    if weight.shape[0] == spec.model_vocab_size:
        trimmed = np.take(weight, spec.kept_model_ids, axis=0)
        if spec.extra_token_ids:
            extra = np.zeros((len(spec.extra_token_ids), weight.shape[1]), dtype=weight.dtype)
            trimmed = np.concatenate([trimmed, extra], axis=0)
        return trimmed
    if weight.shape[1] == spec.model_vocab_size:
        trimmed = np.take(weight, spec.kept_model_ids, axis=1)
        if spec.extra_token_ids:
            extra = np.zeros((weight.shape[0], len(spec.extra_token_ids)), dtype=weight.dtype)
            trimmed = np.concatenate([trimmed, extra], axis=1)
        return trimmed
    raise ValueError(
        f"Expected logits projection with vocab axis {spec.model_vocab_size}, got {weight.shape}"
    )
