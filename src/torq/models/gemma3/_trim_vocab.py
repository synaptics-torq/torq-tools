# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from __future__ import annotations

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

    @property
    def trimmed_vocab_size(self) -> int:
        return len(self.kept_model_ids) + len(self.extra_token_ids)

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

    if config_json is not None:
        _walk(config_json)
    for token_info in tokenizer_json.get("post_processor", {}).get("special_tokens", {}).values():
        for token_id in token_info.get("ids", []):
            if isinstance(token_id, int) and not isinstance(token_id, bool):
                required_ids.add(token_id)
    return required_ids


def build_trimmed_vocab_spec(
    tokenizer: Tokenizer,
    tokenizer_json: dict[str, Any],
    config_json: dict[str, Any] | None,
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

    return TrimmedVocabSpec(
        selected_groups=selected_groups,
        byte_fallback=byte_fallback,
        model_vocab_size=model_vocab_size,
        kept_model_ids=kept_model_ids_sorted,
        extra_token_ids=extra_token_ids,
        byte_token_ids=tuple(base_groups["byte"]),
    )
