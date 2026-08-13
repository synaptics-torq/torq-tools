# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import os
from dataclasses import dataclass
from typing import Final

import numpy as np
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None
try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

from ...inference.runners import (
    InferenceRunner,
    ORTInferenceRunner,
    VMFBInferenceRunner,
)
from ...inference.transformers import (
    DecoderOnlyConfig,
    DynamicDecoderOnlyRunner,
    StaticDecoderOnlyRunner,
    find_single_data_file,
)

DEFAULT_SYS_PROMPT: Final[str] = "You are a helpful AI assistant named SmolLM. Provide all answers as concise responses; use as few words as possible and avoid extra explanation."


def _default_repo_id(instruct_model: bool) -> str:
    repo_id = "HuggingFaceTB/SmolLM2-135M"
    if instruct_model:
        repo_id += "-Instruct"
    return repo_id


def _hf_hub_download(repo_id: str, filename: str) -> str:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub python API not available in environment"
        )
    return hf_hub_download(repo_id, filename)


def _load_tokenizer(tokenizer_path: str | os.PathLike):
    if Tokenizer is None:
        raise RuntimeError(
            "tokenizers python API not available in environment"
        )
    return Tokenizer.from_file(str(tokenizer_path))


@dataclass(frozen=True)
class ModelConfig(DecoderOnlyConfig):
    @classmethod
    def from_json_config(cls, json_file: str | os.PathLike, instruct_model: bool = False) -> "ModelConfig":
        with open(json_file) as f:
            config = json.load(f)
        try:
            return cls(
                config["num_hidden_layers"],
                config["num_key_value_heads"],
                config["hidden_size"] // config["num_attention_heads"],
                config["bos_token_id"],
                config["eos_token_id"],
                config.get("pad_token_id"),
                instruct_model,
            )
        except KeyError as e:
            raise ValueError(f"Model config missing required metadata: {e}")


class SmolLM2Base:
    def _init_model_metadata(self):
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[0]

    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]:
        if not self._instruct_model:
            return self._tokenizer.encode(input).ids
        if role == "assistant":
            return self._tokenizer.encode(self._bos_token + role + "\n").ids
        return self._tokenizer.encode(self._bos_token + role + "\n" + input + self._eos_token + "\n").ids

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model:
            # WARNING: relying on "\n\n" is fragile but is the best we have right now
            return len(gen_tokens) > 2 and all(t == self._nl_token_id for t in gen_tokens[-2:])


class SmolLM2Dynamic(SmolLM2Base, DynamicDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
    ):
        repo_id = repo_id or _default_repo_id(instruct_model)
        DynamicDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(
                _hf_hub_download(repo_id, "config.json"),
                instruct_model,
            ),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(_hf_hub_download(repo_id, "tokenizer.json")),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            include_position_ids=True,
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
    ) -> "SmolLM2Dynamic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
    ) -> "SmolLM2Dynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
        )


class SmolLM2Static(SmolLM2Base, StaticDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
    ):
        repo_id = repo_id or _default_repo_id(instruct_model)
        StaticDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(
                _hf_hub_download(repo_id, "config.json"),
                instruct_model,
            ),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(_hf_hub_download(repo_id, "tokenizer.json")),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            combined_kv_io=combined_kv_io,
            token_embeddings=self._find_token_embeddings(model.model_path),
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
    ) -> "SmolLM2Static":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
    ) -> "SmolLM2Static":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
        )

    @staticmethod
    def _find_token_embeddings(
        model_path: str | os.PathLike,
        emb_pattern: str = "token_embeddings.npy",
    ) -> np.ndarray | None:
        path = find_single_data_file(model_path, emb_pattern, "token embedding")
        if path is None:
            return None
        return np.load(path)


if __name__ == "__main__":
    pass
