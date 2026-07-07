# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import os
from dataclasses import dataclass
from pathlib import Path
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

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant named Gemma. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


def _default_repo_id(instruct_model: bool) -> str:
    repo_id = "google/gemma-3-270m"
    if instruct_model:
        repo_id += "-it"
    return repo_id


def _resolve_asset_path(
    model_path: str | os.PathLike,
    asset_name: str,
    repo_id: str | None,
    instruct_model: bool,
) -> str:
    local_path = Path(model_path).parent / asset_name
    if local_path.exists():
        return str(local_path)
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub python API not available in environment"
        )
    repo_id = repo_id or _default_repo_id(instruct_model)
    try:
        return hf_hub_download(repo_id, asset_name, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id, asset_name)


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
                config["head_dim"],
                config["bos_token_id"],
                config["eos_token_id"],
                config.get("pad_token_id"),
                instruct_model,
            )
        except KeyError as e:
            raise ValueError(f"Model config missing required metadata: {e}")


class Gemma3Base:
    def _init_model_metadata(self):
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id: int = self._tokenizer.encode("\n\n").ids[-1]
        self._end_of_turn_id: int = self._tokenizer.token_to_id("<end_of_turn>")

    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(input).ids
        # Gemma 3 chat format: <start_of_turn>role\ntext<end_of_turn>\n
        # BOS is added once at warmup start; strip auto-prepended BOS here.
        if role == "model":
            ids = self._tokenizer.encode("<start_of_turn>model\n").ids
        else:
            ids = self._tokenizer.encode(
                "<start_of_turn>" + role + "\n" + input + "<end_of_turn>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _prepare_run_input_tokens(self, input: str) -> list[int]:
        inp_tokens = self._tokenize_input(input, "user")
        if self._instruct_model:
            inp_tokens += self._tokenize_input("", "model")
        return inp_tokens

    def _warmup_tokens(self) -> list[int]:
        return [self._bos_token_id] + self._tokenize_input(self._sys_prompt, "system")

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if self._end_of_turn_id is not None and next_token == self._end_of_turn_id:
            return True
        if not self._instruct_model and len(gen_tokens) > 2:
            if next_token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen_tokens[-2:])
        return False


class Gemma3Dynamic(Gemma3Base, DynamicDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
    ):
        config_path = _resolve_asset_path(model.model_path, "config.json", repo_id, instruct_model)
        tokenizer_path = _resolve_asset_path(model.model_path, "tokenizer.json", repo_id, instruct_model)
        DynamicDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(config_path, instruct_model),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            include_position_ids=False,
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
    ) -> "Gemma3Dynamic":
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
    ) -> "Gemma3Dynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
        )


class Gemma3Static(Gemma3Base, StaticDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
    ):
        token_embeddings = self._find_token_embeddings(model.model_path)
        token_id_lut = self._find_token_id_lut(model.model_path)
        config_path = _resolve_asset_path(model.model_path, "config.json", repo_id, instruct_model)
        tokenizer_path = _resolve_asset_path(model.model_path, "tokenizer.json", repo_id, instruct_model)
        StaticDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(config_path, instruct_model),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            combined_kv_io=combined_kv_io,
            token_embeddings=token_embeddings,
            token_id_lut=token_id_lut,
        )
        if self._token_id_lut is not None:
            self._logger.info(
                "Loaded token ID LUT (%d entries) for trimmed vocab remap",
                len(self._token_id_lut),
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
    ) -> "Gemma3Static":
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
    ) -> "Gemma3Static":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
        )

    @staticmethod
    def _find_data_file(
        model_path: str | os.PathLike,
        pattern: str,
        description: str,
    ) -> Path | None:
        return find_single_data_file(model_path, pattern, description)

    @staticmethod
    def _find_token_embeddings(
        model_path: str | os.PathLike,
        emb_pattern: str = "token_embeddings.npy",
    ) -> np.ndarray | None:
        path = Gemma3Static._find_data_file(model_path, emb_pattern, "token embedding")
        if path is None:
            return None
        return np.load(path)

    @staticmethod
    def _find_token_id_lut(
        model_path: str | os.PathLike,
        lut_pattern: str = "token_id_lut.npy",
    ) -> np.ndarray | None:
        path = Gemma3Static._find_data_file(model_path, lut_pattern, "token ID LUT")
        if path is None:
            return None
        return np.load(path)


if __name__ == "__main__":
    pass
