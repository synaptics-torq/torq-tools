# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import os
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
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
)


DEFAULT_HF_REPO = "Qwen/Qwen3-0.6B"


def _resolve_asset_path(
    model_path: str | os.PathLike,
    asset_name: str,
    repo_id: str | None,
) -> str:
    local_path = Path(model_path).parent / asset_name
    if local_path.exists():
        return str(local_path)

    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub Python API is not available in this environment"
        )

    repo_id = repo_id or DEFAULT_HF_REPO

    try:
        return hf_hub_download(repo_id, asset_name, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id, asset_name)


def _load_tokenizer(tokenizer_path: str | os.PathLike):
    if Tokenizer is None:
        raise RuntimeError(
            "tokenizers Python API is not available in this environment"
        )

    return Tokenizer.from_file(str(tokenizer_path))


@dataclass(frozen=True)
class ModelConfig(DecoderOnlyConfig):
    @classmethod
    def from_json_config(
        cls,
        json_file: str | os.PathLike,
    ) -> "ModelConfig":
        with open(json_file, encoding="utf-8") as file:
            config = json.load(file)

        try:
            return cls(
                config["num_hidden_layers"],
                config["num_key_value_heads"],
                config["head_dim"],
                config["bos_token_id"],
                config["eos_token_id"],
                config.get("pad_token_id"),
                False,
            )
        except KeyError as error:
            raise ValueError(
                f"Model config missing required metadata: {error}"
            ) from error


class QwenBase:
    def _init_model_metadata(self) -> None:
        pass

    def _tokenize_input(
        self,
        input_text: str,
        role: str | None = None,
    ) -> list[int]:
        del role
        return self._tokenizer.encode(input_text).ids
 
    def _prepare_run_input_tokens(self, input_text: str) -> list[int]:
        chat_prompt = (
            "<|im_start|>user\n"
            f"{input_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return self._tokenize_input(chat_prompt)

    def _warmup_tokens(self) -> list[int]:
        return [self._bos_token_id]

    def _stop_decoding(
        self,
        next_token: int,
        generated_tokens: list[int],
    ) -> bool:
        del generated_tokens
        return next_token == self._eos_token_id


class QwenDynamic(QwenBase, DynamicDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        repo_id: str | None = None,
    ):
        config_path = _resolve_asset_path(
            model.model_path,
            "config.json",
            repo_id,
        )
        tokenizer_path = _resolve_asset_path(
            model.model_path,
            "tokenizer.json",
            repo_id,
        )

        DynamicDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(config_path),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            None,
            include_position_ids=True,
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        repo_id: str | None = None,
    ) -> "QwenDynamic":
        return cls(
            ORTInferenceRunner(
                model_path,
                n_threads=n_threads,
            ),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            repo_id=repo_id,
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        repo_id: str | None = None,
    ) -> "QwenDynamic":
        return cls(
            VMFBInferenceRunner(
                model_path,
                n_threads=n_threads,
            ),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            repo_id=repo_id,
        )


class _PathInferenceRunner(InferenceRunner):
    def _infer(self, inputs):
        raise RuntimeError("Path-only runner cannot infer")


class QwenStatic(QwenBase, StaticDecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
        lm_head: InferenceRunner | None = None,
        split_parts: tuple[
            InferenceRunner,
            InferenceRunner,
        ]
        | None = None,
    ):
        self._lm_head = lm_head
        self._split_parts = split_parts
        self._split_runtime = split_parts is not None

        if self._split_runtime and not combined_kv_io:
            raise ValueError(
                "Split Qwen VMFBs require combined_kv_io=True"
            )

        config_path = _resolve_asset_path(
            model.model_path,
            "config.json",
            repo_id,
        )
        tokenizer_path = _resolve_asset_path(
            model.model_path,
            "tokenizer.json",
            repo_id,
        )

        token_embeddings = None
        token_id_lut = None

        if self._split_runtime:
            model_dir = Path(model.model_path).parent
            embeddings_path = model_dir / "token_embeddings.npy"
            token_lut_path = model_dir / "token_id_lut.npy"

            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"Split Qwen runtime requires {embeddings_path}"
                )

            if not token_lut_path.exists():
                raise FileNotFoundError(
                    f"Split Qwen runtime requires {token_lut_path}"
                )

            token_embeddings = np.load(
                embeddings_path,
                mmap_mode="r",
            )
            token_id_lut = np.load(token_lut_path)

        StaticDecoderOnlyRunner.__init__(
            self,
            model,
            ModelConfig.from_json_config(config_path),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            None,
            combined_kv_io=combined_kv_io,
            token_embeddings=token_embeddings,
            token_id_lut=token_id_lut,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        if not self._split_runtime:
            return StaticDecoderOnlyRunner._init_cache(self)
        dtype = np.dtype(ml_dtypes.bfloat16)
        if self._combined_kv_io:
            return {f"past_key_values.{i}.key_value": np.zeros([1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim], dtype=dtype) for i in range(self._n_layers)}
        return {f"past_key_values.{i}.{typ}": np.zeros([1, self._n_kv_heads, self._max_gen_tokens, self._head_dim], dtype=dtype) for i in range(self._n_layers) for typ in ("key", "value")}

    def _run_split_part(
        self,
        runner: InferenceRunner,
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return [
            np.asarray(output).copy()
            for output in runner.infer(inputs)
        ]

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
        if not self._split_runtime:
            return StaticDecoderOnlyRunner._llm_step(self, token, curr_seq_len)

        token_embedding = np.expand_dims(self._token_embeddings[token].view(ml_dtypes.bfloat16), axis=(0, 1))
        position_ids = np.array([[curr_seq_len]], dtype=np.int32)
        cache_values = list(self._kv_cache.values())

        a_names = list(self._kv_cache.keys())[:14]
        b_names = list(self._kv_cache.keys())[14:]
        a_inputs = {"token_embedding": token_embedding, "position_ids": position_ids}
        a_inputs.update(dict(zip(a_names, cache_values[:14])))
        a_outputs = self._run_split_part(
            self._split_parts[0],
            a_inputs,
        )
        a_cache = a_outputs[:14]
        hidden_states = a_outputs[14]

        b_inputs = {"token_embedding": hidden_states, "position_ids": position_ids}
        b_inputs.update(dict(zip(b_names, cache_values[14:])))
        b_outputs = self._run_split_part(
            self._split_parts[1],
            b_inputs,
        )
        b_cache = b_outputs[:14]
        last_hidden = b_outputs[14]

        logits = self._lm_head.infer([last_hidden])[0]
        next_token = self.sample_next_token(
            logits[0, -1]
        )

        if self._token_id_lut is not None:
            if next_token >= len(self._token_id_lut):
                raise RuntimeError(f"Sampled compact token index {next_token} outside token ID LUT with {len(self._token_id_lut)} entries")
            next_token = int(self._token_id_lut[next_token])

        return next_token, a_cache + b_cache

    @classmethod
    def from_onnx(cls, model_path: str | os.PathLike, max_gen_tokens: int, max_inp_len: int | None = None, n_threads: int | None = None, repo_id: str | None = None, combined_kv_io: bool = True) -> "QwenStatic":
        model_path = Path(model_path)
        lm_head_path = model_path.parent / "lm_head.onnx"
        lm_head = ORTInferenceRunner(lm_head_path, n_threads=n_threads) if lm_head_path.exists() else None
        return cls(ORTInferenceRunner(model_path, n_threads=n_threads), max_prompt_tokens=max_inp_len, max_gen_tokens=max_gen_tokens, repo_id=repo_id, combined_kv_io=combined_kv_io, lm_head=lm_head)

    @classmethod
    def from_vmfb(cls, model_path: str | os.PathLike, max_gen_tokens: int, max_inp_len: int | None = None, n_threads: int | None = None, repo_id: str | None = None, combined_kv_io: bool = True) -> "QwenStatic":
        model_path = Path(model_path)
        lm_head_path = model_path.parent / "lm_head.vmfb"
        lm_head = VMFBInferenceRunner(lm_head_path, n_threads=n_threads, device_uri="torq") if lm_head_path.exists() else None
        return cls(VMFBInferenceRunner(model_path, n_threads=n_threads, device_uri="torq"), max_prompt_tokens=max_inp_len, max_gen_tokens=max_gen_tokens, repo_id=repo_id, combined_kv_io=combined_kv_io, lm_head=lm_head)

    @classmethod
    def from_split_vmfb(
        cls,
        parts_dir: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
    ) -> "QwenStatic":
        parts_dir = Path(parts_dir)

        part_a_path = parts_dir / "transformer_part_A.vmfb"
        part_b_path = parts_dir / "transformer_part_B.vmfb"
        lm_head_path = parts_dir / "lm_head.vmfb"

        for path in (
            part_a_path,
            part_b_path,
            lm_head_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Split Qwen VMFB not found: {path}"
                )

        config_path = parts_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Split Qwen config not found: {config_path}"
            )

        path_holder = _PathInferenceRunner(config_path)

        part_a = VMFBInferenceRunner(
            part_a_path,
            device_uri="torq",
        )
        part_b = VMFBInferenceRunner(
            part_b_path,
            device_uri="torq",
        )
        lm_head = VMFBInferenceRunner(
            lm_head_path,
            device_uri="torq",
        )

        return cls(
            path_holder,
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
            lm_head=lm_head,
            split_parts=(part_a, part_b),
        )
