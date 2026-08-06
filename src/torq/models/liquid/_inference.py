# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
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

DEFAULT_SYS_PROMPT: Final[str] = "You are a helpful AI assistant. Provide concise answers."


def _download_asset(repo_id: str, asset_name: str) -> str:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub python API not available in environment"
        )
    return hf_hub_download(repo_id, asset_name)


def _load_tokenizer(tokenizer_path: str | os.PathLike):
    if Tokenizer is None:
        raise RuntimeError(
            "tokenizers python API not available in environment"
        )
    return Tokenizer.from_file(str(tokenizer_path))


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    conv_dim: int
    conv_L_cache: int
    layer_types: tuple[str, ...]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None = None
    instruct_model: bool = False

    @classmethod
    def from_json_config(cls, json_file: str | os.PathLike, instruct_model: bool = False) -> "ModelConfig":
        with open(json_file) as f:
            config = json.load(f)
        try:
            head_dim = config.get("head_dim") or (
                config["hidden_size"] // config["num_attention_heads"]
            )
            return cls(
                config["num_hidden_layers"],
                config["num_key_value_heads"],
                head_dim,
                config.get("conv_dim", config["hidden_size"]),
                config.get("conv_L_cache", 3),
                tuple(config["layer_types"]),
                config["bos_token_id"],
                config["eos_token_id"],
                config.get("pad_token_id"),
                instruct_model,
            )
        except KeyError as e:
            raise ValueError(f"Model config missing required metadata: {e}")


class LiquidBase(ABC):

    def __init__(
        self,
        model: InferenceRunner,
        config: ModelConfig,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        tokenizer: Tokenizer,
        sys_prompt: str | None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        sess = getattr(model, "_sess", None)
        self._input_names: set[str] | None = (
            {i.name for i in sess.get_inputs()} if sess is not None else None
        )
        self._max_prompt_tokens = max_prompt_tokens
        self._max_gen_tokens = max_gen_tokens
        self._tokenizer = tokenizer
        self._sys_prompt = sys_prompt
        self._max_total_tokens = self._calc_max_total_tokens(self._max_prompt_tokens, self._max_gen_tokens)
        self._max_user_tokens: int | None = None
        self._temperature = temperature
        self._top_p = top_p

        self._n_layers: int = config.n_layers
        self._n_kv_heads: int = config.n_kv_heads
        self._head_dim: int = config.head_dim
        self._conv_dim: int = config.conv_dim
        self._conv_L_cache: int = config.conv_L_cache
        self._layer_types: tuple[str, ...] = config.layer_types
        self._instruct_model: bool = config.instruct_model
        self._bos_token_id: int = config.bos_token_id
        self._eos_token_id: int = config.eos_token_id
        self._pad_token_id: int = config.pad_token_id or 0
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[-1]
        self._bos_token: str = self._tokenizer.decode([self._bos_token_id], skip_special_tokens=False)
        self._eos_token: str = self._tokenizer.decode([self._eos_token_id], skip_special_tokens=False)
        self._logger.info("Loaded model '%s'", str(self._model.model_path))

        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)
        self._kv_cache = self._init_cache()
        self._warmup_len = self.warmup() if self._instruct_model else 0
        self._reset_cache_state = deepcopy(self._kv_cache)

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else 0.0

    @property
    def max_inp_len(self) -> int | None:
        return self._max_user_tokens if self._max_user_tokens is not None else self._max_prompt_tokens

    @abstractmethod
    def _init_cache(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]: ...

    @abstractmethod
    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool: ...

    @abstractmethod
    def _run(self, input: list[int], max_gen_tokens: int | None = None) -> list[int]: ...

    @staticmethod
    def _calc_max_total_tokens(max_prompt_tokens: int | None, max_gen_tokens: int | None) -> int | None:
        if isinstance(max_prompt_tokens, (int, float)) and isinstance(max_gen_tokens, (int, float)):
            return int(max_prompt_tokens + max_gen_tokens)
        return None

    def _declares_input(self, name: str, default: bool = False) -> bool:
        """Whether the loaded model declares a graph input called `name`.

        Falls back to `default` for runners that cannot report their
        signature (e.g. VMFB, which is fed positionally).
        """
        if self._input_names is None:
            return default
        return name in self._input_names

    def _reset_cache(self):
        self._kv_cache.update(self._reset_cache_state)

    def _format_input_tokens(self, input: list[int]) -> np.ndarray:
        max_len = self.max_inp_len
        if isinstance(max_len, int):
            if len(input) > max_len:
                self._logger.warning("Truncating input from %d to %d", len(input), max_len)
                input = input[: max_len]
            elif len(input) < max_len:
                self._logger.info("Padding input from %d to %d", len(input), max_len)
                input = np.pad(
                    input,
                    (0, max_len - len(input)),
                    constant_values=self._pad_token_id,
                ).tolist()
        return input

    def sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        x = logits.astype(np.float64)
        temperature = temperature or self._temperature
        top_p = top_p or self._top_p
        if temperature <= 0:
            return int(x.argmax())

        x = x / temperature
        x = x - x.max()
        probs = np.exp(x)
        probs = probs / probs.sum()
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cdf = np.cumsum(sorted_probs)
        cut = np.searchsorted(cdf, top_p) + 1
        keep = order[:cut]
        kept_probs = probs[keep]
        kept_probs = kept_probs / kept_probs.sum()
        return int(np.random.choice(keep, p=kept_probs))

    def _prefill_prompt(self, prompt_tokens: list[int], start_seq_len: int = 0) -> tuple[int, int]:
        num_tokens_gen = start_seq_len
        next_token: int | None = None
        for token in prompt_tokens:
            next_token, cache = self._llm_step(token, num_tokens_gen)
            self._update_cache(cache)
            num_tokens_gen += 1
        return next_token, num_tokens_gen

    def _tokenize_input(self, input: str, role: str) -> list[int]:
        if not self._instruct_model:
            return self._tokenizer.encode(input).ids
        # LFM2.5 ChatML format: <|im_start|>role\n{content}<|im_end|>\n
        if role == "assistant":
            ids = self._tokenizer.encode("<|im_start|>assistant\n").ids
        else:
            ids = self._tokenizer.encode(
                f"<|im_start|>{role}\n{input}<|im_end|>\n"
            ).ids
        # Strip any auto-prepended bos — caller prepends it once.
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def run(self, input: str, max_gen_tokens: int | None = None) -> str:
        self._reset_cache()
        if self._instruct_model:
            # ChatML: <|startoftext|> + <|im_start|>user\n…<|im_end|>\n + <|im_start|>assistant\n
            inp_tokens = [self._bos_token_id]
            inp_tokens += self._tokenize_input(input, "user")
            inp_tokens += self._tokenize_input("", "assistant")
        else:
            inp_tokens = self._tokenize_input(input, "user")
        st = time.perf_counter_ns()
        out_tokens = self._run(inp_tokens, max_gen_tokens)
        output = self._tokenizer.decode(out_tokens)
        et = time.perf_counter_ns()
        self._infer_times.append(et - st)
        return output

    def warmup(self) -> int:
        if not self._instruct_model:
            self._logger.warning("Not an instruct model, skipping system prompt warm-up")
            return 0
        # ChatML: <|startoftext|> + <|im_start|>system\n…<|im_end|>\n
        sys_tokens = [self._bos_token_id] + self._tokenize_input(self._sys_prompt, "system")
        if isinstance(self._max_prompt_tokens, int):
            if len(sys_tokens) > self._max_prompt_tokens:
                self._logger.warning("Truncating system prompt from %d to %d", len(sys_tokens), self.max_inp_len)
                sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
        warmup_len = len(sys_tokens)
        self._prefill_prompt(sys_tokens, start_seq_len=0)
        return warmup_len


def _kv_input_name(layer: int, kind: str, combined: bool = False) -> str:
    """Return KV cache input tensor name for an attention layer."""
    if combined:
        return f"past_key_values.{layer}.key_value"
    return f"past_key_values.{layer}.{kind}"


def _conv_input_name(layer: int) -> str:
    return f"past_conv.{layer}"


class LiquidDynamic(LiquidBase):

    DEFAULT_REPO_ID: Final[str] = "LiquidAI/LFM2.5-350M"

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ):
        if config_path is None:
            config_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "config.json")
        if tokenizer_path is None:
            tokenizer_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "tokenizer.json")
        super().__init__(
            model,
            ModelConfig.from_json_config(config_path, instruct_model),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
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
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidDynamic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
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
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidDynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for i, lt in enumerate(self._layer_types):
            if lt == "conv":
                cache[_conv_input_name(i)] = np.zeros(
                    (1, self._conv_dim, self._conv_L_cache), dtype=np.float32
                )
            else:
                cache[_kv_input_name(i, "key")] = np.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
                )
                cache[_kv_input_name(i, "value")] = np.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
                )
        return cache

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    def _llm_step(
        self, token: int, curr_seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([[token]], dtype=np.int64)
        attn_mask = np.ones([1, curr_seq_len + 1], dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        # Source exports disagree on the third input: the upstream LiquidAI
        # ONNX takes a `num_logits_to_keep` scalar, the Synaptics mirror takes
        # `position_ids`.  Feed whichever this model declares, keeping
        # `num_logits_to_keep` when the runner cannot report its signature.
        if self._declares_input("position_ids"):
            inputs["position_ids"] = np.array([[curr_seq_len]], dtype=np.int64)
        if self._declares_input("num_logits_to_keep", default=True):
            inputs["num_logits_to_keep"] = np.array(1, dtype=np.int64)
        inputs.update(self._kv_cache)
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model and len(gen_tokens) > 2:
            return all(t == self._nl_token_id for t in gen_tokens[-2:])
        return False

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None
    ) -> list[int]:
        self._max_gen_tokens = max_gen_tokens or self._max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = [next_token]
        while not self._stop_decoding(next_token, gen_tokens):
            if isinstance(self._max_gen_tokens, int) and len(gen_tokens) >= self._max_gen_tokens:
                self._logger.warning("Max generation tokens reached, stopping early")
                break
            next_token, cache = self._llm_step(next_token, curr_seq_len)
            self._update_cache(cache)
            gen_tokens.append(next_token)
            curr_seq_len += 1
        self._n_tokens_gen = len(gen_tokens)
        return gen_tokens


class LiquidStatic(LiquidBase):

    DEFAULT_REPO_ID: Final[str] = "LiquidAI/LFM2.5-350M"

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ):
        if config_path is None:
            config_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "config.json")
        if tokenizer_path is None:
            tokenizer_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "tokenizer.json")
        self._combined_kv_io = combined_kv_io
        # When `--extract-embeddings` was used at export time the model's
        # input is `token_embedding` rather than `input_ids`; load the LUT
        # from a sibling `token_embeddings.npy` if present.
        self._token_embeddings: np.ndarray | None = self._find_token_embeddings(
            model.model_path
        )
        # The model's KV-cache + attention_mask shape is FIXED at the
        # `max_gen_tokens` chosen at export time.  Generation can stop
        # earlier (smaller `_max_gen_tokens`) but the tensor shapes must
        # always be sized at the compiled length.
        self._kv_cache_len = max_gen_tokens
        super().__init__(
            model,
            ModelConfig.from_json_config(config_path, instruct_model),
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
        )

    @staticmethod
    def _find_token_embeddings(
        model_path: str | os.PathLike,
        emb_pattern: str = "token_embeddings.npy",
    ) -> np.ndarray | None:
        paths = list(Path(model_path).parent.glob(emb_pattern))
        if not paths:
            return None
        if len(paths) > 1:
            raise RuntimeError(f"Found multiple embedding files: {paths}")
        return np.load(paths[0])

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
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidStatic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
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
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidStatic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for i, lt in enumerate(self._layer_types):
            if lt == "conv":
                cache[_conv_input_name(i)] = np.zeros(
                    (1, self._conv_dim, self._conv_L_cache), dtype=np.float32
                )
            else:
                if self._combined_kv_io:
                    cache[f"past_key_values.{i}.key_value"] = np.zeros(
                        (1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
                else:
                    cache[_kv_input_name(i, "key")] = np.zeros(
                        (1, self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
                    cache[_kv_input_name(i, "value")] = np.zeros(
                        (1, self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
        return cache

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    def _llm_step(
        self, token: int, curr_seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        if isinstance(self._token_embeddings, np.ndarray):
            inputs = {
                "token_embedding": np.expand_dims(self._token_embeddings[token], axis=(0, 1)),
            }
        else:
            inputs = {
                "input_ids": np.array([[token]], dtype=np.int64),
            }
        pos_ids = np.array([[curr_seq_len]], dtype=np.int64)
        inputs["position_ids"] = pos_ids
        inputs.update(self._kv_cache)
        # If the static model still exposes attention_mask, supply a full
        # mask sized at the *compiled* KV-cache length (not the runtime
        # generation cap, which may be smaller).
        if self._declares_input("attention_mask"):
            inputs["attention_mask"] = np.ones(
                [1, self._kv_cache_len], dtype=np.int64
            )
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model and len(gen_tokens) > 2:
            return all(t == self._nl_token_id for t in gen_tokens[-2:])
        return False

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None
    ) -> list[int]:
        if isinstance(max_gen_tokens, int) and 0 <= max_gen_tokens < self._max_gen_tokens:
            self._max_gen_tokens = max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = [next_token]
        while not self._stop_decoding(next_token, gen_tokens):
            if curr_seq_len >= self._max_gen_tokens:
                self._logger.warning("Max generation tokens reached, stopping early")
                break
            next_token, cache = self._llm_step(next_token, curr_seq_len)
            self._update_cache(cache)
            gen_tokens.append(next_token)
            curr_seq_len += 1
        self._n_tokens_gen = len(gen_tokens)
        return gen_tokens


if __name__ == "__main__":
    pass
