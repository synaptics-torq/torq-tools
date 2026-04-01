# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from torq.runtime import (
    InferenceRunner,
    VMFBInferenceRunner
)

from ...inference.runners import (
    ORTInferenceRunner,
    TFLiteInferenceRunner
)

DEFAULT_SYS_PROMPT: Final[str] = "You are a helpful AI assistant named SmolLM. Provide all answers as concise responses; use as few words as possible and avoid extra explanation."


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None = None
    instruct_model: bool = False

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
                instruct_model
            )
        except KeyError as e:
            raise ValueError(f"Model config missing required metadata: {e}")


class SmolLM2Base(ABC):

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
        self._instruct_model: bool = config.instruct_model
        self._bos_token_id: int = config.bos_token_id
        self._eos_token_id: int = config.eos_token_id
        self._pad_token_id: int = config.pad_token_id or 0
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[0]
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
        # stable softmax
        x = x - x.max()
        probs = np.exp(x)
        probs = probs / probs.sum()
        # top-p nucleus
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
        if role == "assistant":
            return self._tokenizer.encode(self._bos_token + role + "\n").ids
        return self._tokenizer.encode(self._bos_token + role + "\n" + input + self._eos_token + "\n").ids

    def run(self, input: str, max_gen_tokens: int | None = None) -> str:
        self._reset_cache()
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
        sys_tokens = self._tokenize_input(self._sys_prompt, "system")
        if isinstance(self._max_prompt_tokens, int):
            if len(sys_tokens) > self._max_prompt_tokens:
                self._logger.warning("Truncating system prompt from %d to %d", len(sys_tokens), self.max_inp_len)
                sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
            if self._max_user_tokens < 1:
                self._logger.warning("No tokens left for user prompt")
        warmup_len = len(sys_tokens)
        self._prefill_prompt(sys_tokens, start_seq_len=0)
        if self._max_user_tokens is not None:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt, %d tokens remaining for user input",
                warmup_len,
                self._max_user_tokens,
            )
        else:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt",
                warmup_len,
            )
        return warmup_len


class SmolLM2Dynamic(SmolLM2Base):

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None
    ):
        if repo_id is None:
            repo_id: str = "HuggingFaceTB/SmolLM2-135M"
            if instruct_model:
                repo_id += "-Instruct"
        super().__init__(
            model,
            ModelConfig.from_json_config(
                hf_hub_download(repo_id, "config.json"),
                instruct_model
            ),
            max_prompt_tokens,
            max_gen_tokens,
            Tokenizer.from_file(
                hf_hub_download(repo_id, "tokenizer.json")
            ),
            DEFAULT_SYS_PROMPT if instruct_model else None
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None
    ) -> "SmolLM2Dynamic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            instruct_model=instruct_model,
            repo_id=repo_id
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None
    ) -> "SmolLM2Dynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            instruct_model=instruct_model,
            repo_id=repo_id
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    def _llm_step(
        self, token: int, curr_seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([[token]], dtype=np.int64)
        attn_mask = np.ones([1, curr_seq_len + 1], dtype=np.int64)
        pos_ids = np.array([[curr_seq_len]], dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
            **self._kv_cache
        }
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model:
            # WARNING: relying on "\n\n" is fragile but is the best we have right now
            return len(gen_tokens) > 2 and all(t == self._nl_token_id for t in gen_tokens[-2:])

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


class SmolLM2Static(SmolLM2Base):

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True
    ):
        if repo_id is None:
            repo_id: str = "HuggingFaceTB/SmolLM2-135M"
            if instruct_model:
                repo_id += "-Instruct"
        self._combined_kv_io = combined_kv_io
        super().__init__(
            model,
            ModelConfig.from_json_config(
                hf_hub_download(repo_id, "config.json"),
                instruct_model
            ),
            max_prompt_tokens,
            max_gen_tokens,
            Tokenizer.from_file(
                hf_hub_download(repo_id, "tokenizer.json")
            ),
            DEFAULT_SYS_PROMPT if instruct_model else None
        )
        self._token_embeddings: np.ndarray | None = self._find_token_embeddings()

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True
    ) -> "SmolLM2Static": 
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io
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
        combined_kv_io: bool = True
    ) -> "SmolLM2Static":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io
        )

    def _find_token_embeddings(
        self,
        emb_pattern: str = "token_embeddings.npy",
    ) -> np.ndarray | None:
        paths = []
        paths.extend(Path(self._model.model_path).parent.glob(emb_pattern))
        if not paths:
            return None

        paths = list({p.resolve(): p for p in paths}.values())
        if len(paths) > 1:
            raise RuntimeError(
                f"Expected a single token embedding file, found {len(paths)}: {paths}"
            )
        return np.load(paths[0])

    def _init_cache(self) -> dict[str, np.ndarray]:
        if self._combined_kv_io:
            return {
                f"past_key_values.{i}.key_value": np.zeros(
                    [1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim], dtype=np.float32
                )
                for i in range(self._n_layers)
            }
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                [1, self._n_kv_heads, self._max_gen_tokens, self._head_dim], dtype=np.float32
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    def _llm_step(
        self, token: int, curr_seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        if isinstance(self._token_embeddings, np.ndarray):
            inputs = {
                "token_embedding": np.expand_dims(self._token_embeddings[token], axis=(0, 1))
            }
        else:
            inputs = {
                "input_ids": np.array([[token]], dtype=np.int64)
            }
        pos_ids = np.array([[curr_seq_len]], dtype=np.int64)
        inputs.update({
            "position_ids": pos_ids,
            **self._kv_cache
        })
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model:
            # WARNING: relying on "\n\n" is fragile but is the best we have right now
            return len(gen_tokens) > 2 and all(t == self._nl_token_id for t in gen_tokens[-2:])

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None
    ) -> list[int]:
        if isinstance(max_gen_tokens, int) and 0 <=  max_gen_tokens < self._max_gen_tokens:
            self._max_gen_tokens = max_gen_tokens
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


if __name__ == "__main__":
    pass
