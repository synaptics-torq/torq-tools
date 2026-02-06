# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Final

import numpy as np
import ai_edge_litert.interpreter as lite_rt
import onnxruntime as ort
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


class SmolLMBase(ABC):

    DEFAULT_SYS_PROMPT: Final[str] = "You are a helpful AI assistant named SmolLM. Provide all answers as concise responses; use as few words as possible and avoid extra explanation."

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        sys_prompt: str | None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._max_prompt_tokens = max_prompt_tokens
        self._max_gen_tokens = max_gen_tokens
        self._max_total_tokens = self._calc_max_total_tokens(self._max_prompt_tokens, self._max_gen_tokens)
        self._max_user_tokens: int | None = None
        self._sys_prompt = sys_prompt or self.DEFAULT_SYS_PROMPT
        self._temperature = temperature
        self._top_p = top_p
        self._tokenizer: Tokenizer = Tokenizer.from_file(hf_hub_download("HuggingFaceTB/SmolLM2-135M-Instruct", "tokenizer.json"))

        # from HuggingFaceTB/SmolLM-135M/config.json
        self._n_layers: int = 30
        self._n_kv_heads: int = 3
        self._head_dim: int = 64
        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._pad_token_id: int = 2
        self._encoder_pad_id: int = 2
        self._start_token: str = self._tokenizer.decode([self._start_token_id], skip_special_tokens=False)
        self._end_token: str = self._tokenizer.decode([self._end_token_id], skip_special_tokens=False)

        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)
        self._warmup_len: int = 0

        self._kv_cache: dict[str, np.ndarray] = {
            f"past_key_values.{i}.{typ}": np.zeros(
                (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }
        self._all_cache_names: list[str] = list(self._kv_cache)
        self._dec_cache_names: list[str] = [
            k for k in self._all_cache_names if "encoder" not in k
        ]
        self._gen_start_token = self.warmup()
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
                    constant_values=self._encoder_pad_id,
                ).tolist()

        return input

    @staticmethod
    def _calc_max_total_tokens(max_prompt_tokens: int | None, max_gen_tokens: int | None) -> int | None:
        if isinstance(max_prompt_tokens, (int, float)) and isinstance(max_gen_tokens, (int, float)):
            return int(max_prompt_tokens + max_gen_tokens)
        return None

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
    
    @abstractmethod
    def _llm_step(self, token: int, curr_seq_len: int): ...

    @abstractmethod
    def _run(self, input: list[int], max_gen_tokens: int | None = None) -> list[int]: ...

    def _reset_cache(self):
        self._kv_cache.update(self._reset_cache_state)

    def _prefill_prompt(self, prompt_tokens: list[int], start_seq_len: int = 0) -> tuple[int, int]:
        num_tokens_gen = start_seq_len
        next_token: int | None = None
        for token in prompt_tokens:
            next_token, cache = self._llm_step(token, num_tokens_gen)
            self._update_cache(cache)
            num_tokens_gen += 1
        return next_token, num_tokens_gen
    
    def _tokenize_input(self, input: str, role: str) -> list[int]:
        if role == "assistant":
            return self._tokenizer.encode(self._start_token + role + "\n").ids
        return self._tokenizer.encode(self._start_token + role + "\n" + input + self._end_token + "\n").ids

    def run(self, input: str, max_gen_tokens: int | None = None) -> str:
        self._reset_cache()
        # inp_tokens = self._tokenize_input(self._sys_prompt, "system")
        inp_tokens = self._tokenize_input(input, "user")
        st = time.perf_counter_ns()
        out_tokens = self._run(inp_tokens, max_gen_tokens)
        output = self._tokenizer.decode(out_tokens)
        et = time.perf_counter_ns()
        self._infer_times.append(et - st)
        return output

    def warmup(self):
        sys_tokens = self._tokenize_input(self._sys_prompt, "system")
        if isinstance(self._max_prompt_tokens, int):
            if len(sys_tokens) > self._max_prompt_tokens:
                self._logger.warning("Truncating system prompt from %d to %d", len(sys_tokens), self.max_inp_len)
                sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
            if self._max_user_tokens < 1:
                self._logger.warning("No tokens left for user prompt")
        self._warmup_len = len(sys_tokens)
        gen_start_token, _ = self._prefill_prompt(sys_tokens, start_seq_len=0)
        if self._max_user_tokens is not None:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt, %d tokens remaining for user input",
                self._warmup_len,
                self._max_user_tokens,
            )
        else:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt",
                self._warmup_len,
            )
        return gen_start_token


class SmolLMDynamic(SmolLMBase):

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
    ):
        super().__init__(model, max_prompt_tokens, max_gen_tokens, "You are a helpful AI maths assistant; provide ONLY the final numerical result as the answer, DO NOT elaborate.",)

        self._logger.info("Loaded model '%s'", str(self._model.model_path))

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        n_threads: int | None = None
    ) -> "SmolLMDynamic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_inp_len
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        n_threads: int | None = None
    ) -> "SmolLMDynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_inp_len
        )

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

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None
    ) -> list[int]:
        self._max_gen_tokens = max_gen_tokens or self._max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = []
        while next_token != self._end_token_id:
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
