# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runners import InferenceRunner


@dataclass(frozen=True)
class DecoderOnlyConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None = None
    instruct_model: bool = False


class DecoderOnlyRunner(ABC):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        tokenizer,
        sys_prompt: str | None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._max_prompt_tokens = max_prompt_tokens
        self._max_gen_tokens = max_gen_tokens
        self._tokenizer = tokenizer
        self._sys_prompt = sys_prompt
        self._max_total_tokens = self._calc_max_total_tokens(
            self._max_prompt_tokens, self._max_gen_tokens
        )
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
        self._bos_token: str = self._tokenizer.decode(
            [self._bos_token_id], skip_special_tokens=False
        )
        self._eos_token: str = self._tokenizer.decode(
            [self._eos_token_id], skip_special_tokens=False
        )
        self._init_model_metadata()
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

    def _init_model_metadata(self):
        pass

    @abstractmethod
    def _init_cache(self) -> dict[str, np.ndarray]: ...

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    @abstractmethod
    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]: ...

    @abstractmethod
    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool: ...

    @abstractmethod
    def _run(self, input: list[int], max_gen_tokens: int | None = None) -> list[int]: ...

    @abstractmethod
    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]: ...

    @staticmethod
    def _calc_max_total_tokens(max_prompt_tokens: int | None, max_gen_tokens: int | None) -> int | None:
        if isinstance(max_prompt_tokens, (int, float)) and isinstance(max_gen_tokens, (int, float)):
            return int(max_prompt_tokens + max_gen_tokens)
        return None

    def _reset_cache(self):
        self._kv_cache.update(self._reset_cache_state)

    def _format_input_tokens(self, input: list[int]) -> list[int]:
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

    def _prepare_run_input_tokens(self, input: str) -> list[int]:
        return self._tokenize_input(input, "user")

    def _warmup_tokens(self) -> list[int]:
        return self._tokenize_input(self._sys_prompt, "system")

    def run(self, input: str, max_gen_tokens: int | None = None) -> str:
        self._reset_cache()
        inp_tokens = self._prepare_run_input_tokens(input)
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
        sys_tokens = self._warmup_tokens()
        if isinstance(self._max_prompt_tokens, int):
            if len(sys_tokens) > self._max_prompt_tokens:
                self._logger.warning(
                    "Truncating system prompt from %d to %d",
                    len(sys_tokens),
                    self.max_inp_len,
                )
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


class DynamicDecoderOnlyRunner(DecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        tokenizer,
        sys_prompt: str | None,
        *,
        include_position_ids: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._include_position_ids = include_position_ids
        super().__init__(
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            tokenizer,
            sys_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([[token]], dtype=np.int64)
        attn_mask = np.ones([1, curr_seq_len + 1], dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        if self._include_position_ids:
            inputs["position_ids"] = np.array([[curr_seq_len]], dtype=np.int64)
        inputs.update(self._kv_cache)
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None,
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


class StaticDecoderOnlyRunner(DecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        tokenizer,
        sys_prompt: str | None,
        *,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
        token_id_lut: np.ndarray | None = None,
        lm_head: InferenceRunner | None = None,
        hidden_states_name: str = "last_hidden_states",
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._combined_kv_io = combined_kv_io
        self._token_embeddings = token_embeddings
        self._token_id_lut = token_id_lut
        self._lm_head = lm_head
        self._hidden_states_name = hidden_states_name
        super().__init__(
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            tokenizer,
            sys_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        if self._combined_kv_io:
            return {
                f"past_key_values.{i}.key_value": np.zeros(
                    [1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim],
                    dtype=np.float32,
                )
                for i in range(self._n_layers)
            }
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                [1, self._n_kv_heads, self._max_gen_tokens, self._head_dim],
                dtype=np.float32,
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
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
            **self._kv_cache,
        })
        logits, *cache = self._model.infer(inputs)
        if self._lm_head is not None:
            # With a split LM head the first output is the hidden state, not
            # logits; run it through the standalone head to get the logits.
            logits = self._lm_head.infer({self._hidden_states_name: logits})[0]
        next_token = self.sample_next_token(logits[0, -1])
        if self._token_id_lut is not None:
            if next_token >= len(self._token_id_lut):
                raise RuntimeError(
                    f"Sampled compact token index {next_token} outside token ID LUT "
                    f"with {len(self._token_id_lut)} entries"
                )
            next_token = int(self._token_id_lut[next_token])
        return next_token, cache

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None,
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


@dataclass(frozen=True)
class EncoderDecoderConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    start_token_id: int
    end_token_id: int
    encoder_pad_id: int


class EncoderDecoderRunner(ABC):
    def __init__(
        self,
        config: EncoderDecoderConfig,
        max_inp_len: int | None,
        combined_kv_io: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_inp_len = max_inp_len
        self._combined_kv_io = combined_kv_io
        self._n_layers: int = config.n_layers
        self._n_kv_heads: int = config.n_kv_heads
        if self._combined_kv_io:
            self._n_kv_heads *= 2
        self._head_dim: int = config.head_dim
        self._start_token_id: int = config.start_token_id
        self._end_token_id: int = config.end_token_id
        self._encoder_pad_id: int = config.encoder_pad_id
        self._n_tokens_gen: int = 0
        self._infer_times: deque[int] = deque(maxlen=100)
        self._kv_cache = self._init_cache()
        self._all_cache_names: list[str] = list(self._kv_cache)
        self._dec_cache_names: list[str] = [
            k for k in self._all_cache_names if "encoder" not in k
        ]

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else 0.0

    @property
    def max_inp_len(self) -> int | None:
        return self._max_inp_len

    def _init_cache(self) -> dict[str, np.ndarray]:
        if self._combined_kv_io:
            return {
                f"past_key_values.{i}.{a}.key_value": np.zeros(
                    (1, self._n_kv_heads, 1, self._head_dim),
                    dtype=np.float32,
                )
                for i in range(self._n_layers)
                for a in ("decoder", "encoder")
            }
        return {
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self._n_kv_heads, 1, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }

    def _size_input(self, input: np.ndarray) -> np.ndarray:
        input = input.flatten()
        if len(input) > self._max_inp_len:
            self._logger.warning("Truncating input from %d to %d", len(input), self.max_inp_len)
            input = input[: self._max_inp_len]
        elif len(input) < self._max_inp_len:
            self._logger.info("Padding input from %d to %d", len(input), self.max_inp_len)
            input = np.pad(
                input,
                (0, self._max_inp_len - len(input)),
                constant_values=self._encoder_pad_id,
            )
        return input.reshape((1, self._max_inp_len))

    @abstractmethod
    def run(self, input: np.ndarray, max_tokens: int | None = None) -> np.ndarray: ...


class DynamicEncoderDecoderRunner(EncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        config: EncoderDecoderConfig,
        max_inp_len: int | None = None,
        *,
        decoder_log_name: str = "decoder",
    ):
        super().__init__(config, max_inp_len, combined_kv_io=False)
        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._decoder = decoder
        self._logger.info("Loaded %s '%s'", decoder_log_name, str(self._decoder.model_path))

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        for k, v in zip(self._kv_cache.keys(), new_values):
            if update_all or "decoder" in k:
                self._kv_cache[k] = v

    def _run_decoder(
        self,
        input_tokens: list[int],
        encoder_out: np.ndarray,
        *,
        seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([input_tokens], dtype=np.int64)
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_out,
            **self._kv_cache,
            "use_cache_branch": np.array([seq_len > 0], dtype=np.bool_),
        }
        logits, *cache = self._decoder.infer(decoder_inputs)
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def run(
        self,
        input: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        if max_tokens is None:
            max_tokens = int((input.shape[-1] / 16000) * 6)
        if isinstance(self.max_inp_len, int):
            input = self._size_input(input)

        st = time.time()
        next_token = self._start_token_id
        tokens = [next_token]
        encoder_out = self._encoder.infer({"input_values": input})[0].astype(np.float32)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder([next_token], encoder_out, seq_len=i)
            self._update_cache(cache, update_all=i < 1)

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


class StaticEncoderDecoderRunner(EncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        gen_encoder_cache: InferenceRunner | None,
        decoder: InferenceRunner,
        config: EncoderDecoderConfig,
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None,
        *,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
    ):
        super().__init__(config, max_inp_len, combined_kv_io)
        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._gen_encoder_cache = gen_encoder_cache
        if self._gen_encoder_cache is not None:
            self._logger.info("Loaded gen_encoder_cache '%s'", str(self._gen_encoder_cache.model_path))
        else:
            self._logger.info("Encoder cache folded into encoder")
        self._decoder = decoder
        self._logger.info("Loaded decoder '%s'", str(self._decoder.model_path))
        self._preprocessor = preprocessor
        if self._preprocessor is not None:
            self._logger.info("Loaded preprocessor '%s'", str(self._preprocessor.model_path))
        self._max_dec_len = max_dec_len
        self._dec_cache_shapes: dict[str, tuple[int, ...]] = {
            cache_name: (1, self._n_kv_heads, self._max_dec_len, self._head_dim)
            for cache_name in self._dec_cache_names
        }
        for cache_name, shape in self._dec_cache_shapes.items():
            self._kv_cache[cache_name] = np.zeros(shape, dtype=np.float32)
        self._token_embeddings = token_embeddings

    def _pad_cache_tensor(
        self,
        cache_name: str,
        cache_values: np.ndarray,
    ) -> np.ndarray:
        if not (req_shape := self._dec_cache_shapes.get(cache_name)):
            return cache_values
        if cache_values.shape == req_shape:
            return cache_values
        if cache_values.ndim != len(req_shape):
            raise ValueError(
                f"Invalid cache tensor dims: got {cache_values.ndim}, expected {len(req_shape)}"
            )
        pad_width = []
        for cache_dim, req_dim in zip(cache_values.shape, req_shape):
            if cache_dim > req_dim:
                raise ValueError(
                    f"Unexpected dim for cache tensor: {cache_values.shape}, expected: {req_shape}"
                )
            before = 0
            after = req_dim - cache_dim
            pad_width.append((before, after))

        cache_padded = np.pad(
            cache_values, pad_width, mode="constant", constant_values=0
        )
        return cache_padded

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self._all_cache_names if update_all else self._dec_cache_names
        if (curr_len := len(cache_tensors)) != (new_len := len(new_values)):
            raise RuntimeError(
                f"Cache tensors mismatch: expected {curr_len} new values, got {new_len}"
            )
        for k, v in zip(cache_tensors, new_values):
            self._kv_cache[k] = self._pad_cache_tensor(k, v)

    def _get_decoder_token_input(
        self,
        input_tokens: list[int],
    ) -> dict[str, np.ndarray]:
        if isinstance(self._token_embeddings, np.ndarray):
            last_tok = input_tokens[-1]
            return {"token_embedding": np.expand_dims(self._token_embeddings[last_tok], axis=(0, 1))}
        return {"input_ids": np.array([input_tokens], dtype=np.int64)}

    def _run_decoder(
        self,
        input_tokens: list[int],
        *,
        seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        decoder_inputs = self._get_decoder_token_input(input_tokens)
        decoder_inputs.update({
            **self._kv_cache,
            "current_len": np.array([[seq_len]], dtype=np.int64),
        })
        logits, *cache = self._decoder.infer(decoder_inputs)
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def run(
        self,
        input: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        max_tokens = (
            max_tokens
            if isinstance(max_tokens, int) and max_tokens < self._max_dec_len
            else self._max_dec_len
        )

        st = time.time()
        next_token = self._start_token_id
        tokens = [next_token]
        input = self._size_input(input)
        if self._preprocessor is not None:
            input = self._preprocessor.infer({"input_values": input})[0]
        encoder_input_name = "input_features" if self._preprocessor is not None else "input_values"
        encoder_out = self._encoder.infer({encoder_input_name: input})

        if self._gen_encoder_cache is not None:
            encoder_hidden_states = encoder_out[0].astype(np.float32)
            encoder_cache_outputs = self._gen_encoder_cache.infer(
                {"encoder_hidden_states": encoder_hidden_states}
            )
        else:
            encoder_cache_outputs = encoder_out

        enc_cache_names = [k for k in self._all_cache_names if "encoder" in k]
        for k, v in zip(enc_cache_names, encoder_cache_outputs):
            self._kv_cache[k] = v

        next_token, cache = self._run_decoder(tokens, seq_len=0)
        self._update_cache(cache)
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder(
                [next_token], seq_len=i + 1
            )
            self._update_cache(cache)

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


def find_single_data_file(
    model_path: str | os.PathLike,
    pattern: str,
    description: str,
) -> Path | None:
    paths = list(Path(model_path).parent.glob(pattern))
    if not paths:
        return None

    paths = list({p.resolve(): p for p in paths}.values())
    if len(paths) > 1:
        raise RuntimeError(
            f"Expected a single {description} file, found {len(paths)}: {paths}"
        )
    return paths[0]
