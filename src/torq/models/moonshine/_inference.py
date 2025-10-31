import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import ai_edge_litert.interpreter as lite_rt
import onnxruntime as ort

from torq.runtime import (
    InferenceRunner,
    VMFBInferenceRunner
)

from ...inference.runners import (
    ORTInferenceRunner,
    TFLiteInferenceRunner
)


class MoonshineBase(ABC):

    def __init__(
        self,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_size = model_size
        self._max_inp_len = max_inp_len

        if self._model_size == "tiny":
            self._n_layers: int = 6
            self._n_kv_heads: int = 8
            self._head_dim: int = 36
        else:
            self._n_layers: int = 8
            self._n_kv_heads: int = 8
            self._head_dim: int = 52
        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._encoder_pad_id: int = 2

        self._n_tokens_gen: int = 0
        self._infer_times: deque[int] = deque(maxlen=100)

        self._kv_cache: dict[str, np.ndarray] = {
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self._n_kv_heads, 1, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }
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


class MoonshineDynamic(MoonshineBase):

    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
    ):
        super().__init__(model_size, max_inp_len)

        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._decoder = decoder
        self._logger.info("Loaded merged decoder '%s'", str(self._decoder.model_path))

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
        n_threads: int | None = None
    ) -> "MoonshineDynamic":
        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
        n_threads: int | None = None
    ) -> "MoonshineDynamic":
        return cls(
            VMFBInferenceRunner(encoder_model, n_threads=n_threads),
            VMFBInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len
        )

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        for k, v in zip(self._kv_cache.keys(), new_values):
            if update_all or "decoder" in k:
                self._kv_cache[k] = v

    def _run_decoder(
        self, input_tokens: list[int], encoder_out: np.ndarray, *, seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([input_tokens], dtype=np.int64)
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_out,
            **self._kv_cache,
            "use_cache_branch": np.array([seq_len > 0], dtype=np.bool),
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


class MoonshineStatic(MoonshineBase):

    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        decoder_with_past: InferenceRunner,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None
    ):
        super().__init__(model_size, max_inp_len)

        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._decoder = decoder
        self._logger.info("Loaded decoder '%s'", str(self._decoder.model_path))
        self._decoder_with_past = decoder_with_past
        self._logger.info("Loaded decoder with past '%s'", str(self._decoder_with_past.model_path))
        self._preprocessor = preprocessor
        if self._preprocessor is not None:
            self._logger.info("Loaded preprocessor '%s'", str(self._preprocessor.model_path))
        self._max_dec_len = max_dec_len
        self._dec_cache_shapes: dict[str, tuple[int]] = {
            cache_name: (1, self._n_kv_heads, self._max_dec_len, self._head_dim)
            for cache_name in self._dec_cache_names
        }

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None
    ) -> "MoonshineStatic":
        input_ort = ort.InferenceSession(preprocessor_model or encoder_model, providers=['CPUExecutionProvider'])
        max_inp_len: int = next(
            inp.shape for inp in input_ort.get_inputs() if inp.name == "input_values"
        )[-1]
        decoder_with_past_ort = ort.InferenceSession(decoder_with_past_model, providers=['CPUExecutionProvider'])
        max_dec_len: int = next(
            inp.shape
            for inp in decoder_with_past_ort.get_inputs()
            if "decoder" in inp.name
        )[2]  # assuming shape [B, H, L, D]

        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_with_past_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            ORTInferenceRunner(preprocessor_model) if preprocessor_model is not None else None
        )

    @classmethod
    def from_tflite(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None
    ) -> "MoonshineStatic":
        input_int = lite_rt.Interpreter(preprocessor_model or encoder_model)
        input_int.allocate_tensors()
        max_inp_len: int = next(
            list(inp["shape"]) for inp in input_int.get_input_details() if inp["name"] == "input_values"
        )[-1]
        decoder_with_past_int = lite_rt.Interpreter(decoder_with_past_model)
        decoder_with_past_int.allocate_tensors()
        max_dec_len: int = next(
            list(inp["shape"])
            for inp in decoder_with_past_int.get_input_details()
            if "decoder" in inp["name"]
        )[2] # assuming shape [B, H, L, D]

        return cls(
            TFLiteInferenceRunner(encoder_model, n_threads=n_threads),
            TFLiteInferenceRunner(decoder_model, n_threads=n_threads),
            TFLiteInferenceRunner(decoder_with_past_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            TFLiteInferenceRunner(preprocessor_model) if preprocessor_model is not None else None
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        n_threads: int | None = None
    ) -> "MoonshineStatic":
        return cls(
            VMFBInferenceRunner(encoder_model, n_threads=n_threads),
            VMFBInferenceRunner(decoder_model, n_threads=n_threads),
            VMFBInferenceRunner(decoder_with_past_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len
        )

    def _pad_cache_tensor(
        self, cache_name: str, cache_values: np.ndarray
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

    def _run_decoder(
        self, input_tokens: list[int], encoder_out: np.ndarray, *, seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([input_tokens], dtype=np.int64)
        if seq_len == 0:
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_out,
            }
            logits, *cache = self._decoder.infer(decoder_inputs)
        else:
            decoder_inputs = {
                "input_ids": input_ids,
                **self._kv_cache,
                "current_len": np.array([[seq_len]], dtype=np.int64),
            }
            logits, *cache = self._decoder_with_past.infer(decoder_inputs)
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
        encoder_out = self._encoder.infer(
            {"input_features" if self._preprocessor is not None else "input_values": input}
        )[0].astype(np.float32)

        next_token, init_cache = self._run_decoder(tokens, encoder_out, seq_len=0)
        self._update_cache(init_cache, update_all=True)
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder(
                [next_token], encoder_out, seq_len=i + 1
            )
            self._update_cache(cache)

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


def find_models(model_dir: str | os.PathLike) -> tuple[list[Path], str]:
    p = Path(model_dir)
    formats = {".onnx": "onnx", ".vmfb": "vmfb", ".tflite": "tflite"}
    found: dict[str, list[Path]] = {k: [] for k in formats}

    for f in p.iterdir():
        if f.is_file():
            ext = f.suffix.lower()
            if ext in found:
                found[ext].append(f)

    present = [ext for ext, lst in found.items() if lst]
    if len(present) == 0:
        raise FileNotFoundError(f"No model files found in {p}")
    if len(present) > 1:
        raise ValueError(f"Model directory contains multiple formats: {', '.join(present)}")

    ext = present[0]
    return found[ext], formats[ext]


def load_moonshine(
    model_dir: str | os.PathLike,
    model_size: str,
    max_inp_len: int | None,
    max_dec_len: int | None,
    n_threads: int | None = None
) -> MoonshineDynamic | MoonshineStatic:
    models, kind = find_models(model_dir)
    encoder = None
    decoder_merged = None  # dynamic model
    decoder, decoder_with_past = None, None  # static model
    for m in models:
        if m.stem == "encoder":
            encoder = m
        elif m.stem == "decoder_merged":
            decoder_merged = m
        elif m.stem == "decoder":
            decoder = m
        elif m.stem == "decoder_with_past":
            decoder_with_past = m
    is_static = decoder_merged is None and decoder is not None and decoder_with_past is not None
    if not encoder:
        raise FileNotFoundError(
            f"Missing encoder model 'encoder.{kind}' @ '{model_dir}'"
        )
    if not decoder_merged and not is_static:
        raise FileNotFoundError(
            f"Missing merged decoder model 'decoder_merged.{kind}' @ '{model_dir}'"
        )
    if not decoder_merged:
        if not decoder:
            raise FileNotFoundError(
                f"Missing decoder model 'decoder.{kind}' @ '{model_dir}'"
            )
        if not decoder_with_past:
            raise FileNotFoundError(
                f"Missing decoder with past model 'decoder_with_past.{kind}' @ '{model_dir}'"
            )

    if is_static:
        if kind == "vmfb":
            if not isinstance(max_inp_len, int) or not isinstance(max_dec_len, int):
                raise ValueError(
                    f"Valid maximum input length and maximum decoder length are required for static VMFB models, received ({max_inp_len}, {max_dec_len})"
                )
            return MoonshineStatic.from_vmfb(
                encoder,
                decoder,
                decoder_with_past,
                model_size,
                max_inp_len,
                max_dec_len,
                n_threads=n_threads
            )
        elif kind == "tflite":
            return MoonshineStatic.from_tflite(
                encoder, decoder, decoder_with_past, model_size, n_threads=n_threads
            )
        return MoonshineStatic.from_onnx(
            encoder, decoder, decoder_with_past, model_size, n_threads=n_threads
        )
    else:
        if kind == "vmfb":
            return MoonshineDynamic(
                VMFBInferenceRunner(encoder, n_threads=n_threads),
                VMFBInferenceRunner(decoder_merged, n_threads=n_threads, function="merged"),
                model_size,
                max_inp_len
            )
        else:
            return MoonshineDynamic.from_onnx(encoder, decoder_merged, model_size, max_inp_len, n_threads=n_threads)


if __name__ == "__main__":
    pass
