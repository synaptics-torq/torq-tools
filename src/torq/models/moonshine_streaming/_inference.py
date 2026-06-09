# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

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

from ...inference.runners import (
    InferenceRunner,
    ORTInferenceRunner,
    TFLiteInferenceRunner,
    VMFBInferenceRunner
)


class MoonshineStreamingBase(ABC):

    def __init__(
        self,
        model_size: Literal["tiny", "small"],
        max_inp_len: int | None,
        combined_kv_io: bool = False
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_size = model_size
        self._max_inp_len = max_inp_len
        self._combined_kv_io = combined_kv_io

        if self._model_size == "tiny":
            self._n_layers: int = 6
            self._n_kv_heads: int = 8
            self._head_dim: int = 40
        else: # small
            self._n_layers: int = 8
            self._n_kv_heads: int = 8
            self._head_dim: int = 52

        if self._combined_kv_io:
            self._n_kv_heads *= 2

        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._encoder_pad_id: int = 2

        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)

        # In streaming, we define self KV caches first, then cross KV caches
        self._kv_cache: dict[str, np.ndarray] = {}
        self._all_cache_names: list[str] = []

        for i in range(self._n_layers):
            self._all_cache_names.append(f"past_self_key_{i}")
            self._all_cache_names.append(f"past_self_value_{i}")
        for i in range(self._n_layers):
            self._all_cache_names.append(f"past_cross_key_{i}")
            self._all_cache_names.append(f"past_cross_value_{i}")

        for k in self._all_cache_names:
            self._kv_cache[k] = np.zeros((1, self._n_kv_heads, 1, self._head_dim), dtype=np.float32)

        self._dec_cache_names: list[str] = self._all_cache_names[:2 * self._n_layers]

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
            self._logger.warning("Truncating input from %d to %d", len(input), self._max_inp_len)
            input = input[: self._max_inp_len]
        elif len(input) < self._max_inp_len:
            self._logger.info("Padding input from %d to %d", len(input), self._max_inp_len)
            input = np.pad(
                input,
                (0, self._max_inp_len - len(input)),
                constant_values=self._encoder_pad_id,
            )
        return input.reshape((1, self._max_inp_len))

    def _expects_token_embedding(self) -> bool:
        if not hasattr(self, "_use_token_embedding"):
            self._use_token_embedding = False
            decoder = getattr(self, "_decoder", None)
            if decoder is not None:
                if hasattr(decoder, "_sess") and hasattr(decoder._sess, "get_inputs"):
                    try:
                        self._use_token_embedding = any(inp.name == "token_embedding" for inp in decoder._sess.get_inputs())
                    except Exception:
                        pass
                elif hasattr(decoder, "_interpreter") and hasattr(decoder._interpreter, "get_input_details"):
                    try:
                        self._use_token_embedding = any(inp["name"] == "token_embedding" for inp in decoder._interpreter.get_input_details())
                    except Exception:
                        pass
                elif hasattr(decoder, "inputs_info"):
                    try:
                        info = decoder.inputs_info
                        if isinstance(info, dict):
                            self._use_token_embedding = "token_embedding" in info
                        elif isinstance(info, (list, tuple)):
                            self._use_token_embedding = any(
                                (item == "token_embedding" or (isinstance(item, dict) and item.get("name") == "token_embedding"))
                                for item in info
                            )
                        elif hasattr(info, "__contains__"):
                            self._use_token_embedding = "token_embedding" in info
                    except Exception:
                        pass
        return self._use_token_embedding

    def _get_decoder_token_input(self, last_tok: int) -> dict[str, np.ndarray]:
        if self._expects_token_embedding() and isinstance(self._token_embeddings, np.ndarray):
            emb = self._token_embeddings[last_tok]
            return {"token_embedding": np.expand_dims(emb, axis=(0, 1))}
        return {"decoder_input_ids": np.array([[last_tok]], dtype=np.int64)}

    @abstractmethod
    def run(self, input: np.ndarray, max_tokens: int | None = None) -> np.ndarray: ...


class MoonshineStreamingDynamic(MoonshineStreamingBase):

    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        decoder_with_past: InferenceRunner,
        model_size: Literal["tiny", "small"],
        max_inp_len: int | None = None,
        preprocessor: InferenceRunner | None = None,
    ):
        super().__init__(model_size, max_inp_len, combined_kv_io=False)

        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._decoder = decoder
        self._logger.info("Loaded decoder '%s'", str(self._decoder.model_path))
        self._decoder_with_past = decoder_with_past
        self._logger.info("Loaded decoder with past '%s'", str(self._decoder_with_past.model_path))
        self._preprocessor = preprocessor
        if self._preprocessor is not None:
            self._logger.info("Loaded preprocessor '%s'", str(self._preprocessor.model_path))

        self._token_embeddings: np.ndarray = self._find_token_embeddings()

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None,
    ) -> "MoonshineStreamingDynamic":
        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_with_past_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            ORTInferenceRunner(preprocessor_model) if preprocessor_model is not None else None,
        )

    def _find_token_embeddings(self) -> np.ndarray:
        paths = list(self._decoder.model_path.parent.glob("decoder*token_embeddings.npy"))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")
        return np.load(paths[0])

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self._all_cache_names if update_all else self._dec_cache_names
        for k, v in zip(cache_tensors, new_values):
            self._kv_cache[k] = v

    def _run_decoder(
        self, input_tokens: list[int], encoder_out: np.ndarray, *, seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        decoder_inputs = self._get_decoder_token_input(input_tokens[-1])
        if seq_len == 0:
            decoder_inputs["encoder_hidden_states"] = encoder_out
            hidden, *cache = self._decoder.infer(decoder_inputs)
        else:
            decoder_inputs["encoder_hidden_states"] = encoder_out
            decoder_inputs.update(self._kv_cache)
            hidden, *cache = self._decoder_with_past.infer(decoder_inputs)

        logits = hidden[0, -1, :] @ self._token_embeddings.T
        next_token = logits.argmax().item()
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

        if self._preprocessor is not None:
            attention_mask = np.ones((1, input.shape[-1]), dtype=np.int64)
            features, padding_mask = self._preprocessor.infer({
                "input_values": input,
                "attention_mask": attention_mask
            })
            encoder_out = self._encoder.infer({
                "input_features": features,
                "attention_mask": padding_mask
            })[0].astype(np.float32)
        else:
            encoder_out = self._encoder.infer({"input_values": input})[0].astype(np.float32)

        next_token, init_cache = self._run_decoder(tokens, encoder_out, seq_len=0)
        self._update_cache(init_cache, update_all=True)
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(1, max_tokens):
            next_token, cache = self._run_decoder([next_token], encoder_out, seq_len=i)
            self._update_cache(cache[:2 * self._n_layers])

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


class MoonshineStreamingStatic(MoonshineStreamingBase):

    def __init__(
        self,
        encoder: InferenceRunner,
        gen_encoder_cache: InferenceRunner | None,
        decoder: InferenceRunner,
        model_size: Literal["tiny", "small"],
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None,
    ):
        super().__init__(model_size, max_inp_len, combined_kv_io=False)

        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._gen_encoder_cache = gen_encoder_cache
        if self._gen_encoder_cache is not None:
            self._logger.info("Loaded gen_encoder_cache '%s'", str(self._gen_encoder_cache.model_path))
        self._decoder = decoder
        self._logger.info("Loaded unified decoder '%s'", str(self._decoder.model_path))
        self._preprocessor = preprocessor
        if self._preprocessor is not None:
            self._logger.info("Loaded preprocessor '%s'", str(self._preprocessor.model_path))
        self._max_dec_len = max_dec_len
        self._dec_cache_shapes: dict[str, tuple[int, ...]] = {
            cache_name: (1, self._n_kv_heads, self._max_dec_len, self._head_dim)
            for cache_name in self._dec_cache_names
        }
        # Initialize decoder (self-attn) KV caches at padded size
        for cache_name, shape in self._dec_cache_shapes.items():
            self._kv_cache[cache_name] = np.zeros(shape, dtype=np.float32)
        self._token_embeddings: np.ndarray = self._find_token_embeddings()

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None,
    ) -> "MoonshineStreamingStatic":
        input_ort = ort.InferenceSession(preprocessor_model or encoder_model, providers=['CPUExecutionProvider'])
        max_inp_len: int = next(
            inp.shape for inp in input_ort.get_inputs() if inp.name == "input_values"
        )[-1]
        decoder_ort = ort.InferenceSession(decoder_model, providers=['CPUExecutionProvider'])
        max_dec_len: int = next(
            inp.shape
            for inp in decoder_ort.get_inputs()
            if "past_self" in inp.name
        )[2]  # assuming shape [B, H, L, D]

        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model is not None else None,
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            ORTInferenceRunner(preprocessor_model) if preprocessor_model is not None else None,
        )

    @classmethod
    def from_tflite(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None,
    ) -> "MoonshineStreamingStatic":
        input_int = lite_rt.Interpreter(preprocessor_model or encoder_model)
        input_int.allocate_tensors()
        max_inp_len: int = next(
            list(inp["shape"]) for inp in input_int.get_input_details() if inp["name"] == "input_values"
        )[-1]
        decoder_int = lite_rt.Interpreter(decoder_model)
        decoder_int.allocate_tensors()
        max_dec_len: int = next(
            list(inp["shape"])
            for inp in decoder_int.get_input_details()
            if "past_self" in inp["name"]
        )[2] # assuming shape [B, H, L, D]

        return cls(
            TFLiteInferenceRunner(encoder_model, n_threads=n_threads),
            TFLiteInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model is not None else None,
            TFLiteInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            TFLiteInferenceRunner(preprocessor_model) if preprocessor_model is not None else None,
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["tiny", "small"],
        max_inp_len: int,
        max_dec_len: int,
        n_threads: int | None = None,
    ) -> "MoonshineStreamingStatic":
        return cls(
            VMFBInferenceRunner(encoder_model, n_threads=n_threads),
            VMFBInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model is not None else None,
            VMFBInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
        )

    def _find_token_embeddings(
        self,
        emb_pattern: str = "decoder*token_embeddings.npy",
    ) -> np.ndarray:
        paths = list(self._decoder.model_path.parent.glob(emb_pattern))
        if not paths:
            raise FileNotFoundError("Missing token embeddings file 'decoder_token_embeddings.npy'")

        paths = list({p.resolve(): p for p in paths}.values())
        return np.load(paths[0])

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
        self, input_tokens: list[int], *, seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        decoder_inputs = self._get_decoder_token_input(input_tokens[-1])
        decoder_inputs.update({
            **self._kv_cache,
            "current_len": np.array([[seq_len]], dtype=np.int64),
        })
        hidden, *cache = self._decoder.infer(decoder_inputs)

        logits = hidden[0, -1, :] @ self._token_embeddings.T
        next_token = logits.argmax().item()
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
            attention_mask = np.ones((1, input.shape[-1]), dtype=np.int64)
            features, padding_mask = self._preprocessor.infer({
                "input_values": input,
                "attention_mask": attention_mask
            })
            encoder_out = self._encoder.infer({
                "input_features": features,
                "attention_mask": padding_mask
            })
        else:
            encoder_out = self._encoder.infer({"input_values": input})

        # Generate cross-attn KV cache
        if self._gen_encoder_cache is not None:
            # Unfolded: encoder produces hidden states, gen_encoder_cache produces KV
            encoder_hidden_states = encoder_out[0].astype(np.float32)
            encoder_cache_outputs = self._gen_encoder_cache.infer(
                {"encoder_hidden_states": encoder_hidden_states}
            )
        else:
            # Folded: encoder directly outputs cross-attn KV caches
            encoder_cache_outputs = encoder_out

        # Populate encoder (cross-attn) cache entries
        enc_cache_names = [k for k in self._all_cache_names if "cross" in k]
        for k, v in zip(enc_cache_names, encoder_cache_outputs):
            self._kv_cache[k] = v

        # First decoder call with seq_len=0 (zero-initialized self-attn KV)
        next_token, cache = self._run_decoder(tokens, seq_len=0)
        self._update_cache(cache[:2 * self._n_layers])
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder(
                [next_token], seq_len=i + 1
            )
            self._update_cache(cache[:2 * self._n_layers])

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
) -> MoonshineStreamingDynamic | MoonshineStreamingStatic:
    models, kind = find_models(model_dir)
    encoder = None
    gen_encoder_cache = None
    decoder = None
    decoder_with_past = None
    preprocessor = None

    for m in models:
        if m.name == f"preprocessor.{kind}":
            preprocessor = m
        elif m.name == f"encoder.{kind}":
            encoder = m
        elif m.name == f"gen_encoder_cache.{kind}":
            gen_encoder_cache = m
        elif m.name == f"decoder.{kind}":
            decoder = m
        elif m.name == f"decoder_with_past.{kind}":
            decoder_with_past = m

    is_static = decoder_with_past is None and decoder is not None

    if not is_static and max_dec_len is not None:
        is_static = True

    if not encoder:
        raise FileNotFoundError(f"Missing encoder model 'encoder.{kind}' @ '{model_dir}'")

    if is_static:
        if not decoder:
            raise FileNotFoundError(f"Missing decoder model 'decoder.{kind}' @ '{model_dir}'")
        if kind == "vmfb":
            if not isinstance(max_inp_len, int) or not isinstance(max_dec_len, int):
                raise ValueError(
                    f"Valid maximum input length and maximum decoder length are required for static VMFB models, received ({max_inp_len}, {max_dec_len})"
                )
            return MoonshineStreamingStatic.from_vmfb(
                encoder,
                gen_encoder_cache,
                decoder,
                model_size,
                max_inp_len,
                max_dec_len,
                n_threads=n_threads
            )
        elif kind == "tflite":
            return MoonshineStreamingStatic.from_tflite(
                encoder, gen_encoder_cache, decoder, model_size, n_threads=n_threads, preprocessor_model=preprocessor
            )
        return MoonshineStreamingStatic.from_onnx(
            encoder, gen_encoder_cache, decoder, model_size, n_threads=n_threads, preprocessor_model=preprocessor
        )
    else:
        if not decoder:
            raise FileNotFoundError(f"Missing decoder model 'decoder.{kind}' @ '{model_dir}'")
        if not decoder_with_past:
            raise FileNotFoundError(f"Missing decoder with past model 'decoder_with_past.{kind}' @ '{model_dir}'")
        return MoonshineStreamingDynamic.from_onnx(
            encoder, decoder, decoder_with_past, model_size, max_inp_len, n_threads=n_threads, preprocessor_model=preprocessor
        )
