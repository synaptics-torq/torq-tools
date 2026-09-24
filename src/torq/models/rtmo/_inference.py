# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

try:
    import ai_edge_litert.interpreter as lite_rt
except ImportError:
    try:
        import tensorflow.lite as lite_rt
    except ImportError:
        lite_rt = None

from ...inference.runners import (
    InferenceRunner,
    ORTInferenceRunner,
    TFLiteInferenceRunner,
    VMFBInferenceRunner,
)
from ...inference.transformers import (
    DynamicEncoderDecoderRunner,
    EncoderDecoderConfig,
    StaticEncoderDecoderRunner,
    find_single_data_file,
)


def _model_config(model_size: Literal["base", "tiny"]) -> EncoderDecoderConfig:
    if model_size == "tiny":
        n_layers = 6
        head_dim = 36
    else:
        n_layers = 8
        head_dim = 52
    return EncoderDecoderConfig(
        n_layers=n_layers,
        n_kv_heads=8,
        head_dim=head_dim,
        start_token_id=1,
        end_token_id=2,
        encoder_pad_id=2,
    )


class MoonshineDynamic(DynamicEncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
    ):
        self._model_size = model_size
        super().__init__(
            encoder,
            decoder,
            _model_config(model_size),
            max_inp_len,
            decoder_log_name="merged decoder",
        )

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
        n_threads: int | None = None,
    ) -> "MoonshineDynamic":
        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int | None = None,
        n_threads: int | None = None,
    ) -> "MoonshineDynamic":
        return cls(
            VMFBInferenceRunner(encoder_model, n_threads=n_threads),
            VMFBInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
        )


class MoonshineStatic(StaticEncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        gen_encoder_cache: InferenceRunner | None,
        decoder: InferenceRunner,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None,
        combined_kv_io: bool = True,
    ):
        self._model_size = model_size
        super().__init__(
            encoder,
            gen_encoder_cache,
            decoder,
            _model_config(model_size),
            max_inp_len,
            max_dec_len,
            preprocessor,
            combined_kv_io=combined_kv_io,
        )
        self._token_embeddings: np.ndarray | None = self._find_token_embeddings()

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None,
        combined_kv_io: bool = True,
    ) -> "MoonshineStatic":
        input_ort = ort.InferenceSession(preprocessor_model or encoder_model, providers=["CPUExecutionProvider"])
        max_inp_len: int = next(
            inp.shape for inp in input_ort.get_inputs() if inp.name == "input_values"
        )[-1]
        decoder_ort = ort.InferenceSession(decoder_model, providers=["CPUExecutionProvider"])
        max_dec_len: int = next(
            inp.shape
            for inp in decoder_ort.get_inputs()
            if "decoder" in inp.name
        )[2]  # assuming shape [B, H, L, D]

        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model else None,
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            ORTInferenceRunner(preprocessor_model) if preprocessor_model is not None else None,
            combined_kv_io=combined_kv_io,
        )

    @classmethod
    def from_tflite(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        n_threads: int | None = None,
        preprocessor_model: str | os.PathLike | None = None,
        combined_kv_io: bool = True,
    ) -> "MoonshineStatic":
        if lite_rt is None:
            raise RuntimeError(
                "TFLite python API not available in environment"
            )

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
            if "decoder" in inp["name"]
        )[2]  # assuming shape [B, H, L, D]

        return cls(
            TFLiteInferenceRunner(encoder_model, n_threads=n_threads),
            TFLiteInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model else None,
            TFLiteInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            TFLiteInferenceRunner(preprocessor_model) if preprocessor_model is not None else None,
            combined_kv_io=combined_kv_io,
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        gen_encoder_cache_model: str | os.PathLike | None,
        decoder_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        n_threads: int | None = None,
        combined_kv_io: bool = True,
    ) -> "MoonshineStatic":
        return cls(
            VMFBInferenceRunner(encoder_model, n_threads=n_threads),
            VMFBInferenceRunner(gen_encoder_cache_model, n_threads=n_threads) if gen_encoder_cache_model else None,
            VMFBInferenceRunner(decoder_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len,
            combined_kv_io=combined_kv_io,
        )

    def _find_token_embeddings(
        self,
        emb_pattern: str = "decoder*token_embeddings.npy",
    ) -> np.ndarray | None:
        path = find_single_data_file(self._decoder.model_path, emb_pattern, "token embedding")
        if path is None:
            return None
        return np.load(path)


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
    n_threads: int | None = None,
) -> MoonshineDynamic | MoonshineStatic:
    models, kind = find_models(model_dir)
    encoder = None
    gen_encoder_cache = None
    decoder_merged = None
    decoder = None
    for m in models:
        if m.stem == "encoder":
            encoder = m
        elif m.stem == "gen_encoder_cache":
            gen_encoder_cache = m
        elif m.stem == "decoder_merged":
            decoder_merged = m
        elif m.stem == "decoder":
            decoder = m

    is_static = decoder_merged is None and decoder is not None
    if not encoder:
        raise FileNotFoundError(
            f"Missing encoder model 'encoder.{kind}' @ '{model_dir}'"
        )
    if not decoder_merged and not is_static:
        raise FileNotFoundError(
            f"Missing model files @ '{model_dir}': need either 'decoder_merged.{kind}' or 'encoder.{kind}' + 'decoder.{kind}'"
        )

    if is_static:
        if kind == "vmfb":
            if not isinstance(max_inp_len, int) or not isinstance(max_dec_len, int):
                raise ValueError(
                    f"Valid maximum input length and maximum decoder length are required for static VMFB models, received ({max_inp_len}, {max_dec_len})"
                )
            return MoonshineStatic.from_vmfb(
                encoder,
                gen_encoder_cache,
                decoder,
                model_size,
                max_inp_len,
                max_dec_len,
                n_threads=n_threads,
            )
        elif kind == "tflite":
            return MoonshineStatic.from_tflite(
                encoder, gen_encoder_cache, decoder, model_size, n_threads=n_threads
            )
        return MoonshineStatic.from_onnx(
            encoder, gen_encoder_cache, decoder, model_size, n_threads=n_threads
        )

    if kind == "vmfb":
        return MoonshineDynamic(
            VMFBInferenceRunner(encoder, n_threads=n_threads),
            VMFBInferenceRunner(decoder_merged, n_threads=n_threads, function="merged"),
            model_size,
            max_inp_len,
        )
    return MoonshineDynamic.from_onnx(encoder, decoder_merged, model_size, max_inp_len, n_threads=n_threads)


if __name__ == "__main__":
    pass
