# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import json
import logging
import os
import shutil
from math import floor
from pathlib import Path
from subprocess import check_output, CalledProcessError, STDOUT
from typing import Literal, Final

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from transformers import AutoConfig

_FP_EXPORT_DTYPE_MAPPING: Final[dict] = {
    "float": onnx.TensorProto.FLOAT,
    "fp32" : onnx.TensorProto.FLOAT,
    "fp16" : onnx.TensorProto.FLOAT16,
    "bf16" : onnx.TensorProto.BFLOAT16
}


class SmolLM2ModelExporter:

    def __init__(
        self,
        model_size: Literal["135M", "360M", "1.7B"] = "135M",
        instruct_model: bool = False,
        extract_embeddings: bool = False,
        static_models: bool = True,
        *,
        input_tokens: int = 128,
        output_ratio: float = 0.5,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtype: str | None = None,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        if model_size not in ["135M", "360M", "1.7B"]:
            raise ValueError(
                f"Invalid model size '{model_size}', choose one of: ['135M', '360M', '1.7B']"
            )

        self._extract_embeddings = extract_embeddings
        self._static_models = static_models
        self._input_tokens = input_tokens
        self._output_tokens = floor(output_ratio * input_tokens)
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._convert_dtype = convert_dtype
        self._onnx_export_dtype = _FP_EXPORT_DTYPE_MAPPING.get(
            self._model_dtype,
            onnx.TensorProto.FLOAT
        )
        self._skip_export = set(skip_export or [])
        self._hf_repo = f"HuggingFaceTB/SmolLM2-{model_size}"
        if instruct_model:
            self._hf_repo += "-Instruct"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        if onnx_source_dir and (onnx_source_dir := Path(onnx_source_dir)).exists():
            self._onnx_dir = onnx_source_dir
        else:
            self._onnx_dir = self._models_dir / "source"
            self._onnx_dir.mkdir(parents=True, exist_ok=True)
            self._optimum_export_model()
        
        self._model: onnx.ModelProto = self._load_onnx()
        self._export_dir = self._onnx_dir / "export" / ("static" if self._static_models else "dynamic")
        if self._export_dir.exists():
            shutil.rmtree(self._export_dir, ignore_errors=True)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._export_paths: Path | None = None

    @property
    def export_dir(self) -> Path:
        return self._export_dir

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = False) -> onnx.ModelProto:
        if model.ir_version > 10:
            self._logger.warning(
                "Warning: Model IR version is > 10 (%d), which might be unsupported by onnxruntime",
                model.ir_version
            )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=not skip_data_prop
        )
        onnx.checker.check_model(model, full_check=True)
        return model

    def _optimum_export_model(self):
        if not (self._onnx_dir /  "model.onnx").exists():
            try:
                check_output(
                    [
                        "optimum-cli", "export", "onnx",
                        str(self._onnx_dir),
                        "--model", f"{self._hf_repo}-{self._model_size}",
                        "--dtype", self._model_dtype,
                        "--opset", "22",
                        "--optimize", "O1",
                    ],
                    text=True,
                    stderr=STDOUT,
                )
            except CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                    + "\n    ".join(e.output.strip().splitlines())
                ) from None

    def _load_onnx(self) -> onnx.ModelProto:
        model_path = self._onnx_dir /  "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.onnx @ '{self._onnx_dir}'")
        return onnx.load(model_path)
