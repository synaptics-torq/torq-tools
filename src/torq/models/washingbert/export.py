# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Final

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from onnxruntime.transformers.optimizer import optimize_model
from torq.compile import (
    process_iree_args,
    export_iree,
)
from torq.utils.logging import configure_logging

from huggingface_hub import hf_hub_download

from . import (
    HF_MODEL_REPO,
    HF_MODEL_FILES,
    LABEL_FILES,
    MODEL_COMPONENTS,
    ONNX_DTYPES,
    DEFAULT_MAX_SEQ_LEN,
    add_washingbert_export_args,
)
from ._inference import WashingBERTRunner, LabelMap
from ...utils.onnx import (
    get_model_opset,
    get_model_ops_count,
    check_dynamic_shapes,
    print_onnx_model_inputs_outputs_info,
)
from ...tools.convert_dtype.onnx import convert_model


class WashingBERTModelExporter:
    """Export and prepare WashingBERT ONNX model for the Torq NPU.

    Handles loading the source ONNX model from disk, making shapes static,
    optimizing, converting dtypes, and compiling to IREE.
    """

    def __init__(
        self,
        model_dtype: str = "float",
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        *,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        no_optimize: bool = False,
        convert_dtypes: bool = False,
        skip_validation: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model_dtype = model_dtype
        self._max_seq_len = max_seq_len
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._no_optimize = no_optimize
        self._convert_dtypes = convert_dtypes
        self._skip_validation = skip_validation

        if onnx_source_dir and (onnx_source_dir := Path(onnx_source_dir)).exists():
            self._onnx_dir = onnx_source_dir
        else:
            self._onnx_dir = self._models_dir / "WashingBERT" / "source" / "onnx"
            self._onnx_dir.mkdir(parents=True, exist_ok=True)
            self._hf_download_model()

        self._label_map = LabelMap.from_dir(self._onnx_dir)
        self._logger.info(
            "Labels: %d intents, %d type1, %d type2",
            len(self._label_map.intents),
            len(self._label_map.type1),
            len(self._label_map.type2),
        )

        self._model = self._load_onnx()

        self._export_dir = (
            self._models_dir
            / "WashingBERT"
            / "export"
            / "onnx"
            / self._model_dtype
        )
        if self._export_dir.exists():
            shutil.rmtree(self._export_dir, ignore_errors=True)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._export_path: Path | None = None

    @property
    def export_dir(self) -> Path:
        return self._export_dir

    def check_model(
        self, model: onnx.ModelProto, skip_data_prop: bool = False
    ) -> onnx.ModelProto:
        if model.ir_version > 10:
            self._logger.warning(
                "Model IR version is > 10 (%d), which might be unsupported by onnxruntime",
                model.ir_version,
            )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=not skip_data_prop
        )
        onnx.checker.check_model(model, full_check=True)
        return model

    def _hf_download_model(self):
        """Download WashingBERT model and label files from HuggingFace."""
        for filename in HF_MODEL_FILES:
            dest = self._onnx_dir / filename
            if dest.exists():
                self._logger.debug("Already exists: '%s'", dest)
                continue
            self._logger.info("Downloading '%s' from %s...", filename, HF_MODEL_REPO)
            hf_hub_download(
                HF_MODEL_REPO,
                filename,
                local_dir=self._onnx_dir,
            )

    def _load_onnx(self) -> onnx.ModelProto:
        model_path = self._onnx_dir / MODEL_COMPONENTS["model"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"WashingBERT ONNX model not found at '{model_path}'. "
                f"Download failed or use --onnx-source-dir to point to a local copy."
            )
        self._logger.info("Loading ONNX model from '%s'", model_path)
        return onnx.load(model_path)

    def _make_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Replace dynamic dimensions with fixed values for NPU compatibility."""
        graph = gs.import_onnx(model)

        static_shapes: Final[dict[str, list[int]]] = {
            "input_ids": [1, self._max_seq_len],
            "attention_mask": [1, self._max_seq_len],
        }

        for inp in graph.inputs:
            if inp.name in static_shapes:
                old_shape = inp.shape
                inp.shape = static_shapes[inp.name]
                self._logger.info(
                    "Fixed input '%s' shape: %s -> %s", inp.name, old_shape, inp.shape
                )
            elif inp.name == "token_type_ids":
                old_shape = inp.shape
                inp.shape = [1, self._max_seq_len]
                self._logger.info(
                    "Fixed input '%s' shape: %s -> %s", inp.name, old_shape, inp.shape
                )

        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()

        new_model = gs.export_onnx(graph)
        new_model.ir_version = model.ir_version
        return new_model

    def optimize_model(self, model_path: str | os.PathLike):
        optimized = optimize_model(
            str(model_path),
            model_type="bert",
            num_heads=0,
            hidden_size=0,
            only_onnxruntime=True,
            verbose=False,
        )
        optimized.save_model_to_file(str(model_path))
        optimized_model = onnx.load(model_path)
        optimized_model = onnx.shape_inference.infer_shapes(
            optimized_model, check_type=True, strict_mode=True, data_prop=False
        )
        onnx.save(optimized_model, model_path)

    def export_onnx(self, validate: bool = True):
        self._logger.info("Making model static (max_seq_len=%d)...", self._max_seq_len)
        static_model = self._make_static(self._model)

        self._logger.info("Checking model...")
        static_model = self.check_model(static_model)

        self._export_path = self._export_dir / MODEL_COMPONENTS["model"]
        onnx.save(static_model, self._export_path)

        if not self._no_optimize:
            self._logger.info("Optimizing model...")
            self.optimize_model(self._export_path)

        self.check_model(onnx.load(self._export_path))

        self._logger.info("Verifying static shapes...")
        dynamic_shapes = check_dynamic_shapes(onnx.load(self._export_path))
        if dynamic_shapes:
            raise ValueError(
                f"Model still has dynamic shapes: {json.dumps(dynamic_shapes)}"
            )

        if self._show_model_info:
            print(f"\n\nInfo for model '{self._export_path}':")
            print_onnx_model_inputs_outputs_info(self._export_path)
            print(f"\nModel ops summary:")
            print(
                json.dumps(
                    get_model_ops_count(onnx.load(self._export_path)), indent=4
                ),
                end="\n\n",
            )

        self._logger.info("Saved model to '%s'", self._export_path)

        self._copy_label_files()

        if validate and not self._skip_validation:
            self.validate_onnx()

    def _copy_label_files(self):
        """Copy label JSON files alongside the exported model."""
        for filename in LABEL_FILES.values():
            src = self._onnx_dir / filename
            if src.exists():
                dst = self._export_dir / filename
                shutil.copy2(src, dst)
                self._logger.info("Copied label file '%s' -> '%s'", src, dst)

    def validate_onnx(self, n_iters: int = 3):
        """Validate exported model produces same outputs as source."""
        source_runner = WashingBERTRunner.from_onnx(
            self._onnx_dir / MODEL_COMPONENTS["model"],
            max_seq_len=self._max_seq_len,
        )
        export_runner = WashingBERTRunner.from_onnx(
            self._export_path,
            max_seq_len=self._max_seq_len,
        )

        rng = np.random.default_rng(42)
        for i in range(n_iters):
            seq_len = rng.integers(1, self._max_seq_len + 1)
            input_ids = rng.integers(0, 30000, size=(1, seq_len), dtype=np.int64)
            attention_mask = np.ones((1, seq_len), dtype=np.int64)

            source_outputs = source_runner.run_raw(input_ids, attention_mask)
            export_outputs = export_runner.run_raw(input_ids, attention_mask)

            all_close = all(
                np.allclose(s, e, rtol=1e-3, atol=1e-4)
                for s, e in zip(source_outputs, export_outputs)
            )
            status = "Validation successful" if all_close else "WARNING: output mismatch"
            self._logger.info(
                "(ONNX-validation) [iter %d, seq_len=%d]: %s", i, seq_len, status
            )

    def convert_models(self, convert_dir: str | os.PathLike | None = None):
        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
            return
        if not self._export_path:
            raise RuntimeError("Must run export_onnx() before convert_models()")

        convert_dir = Path(convert_dir) if convert_dir else (
            self._models_dir / "WashingBERT" / "export" / "onnx" / "converted"
        )
        convert_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info("Converting model to bf16...")
        converted_path = convert_dir / self._export_path.name
        convert_model(self._export_path, converted_path, "bf16")
        self._logger.info("Converted model saved to '%s'", converted_path)

        self._logger.info("Converting model int types to int32...")
        convert_model(converted_path, converted_path, "int32")
        self._logger.info("Successfully converted model @ '%s'", converted_path)
        self._export_path = converted_path

    def export_iree(
        self,
        iree_export_dir: str | os.PathLike | None = None,
        iree_compile_args: list[str] | None = None,
        use_iree_cli: bool = False,
    ):
        if not self._export_path:
            raise RuntimeError("Must run export_onnx() before export_iree()")

        iree_export_dir = Path(iree_export_dir) if iree_export_dir else (
            self._models_dir
            / "WashingBERT"
            / "export"
            / "iree"
            / ("converted" if self._convert_dtypes else self._model_dtype)
        )

        onnx_path = self._export_path
        self._logger.info("Exporting model @ '%s' to IREE...", onnx_path)
        model = onnx.load(onnx_path)
        graph = gs.import_onnx(model)
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        model = gs.export_onnx(graph)
        self.check_model(model)
        onnx.save(model, onnx_path)
        export_iree(
            onnx_path,
            iree_export_dir,
            opset=get_model_opset(model),
            compiler_args=iree_compile_args,
            use_iree_cli=use_iree_cli,
        )
        self._logger.info(
            "Successfully exported '%s/%s.vmfb'", iree_export_dir, onnx_path.stem
        )


def export_washingbert_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = WashingBERTModelExporter(
        model_dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        no_optimize=getattr(args, "no_optimize", False),
        convert_dtypes=args.convert_dtypes,
        skip_validation=args.skip_validation,
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models()
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))


def main():
    parser = argparse.ArgumentParser(description="Export WashingBERT to Torq")
    add_washingbert_export_args(parser)
    export_washingbert_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
