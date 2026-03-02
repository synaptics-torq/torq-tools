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
import ast
import time

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from torq.compile import (
    process_iree_args,
    export_iree
)
from torq.utils.logging import (
    configure_logging,
)

from . import (
    ONNX_DTYPES,
    OPTIMUM_DTYPES,
    add_fumi_export_args,
)

from ._graph import FumiOnnxGraphEditor
from ._inference import MoonshineDynamic, MoonshineStatic
from ...utils.onnx import (
    get_model_opset,
    get_model_ops_count,
    print_onnx_model_inputs_outputs_info,
    check_dynamic_shapes,
)
from ...tools.convert_dtype.onnx import (
    convert_model
)

_FP_EXPORT_DTYPE_MAPPING: Final[dict] = {
    "float": onnx.TensorProto.FLOAT,
    "fp32" : onnx.TensorProto.FLOAT,
    "fp16" : onnx.TensorProto.FLOAT16,
    "bf16" : onnx.TensorProto.BFLOAT16
}

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.FloorDiv, ast.Div, ast.Mod, ast.USub, ast.UAdd, ast.Constant, ast.Name
)

def _safe_eval_dim_expr(expr: str, vars: dict[str, int]) -> int:
    """
    Safely eval expressions like '20*u0 + 1' using only + -
    """
    if "unk" in expr: 
        return "unknown"

    if '+' in expr:
        op = '+'
    elif '-' in expr:
        op = '-'
    else:
        op = None

    expression = expr.split(op) if op is not None else [expr,0]
    var, const = expression
    var = var.split('*')
    varValue = int(var[0]) * int(vars[var[1]])
    varValue += int(const)
    
    return varValue

def force_all_dim_params_to_values(model: onnx.ModelProto, vars: dict[str, int], unknown) -> onnx.ModelProto:
    def _fix_shape(shape, name):
        if shape is None:
            return
        
        # Iterate over current tensor dimensions (dim_param and dim_value)
        for d in shape.dim:
            if not d.dim_param:
                continue

            s = d.dim_param.strip()
            s = s.replace(" ", "")
            
            # Set a value when the shape is only the variable. e.g. 's26' or 'u0'
            if s in vars:
                d.dim_value = int(vars[s])
                d.ClearField("dim_param")
                continue

            try:
                # Evaluate expressions like '300*u0 + 1' and set the resulting value to the tensor
                expression = _safe_eval_dim_expr(s, vars)
                if  expression != "unknown":
                    d.dim_value = expression
                    d.ClearField("dim_param")
                    continue

                # Below are tensors with unknown (unk__) shapes, most of them take a value 
                # when onnx.shape_inference.infer_shapes is executed, but below is for
                # tensors that have a value directly related with the output (u0).
                
                # This tensor dimension exptects a value that is u0 * 300
                if name == 'val_13092':
                    d.dim_value = 300 * int(vars['u0']) 
                    d.ClearField("dim_param")
                    continue

                # Rest of tensors expects the output value (u0).
                if unknown == True:
                    d.dim_value = int(vars['u0'])
                    d.ClearField("dim_param")

            except Exception:
                # leave it if we can't resolve it
                pass

    g = model.graph
    # Iterate over each tensor
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            _fix_shape(tt.shape, vi.name)

    return model


class FumiModelExporter:
    COMPONENTS = {
        "model": "fumi_f_ja.onnx"
    }

    def __init__(
        self,
        model_dtype: str = "float",
        static_models: bool = True,
        *,
        text_len: int,
        audio_u0: int,
        onnx_model: str,
        models_dir: str | os.PathLike = "models",
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._enable_ort_opt = False

        self._text_len = int(text_len)
        self._audio_u0 = int(audio_u0)
        self._audio_u3 = int(audio_u0)
        self._onnx_model_path = Path(onnx_model)
        if not self._onnx_model_path.exists():
            raise ValueError(f"--onnx-model not found: {self._onnx_model_path}")

        self._model_dtype = model_dtype
        self._static_models = static_models
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._convert_dtypes = convert_dtypes
        self._onnx_export_dtype = _FP_EXPORT_DTYPE_MAPPING.get(
            self._model_dtype,
            onnx.TensorProto.FLOAT
        )
        self._skip_export = set(skip_export or [])
        self._hf_repo = "fumi_f_ja"

        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        self._components = self._load_onnx()

        self._export_dir = (
            self._models_dir
            / "fumi_f_ja"
            / "export"
            / "onnx"
            / self._model_dtype
            / ("static" if self._static_models else "dynamic")
        )
        if self._export_dir.exists():
            shutil.rmtree(self._export_dir, ignore_errors=True)

        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._export_paths: dict[str, Path] = {}

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

    def _optimum_export_models(self):
        if not all(
            (self._onnx_dir / comp_model_name).exists()
            for comp_model_name in FumiModelExporter.COMPONENTS.values()
        ):
            try:
                check_output(
                    [
                        "optimum-cli", "export", "onnx",
                        str(self._onnx_dir),
                        "--model", f"{self._hf_repo}-{self._model_size}",
                        "--dtype", self._model_dtype,
                        "--opset", "17",
                    ],
                    text=True,
                    stderr=STDOUT,
                )
            except CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                    + "\n    ".join(e.output.strip().splitlines())
                ) from None


    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        return {"model": onnx.load(self._onnx_model_path)}

    def _make_fumi_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = FumiOnnxGraphEditor.from_onnx(model, "model", self._onnx_export_dtype)

        # Convert I/O to static
        editor.fix_fumi_io(self._text_len, self._audio_u0, self._audio_u3)
        static_model = editor.to_onnx(override_ir=model.ir_version)

        # Populate value_info as much as possible before rewriting
        static_model = onnx.shape_inference.infer_shapes(static_model, check_type=True, strict_mode=False, data_prop=True)

        # Now eliminate remaining symbolic dim_params everywhere (including "20*u0 + 1")
        static_model = force_all_dim_params_to_values(static_model, {
            "s26": self._text_len,
            "u0": self._audio_u0,
            "u3": self._audio_u3,
        }, unknown = False)

        # Infer unknown shapes (unk__) after setting a value to variables (s26, u0, u3)
        static_model = onnx.shape_inference.infer_shapes(static_model, check_type=True, strict_mode=False, data_prop=True)

        # Set a value to remaining unknown shapes (unk__)
        static_model = force_all_dim_params_to_values(static_model, {
            "s26": self._text_len,
            "u0": self._audio_u0,
            "u3": self._audio_u3,
        }, unknown = True)

        return static_model


    def _replace_int_to_bf16_casts(self, model_path: str | os.PathLike, component: str):
        model = onnx.load(model_path)
        editor = FumiOnnxGraphEditor.from_onnx(model, component, self._onnx_export_dtype)

        # Repalce potentially unsupported int64 -> float cast with lookup table
        editor.replace_int64_float_cast(max_int=self._max_tokens)

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self._make_fumi_model_static(self._components["model"])

    def optimize_model(self, model_path: str | os.PathLike, component: str):
        if not self._enable_ort_opt:
            self._logger.info("(%s) Skipping ORT optimization", component)
            return
        
        optimized = optimize_model(
            str(model_path),
            model_type="bert",
            hidden_size=self._config.hidden_size,
            only_onnxruntime=True,
            verbose=False,
        )
        optimized.save_model_to_file(str(model_path))
        optimized_model = onnx.load(model_path)
        optimized_model = onnx.shape_inference.infer_shapes(
            optimized_model, check_type=True, strict_mode=True, data_prop=False
        )
        onnx.save(optimized_model, model_path)

        m = onnx.load(model_path)
        m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=False)
        onnx.save(m, model_path)

    def export_onnx(self, validate: bool = True):
        if self._static_models:
            self.make_static()

        for comp, model in self._components.items():
            if comp in self._skip_export:
                self._logger.info("Skipping export of component %s", comp)
                continue
            self._export_paths[comp] = self._export_dir / f"{comp}.onnx"
            self._logger.info("(%s) Checking model...", comp)
            model = self.check_model(model, skip_data_prop="decoder" in comp and self._merged_decoder)
            onnx.save(model, self._export_paths[comp])
            self._logger.info("(%s) Optimizing model...", comp)
            self.optimize_model(self._export_paths[comp], comp)
 
            if self._static_models:
                self._logger.info("(%s) Verifying static shapes...", comp)
                dynamic_shapes = check_dynamic_shapes(onnx.load(self._export_paths[comp]))
                if dynamic_shapes:
                    raise ValueError(
                        f"Model '{comp}' still has dynamic shapes: {json.dumps(dynamic_shapes)}"
                    )
            if self._show_model_info:
                print(f"\n\nInfo for model '{self._export_paths[comp]}':")
                print_onnx_model_inputs_outputs_info(self._export_paths[comp])
                print(f"\nModel ops summary:")
                print(
                    json.dumps(
                        get_model_ops_count(onnx.load(self._export_paths[comp])), indent=4
                    ),
                    end="\n\n",
                )
            self._logger.info("(%s) Saved model to '%s'", comp, str(self._export_paths[comp]))

        if validate:
            if self._skip_export:
                self._logger.warning("Skipping validation as components %s have not been exported", str(self._skip_export))
            else:
                self.validate_onnx()

    def validate_onnx(self, n_iters: int = 5):

        def _sample_input(idx: int) -> np.ndarray:
            sample = dataset[idx]["audio"]
            inputs: np.ndarray = processor(
                sample["array"],
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="np",
            )
            return inputs["input_values"]

        if self._static_models:
            runner = MoonshineStatic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                decoder_model=self._export_dir / "decoder.onnx",
                decoder_with_past_model=self._export_dir / "decoder_with_past.onnx",
                model_size=self._model_size,
                preprocessor_model=self._export_dir / "preprocessor.onnx" if self._split_encoder else None
            )
        else:
            runner = MoonshineDynamic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                decoder_model=self._export_dir / "decoder_merged.onnx",
                model_size=self._model_size
            )
        val_runner = MoonshineDynamic.from_onnx(
            encoder_model=self._onnx_dir / "encoder_model.onnx",
            decoder_model=self._onnx_dir / "decoder_model_merged.onnx",
            model_size=self._model_size,
            max_inp_len=runner.max_inp_len
        )

        processor = AutoProcessor.from_pretrained(f"{self._hf_repo}-{self._model_size}")
        dataset = load_dataset(
            path="hf-internal-testing/librispeech_asr_dummy",
            name="clean",
            split="validation",
        )
        dataset = dataset.cast_column(
            "audio", Audio(processor.feature_extractor.sampling_rate)
        )
        self._logger.debug("(ONNX-validation) Loaded dataset 'hf-internal-testing/librispeech_asr_dummy', details: %s", str(dataset))

        for i in range(n_iters):
            if i >= len(dataset):
                self._logger.warning("(ONNX-validation) No more samples to validate, stopping")
                break

            input = _sample_input(i)
            tokens = runner.run(input)
            val_tokens = val_runner.run(input)
            if not np.array_equal(tokens, val_tokens):
                result = f"Warning: Validation failed, mismatched outputs\nExpected:\n{val_tokens},\nGenerated:\n{tokens}"
            else:
                result = f"Validation successful, identical outputs"
            self._logger.info(
                "(ONNX-validation) [iter %d, %.3f ms]: %s",
                i,
                runner.last_infer_time * 1000,
                result
            )
        self._logger.info(
            "(ONNX-validation) Avg. inference time: %.3f ms",
            runner.avg_infer_time * 1000
        )

    def convert_models(self, convert_dir: str | os.PathLike | None = None):
        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
        convert_dir = convert_dir or (
            self._models_dir
            / self._hf_repo
            / "export"
            / "onnx"
            / self._model_size
            / "converted"
            / ("static" if self._static_models else "dynamic")
        )
        for comp, model_path in self._export_paths.items():
            if comp == "preprocessor":
                shutil.copy2(model_path, convert_dir)
                continue
            self._logger.info("(ONNX-convert) Converting model '%s' to dtype bf16...", str(model_path))
            converted_model_path = convert_dir / model_path.name
            convert_model(model_path, converted_model_path, "bf16")
            self._logger.info("(ONNX-convert) Successfully converted model to dtype bf16 @ '%s'", str(converted_model_path))
            self._logger.info("(ONNX-convert) Converting model '%s' to dtype int32...", str(model_path))
            convert_model(converted_model_path, converted_model_path, "int32")
            self._logger.info("(ONNX-convert) Successfully converted model to dtype int32 @ '%s'", str(converted_model_path))
            self._export_paths[comp] = converted_model_path
            self._logger.debug("(ONNX-convert) Update %s model path to '%s'", comp, str(converted_model_path))

    def export_iree(
        self,
        iree_export_dir: str | os.PathLike | None = None,
        iree_compile_args: list[str] | None = None,
        use_iree_cli: bool = False,
    ):
        iree_export_dir = iree_export_dir or (
            self._models_dir
            / self._hf_repo
            / "export"
            / "iree"
            / self._model_size
            / ("converted" if (self._convert_dtypes and self._model_dtype == "float") else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )
        for comp, onnx_path in self._export_paths.items():
            if comp == "preprocessor":
                continue
            if (self._model_dtype == "bf16" or self._convert_dtypes) and self._replace_int_bf16_cast:
                self._replace_int_to_bf16_casts(onnx_path, comp)
            self._logger.info("(IREE-export) Exporting %s model @ '%s' to IREE...", comp, str(onnx_path))
            model = onnx.load(onnx_path)
            graph = gs.import_onnx(model)
            graph.name = "main"
            graph.cleanup(
                remove_unused_graph_inputs=True, remove_unused_node_outputs=True
            ).toposort()
            model = gs.export_onnx(graph)
            self.check_model(model, skip_data_prop="decoder" in comp and self._merged_decoder)
            onnx.save(model, onnx_path)
            export_iree(
                onnx_path,
                iree_export_dir,
                opset=get_model_opset(model),
                compiler_args=iree_compile_args,
                use_iree_cli=use_iree_cli
            )
            self._logger.info("(IREE-export) Successfully exported '%s/%s.vmfb'", str(iree_export_dir), onnx_path.stem)


def export_fumi_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = FumiModelExporter(
        args.dtype,
        not args.dynamic_models,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        use_optimum=args.use_optimum,
        convert_dtypes=args.convert_dtypes,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops,
        text_len=args.text_len,
        audio_u0=args.audio_len,
        onnx_model=args.onnx_model,
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models()
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))

def main():
    parser = argparse.ArgumentParser(description="Export Fumi to Torq")

    add_fumi_export_args(parser)
    export_fumi_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
