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
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoConfig
from torq.utils.logging import (
    configure_logging,
)

from . import add_smollm2_export_args
from ._graph import SmolLM2OnnxGraphEditor
from ._inference import SmolLM2Dynamic, SmolLM2Static
from ...utils.onnx import (
    get_model_opset,
    get_model_ops_count,
    print_onnx_model_inputs_outputs_info,
    check_dynamic_shapes,
)

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
        max_gen_tokens: int = 64,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtype: str | None = None,
        **edit_args
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        if model_size not in ["135M", "360M", "1.7B"]:
            raise ValueError(
                f"Invalid model size '{model_size}', choose one of: ['135M', '360M', '1.7B']"
            )

        self._instruct_model = instruct_model
        self._extract_embeddings = extract_embeddings
        self._static_models = static_models
        self._max_gen_tokens = max_gen_tokens
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._convert_dtype = convert_dtype
        self._onnx_export_dtype = _FP_EXPORT_DTYPE_MAPPING.get(
            "fp32",
            onnx.TensorProto.FLOAT
        )
        self._hf_repo = f"HuggingFaceTB/SmolLM2-{model_size}"
        if self._instruct_model:
            self._hf_repo += "-Instruct"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        if onnx_source_dir and (onnx_source_dir := Path(onnx_source_dir)).exists():
            self._onnx_dir = onnx_source_dir
        else:
            self._onnx_dir = self._models_dir / self._hf_repo / "source" / "fp32"
            self._onnx_dir.mkdir(parents=True, exist_ok=True)
            self._optimum_export_model()
        
        self._model: onnx.ModelProto = self._load_onnx()
        self._export_dir = self._models_dir / self._hf_repo / "export" / "fp32" / ("static" if self._static_models else "dynamic")
        if self._export_dir.exists():
            shutil.rmtree(self._export_dir, ignore_errors=True)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._export_path: Path | None = None

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
                        "--model", f"{self._hf_repo}",
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

    def _make_model_static(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """
        Make model static by replacing dynamic dimensions with fixed values and applying necessary transformations.

        Replaces KV caching and other dynamic operations with static equivalents.

        Args:
            model (onnx.ModelProto): ONNX decoder model to modify

        Returns:
            onnx.ModelProto: The modified decoder model with static dimensions and transformations applied

        Raises:
            ValueError: If an unexpected dynamic dimension is found in the model inputs, outputs, or nodes
        """

        graph: gs.Graph = gs.import_onnx(model)
        self._logger.debug(
            "Set export data type to %s for model data type fp32",
            onnx.helper.tensor_dtype_to_string(self._onnx_export_dtype),
        )
        
        editor = SmolLM2OnnxGraphEditor(graph, self._onnx_export_dtype)
        editor.fix_io(self._max_gen_tokens)

        # Remove isNaN ops
        editor.remove_isNaN()

        for inp in graph.inputs:
            if inp.name == "position_ids":
                cur_len_2d = inp
                break
        else:
            raise ValueError("Position ID input ('position_ids') not found in graph")
        cur_len = graph.layer(
            name="current_len_to_1d",
            op="Squeeze",
            inputs=[cur_len_2d, [0]],
            outputs=[gs.Variable(cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1])],
        )[0]

        (
            editor
            # Replace dynamic KV cache
            .replace_dynamic_kv_cache(cur_len, self._max_gen_tokens)
            # Add causal attention score mask
            .mask_future_attn_scores(cur_len, self._max_gen_tokens)
            # Replace dynamic sequence length getter with `cur_len`
            .add_curr_len_input(cur_len)
            # Replace dynamic index computation `Range(start, start + 1, 1) -> index`
            .convert_to_static_index()
        )

        new_model = editor.to_onnx(override_ir=model.ir_version)
        return new_model

    def _patch_static_model(self, model_path: str | os.PathLike):
        model = onnx.load(model_path)
        editor = SmolLM2OnnxGraphEditor.from_onnx(model, self._onnx_export_dtype)

        # Fold MatMul A @ B where B is a scalar into Mul
        editor.fold_scalar_matmul()
        # Broadcast op inputs to match output shape
        if self._broadcast_ops is not None:
            editor.broadcast_op_inputs(
                ops=self._broadcast_ops,
            )

        if self._extract_embeddings:
            # Extract token embeddings LUT
            embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
            embeddings_inp = "token_embedding"
            editor.extract_token_embeddings(
                self._hidden_size,
                self._vocab_size,
                embeddings_npy,
                inp_name=embeddings_inp
            )
            editor.reorder_graph_input(embeddings_inp, 0)

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def make_static(self):
        self._logger.info("(decoder_with_past) Making graph static...")
        self._model = self.check_model(self._model)
        self._model = self._make_model_static(self._model)

    def optimize_model(self, model_path: str | os.PathLike):
        optimized = optimize_model(
            str(model_path),
            model_type="bert",
            num_heads=self._config.num_attention_heads,
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

    def apply_post_static_patches(self, model_path: str | os.PathLike):
        self._patch_static_model(model_path)

    def export_onnx(self, validate: bool = True):
        if self._static_models:
            self.make_static()


        self._export_path = self._export_dir / f"model.onnx"
        self._logger.info("(decoder_with_past) Checking model...")
        self._model = self.check_model(self._model)
        onnx.save(self._model, self._export_path)
        self._logger.info("(decoder_with_past) Optimizing model...")
        self.optimize_model(self._export_path)
        if self._static_models:
            self._logger.info("(decoder_with_past) Applying post-static conversion patches...")
            self.apply_post_static_patches(self._export_path)
        self.check_model(onnx.load(self._export_path))
        if self._static_models:
            self._logger.info("(decoder_with_past) Verifying static shapes...")
            dynamic_shapes = check_dynamic_shapes(onnx.load(self._export_path))
            if dynamic_shapes:
                raise ValueError(
                    f"Model 'decoder_with_past' still has dynamic shapes: {json.dumps(dynamic_shapes)}"
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
        self._logger.info("(decoder_with_past) Saved model to '%s'", str(self._export_path))

        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 5):
        # simple dataset to test functional equivalence
        prompts = [
            # very short (position_ids = 0 edge case)
            "Hello",

            # normal medium-length prompt
            "The quick brown fox jumps over the lazy dog.",

            # repetitive tokens (attention accumulation / stability)
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",

            # non-ASCII / multi-token UTF-8
            "こんにちは世界",

            # structured / punctuation-heavy (tokenizer edge cases)
            "def foo(x): return x * 2 # simple test"
        ]
        n_threads: int = os.cpu_count()

        if self._static_models:
            runner = SmolLM2Static.from_onnx(
                self._export_path,
                self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo
            )
        else:
            runner = SmolLM2Dynamic.from_onnx(
                self._export_path,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo
            )
        val_runner = SmolLM2Dynamic.from_onnx(
            self._onnx_dir /  "model.onnx",
            n_threads=n_threads,
            instruct_model=self._instruct_model,
            repo_id=self._hf_repo
        )

        for i in range(n_iters):
            if i >= len(prompts):
                self._logger.warning("(ONNX-validation) No more samples to validate, stopping")
                break
        
            input = prompts[i]
            output = runner.run(input)
            val_output = val_runner.run(input)
            if output != val_output:
                result = f"Warning: Validation failed, mismatched outputs\nExpected:\n{val_output},\nGenerated:\n{output}"
            else:
                result = f"Validation successful, identical outputs"
            self._logger.info(
                "(ONNX-validation) [iter %d, %.3f ms]: %s",
                i,
                runner.last_infer_time / 1e6,
                result
            )
        self._logger.info(
            "(ONNX-validation) Avg. inference time: %.3f ms",
            runner.avg_infer_time / 1e6
        )


def export_smollm2_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = SmolLM2ModelExporter(
        args.model_size,
        args.instruct_model,
        args.extract_embeddings,
        not args.dynamic_models,
        max_gen_tokens=args.max_gen_tokens,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtype=args.convert_dtype,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.export_onnx(validate=not args.skip_validation)


def main():
    parser = argparse.ArgumentParser(description="Export SmolLM2 to Torq")
    add_smollm2_export_args(parser)
    export_smollm2_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
