# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import onnx_graphsurgeon as gs
from transformers import AutoConfig
import json

from tokenizers import Tokenizer

from ._trim_vocab import build_qwen_trimmed_vocab_spec
from . import add_qwen_export_args
from ._graph import QwenOnnxGraphEditor
from ._inference import QwenDynamic, QwenStatic
from ...utils.compile import export_torq as compile_torq_model
from ...tools.convert_dtype.onnx import (
    FP32Converter,
    Int64Converter,
)
from ...tools.quantization.weight_quantization.quantize import WeightQuantizer
from ...tools.quantization.weight_quantization.config import QuantizationConfig
from ...model_export.hf import (
    hf_download_source_model,
    optimum_export_onnx,
)
from ...model_export.onnx import (
    ORTOptimizerConfig,
    OnnxModelExporterBase,
)
from ...utils.logging import configure_logging


DEFAULT_HF_REPO = "Qwen/Qwen3-0.6B"

class _QwenLargeModelConverterMixin:
    """
    Dtype conversion path for Qwen's multi-gigabyte external-data ONNX model.

    The shared dtype converter performs ONNX in-memory shape inference before
    and after conversion. Qwen3-0.6B exceeds protobuf's serialization limit
    when that path is used.

    Qwen's static model has already passed strict static-shape verification,
    so conversion can operate directly on the GraphSurgeon graph. Path-based
    shape inference is performed after the converted model is saved.
    """

    def convert_model(
        self,
        input_model: onnx.ModelProto,
    ) -> onnx.ModelProto:
        root, *subgraphs = self._collect_all_graphs(
            gs.import_onnx(input_model)
        )

        for graph in subgraphs:
            self._fold_literal_constants(graph)
            self._convert_graph(graph)
            self._update_inputs(graph)
            self._update_outputs(graph)
            self._check_conversion(graph)

        self._fold_literal_constants(root)
        self._convert_graph(root)
        self._update_inputs(root)
        self._update_outputs(root)
        self._check_conversion(root)

        converted_model = gs.export_onnx(root)
        converted_model.ir_version = input_model.ir_version

        return converted_model


class _QwenFP32Converter(
    _QwenLargeModelConverterMixin,
    FP32Converter,
):
    pass


class _QwenInt64Converter(
    _QwenLargeModelConverterMixin,
    Int64Converter,
):
    pass

class _QwenDQLOnlyQuantizer(WeightQuantizer):
    """Insert quantized DQL weights without running the generic dtype converters."""

    def _convert_non_weight_to_bf16(self, model) -> None:
        # Qwen must use its large-model-safe conversion pipeline afterward.
        pass


class QwenModelExporter(OnnxModelExporterBase):
    """Export Qwen models to static ONNX and Torq artifacts."""

    def __init__(
        self,
        static_models: bool = True,
        *,
        hf_repo: str = DEFAULT_HF_REPO,
        hf_repo_subdir: str | os.PathLike | None = None,
        max_gen_tokens: int = 256,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        keep_individual_kv_io: bool = False,
        split_lm_head: bool = False,
        split_model: bool = False,
        weight_quantization: str | None = None,
    ):
        self._hf_repo = hf_repo
        self._hf_repo_subdir = hf_repo_subdir
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = onnx_source_dir
        self._keep_individual_kv_io = keep_individual_kv_io
        self._split_lm_head = split_lm_head
        self._split_model = split_model
        self._weight_quantization = weight_quantization

        if self._split_model and not static_models:
            raise ValueError(
                "`--split-model` is currently supported only for static Qwen exports"
            )
        
        if self._weight_quantization is not None and not self._split_model:
            raise ValueError(
                "`--weight-quantization` currently requires `--split-model`"
            )

        if self._onnx_source_dir:
            local_config_dir = Path(self._onnx_source_dir)
        else:
            local_config_dir = None

        if (
            local_config_dir is not None
            and (local_config_dir / "config.json").exists()
        ):
            self._config = AutoConfig.from_pretrained(
                local_config_dir,
                local_files_only=True,
            )
        else:
            try:
                self._config = AutoConfig.from_pretrained(
                    self._hf_repo,
                    local_files_only=True,
                )
            except OSError:
                self._config = AutoConfig.from_pretrained(self._hf_repo)

        super().__init__(
            "fp32",
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs={
                "model": ORTOptimizerConfig(
                    num_heads=self._config.num_attention_heads,
                    hidden_size=self._config.hidden_size,
                )
            },
        )

    def _setup_dirs(self) -> list[Path]:
        if (
            self._onnx_source_dir
            and Path(self._onnx_source_dir).exists()
        ):
            onnx_dir = Path(self._onnx_source_dir)
        else:
            onnx_dir = self._models_dir / "source" / self._model_dtype
            onnx_dir.mkdir(parents=True, exist_ok=True)

            try:
                hf_download_source_model(
                    self._hf_repo,
                    "model.onnx",
                    onnx_dir,
                    subfolder=self._hf_repo_subdir,
                    peripheral_files=[
                        "config.json",
                        "generation_config.json",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ],
                )
                self._logger.info(
                    "Downloaded Qwen ONNX model from %s",
                    self._hf_repo,
                )
            except Exception:
                optimum_export_onnx(
                    onnx_dir,
                    self._hf_repo,
                    self._model_dtype,
                    ["model.onnx"],
                    opt_level=None,
                )
                self._logger.info(
                    "Exported %s to ONNX at '%s'",
                    self._hf_repo,
                    onnx_dir,
                )

            if self._hf_repo_subdir:
                onnx_dir /= Path(self._hf_repo_subdir)

        export_dir = (
            self._models_dir
            / "export"
            / "onnx"
            / self._model_dtype
            / ("static" if self._static_models else "dynamic")
        )

        convert_dir = (
            self._models_dir
            / "export"
            / "onnx"
            / "converted"
            / ("static" if self._static_models else "dynamic")
        )

        torq_dir = (
            self._models_dir
            / "export"
            / "torq"
            / ("converted" if self._convert_dtypes else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )

        return onnx_dir, export_dir, convert_dir, torq_dir

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        model_path = self._onnx_dir / "model.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected Qwen model at '{model_path}'"
            )

        model = onnx.load(model_path)
        original_ir_version = model.ir_version

        graph = gs.import_onnx(model)
        graph.name = "main"

        graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True,
        ).toposort()

        model = gs.export_onnx(graph)
        model.ir_version = original_ir_version

        return {"model": model}
    
    def optimize_model(self,
        model_path: str | os.PathLike,
        opt_config,
    ) -> None:
        """
        Skip ONNX Runtime optimization for the initial Qwen exporter.

        ONNX Runtime currently reloads and saves the complete multi-gigabyte
        Qwen model as one protobuf message, which exceeds the serialization
        limit. The static graph will be tested without this optional stage.
        """
        del model_path, opt_config
        self._logger.info(
            "(model) Skipping ONNX Runtime optimization for large external-data model"
        )

    def _copy_runtime_assets(
        self,
        destination_dir: str | os.PathLike,
        source_dir: str | os.PathLike | None = None,
    ) -> None:
        source_dir = Path(source_dir or self._onnx_dir)
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        for asset_name in (
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            source_path = source_dir / asset_name

            if source_path.exists():
                shutil.copy2(
                    source_path,
                    destination_dir / asset_name,
                )

    def _make_model_static(
        self,
        model: onnx.ModelProto,
    ) -> onnx.ModelProto:
        """Replace Qwen dynamic dimensions and decoder state with static forms."""

        graph = gs.import_onnx(model)

        editor = QwenOnnxGraphEditor(
            graph,
            self._onnx_export_dtype,
        )

        editor.fix_io(self._max_gen_tokens)
        editor.remove_redundant_casts()

        position_ids = next(
            (
                graph_input
                for graph_input in graph.inputs
                if graph_input.name == "position_ids"
            ),
            None,
        )

        if position_ids is None:
            position_ids = gs.Variable(
                "position_ids",
                dtype=np.int64,
                shape=[1, 1],
            )
            graph.inputs.append(position_ids)
        else:
            position_ids.dtype = np.int64
            position_ids.shape = [1, 1]

        current_length = graph.layer(
            name="current_len_to_1d",
            op="Squeeze",
            inputs=[position_ids, [0]],
            outputs=[
                gs.Variable(
                    "position_ids_squeezed",
                    dtype=np.int64,
                    shape=[1],
                )
            ],
        )[0]

        (
            editor
            .replace_dynamic_kv_cache(
                current_length,
                self._max_gen_tokens,
            )
            .mask_future_attn_scores(
                current_length,
                self._max_gen_tokens,
            )
            .add_curr_len_input(current_length)
            .convert_to_static_index()
        )

        editor.resolve_static_shape_inputs()

        static_model = editor.to_onnx(
            override_ir=model.ir_version,
        )

        convert_model_to_external_data(
            static_model,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )

        return static_model

    def _patch_static_model(
        self,
        model_path: str | os.PathLike,
    ) -> None:
        model = onnx.load(model_path)

        editor = QwenOnnxGraphEditor.from_onnx(
            model,
            self._onnx_export_dtype,
        )

        if self._split_model:
            embeddings_npy = (
                Path(model_path).parent / "token_embeddings.npy"
            )

            editor.extract_token_embeddings(
                1024,
                151936,
                embeddings_npy,
                inp_name="token_embedding",
            )

            token_embedding = editor.graph.tensors()["token_embedding"]
            token_embedding.dtype = np.float32
            token_embedding.shape = [1, 1, 1024]

            editor.reorder_graph_input(
                "token_embedding",
                0,
            )

            if not self._keep_individual_kv_io:
                editor.combine_kv_io_tensors(
                    [
                        1,
                        self._config.num_key_value_heads,
                        self._max_gen_tokens,
                        self._config.head_dim,
                    ]
                )

            self._logger.info(
                "(model) Extracted Qwen token embeddings to '%s'",
                str(embeddings_npy),
            )

        editor.eliminate_transposes()
        editor.collapse_reshape_chains()
        editor.fold_scalar_matmul()

        if self._split_model:
            tokenizer_path = (
                Path(model_path).parent / "tokenizer.json"
            )

            config_path = (
                Path(model_path).parent / "config.json"
            )

            tokenizer = Tokenizer.from_file(
                str(tokenizer_path)
            )

            tokenizer_json = json.loads(
                tokenizer_path.read_text()
            )

            config_json = json.loads(
                config_path.read_text()
            )

            spec = build_qwen_trimmed_vocab_spec(
                tokenizer,
                tokenizer_json,
                config_json,
                selected_groups=("latin", "punct", "digits"),
                byte_fallback=True,
            )

            token_id_lut_path = (
                Path(model_path).parent / "token_id_lut.npy"
            )

            editor.trim_lm_head_vocab(
                kept_token_ids=np.array(
                    list(spec.kept_model_ids)
                    + list(spec.extra_token_ids),
                    dtype=np.int64,
                ),
                save_lut=token_id_lut_path,
            )

            lm_head_path = (
                Path(model_path).parent / "lm_head.onnx"
            )

            lm_head_matmul = next(
                node
                for node in editor.graph.nodes
                if node.op == "MatMul"
                and node.outputs
                and node.outputs[0].name == "logits"
            )

            lm_head_matmul.inputs[0].dtype = np.float32
            lm_head_matmul.inputs[0].shape = [1, 1, 1024]

            editor.split_lm_head(
                lm_head_path
            )

            self._export_paths["lm_head"] = lm_head_path

            self._logger.info(
                "(lm_head) Saved Qwen trimmed LM head to '%s' "
                "(%d tokens)",
                str(lm_head_path),
                spec.trimmed_vocab_size,
            )

        editor.reorder_graph_input("position_ids", 1)
        editor.remove_isNaN()

        editor.resolve_static_shape_inputs()
        editor.replace_current_len_squeeze()

        if self._split_model:
            split_dir = Path(model_path).parent

            part_a_graph, part_b_graph = (
                editor.split_transformer_layers()
            )

            part_a_path = (
                split_dir / "transformer_part_A.onnx"
            )

            part_b_path = (
                split_dir / "transformer_part_B.onnx"
            )

            part_a_model = gs.export_onnx(part_a_graph)
            part_b_model = gs.export_onnx(part_b_graph)

            part_a_model.ir_version = model.ir_version
            part_b_model.ir_version = model.ir_version

            onnx.save(
                part_a_model,
                part_a_path,
            )

            onnx.save(
                part_b_model,
                part_b_path,
            )

            self._export_paths["transformer_part_A"] = (
                part_a_path
            )

            self._export_paths["transformer_part_B"] = (
                part_b_path
            )

            self._logger.info(
                "(split) Saved transformer Part A to '%s'",
                str(part_a_path),
            )

            self._logger.info(
                "(split) Saved transformer Part B to '%s'",
                str(part_b_path),
            )

        patched_model = editor.to_onnx(
            override_ir=model.ir_version,
        )

        patched_model = self.check_model(patched_model)

        convert_model_to_external_data(
            patched_model,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )

        external_data_path = Path(model_path).parent / "model.onnx_data"

        if external_data_path.exists():
            external_data_path.unlink()

        onnx.save_model(
            patched_model,
            model_path,
        )

    def _save_large_external_model(
        self,
        model: onnx.ModelProto,
        model_path: str | os.PathLike,
    ) -> None:
        """
        Save a large Qwen model using one external-data file.

        Remove any previous external-data file before saving so ONNX does not
        append new tensor data to stale contents.
        """

        model_path = Path(model_path)

        external_data_name = f"{model_path.name}_data"

        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
            convert_attribute=False,
        )

        external_data_path = (
            model_path.parent / external_data_name
        )

        if external_data_path.exists():
            external_data_path.unlink()

        onnx.save_model(
            model,
            model_path,
        )

    def _finalize_static_shapes(
        self,
        model_path: str | os.PathLike,
    ) -> None:
        """
        Resolve Qwen internal static shapes using path-based ONNX inference.

        Qwen3-0.6B is too large for ONNX's in-memory shape inference because
        its weights are stored as multi-gigabyte external data.

        A two-pass process is required:

        1. Run path-based ONNX shape inference so intermediate tensor metadata
           becomes available.
        2. Reload the inferred graph without loading external weights and
           materialize statically-computable Expand/Slice shape expressions.
        3. Run path-based shape inference again so the rewritten graph
           propagates concrete shapes through all downstream tensors.

        This avoids in-memory protobuf serialization while still producing a
        graph that passes Torq's strict static-shape verification.
        """

        model_path = Path(model_path)

        first_inferred_path = model_path.with_name(
            f"{model_path.stem}_shape_pass1.onnx"
        )

        resolved_path = model_path.with_name(
            f"{model_path.stem}_shape_resolved.onnx"
        )

        final_inferred_path = model_path.with_name(
            f"{model_path.stem}_shape_final.onnx"
        )

        try:
            self._logger.info(
                "(model) Running first path-based shape inference pass..."
            )

            onnx.shape_inference.infer_shapes_path(
                model_path,
                first_inferred_path,
                data_prop=True,
            )

            inferred_model = onnx.load(
                first_inferred_path,
                load_external_data=False,
            )

            editor = QwenOnnxGraphEditor.from_onnx(
                inferred_model,
                self._onnx_export_dtype,
            )

            editor.resolve_static_shape_inputs()

            resolved_model = editor.to_onnx(
                override_ir=inferred_model.ir_version,
            )

            onnx.save(
                resolved_model,
                resolved_path,
            )

            self._logger.info(
                "(model) Running final path-based shape inference pass..."
            )

            onnx.shape_inference.infer_shapes_path(
                resolved_path,
                final_inferred_path,
                data_prop=True,
            )

            os.replace(
                final_inferred_path,
                model_path,
            )

        finally:
            for temporary_path in (
                first_inferred_path,
                resolved_path,
                final_inferred_path,
            ):
                if temporary_path.exists():
                    temporary_path.unlink()
    
    def check_model(self, model: onnx.ModelProto,) -> onnx.ModelProto:
        """
        Return the model without in-memory shape inference.

        Qwen3-0.6B uses external ONNX weight data and is too large to
        serialize through ONNX's in-memory shape-inference API.
        Path-based validation will be performed after the model is saved.
        """
        return model

    def make_static(self) -> None:
        self._logger.info("(model) Making Qwen graph static...")

        self._components["model"] = self.check_model(
            self._components["model"]
        )

        self._components["model"] = self._make_model_static(
            self._components["model"]
        )

    def apply_post_static_patches(
        self,
        model_path: str | os.PathLike,
        _,
    ) -> None:
        self._copy_runtime_assets(
            Path(model_path).parent,
            self._onnx_dir,
        )

        self._patch_static_model(model_path)

        self._finalize_static_shapes(model_path)

    def validate_onnx(self, n_iters: int = 3) -> None:
        prompts = [
            "Hello",
            "What is machine learning?",
            "Write one sentence about embedded systems.",
        ]

        n_threads = os.cpu_count()

        if self._static_models:
            runner = QwenStatic.from_onnx(
                self._export_paths["model"],
                max_gen_tokens=self._max_gen_tokens,
                n_threads=n_threads,
                repo_id=self._hf_repo,
                combined_kv_io=not self._keep_individual_kv_io,
            )
        else:
            runner = QwenDynamic.from_onnx(
                self._export_paths["model"],
                max_gen_tokens=self._max_gen_tokens,
                n_threads=n_threads,
                repo_id=self._hf_repo,
            )

        reference_runner = QwenDynamic.from_onnx(
            self._onnx_dir / "model.onnx",
            max_gen_tokens=self._max_gen_tokens,
            n_threads=n_threads,
            repo_id=self._hf_repo,
        )

        for iteration, prompt in enumerate(prompts[:n_iters]):
            generated_output = runner.run(prompt)
            reference_output = reference_runner.run(prompt)

            minimum_length = min(
                len(generated_output),
                len(reference_output),
            )

            if (
                generated_output[:minimum_length]
                != reference_output[:minimum_length]
            ):
                result = (
                    "Validation failed.\n"
                    f"Expected:\n{reference_output}\n"
                    f"Generated:\n{generated_output}"
                )
            else:
                result = "Validation successful: outputs match"

                if len(generated_output) != len(reference_output):
                    result += (
                        " for their common prefix "
                        f"({len(generated_output)} vs "
                        f"{len(reference_output)} characters)"
                    )

            self._logger.info(
                "(ONNX-validation) Iteration %d: %s",
                iteration,
                result,
            )

    def _infer_converted_shapes(
        self,
        model_path: str | os.PathLike,
    ) -> None:
        """Run large-model-safe path-based shape inference."""

        model_path = Path(model_path)

        inferred_path = model_path.with_name(
            f"{model_path.stem}_inferred.onnx"
        )

        try:
            onnx.shape_inference.infer_shapes_path(
                model_path,
                inferred_path,
                data_prop=True,
            )

            os.replace(
                inferred_path,
                model_path,
            )

        finally:
            if inferred_path.exists():
                inferred_path.unlink()

    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
    ):
        """
        Convert Qwen fp32 -> bf16 and int64 -> int32 without using
        in-memory ONNX shape inference.

        Qwen3-0.6B uses multi-gigabyte external tensor data, which exceeds
        protobuf serialization limits in the shared converter's shape-
        inference path.
        """

        if not self._convert_dtypes:
            self._logger.warning(
                "Skipping conversion as convert_dtypes==False"
            )
            return

        self._convert_dir = Path(
            convert_dir or self._convert_dir
        )

        if self._convert_dir.exists():
            shutil.rmtree(
                self._convert_dir,
                ignore_errors=True,
            )

        self._convert_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_dir = self._export_paths["model"].parent

        for comp, model_path in self._export_paths.items():
            converted_model_path = (
                self._convert_dir / model_path.name
            )

            self._logger.info(
                "(ONNX-convert) Converting model '%s' to dtype bf16...",
                str(model_path),
            )

            # Populate intermediate tensor dtype metadata before BF16 conversion.
            # Qwen3-0.6B is too large for in-memory shape inference, so use
            # ONNX's path-based inference on a temporary FP32 model.
            inferred_fp32_path = (
                Path(model_path).with_name(
                    f"{Path(model_path).stem}_fp32_inferred.onnx"
                )
            )

            try:
                onnx.shape_inference.infer_shapes_path(
                    model_path,
                    inferred_fp32_path,
                    data_prop=True,
                )

                fp32_model = onnx.load(inferred_fp32_path)
            finally:
                if inferred_fp32_path.exists():
                    inferred_fp32_path.unlink()

            # Quantize supported split-component MatMul weights to INT8 DQL
            # before the Qwen-specific BF16/int32 conversion pipeline.
            if (
                self._split_model
                and self._weight_quantization == "int8"
                and model_path.name in {
                    "transformer_part_A.onnx",
                    "transformer_part_B.onnx",
                    "lm_head.onnx",
                }
            ):
                self._logger.info(
                    "(ONNX-quantize) Quantizing %s weights to INT8...",
                    model_path.name,
                )

                quantizer = _QwenDQLOnlyQuantizer(
                    model_path=inferred_fp32_path,
                    output_path=converted_model_path,
                )

                # Reuse the already shape-inferred model instead of loading
                # the original uninferred graph again.
                quantizer._model = fp32_model

                quant_config = QuantizationConfig.uniform(
                    bits=8,
                    block_size=32,
                )

                matmul_info = quantizer._find_matmul_weights(
                    fp32_model
                )

                quantizer._quantize_to_dql(
                    fp32_model,
                    matmul_info,
                    quant_config,
                )

                self._logger.info(
                    "(ONNX-quantize) Quantized %d MatMul weight layers in %s",
                    len(matmul_info),
                    model_path.name,
                )

            bf16_converter = _QwenFP32Converter(
                "bf16",
                convert_io=not preserve_io,
            )

            bf16_model = bf16_converter.convert_model(
                fp32_model
            )

            self._save_large_external_model(
                bf16_model,
                converted_model_path,
            )

            self._infer_converted_shapes(
                converted_model_path,
            )

            self._logger.info(
                "(ONNX-convert) Successfully converted model to dtype bf16 @ '%s'",
                str(converted_model_path),
            )

            self._logger.info(
                "(ONNX-convert) Converting model '%s' to dtype int32...",
                str(converted_model_path),
            )

            bf16_model = onnx.load(
                converted_model_path
            )

            int64_converter = _QwenInt64Converter(
                "int32",
                convert_io=not preserve_io,
                enforce_io_casts=True,
            )

            converted_model = int64_converter.convert_model(
                bf16_model
            )

            if self._split_model:
                converted_editor = QwenOnnxGraphEditor.from_onnx(
                    converted_model,
                    self._onnx_export_dtype,
                )
                converted_editor.convert_static_shape_params_to_int32()

                if model_path.name in {
                    "transformer_part_A.onnx",
                    "transformer_part_B.onnx",
                }:
                    converted_editor.split_rope_qk()

                converted_model = converted_editor.to_onnx(
                    override_ir=converted_model.ir_version,
                )

            self._save_large_external_model(
                converted_model,
                converted_model_path,
            )

            self._infer_converted_shapes(
                converted_model_path,
            )

            self._logger.info(
                "(ONNX-convert) Successfully converted model to dtype int32 @ '%s'",
                str(converted_model_path),
            )

            self._export_paths[comp] = (
                converted_model_path
            )

        if self._split_model:
            embeddings_src = source_dir / "token_embeddings.npy"
            embeddings_dst = self._convert_dir / "token_embeddings.npy"

            if not embeddings_src.exists():
                raise FileNotFoundError(
                    f"Qwen split runtime requires {embeddings_src}"
                )

            embeddings = np.load(
                embeddings_src,
                mmap_mode="r",
            )

            embeddings_bf16 = np.asarray(
                embeddings,
                dtype=ml_dtypes.bfloat16,
            )

            np.save(
                embeddings_dst,
                embeddings_bf16,
            )

            self._logger.info(
                "(ONNX-convert) Converted token embeddings to BF16 @ '%s'",
                str(embeddings_dst),
            )

            lut_src = source_dir / "token_id_lut.npy"
            lut_dst = self._convert_dir / "token_id_lut.npy"

            if not lut_src.exists():
                raise FileNotFoundError(
                    f"Qwen split runtime requires {lut_src}"
                )

            lut = np.load(lut_src)

            np.save(
                lut_dst,
                np.asarray(lut, dtype=np.int32),
            )

            self._logger.info(
                "(ONNX-convert) Converted token ID LUT to int32 @ '%s'",
                str(lut_dst),
            )

        self._copy_runtime_assets(
            self._convert_dir,
            self._export_dir,
        )

    def export_torq(
        self,
        torq_export_dir: str | os.PathLike | None = None,
        torq_compile_args: list[str] | None = None,
        use_binary: bool = False,
        skip: list[str] | None = None,
        local_compile: bool = False,
        compiler_path: str | Path | None = None,
    ):
        """
        Compile the Qwen ONNX model directly from its external-data file.

        The shared exporter reloads the full model into GraphSurgeon and
        re-saves it before compilation. Qwen3-0.6B uses multi-gigabyte
        external tensor data, so that preprocessing path is avoided here.

        The Qwen static/converted ONNX model has already been cleaned,
        shape-finalized, and validated before reaching this stage.
        """

        self._torq_dir = Path(
            torq_export_dir or self._torq_dir
        )

        skip = skip or []

        if self._split_model:
            skip = list(skip) + ["model"]

        if self._torq_dir.exists():
            shutil.rmtree(
                self._torq_dir,
                ignore_errors=True,
            )

        self._torq_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for comp, onnx_path in self._export_paths.items():
            if comp in skip:
                self._logger.info(
                    "(Torq-export) Skipping %s",
                    comp,
                )
                continue

            self._logger.info(
                "(Torq-export) Exporting %s model @ '%s' to Torq...",
                comp,
                str(onnx_path),
            )

            # Reading the protobuf metadata without loading external tensor
            # values is safe and gives us the model's existing opset.
            metadata_model = onnx.load(
                onnx_path,
                load_external_data=False,
            )

            opset = next(
                (
                    entry.version
                    for entry in metadata_model.opset_import
                    if entry.domain in ("", "ai.onnx")
                ),
                None,
            )

            compile_torq_model(
                onnx_path,
                self._torq_dir,
                opset=opset,
                compiler_args=torq_compile_args,
                use_binary=use_binary,
                local_compile=local_compile,
                compiler_path=compiler_path,
            )

            self._logger.info(
                "(Torq-export) Successfully exported '%s/%s.vmfb'",
                str(self._torq_dir),
                Path(onnx_path).stem,
            )

        self._copy_runtime_assets(
            self._torq_dir,
            self._export_paths["model"].parent,
        )


def export_qwen_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)

    exporter = QwenModelExporter(
        static_models=not args.dynamic_models,
        hf_repo=args.hf_repo,
        hf_repo_subdir=args.hf_repo_subdir,
        max_gen_tokens=args.max_gen_tokens,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        keep_individual_kv_io=args.keep_individual_kv_io,
        split_model=args.split_model,
        weight_quantization=args.weight_quantization,
    )

    exporter.export_onnx(
        validate=not args.skip_validation,
    )

    if args.convert_dtypes:
        exporter.convert_models(
            preserve_io=args.preserve_io_dtypes,
        )

    if not args.skip_torq:
        exporter.export_torq(
            torq_compile_args=args.compile_flags or [],
            use_binary=args.use_binary,
            local_compile=args.local_compile,
            compiler_path=args.compiler_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 to ONNX and Torq"
    )

    add_qwen_export_args(parser)
    export_qwen_from_args(parser.parse_args())


if __name__ == "__main__":
    main()