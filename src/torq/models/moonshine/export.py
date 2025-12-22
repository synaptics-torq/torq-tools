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

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from onnxruntime.transformers.optimizer import optimize_model
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
    STATIC_MODEL_COMPONENTS,
    add_moonshine_export_args,
)

from ._graph import MoonshineOnnxGraphEditor
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


class MoonshineModelExporter:

    COMPONENTS: Final[dict[str, str]] = {
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model.onnx",
        "decoder_with_past": "decoder_with_past_model.onnx",
    }

    COMPONENTS_MERGED: Final[dict[str, str]] = {
        "encoder": "encoder_model.onnx",
        "decoder_merged": "decoder_model_merged.onnx",
    }

    def __init__(
        self,
        model_size: Literal["base", "tiny"] = "tiny",
        model_dtype: str = "float",
        split_encoder: bool = False,
        extract_embeddings: bool = False,
        static_models: bool = True,
        *,
        max_audio_s: int = 5,
        max_tok_per_s: int = 6,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        use_optimum: bool = False,
        convert_dtype: str | None = None,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        if model_size not in ["base", "tiny"]:
            raise ValueError(
                f"Invalid model size '{model_size}', choose one of: ['base', 'tiny']"
            )

        self._model_size = model_size
        self._model_dtype = model_dtype
        self._split_encoder = split_encoder
        self._extract_embeddings = extract_embeddings
        self._static_models = static_models
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._convert_dtype = convert_dtype
        self._onnx_export_dtype = _FP_EXPORT_DTYPE_MAPPING.get(
            self._model_dtype,
            onnx.TensorProto.FLOAT
        )
        self._skip_export = set(skip_export or [])
        self._hf_repo = "UsefulSensors/moonshine"
        self._config = AutoConfig.from_pretrained(f"{self._hf_repo}-{self._model_size}")
        self._num_samples = max_audio_s * 16_000
        self._max_tokens = max_audio_s * max_tok_per_s
        self._enc_seq_len = (
            floor(floor(floor(self._num_samples / 64 - 127 / 64) / 3) / 2) - 1
        )
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        if onnx_source_dir and (onnx_source_dir := Path(onnx_source_dir)).exists():
            self._onnx_dir = onnx_source_dir
        else:
            if use_optimum or self._model_dtype in OPTIMUM_DTYPES:
                self._model_dtype = "fp32" if self._model_dtype == "float" else self._model_dtype
                if self._model_dtype not in OPTIMUM_DTYPES:
                    raise ValueError(f"'{self._model_dtype}' is an invalid dtype for optimium export, choose one of {OPTIMUM_DTYPES}")
                self._onnx_dir = self._models_dir / self._hf_repo / "source" / "onnx" / self._model_size / self._model_dtype
                self._onnx_dir.mkdir(parents=True, exist_ok=True)
                self._optimum_export_models()
            else:
                if self._model_dtype not in ONNX_DTYPES:
                    raise ValueError(f"'{self._model_dtype}' is an invalid dtype for pre-existing ONNX models, choose one of {ONNX_DTYPES}")
                self._onnx_dir = self._models_dir / self._hf_repo / "source" / "onnx" / "merged" / self._model_size / self._model_dtype
                self._onnx_dir.mkdir(parents=True, exist_ok=True)
                self._hf_download_models()

        self._components, self._merged_decoder = self._load_onnx()
        self._export_dir = (
            self._models_dir
            / self._hf_repo
            / "export"
            / "onnx"
            / self._model_size
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

    @staticmethod
    def split_merged_encoder(merged_model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
        assert merged_model.ir_version <= 10
        graph = gs.import_onnx(merged_model)

        preproc_out: gs.Node | None = None
        for node in graph.nodes:
            if node.op != "Mul":
                continue
            if any(isinstance(inp, gs.Constant) for inp in node.inputs):
                continue
            inp_A: gs.Node = node.i(tensor_idx=0)
            inp_B: gs.Node = node.i(tensor_idx=1)
            consumer: gs.Node = node.o().o()
            if (inp_A.op == "Conv" and inp_B.op == "Add" and consumer.op == "Transpose"):
                preproc_out = node
                break
        
        if preproc_out is None:
            raise ValueError("Unable to split encoder model: preprocessor boundary not found")

        import tempfile
        with tempfile.TemporaryDirectory() as t_dir:
            merged_model_path = Path(t_dir) / "merged_encoder.onnx"
            preprocessor_path = Path(t_dir) / "preprocessor.onnx"
            encoder_path = Path(t_dir) / "encoder.onnx"
            onnx.save(merged_model, merged_model_path) 
            onnx.utils.extract_model(
                merged_model_path,
                preprocessor_path,
                input_names=[i.name for i in graph.inputs],
                output_names=[preproc_out.outputs[0].name]
            )
            onnx.utils.extract_model(
                merged_model_path,
                encoder_path,
                input_names=[preproc_out.outputs[0].name],
                output_names=[o.name for o in graph.outputs]
            )

            preprocessor_ext = gs.import_onnx(onnx.load(preprocessor_path))
            preprocessor_ext.name = graph.name
            preprocessor_ext.outputs[0].name = "input_features"
            preprocessor_ext.cleanup(
                remove_unused_graph_inputs=True, remove_unused_node_outputs=True
            ).toposort()
            preprocessor_model = gs.export_onnx(preprocessor_ext)
            preprocessor_model.ir_version = merged_model.ir_version

            encoder_ext = gs.import_onnx(onnx.load(encoder_path))
            encoder_ext.name = graph.name
            encoder_ext.inputs[0].name = "input_features"
            encoder_ext.cleanup(
                remove_unused_graph_inputs=True, remove_unused_node_outputs=True
            ).toposort()
            encoder_model = gs.export_onnx(encoder_ext)
            encoder_model.ir_version = merged_model.ir_version

            return preprocessor_model, encoder_model

    @staticmethod
    def split_merged_decoder(merged_model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
        assert merged_model.ir_version <= 10
        if_node = next(n for n in merged_model.graph.node if n.op_type == "If")
        then_branch = None
        else_branch = None
        for attr in if_node.attribute:
            if attr.name == "then_branch":
                then_branch = attr.g
            elif attr.name == "else_branch":
                else_branch = attr.g
        if not then_branch or not else_branch:
            raise ValueError("Merged decoder If node missing branches")
        
        outputs = merged_model.graph.output
        same_outputs: bool = all([
            out_merged == out == out_with_past 
            for out_merged, out, out_with_past 
            in zip(
                [out.name for out in outputs],
                [out.name for out in then_branch.output],
                [out.name for out in else_branch.output]
            )
        ])

        decoder_graph = onnx.helper.make_graph(
            nodes=else_branch.node,
            name="main",
            inputs=[input for input in merged_model.graph.input if input.name in ("input_ids", "encoder_hidden_states")],
            outputs=outputs if same_outputs else else_branch.output,
            initializer=list(merged_model.graph.initializer) + list(else_branch.initializer)
        )
        decoder_model = onnx.helper.make_model(decoder_graph, opset_imports=merged_model.opset_import)
        decoder_model.ir_version = merged_model.ir_version

        decoder_with_past_graph = onnx.helper.make_graph(
            nodes=then_branch.node,
            name="main",
            inputs=[input for input in merged_model.graph.input if input.name not in ("encoder_hidden_states", "use_cache_branch")],
            outputs=[out for out in (outputs if same_outputs else then_branch.output) if "encoder" not in out.name],
            initializer=list(merged_model.graph.initializer) + list(then_branch.initializer)
        )
        decoder_with_past_model = onnx.helper.make_model(decoder_with_past_graph, opset_imports=merged_model.opset_import)
        decoder_with_past_model.ir_version = merged_model.ir_version

        return decoder_model, decoder_with_past_model

    def _optimum_export_models(self):
        if not all(
            (self._onnx_dir / comp_model_name).exists()
            for comp_model_name in MoonshineModelExporter.COMPONENTS.values()
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

    def _hf_download_models(self):
        for comp_model_name in MoonshineModelExporter.COMPONENTS_MERGED.values():
            hf_hub_download(
                self._hf_repo,
                comp_model_name,
                subfolder=f"onnx/merged/{self._model_size}/{self._model_dtype}",
                local_dir=self._models_dir / self._hf_repo / "source",
            )

    def _load_onnx(self) -> tuple[dict[str, onnx.ModelProto], bool]:
        unmerged_model_names: set[str] = set(self.COMPONENTS.values())
        merged_model_names: set[str] = set(self.COMPONENTS_MERGED.values())
        model_names: set[str] = set(list(p.name for p in self._onnx_dir.glob("*.onnx")))
        if merged_model_names.issubset(model_names) and unmerged_model_names.issubset(model_names):
            self._logger.warning("(ONNX-load) Found both merged and un-merged decoder models @ '%s', defaulting to loading merged", str(self._onnx_dir))
            model_names = merged_model_names
            merged_decoder = True
        elif unmerged_model_names.issubset(model_names):
            self._logger.info("(ONNX-load) Found encoder and un-merged decoder models @ '%s'", str(self._onnx_dir))
            merged_decoder = False
        elif merged_model_names.issubset(model_names):
            self._logger.info("(ONNX-load) Found encoder and merged decoder model @ '%s'", str(self._onnx_dir))
            merged_decoder = True
        else:
            raise ValueError(
                f"Expected merged models {merged_model_names} or un-merged models {unmerged_model_names} @ '{self._onnx_dir}'"
            )
        comps = MoonshineModelExporter.COMPONENTS_MERGED if merged_decoder else MoonshineModelExporter.COMPONENTS
        onnx_models: dict[str, onnx.ModelProto] = {
            comp: onnx.load(self._onnx_dir / comp_model_name)
            for comp, comp_model_name in comps.items()
        }
        return onnx_models, merged_decoder

    def _make_encoder_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Make the encoder model static by replacing dynamic dimensions with fixed values.

        Args:
            model: Encoder model with dynamic I/O

        Returns:
            onnx.ModelProto: The modified encoder model with static dimensions
        """

        editor = MoonshineOnnxGraphEditor.from_onnx(model, "encoder", self._onnx_export_dtype)
        editor.fix_encoder_io(self._num_samples, self._enc_seq_len)
        new_encoder = editor.to_onnx(override_ir=model.ir_version)
        return new_encoder

    def _make_decoder_model_static(
        self, decoder_model: onnx.ModelProto, with_past: bool
    ) -> onnx.ModelProto:
        """
        Make decoder models static by replacing dynamic dimensions with fixed values and applying necessary transformations.

        Replaces KV caching and other dynamic operations with static equivalents in the cached decoder model.

        Args:
            decoder_model (onnx.ModelProto): ONNX decoder model to modify
            with_past (bool): Whether the model is the cached branch of the decoder

        Returns:
            onnx.ModelProto: The modified decoder model with static dimensions and transformations applied

        Raises:
            ValueError: If an unexpected dynamic dimension is found in the model inputs, outputs, or nodes
        """
        graph: gs.Graph = gs.import_onnx(decoder_model)
        self._logger.debug(
            "Set export data type to %s for model data type %s",
            onnx.helper.tensor_dtype_to_string(self._onnx_export_dtype),
            self._model_dtype
        )
        output_names = {o.name for o in graph.outputs}
        comp = "decoder" + ("_with_past" if with_past else "")
        pad_len = (
            self._config.hidden_size // self._config.decoder_num_attention_heads
        ) % 8

        editor = MoonshineOnnxGraphEditor(graph, comp, self._onnx_export_dtype)
        editor.fix_decoder_io(self._enc_seq_len, self._max_tokens, with_past)

        # Remove isNaN ops
        editor.remove_isNaN()
        # Move model output if it's fed by a Concat node which has a Pad consumer
        if not with_past:
            editor.move_output_from_concat(pad_len=pad_len)

        if with_past:
            cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
            graph.inputs.append(cur_len_2d)
            cur_len = graph.layer(
                name="current_len_to_1d",
                op="Squeeze",
                inputs=[cur_len_2d, [0]],
                outputs=[gs.Variable(cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1])],
            )[0]

            (
                editor
                # Replace dynamic KV cache
                .replace_dynamic_kv_cache(cur_len, self._max_tokens)
                # Add causal attention score mask
                .mask_future_attn_scores(cur_len, self._max_tokens)
                # Replace dynamic sequence length getter with `cur_len`
                .add_curr_len_input(cur_len)
                # Replace dynamic index computation `Range(start, start + 1, 1) -> index`
                .convert_to_static_index()
            )

        new_model = editor.to_onnx(override_ir=decoder_model.ir_version)
        return new_model

    def _replace_int_to_bf16_casts(self, model_path: str | os.PathLike, component: str):
        model = onnx.load(model_path)
        editor = MoonshineOnnxGraphEditor.from_onnx(model, component, self._onnx_export_dtype)

        # Repalce potentially unsupported int64 -> float cast with lookup table
        editor.replace_int64_float_cast(max_int=self._max_tokens)

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def _patch_static_encoder(self, model_path: str | os.PathLike, component: str):
        model = onnx.load(model_path)
        editor = MoonshineOnnxGraphEditor.from_onnx(model, component, self._onnx_export_dtype)

        # Broadcast op inputs to match output shape
        if self._broadcast_ops is not None:
            editor.broadcast_op_inputs(
                ops=self._broadcast_ops,
            )

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def _patch_static_decoder(self, model_path: str | os.PathLike, component: str):
        model = onnx.load(model_path)
        editor = MoonshineOnnxGraphEditor.from_onnx(model, component, self._onnx_export_dtype)

        # Fold MatMul A @ B where B is a scalar into Mul
        editor.fold_scalar_matmul()
        # Manually dequantize projection scores
        if self._model_dtype in ("quantized", "quantized_4bit"):
            editor.dequantize_projections_matmul(
                hidden_size=self._hidden_size,
                vocab_size=self._vocab_size
            )
        # Broadcast op inputs to match output shape
        if self._broadcast_ops is not None:
            editor.broadcast_op_inputs(
                ops=self._broadcast_ops,
            )

        if self._extract_embeddings:
            # Extract token embeddings LUT
            embeddings_npy = Path(model_path).parent / f"{component}_token_embeddings.npy"
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

    def _dedup_decoder_embeddings_npy(self, emb_dir: str | os.PathLike):
        emb_dir = Path(emb_dir)
        if (d_emb_p := emb_dir / f"decoder_token_embeddings.npy").exists() \
            and (dp_emb_p := emb_dir / f"decoder_with_past_token_embeddings.npy").exists():
            d_emb = np.load(d_emb_p)
            dp_emb = np.load(dp_emb_p)
            if np.array_equal(d_emb, dp_emb):
                dp_emb_p.unlink()

    def make_static(self):
        if self._merged_decoder:
            decoder, decoder_with_past = self.split_merged_decoder(self._components["decoder_merged"])
            self._components["decoder"] = self.check_model(decoder)
            self._components["decoder_with_past"] = self.check_model(decoder_with_past)
            del self._components["decoder_merged"]
            assert set(self._components) == set(STATIC_MODEL_COMPONENTS)
            self._logger.info("(decoder_merged) Decoder split into regular and with_past models")

        if self._split_encoder:
            self._components["preprocessor"], self._components["encoder"] = \
                self.split_merged_encoder(self._components["encoder"])
            self._components["preprocessor"] = self._make_encoder_model_static(self._components["preprocessor"])
            self._logger.info("(encoder) Encoder split into preprocessor and encoder models")
        self._logger.info("(encoder) Making graph static...")
        self._components["encoder"] = self._make_encoder_model_static(self._components["encoder"])
        self._logger.info("(decoder) Making graph static...")
        self._components["decoder"] = self._make_decoder_model_static(
            self._components["decoder"], False
        )
        self._logger.info("(decoder_with_past) Making graph static...")
        self._components["decoder_with_past"] = self._make_decoder_model_static(
            self._components["decoder_with_past"], True
        )

    def optimize_model(self, model_path: str | os.PathLike, component: str):
        optimized = optimize_model(
            str(model_path),
            model_type="bert",
            num_heads=(
                self._config.encoder_num_attention_heads
                if "encoder" in component
                else self._config.decoder_num_attention_heads
            ),
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

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        if component == "encoder":
            self._patch_static_encoder(model_path, component)
        else:
            self._patch_static_decoder(model_path, component)
            self._dedup_decoder_embeddings_npy(Path(model_path).parent)

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
                self._logger.info("(%s) Applying post-static conversion patches...", comp)
                self.apply_post_static_patches(self._export_paths[comp], comp)
            self.check_model(onnx.load(self._export_paths[comp]), skip_data_prop="decoder" in comp and self._merged_decoder)
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
        if not self._convert_dtype:
            self._logger.warning("Skipping conversion as convert dtype is not set")
        convert_dir = convert_dir or (
            self._models_dir
            / self._hf_repo
            / "export"
            / "onnx"
            / self._model_size
            / self._convert_dtype
            / ("static" if self._static_models else "dynamic")
        )
        for comp, model_path in self._export_paths.items():
            if comp == "preprocessor":
                shutil.copy2(model_path, convert_dir)
                continue
            self._logger.info("(ONNX-convert) Converting model '%s' to dtype %s...", str(model_path), self._convert_dtype)
            converted_model_path = convert_dir / model_path.name
            convert_model(model_path, converted_model_path, self._convert_dtype)
            self._logger.info("(ONNX-convert) Successfully converted model to dtype %s @ '%s'", self._convert_dtype, str(converted_model_path))
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
            / (self._convert_dtype if (self._convert_dtype and self._model_dtype == "float") else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )
        for comp, onnx_path in self._export_paths.items():
            if comp == "preprocessor":
                continue
            if (self._model_dtype == "bf16" or self._convert_dtype == "bf16") and self._replace_int_bf16_cast:
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


def export_moonshine_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = MoonshineModelExporter(
        args.model_size,
        args.dtype,
        args.split_encoder,
        args.extract_embeddings,
        not args.dynamic_models,
        max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        use_optimum=args.use_optimum,
        convert_dtype=args.convert_dtype,
        skip_export=args.skip_export,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtype:
        exporter.convert_models()
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))

def main():
    parser = argparse.ArgumentParser(description="Export Moonshine to Torq")
    add_moonshine_export_args(parser)
    export_moonshine_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
