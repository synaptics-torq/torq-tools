# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from math import floor
from pathlib import Path
from typing import Literal, Final

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import ml_dtypes
from datasets import load_dataset, Audio
from transformers import AutoConfig, AutoProcessor
from torq.compile import process_iree_args
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
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.hf import hf_download_models, optimum_export_onnx


class MoonshineModelExporter(OnnxModelExporterBase):

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
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        self._model_size = model_size
        self._split_encoder = split_encoder
        self._extract_embeddings = extract_embeddings
        self._onnx_source_dir = onnx_source_dir
        self._use_optimum = use_optimum
        self._hf_repo = f"UsefulSensors/moonshine-{self._model_size}"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._num_samples = max_audio_s * 16_000
        self._max_tokens = max_audio_s * max_tok_per_s
        self._enc_seq_len = (
            floor(floor(floor(self._num_samples / 64 - 127 / 64) / 3) / 2) - 1
        )
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._merged_decoder = True
        opt_configs = {
            comp: ORTOptimizerConfig(
                num_heads=self._config.encoder_num_attention_heads 
                if comp == "encoder" else
                self._config.decoder_num_attention_heads,
                hidden_size=self._config.hidden_size
            ) for comp in STATIC_MODEL_COMPONENTS
        }

        super().__init__(
            model_dtype,
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs=opt_configs,
            skip_export=skip_export,
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir, export_dir, convert_dir, iree_dir = [None] * 4
        if self._onnx_source_dir and (onnx_source_dir := Path(self._onnx_source_dir)).exists():
            onnx_dir = onnx_source_dir
        else:
            if self._use_optimum or self._model_dtype in OPTIMUM_DTYPES:
                self._model_dtype = "fp32" if self._model_dtype == "float" else self._model_dtype
                if self._model_dtype not in OPTIMUM_DTYPES:
                    raise ValueError(f"'{self._model_dtype}' is an invalid dtype for optimium export, choose one of {OPTIMUM_DTYPES}")
                onnx_dir = self._models_dir / "source" / "onnx" / self._model_size / self._model_dtype
                onnx_dir.mkdir(parents=True, exist_ok=True)
                optimum_export_onnx(
                    onnx_dir, self._hf_repo, self._model_dtype, list(MoonshineModelExporter.COMPONENTS.values())
                )
            else:
                if self._model_dtype not in ONNX_DTYPES:
                    raise ValueError(f"'{self._model_dtype}' is an invalid dtype for pre-existing ONNX models, choose one of {ONNX_DTYPES}")
                onnx_dir = self._models_dir / "source" / "onnx" / "merged" / self._model_size / self._model_dtype
                onnx_dir.mkdir(parents=True, exist_ok=True)
                hf_download_models(
                    "UsefulSensors/moonshine",
                    list(MoonshineModelExporter.COMPONENTS_MERGED.values()),
                    subfolder=f"onnx/merged/{self._model_size}/{self._model_dtype}",
                    local_dir=self._models_dir / "source",
                )
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
        iree_dir = (
            self._models_dir
            / "export"
            / "iree"
            / ("converted" if self._convert_dtypes else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )
        return onnx_dir, export_dir, convert_dir, iree_dir

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        unmerged_model_names: set[str] = set(MoonshineModelExporter.COMPONENTS.values())
        merged_model_names: set[str] = set(MoonshineModelExporter.COMPONENTS_MERGED.values())
        model_names: set[str] = set(list(p.name for p in self._onnx_dir.glob("*.onnx")))
        if merged_model_names.issubset(model_names) and unmerged_model_names.issubset(model_names):
            self._logger.warning("(ONNX-load) Found both merged and un-merged decoder models @ '%s', defaulting to loading merged", str(self._onnx_dir))
            model_names = merged_model_names
            self._merged_decoder = True
        elif unmerged_model_names.issubset(model_names):
            self._logger.info("(ONNX-load) Found encoder and un-merged decoder models @ '%s'", str(self._onnx_dir))
            self._merged_decoder = False
        elif merged_model_names.issubset(model_names):
            self._logger.info("(ONNX-load) Found encoder and merged decoder model @ '%s'", str(self._onnx_dir))
            self._merged_decoder = True
        else:
            raise ValueError(
                f"Expected merged models {merged_model_names} or un-merged models {unmerged_model_names} @ '{self._onnx_dir}'"
            )
        comps = MoonshineModelExporter.COMPONENTS_MERGED if self._merged_decoder else MoonshineModelExporter.COMPONENTS
        return {
            comp: onnx.load(self._onnx_dir / comp_model_name)
            for comp, comp_model_name in comps.items()
        }

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
        graph = gs.import_onnx(new_encoder)
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_encoder = gs.export_onnx(graph)
        new_encoder.ir_version = model.ir_version
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
        comp = "decoder" + ("_with_past" if with_past else "")
        pad_len = (
            self._config.hidden_size // self._config.decoder_num_attention_heads
        ) % 8

        editor = MoonshineOnnxGraphEditor(graph, comp, self._onnx_export_dtype)
        editor.fix_decoder_io(self._enc_seq_len, self._max_tokens, with_past)

        # Remove redundant Cast ops
        editor.remove_redundant_casts()
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

        # Replace Pad ops with Concat ops
        editor.replace_pad_with_concat()

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
            encoder_ext.name = "main"
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

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        if component == "encoder":
            self._patch_static_encoder(model_path, component)
        elif "decoder" in component:
            self._patch_static_decoder(model_path, component)
            self._dedup_decoder_embeddings_npy(Path(model_path).parent)

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

        processor = AutoProcessor.from_pretrained(f"{self._hf_repo}")
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

    def convert_models(
        self, 
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
        skip: list[str] | None = None,
    ):
        skip = skip or []
        skip.append("preprocessor")
        external_data = None if any(m in self._skip_export for m in ("decoder", "decoder_with_past")) else \
        [(self._export_paths["decoder"].parent / "decoder_token_embeddings.npy", np.dtype(ml_dtypes.bfloat16))]
        super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            skip=skip,
            external_data=external_data,
        )
        for comp, model_path in self._export_paths.items():
            if comp == "preprocessor":
                shutil.copy2(model_path, self._convert_dir)
                break

    def export_iree(
        self,
        iree_export_dir: str | os.PathLike | None = None,
        iree_compile_args: list[str] | None = None,
        use_iree_cli: bool = False,
        skip: list[str] | None = None,
    ):
        skip = skip or []
        skip.append("preprocessor")
        for comp, onnx_path in self._export_paths.items():
            if comp in skip:
                continue
            if (self._model_dtype == "bf16" or self._convert_dtypes) and self._replace_int_bf16_cast:
                self._replace_int_to_bf16_casts(onnx_path, comp)
        return super().export_iree(
            iree_export_dir,
            iree_compile_args,
            use_iree_cli,
            skip
        )

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
        convert_dtypes=args.convert_dtypes,
        skip_export=args.skip_export,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))

def main():
    parser = argparse.ArgumentParser(description="Export Moonshine to Torq")
    add_moonshine_export_args(parser)
    export_moonshine_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
