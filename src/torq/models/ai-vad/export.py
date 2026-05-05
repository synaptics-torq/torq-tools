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
    add_aivad_export_args,
)

from ._graph import AiVadOnnxGraphEditor
from ._inference import MoonshineDynamic, MoonshineStatic
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.hf import hf_download_models, optimum_export_onnx

class DummyConfig:
    pass

def force_all_dim_params_to_values(model: onnx.ModelProto, vars: dict[str, int], unknown) -> onnx.ModelProto:
    def _fix_shape(shape, name):
        if shape is None:
            return
        
        print(shape, name)
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
                # Rest of tensors expects the output value (u0).
                if unknown == True:

                    d.dim_value = 1
                    d.ClearField("dim_param")

            except Exception:
                # leave it if we can't resolve it
                pass

    g = model.graph
    # Iterate over each tensor
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        #print("name: ", vi.name)
        #print(tt.shape)
        if tt.HasField("shape"):
            _fix_shape(tt.shape, vi.name)

    return model


class AiVadModelExporter(OnnxModelExporterBase):

    COMPONENTS = {
        "model": "aec_vad_exp12_d4_model_epoch_t710.onnx"
    }

    def __init__(
        self,

        model_dtype: str = "float",
        split_encoder: bool = False,
        extract_embeddings: bool = False,

        static_models: bool = True,
        *,

        onnx_model: str,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        use_optimum: bool = False,
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        
        self._split_encoder = split_encoder
        self._extract_embeddings = extract_embeddings
        self._static_models = static_models
        self._onnx_source_dir = onnx_source_dir
        self._use_optimum = use_optimum
        


        self._onnx_model_path = Path(onnx_model)
        if not self._onnx_model_path.exists():
            raise ValueError(f"--onnx-model not found: {self._onnx_model_path}")

        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._config = DummyConfig()

        opt_configs = {}

        super().__init__(
            model_dtype,
            static_models,
            self._config,
            Path(models_dir),
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs=opt_configs,
            skip_export=skip_export,
        )


    def _setup_dirs(self) -> list[Path]:
        # Use the directory of the provided ONNX model as source
        onnx_dir = self._onnx_model_path.parent

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

        # Ensure dirs exist
        export_dir.mkdir(parents=True, exist_ok=True)
        convert_dir.mkdir(parents=True, exist_ok=True)
        iree_dir.mkdir(parents=True, exist_ok=True)

        return onnx_dir, export_dir, convert_dir, iree_dir

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        return {"model": onnx.load(self._onnx_model_path)}

    def _make_aivad_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Make the encoder model static by replacing dynamic dimensions with fixed values.

        Args:
            model: Encoder model with dynamic I/O

        Returns:
            onnx.ModelProto: The modified encoder model with static dimensions
        """

        editor = AiVadOnnxGraphEditor.from_onnx(model, "model", self._onnx_export_dtype)

        # output vad: Reshapevad_dim_0, Reshapevad_dim_1
        # output hidden state: Concathidden_state_dim_1
        editor.fix_encoder_io(1,1,16)
        # Widen narrow strided-depthwise Convs so the Torq compiler does not
        # pick the DEDR scatter-gather (G(L)[>1]) path that hangs the
        # precompiled CModel simulator. See WidenSmallStridedDepthwiseConv.
        editor.widen_small_strided_depthwise_conv()
        static_model = editor.to_onnx(override_ir=model.ir_version)
        static_model = onnx.shape_inference.infer_shapes(static_model)

        graph = gs.import_onnx(static_model) 
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_encoder = gs.export_onnx(graph)

        new_encoder.ir_version = model.ir_version
        return new_encoder

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

        if not self._keep_individual_kv_io:
            n_kv_heads = self._config.decoder_num_attention_heads
            head_dim = self._hidden_size // n_kv_heads
            with_past = "with_past" in component
            dec_seq_len = self._max_tokens if with_past else 1
            # Combine decoder (self-attn) key+value into single tensor per layer
            editor.combine_kv_io_tensors(
                [1, n_kv_heads, dec_seq_len, head_dim],
                kv_layer_re=r"\.(\d+)\.decoder\.(key|value)$",
                combined_name_fmt="{prefix}.{layer}.decoder"
            )
            # Combine encoder (cross-attn) key+value into single tensor per layer
            editor.combine_kv_io_tensors(
                [1, n_kv_heads, self._enc_seq_len, head_dim],
                kv_layer_re=r"\.(\d+)\.encoder\.(key|value)$",
                combined_name_fmt="{prefix}.{layer}.encoder"
            )

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
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self._make_aivad_model_static(self._components["model"])


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
                preprocessor_model=self._export_dir / "preprocessor.onnx" if self._split_encoder else None,
                combined_kv_io=not self._keep_individual_kv_io
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

def export_aivad_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = AiVadModelExporter(
        args.dtype,
        args.extract_embeddings,
        not args.dynamic_models,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        use_optimum=args.use_optimum,
        convert_dtypes=args.convert_dtypes,
        skip_export=args.skip_export,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops,
        onnx_model=args.onnx_model,
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))

def main():
    parser = argparse.ArgumentParser(description="Export Moonshine to Torq")
    add_aivad_export_args(parser)
    export_aivad_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
