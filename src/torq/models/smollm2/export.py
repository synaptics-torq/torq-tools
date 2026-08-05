# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import ml_dtypes
from transformers import AutoConfig

from . import add_smollm2_export_args
from ._graph import SmolLM2OnnxGraphEditor
from ._inference import SmolLM2Dynamic, SmolLM2Static
from ...graph_edit.harness import EditSpec, GraphEditHarness, ctx, render_graph_edit_plan
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.hf import optimum_export_onnx

from ...utils.logging import (
    configure_logging,
)


class SmolLM2ModelExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["135M", "360M", "1.7B"] = "135M",
        instruct_model: bool = False,
        extract_embeddings: bool = False,
        keep_individual_kv_io: bool = False,
        static_models: bool = True,
        *,
        max_gen_tokens: int = 64,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        **edit_args
    ):
        self._instruct_model = instruct_model
        self._extract_embeddings = extract_embeddings
        self._keep_individual_kv_io = keep_individual_kv_io
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = onnx_source_dir
        self._hf_repo = f"HuggingFaceTB/SmolLM2-{model_size}"
        if self._instruct_model:
            self._hf_repo += "-Instruct"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        super().__init__(
            "fp32",
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs={"model": ORTOptimizerConfig(
                num_heads=self._config.num_attention_heads,
                hidden_size=self._config.hidden_size
            )}
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir, export_dir, convert_dir, torq_dir = [None] * 4
        if self._onnx_source_dir and (onnx_source_dir := Path(self._onnx_source_dir)).exists():
            onnx_dir = onnx_source_dir
        else:
            onnx_dir = self._models_dir / "source" / self._model_dtype
            onnx_dir.mkdir(parents=True, exist_ok=True)
            optimum_export_onnx(
                onnx_dir, self._hf_repo, self._model_dtype, ["model.onnx"]
            )
        export_dir = (
            self._models_dir / 
            "export" / 
            "onnx" / 
            self._model_dtype / 
            ("static" if self._static_models else "dynamic")
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
        model_path = self._onnx_dir /  "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.onnx @ '{self._onnx_dir}'")
        model = onnx.load(model_path)
        orig_ir = model.ir_version
        graph = gs.import_onnx(model)
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        model = gs.export_onnx(graph)
        model.ir_version = orig_ir
        return {"model": model}

    def graph_edit_blocks(self) -> dict[str, list[EditSpec]]:
        make_static = [EditSpec("RemoveIsNaN")]
        make_static_kv = [
            EditSpec("ReplaceDynamicKVCache", (ctx("cur_len"), ctx("max_tokens"))),
            EditSpec("MaskFutureAttentionScores", (ctx("cur_len"), ctx("max_tokens"), ctx("export_dtype"))),
            EditSpec("AddCurrLenInput", (ctx("cur_len"),)),
            EditSpec("ConvertToStaticIndex"),
        ]
        patch = [
            EditSpec("EliminateTranspose"),
            EditSpec("CollapseReshapeChain"),
            EditSpec("FoldScalarMatMul"),
        ]
        if self._broadcast_ops is not None:
            patch.append(EditSpec("BroadcastOpInputs", (self._broadcast_ops,)))
        if self._extract_embeddings:
            patch.append(EditSpec(
                "ExtractConstantLUT",
                ((self._vocab_size, self._hidden_size), ctx("embeddings_path"), "token_embedding"),
            ))
        return {
            "model.static": make_static,
            "model.static (KV cache)": make_static_kv,
            "model.patch": patch,
        }

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
            "Set export data type to %s for model data type %s",
            onnx.helper.tensor_dtype_to_string(self._onnx_export_dtype), self._model_dtype
        )
        
        editor = SmolLM2OnnxGraphEditor(graph, self._onnx_export_dtype)
        editor.fix_io(self._max_gen_tokens)

        blocks = self.graph_edit_blocks()
        # Remove isNaN ops
        editor.apply_specs(blocks["model.static"], self._harness)

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

        editor.apply_specs(
            blocks["model.static (KV cache)"],
            self._harness,
            {
                "cur_len": cur_len,
                "max_tokens": self._max_gen_tokens,
                "export_dtype": self._onnx_export_dtype,
            },
        )

        new_model = editor.to_onnx(override_ir=model.ir_version)
        return new_model

    def _patch_static_model(self, model_path: str | os.PathLike):
        model = onnx.load(model_path)
        editor = SmolLM2OnnxGraphEditor.from_onnx(model, self._onnx_export_dtype)

        embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
        editor.apply_specs(
            self.graph_edit_blocks()["model.patch"],
            self._harness,
            {"embeddings_path": embeddings_npy},
        )

        if self._extract_embeddings:
            editor.reorder_graph_input("token_embedding", 0)

        if not self._keep_individual_kv_io:
            editor.combine_kv_io_tensors([
                1,                                                              # B
                self._config.num_key_value_heads,                               # H
                self._max_gen_tokens,                                           # L
                self._config.hidden_size // self._config.num_attention_heads    # D
            ])

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self.check_model(self._components["model"])
        self._components["model"] = self._make_model_static(self._components["model"])

    def apply_post_static_patches(self, model_path: str | os.PathLike, _):
        self._patch_static_model(model_path)

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
                self._export_paths["model"],
                self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo
            )
        else:
            runner = SmolLM2Dynamic.from_onnx(
                self._export_paths["model"],
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
            min_len = min(len(output), len(val_output))
            if output[:min_len] != val_output[:min_len]:
                result = f"Warning: Validation failed, mismatched outputs\nExpected:\n{val_output},\nGenerated:\n{output}"
            else:
                result = f"Validation successful, identical outputs"
                if len(output) != len(val_output):
                    result += f" (output lengths differ: {len(output)} vs {len(val_output)})"
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

    def convert_models(
        self, 
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
    ):
        return super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            external_data=[
                (self._export_paths["model"].parent / "token_embeddings.npy", np.dtype(ml_dtypes.bfloat16))
            ]
        )

    def _copy_runtime_assets(
        self,
        dst_dir: str | os.PathLike,
        src_dir: str | os.PathLike | None = None,
        *,
        include_npy_data: bool = True,
    ) -> None:
        src_dir = Path(src_dir or self._export_paths["model"].parent)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        asset_names = ["config.json", "tokenizer.json"]
        if include_npy_data:
            asset_names.extend(p.name for p in src_dir.glob("*.npy"))
        for asset_name in asset_names:
            src_path = src_dir / asset_name
            if not src_path.exists():
                continue
            shutil.copy2(src_path, dst_dir / asset_name)

    def export_torq(
        self,
        torq_export_dir: str | os.PathLike | None = None,
        torq_compile_args: list[str] | None = None,
        use_binary: bool = False,
        skip: list[str] | None = None,
        local_compile: bool = False,
        compiler_path: str | Path | None = None,
    ):
        result = super().export_torq(
            torq_export_dir=torq_export_dir,
            torq_compile_args=torq_compile_args,
            use_binary=use_binary,
            skip=skip,
            local_compile=local_compile,
            compiler_path=compiler_path,
        )
        self._copy_runtime_assets(self._torq_dir, self._export_paths["model"].parent)
        return result

def export_smollm2_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = SmolLM2ModelExporter(
        args.model_size,
        args.instruct_model,
        args.extract_embeddings,
        args.keep_individual_kv_io,
        not args.dynamic_models,
        max_gen_tokens=args.max_gen_tokens,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(args))
    if args.view_graph_edits:
        print(render_graph_edit_plan(exporter.describe_graph_edits()))
        return
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if not args.skip_torq:
        exporter.export_torq(
            torq_compile_args=args.compile_flags or [],
            use_binary=args.use_binary,
            local_compile=args.local_compile,
            compiler_path=args.compiler_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Export SmolLM2 to Torq")
    add_smollm2_export_args(parser)
    export_smollm2_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
