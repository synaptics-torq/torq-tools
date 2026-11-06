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
from ._inference import SmolLM2Dynamic, SmolLM2Static, _repo_id_for_size
from ...graph_edit.harness import EditSpec, GraphEditHarness, ctx, render_graph_edit_plan
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.validation import validate_decoder_only_onnx
from ...model_export.hf import optimum_export_onnx
from ...utils.onnx import load_onnx_lazy, validate_onnx_source_dir

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
        batch_prefill: int | None = None,
        native_batch_prefill: bool = False,
        batch_prefill_sizes: tuple[int, ...] | None = None,
        split_lm_head: bool = False,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        dynamic_quantize: bool = False,
        convert_dtypes: bool = False,
        split_weights: bool = False,
        read_only_kv_cache: bool = True,
        **edit_args
    ):
        if read_only_kv_cache and keep_individual_kv_io:
            raise ValueError(
                "`--read-only-kv-cache` needs the combined KV cache I/O; "
                "add `--no-read-only-kv-cache` to `--keep-individual-kv-io`"
            )
        self._instruct_model = instruct_model
        self._extract_embeddings = extract_embeddings
        self._keep_individual_kv_io = keep_individual_kv_io
        self._read_only_kv_cache = read_only_kv_cache
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = validate_onnx_source_dir(
            onnx_source_dir, required_files=("tokenizer.json",)
        )
        self._hf_repo = _repo_id_for_size(model_size, instruct_model)
        self._config = AutoConfig.from_pretrained(
            self._onnx_source_dir if self._onnx_source_dir is not None else self._hf_repo,
            local_files_only=self._onnx_source_dir is not None,
        )
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
            dynamic_quantize=dynamic_quantize,
            convert_dtypes=convert_dtypes,
            split_weights=split_weights,
            opt_configs={"model": ORTOptimizerConfig(
                num_heads=self._config.num_attention_heads,
                hidden_size=self._config.hidden_size
            )},
            max_gen_tokens=max_gen_tokens,
            extract_embeddings=extract_embeddings,
            split_lm_head=split_lm_head,
            read_only_kv_cache=read_only_kv_cache,
            keep_individual_kv_io=keep_individual_kv_io,
            batch_prefill=batch_prefill,
            native_batch_prefill=native_batch_prefill,
            batch_prefill_sizes=batch_prefill_sizes,
            native_minimum=4,
            native_supports_read_only=True,
            native_unsupported=(
                model_size not in ("135M", "360M")
                or dynamic_quantize
                or convert_dtypes
                or split_weights
                or not read_only_kv_cache
            ),
            native_supported=(
                "SmolLM2 135M and 360M only, with FP32 weights, "
                "--extract-embeddings, --split-lm-head and --read-only-kv-cache; "
                "1.7B, dynamic quantization, dtype conversion and "
                "--split-weights are unsupported"
            ),
            # 30/32 layers make this the deepest native family: the read-only
            # attention past/new split's FP32 reduction-order noise amplifies
            # through the residual stack to ~1e-3 at maximum history depth
            # (taps verified the attention itself matches sequential decode
            # to <1e-6), so the gate uses a wider rtol.
            native_validation_rtol=1e-3,
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir, export_dir, quantize_dir, convert_dir, torq_dir = [None] * 5
        if self._onnx_source_dir is not None:
            onnx_dir = self._onnx_source_dir
        else:
            onnx_dir = self._models_dir / "source" / self._model_dtype
            onnx_dir.mkdir(parents=True, exist_ok=True)
            optimum_export_onnx(
                onnx_dir, self._hf_repo, self._model_dtype, ["model.onnx"]
            )
        suffix = "static" if self._static_models else "dynamic"
        root = self._models_dir / "export"
        export_dir = root / self._model_dtype / suffix
        quantize_dir = root / "quantized" / suffix
        convert_dir = root / "converted" / suffix
        # Compiled artifacts live in the variant that is actually compiled
        # (see the base class contract for _setup_dirs).
        variant_dir = convert_dir if self._convert_dtypes else (
            quantize_dir if self._dynamic_quantize else export_dir)
        torq_dir = variant_dir / "compiled"
        return onnx_dir, export_dir, quantize_dir, convert_dir, torq_dir

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        model_path = self._onnx_dir /  "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.onnx @ '{self._onnx_dir}'")
        model = load_onnx_lazy(model_path)
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
            EditSpec("ReplaceDynamicKVCache", (ctx("cur_len"), ctx("max_tokens"), ctx("chunk_len"))),
            EditSpec("MaskFutureAttentionScores", (ctx("cur_len"), ctx("max_tokens"), ctx("export_dtype"), False, ctx("chunk_len"))),
            EditSpec("AddCurrLenInput", (ctx("cur_len"),)),
            EditSpec("ConvertToStaticIndex", (ctx("chunk_len"),)),
        ]
        patch = [
            EditSpec("EliminateTranspose"),
            EditSpec("CollapseReshapeChain"),
            EditSpec("FoldScalarMatMul"),
            EditSpec("GroupQueryHeads", (
                self._config.num_attention_heads,
                self._config.num_key_value_heads,
                self._config.hidden_size // self._config.num_attention_heads,
                ctx("chunk_len"),
            )),
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
        self, model: onnx.ModelProto, chunk_len: int = 1
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
        
        editor = SmolLM2OnnxGraphEditor(
            graph,
            self._onnx_export_dtype,
            **self._editor_dump_kwargs(self._export_path_for_component("model")),
        )
        editor.fix_io(self._max_gen_tokens, chunk_len=chunk_len)
        # The chunk's queries span start..start+B-1; decode is the single start.
        # Fail closed: a chunk whose rotary chain was not recognized would
        # export wrong positions that only the ORT gate could catch.
        if chunk_len > 1 and not editor.fix_native_rotary_positions(chunk_len):
            raise ValueError(
                "SmolLM2 native chunk: the rotary position chain was not recognized"
            )

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
                "chunk_len": chunk_len,
            },
        )

        new_model = editor.to_onnx(override_ir=model.ir_version)
        return new_model

    def _patch_static_model(self, model_path: str | os.PathLike, component: str = "model"):
        model = load_onnx_lazy(model_path)
        editor = SmolLM2OnnxGraphEditor.from_onnx(
            model,
            self._onnx_export_dtype,
            **self._editor_dump_kwargs(Path(model_path)),
        )

        # Native prefill chunks carry B tokens; decode is the single-token body.
        chunk_len = (
            int(component.removeprefix("model_prefill_"))
            if component.startswith("model_prefill_")
            else 1
        )

        embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
        editor.apply_specs(
            self.graph_edit_blocks()["model.patch"],
            self._harness,
            # The chunk reuses decode's extracted table; only decode saves it.
            {"embeddings_path": embeddings_npy if component == "model" else None,
             "chunk_len": chunk_len},
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
            if self._read_only_kv_cache:
                editor.read_only_kv_cache(
                    self._config.num_attention_heads,
                    self._config.num_key_value_heads,
                    self._config.hidden_size // self._config.num_attention_heads,
                    chunk_len,
                )

        if self._split_lm_head:
            if component == "model":
                lm_head_path = Path(model_path).parent / "lm_head.onnx"
                editor.split_lm_head(lm_head_path)
                self._export_paths["lm_head"] = lm_head_path
            else:
                # The chunk reuses decode's lm_head.onnx: cut the graph at the
                # head and keep the last token's hidden row for the family ABI.
                editor.split_lm_head(None)
                editor.take_last_token_hidden(chunk_len)

        new_model = editor.to_onnx(override_ir=model.ir_version)
        self._save_component_model(new_model, model_path)

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self.check_model(self._components["model"])
        self._components["model"] = self._make_model_static(self._components["model"])

    def _make_native_prefill(self) -> dict[str, onnx.ModelProto]:
        components = {}
        source_model = self._load_onnx()["model"]
        for size in self._native_batch_prefill:
            self._logger.info(
                "(model_prefill_%d) Making graph static with %d tokens...",
                size,
                size,
            )
            components[f"model_prefill_{size}"] = self._make_model_static(
                source_model,
                chunk_len=size,
            )
        return components

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        self._patch_static_model(model_path, component)
        self._copy_runtime_assets(
            Path(model_path).parent,
            self._onnx_dir,
            include_npy_data=False,
        )

    def validate_onnx(self, n_iters: int = 5):
        validate_decoder_only_onnx(
            self._logger,
            SmolLM2Static,
            SmolLM2Dynamic,
            self._export_paths["model"],
            self._onnx_dir / "model.onnx",
            static_models=self._static_models,
            max_gen_tokens=self._max_gen_tokens,
            instruct_model=self._instruct_model,
            repo_id=self._hf_repo,
            n_iters=n_iters,
            lm_head_path=self._export_paths.get("lm_head"),
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

def export_smollm2_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = SmolLM2ModelExporter(
        args.model_size,
        args.instruct_model,
        args.extract_embeddings,
        args.keep_individual_kv_io,
        not args.dynamic_models,
        max_gen_tokens=args.max_gen_tokens,
        # SmolLM2 has no legacy prefill builder: the shared --batch-prefill
        # flag is not registered for it, so there is nothing to forward.
        native_batch_prefill=args.native_batch_prefill,
        batch_prefill_sizes=args.batch_prefill_sizes,
        split_lm_head=args.split_lm_head,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        dynamic_quantize=args.dynamic_quantize,
        convert_dtypes=args.convert_dtypes,
        split_weights=args.split_weights,
        read_only_kv_cache=args.read_only_kv_cache,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(args))
    if args.view_graph_edits:
        print(render_graph_edit_plan(exporter.describe_graph_edits()))
        return
    from ...model_export.prefill import check_native_cli_args
    check_native_cli_args(args)
    exporter.export_onnx(validate=not args.skip_validation, cleanup=not args.no_onnx_cleanup)
    if args.dynamic_quantize:
        exporter.dynamic_quantize_models(
            skip=args.dynamic_quantization_skip_model,
            analyze_nodes=args.dynamic_quantize_analyze_nodes,
            uint8_weights=args.dynamic_quantize_uint8_weights,
            per_tensor=args.dynamic_quantize_per_tensor,
        )
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
