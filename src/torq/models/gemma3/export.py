# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import ml_dtypes
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from tokenizers import Tokenizer
from transformers import AutoConfig

from . import add_gemma3_export_args
from ._graph import Gemma3OnnxGraphEditor
from ._inference import Gemma3Dynamic, Gemma3Static
from ._trim_vocab import (
    TrimmedVocabSpec,
    build_trimmed_vocab_spec,
    load_json,
)
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.hf import optimum_export_onnx, hf_download_source_model

from ...utils.logging import (
    configure_logging,
)


class Gemma3ModelExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["270m", "1b"] = "270m",
        instruct_model: bool = False,
        extract_embeddings: bool = False,
        keep_individual_kv_io: bool = False,
        static_models: bool = True,
        *,
        hf_repo: str | None = None,
        hf_repo_subdir: str | os.PathLike | None = None,
        max_gen_tokens: int = 256,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        trim_vocab: bool = False,
        split_lm_head: bool = False,
        trim_vocab_groups: list[str] | None = None,
        trim_byte_fallback: bool = True,
        **edit_args
    ):
        self._instruct_model = instruct_model
        self._extract_embeddings = extract_embeddings
        self._keep_individual_kv_io = keep_individual_kv_io
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = onnx_source_dir
        self._trim_vocab = trim_vocab
        self._split_lm_head = split_lm_head
        self._trim_vocab_groups = tuple(trim_vocab_groups or ("latin", "punct", "digits"))
        self._trim_byte_fallback = trim_byte_fallback
        self._hf_repo_subdir = hf_repo_subdir
        if hf_repo:
            self._hf_repo = hf_repo
        else:
            self._hf_repo = f"google/gemma-3-{model_size}"
            if self._instruct_model:
                self._hf_repo += "-it"
        self._source_asset_dirs = [
            path
            for path in (
                Path(self._onnx_source_dir) if self._onnx_source_dir else None,
                Path(models_dir) / self._hf_repo / "source" / "fp32",
            )
            if isinstance(path, Path) and path.exists()
        ]
        local_config_dir = next(
            (path for path in self._source_asset_dirs if (path / "config.json").exists()),
            None,
        )
        if local_config_dir is not None:
            self._config = AutoConfig.from_pretrained(local_config_dir, local_files_only=True)
        else:
            try:
                self._config = AutoConfig.from_pretrained(self._hf_repo, local_files_only=True)
            except OSError:
                self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._trimmed_vocab_spec: TrimmedVocabSpec | None = None
        self._source_tokenizer_json: dict | None = None
        self._source_config_json: dict | None = None
        if self._trim_vocab and not static_models:
            raise ValueError("`--trim-vocab` is currently supported only for static Gemma exports")
        if self._split_lm_head and not static_models:
            raise ValueError("`--split-lm-head` is currently supported only for static Gemma exports")

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
                self._logger.info("Downloaded model.onnx and peripheral files from %s", self._hf_repo)
            except (FileNotFoundError, Exception):
                optimum_export_onnx(
                    onnx_dir, self._hf_repo, self._model_dtype, ["model.onnx"], opt_level=None
                )
                self._logger.info("Exported %s to ONNX @ '%s' via optimum-cli", self._hf_repo, onnx_dir)
            if self._hf_repo_subdir:
                onnx_dir /= Path(self._hf_repo_subdir)
                self._config = AutoConfig.from_pretrained(onnx_dir / "config.json")
                self._hidden_size = int(self._config.hidden_size)
                self._vocab_size = int(self._config.vocab_size)
        model_type = "trim" if self._trim_vocab else "full"
        model_topology = "split_lm_head" if self._split_lm_head else "unified"
        export_dir = (
            self._models_dir / 
            "export" / 
            model_type /
            model_topology /
            "onnx" / 
            self._model_dtype / 
            ("static" if self._static_models else "dynamic")
        )
        convert_dir = (
            self._models_dir 
            / "export"
            / model_type
            / model_topology
            / "onnx"
            / "converted"
            / ("static" if self._static_models else "dynamic")
        )
        torq_dir = (
            self._models_dir
            / "export"
            / model_type
            / model_topology
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

    def _resolve_source_asset_path(self, asset_name: str) -> Path:
        for asset_dir in self._source_asset_dirs:
            asset_path = asset_dir / asset_name
            if asset_path.exists():
                return asset_path
        try:
            return Path(hf_hub_download(self._hf_repo, asset_name, local_files_only=True))
        except Exception:
            return Path(hf_hub_download(self._hf_repo, asset_name))

    def _get_trimmed_vocab_spec(self) -> TrimmedVocabSpec:
        if not self._trim_vocab:
            raise RuntimeError("Trimmed vocab spec requested when trim-vocab export is disabled")
        if self._trimmed_vocab_spec is not None:
            return self._trimmed_vocab_spec

        tokenizer_json_path = self._resolve_source_asset_path("tokenizer.json")
        config_json_path = self._resolve_source_asset_path("config.json")
        self._source_tokenizer_json = load_json(tokenizer_json_path)
        self._source_config_json = load_json(config_json_path)
        tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        self._trimmed_vocab_spec = build_trimmed_vocab_spec(
            tokenizer=tokenizer,
            tokenizer_json=self._source_tokenizer_json,
            config_json=self._source_config_json,
            selected_groups=self._trim_vocab_groups,
            byte_fallback=self._trim_byte_fallback,
        )
        self._logger.info(
            "(model) Trimmed vocab enabled: %d -> %d tokens",
            self._trimmed_vocab_spec.model_vocab_size,
            self._trimmed_vocab_spec.trimmed_vocab_size,
        )
        return self._trimmed_vocab_spec

    @staticmethod
    def _replace_initializer(
        model: onnx.ModelProto,
        initializer_name: str,
        values: np.ndarray,
    ) -> None:
        for idx, initializer in enumerate(model.graph.initializer):
            if initializer.name == initializer_name:
                model.graph.initializer[idx].CopyFrom(
                    numpy_helper.from_array(values, name=initializer_name)
                )
                return
        raise ValueError(f"Initializer '{initializer_name}' not found in ONNX model")

    @staticmethod
    def _update_value_info_shape(
        value_info: onnx.ValueInfoProto,
        shape: Sequence[int | str],
    ) -> None:
        dims = value_info.type.tensor_type.shape.dim
        del dims[:]
        for dim in shape:
            new_dim = dims.add()
            if isinstance(dim, int):
                new_dim.dim_value = int(dim)
            else:
                new_dim.dim_param = str(dim)

    @staticmethod
    def _update_logits_metadata(
        model: onnx.ModelProto,
        trimmed_vocab_size: int,
    ) -> None:
        updated = False
        for value_info in model.graph.output:
            if value_info.name != "logits":
                continue
            Gemma3ModelExporter._update_value_info_shape(value_info, [1, 1, trimmed_vocab_size])
            updated = True
        for value_info in model.graph.value_info:
            if value_info.name != "logits":
                continue
            Gemma3ModelExporter._update_value_info_shape(value_info, [1, 1, trimmed_vocab_size])
            updated = True
        if not updated:
            raise ValueError("Could not find logits output metadata for trimmed-vocab export")

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
        
        editor = Gemma3OnnxGraphEditor(graph, self._onnx_export_dtype)
        editor.fix_io(self._max_gen_tokens)

        # Remove redundant Cast ops
        editor.remove_redundant_casts()
        # Remove isNaN ops
        editor.remove_isNaN()

        cur_len_2d = gs.Variable("position_ids", dtype=np.int64, shape=[1, 1])
        graph.inputs.append(cur_len_2d)
        cur_len = graph.layer(
            name="current_len_to_1d",
            op="Squeeze",
            inputs=[cur_len_2d, [0]],
            outputs=[gs.Variable(cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1])],
        )[0]
        cur_len_scalar = graph.layer(
            name="current_len_to_scalar",
            op="Squeeze",
            inputs=[cur_len, [0]],
            outputs=[gs.Variable(cur_len.name + "_squeezed", dtype=np.int64, shape=[])],
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
        editor = Gemma3OnnxGraphEditor.from_onnx(model, self._onnx_export_dtype)

        # Eliminate data-preserving Transpose ops (no-op K/V head transposes, K^T when seq==head_dim)
        editor.eliminate_transposes()
        # Collapse consecutive Reshape chains
        editor.collapse_reshape_chains()
        # Collapse Unsqueeze→Expand→Reshape GQA broadcast into single Expand (KV_heads=1)
        editor.collapse_gqa_broadcast()
        # Fold MatMul A @ B where B is a scalar into Mul
        editor.fold_scalar_matmul()
        # Broadcast op inputs to match output shape
        if self._broadcast_ops is not None:
            editor.broadcast_op_inputs(
                ops=self._broadcast_ops,
            )
        else:
            # Eliminate explicit Expand ops
            editor.eliminate_expands([
                # supports kernel broadcast in Torq-compiler
                "Mul", "MatMul", "Where",
                # produced by collapse_gqa_broadcast()
                "Transpose",
            ])

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

        if not self._keep_individual_kv_io:
            editor.combine_kv_io_tensors([
                1,                                                              # B
                self._config.num_key_value_heads,                               # H
                self._max_gen_tokens,                                           # L
                self._config.head_dim                                           # D
            ])

        if self._trim_vocab:
            spec = self._get_trimmed_vocab_spec()
            token_id_lut_path = Path(model_path).parent / "token_id_lut.npy"
            editor.trim_lm_head_vocab(
                kept_token_ids=np.array(spec.kept_model_ids, dtype=np.int64),
                save_lut=token_id_lut_path,
            )

        editor.reorder_graph_input("position_ids", 1)

        if self._split_lm_head:
            lm_head_path = Path(model_path).parent / "lm_head.onnx"
            editor.split_lm_head(lm_head_path)
            lm_head_model = onnx.load(lm_head_path)
            lm_head_model.ir_version = model.ir_version
            onnx.save(self.check_model(lm_head_model), lm_head_path)
            self._export_paths["lm_head"] = lm_head_path
            self._logger.info("(lm_head) Saved split LM head to '%s'", str(lm_head_path))

        new_model = editor.to_onnx(override_ir=model.ir_version)
        onnx.save(new_model, model_path)

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self.check_model(self._components["model"])
        self._components["model"] = self._make_model_static(self._components["model"])

    def apply_post_static_patches(self, model_path: str | os.PathLike, _):
        self._patch_static_model(model_path)
        self._copy_runtime_assets(
            Path(model_path).parent,
            self._onnx_dir,
            include_npy_data=False,
        )

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
            runner = Gemma3Static.from_onnx(
                self._export_paths["model"],
                self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo
            )
        else:
            runner = Gemma3Dynamic.from_onnx(
                self._export_paths["model"],
                max_gen_tokens=self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo
            )
        val_runner = Gemma3Dynamic.from_onnx(
            self._onnx_dir /  "model.onnx",
            max_gen_tokens=self._max_gen_tokens,
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
        external_data = [
            (self._export_paths["model"].parent / "token_embeddings.npy", np.dtype(ml_dtypes.bfloat16)),
        ]
        if self._trim_vocab:
            external_data.append((self._export_paths["model"].parent / "token_id_lut.npy", np.dtype(np.int32)))
        result = super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            external_data=external_data
        )
        self._copy_runtime_assets(self._convert_dir, self._export_dir, include_npy_data=False)
        return result

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

def export_gemma3_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = Gemma3ModelExporter(
        args.model_size,
        args.instruct_model,
        args.extract_embeddings,
        args.keep_individual_kv_io,
        not args.dynamic_models,
        hf_repo=args.hf_repo,
        hf_repo_subdir=args.hf_repo_subdir,
        max_gen_tokens=args.max_gen_tokens,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        trim_vocab=args.trim_vocab,
        split_lm_head=args.split_lm_head,
        trim_vocab_groups=args.trim_vocab_groups,
        trim_byte_fallback=args.trim_byte_fallback,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
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
    parser = argparse.ArgumentParser(description="Export Gemma3 to Torq")
    add_gemma3_export_args(parser)
    export_gemma3_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
