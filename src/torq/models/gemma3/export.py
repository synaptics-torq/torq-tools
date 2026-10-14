# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
from pathlib import Path
from typing import Final, Literal
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
from ...graph_edit.harness import EditSpec, GraphEditHarness, ctx, render_graph_edit_plan
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.validation import validate_decoder_only_onnx
from ...model_export.hf import optimum_export_onnx, hf_download_source_model
from ...utils.onnx import load_onnx_lazy, validate_onnx_source_dir

from ...utils.logging import (
    configure_logging,
)


_GEMMA3_MODEL_FILENAMES: Final[tuple[tuple[str, ...], tuple[str, ...]]] = (
    ("transformer.onnx", "lm_head.onnx"),
    ("model.onnx",),
)

_GEMMA3_PTD_MODEL_FILENAMES: Final[tuple[tuple[str, ...], tuple[str, ...]]] = (
    ("ptd_transformer.onnx", "ptd_lm_head.onnx"),
    ("ptd_model.onnx",),
)

# Sizes whose base (non-instruct) checkpoint is published with a `-pt` suffix.
_GEMMA3_PRETRAINED_SUFFIX_SIZES: Final[frozenset[str]] = frozenset({"1b"})


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
        batch_prefill: int | None = None,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        dynamic_quantize: bool = False,
        convert_dtypes: bool = False,
        trim_vocab: bool = False,
        split_lm_head: bool = False,
        trim_vocab_groups: list[str] | None = None,
        trim_byte_fallback: bool = True,
        split_weights: bool = False,
        ptd_rows: int | None = None,
        **edit_args
    ):
        self._instruct_model = instruct_model
        self._extract_embeddings = extract_embeddings
        self._keep_individual_kv_io = keep_individual_kv_io
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = validate_onnx_source_dir(
            onnx_source_dir, required_files=("tokenizer.json",)
        )
        self._trim_vocab = trim_vocab
        self._split_lm_head = split_lm_head
        self._export_model_filenames = _GEMMA3_MODEL_FILENAMES[0 if split_lm_head else 1]
        self._trim_vocab_groups = tuple(trim_vocab_groups or ("latin", "punct", "digits"))
        self._trim_byte_fallback = trim_byte_fallback
        self._hf_repo_subdir = hf_repo_subdir
        if hf_repo:
            self._hf_repo = hf_repo
        else:
            self._hf_repo = f"google/gemma-3-{model_size}"
            if self._instruct_model:
                self._hf_repo += "-it"
            elif model_size in _GEMMA3_PRETRAINED_SUFFIX_SIZES:
                # Upstream publishes the base 1B checkpoint as `-pt`; 270m has a
                # bare repo, so the suffix cannot be applied unconditionally.
                self._hf_repo += "-pt"
        self._source_asset_dirs = (
            [self._onnx_source_dir]
            if self._onnx_source_dir is not None
            else [Path(models_dir) / self._hf_repo / "source" / "fp32"]
        )
        if ptd_rows is not None and ptd_rows > 0:
            if not static_models:
                raise ValueError("`--ptd-rows` is currently supported only for static Gemma exports")
            if ptd_rows < 2:
                raise ValueError(f"`--ptd-rows` must be at least 2 (anchor + 1 tree row), got {ptd_rows}")
            if ptd_rows > max_gen_tokens:
                raise ValueError(
                    f"`--ptd-rows` ({ptd_rows}) cannot exceed `--max-gen-tokens` ({max_gen_tokens})"
                )
            self._ptd_rows = int(ptd_rows)
        else:
            self._ptd_rows = None
        if self._onnx_source_dir is not None:
            self._config = AutoConfig.from_pretrained(self._onnx_source_dir, local_files_only=True)
        else:
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
        if batch_prefill is not None:
            if batch_prefill < 1:
                raise ValueError(f"`--batch-prefill` must be positive, got {batch_prefill}")
            if batch_prefill > max_gen_tokens:
                raise ValueError(
                    f"`--batch-prefill` ({batch_prefill}) cannot exceed `--max-gen-tokens` ({max_gen_tokens})"
                )
            if not self._split_lm_head:
                raise ValueError(
                    "`--batch-prefill` requires `--split-lm-head` "
                    "(disable the prefill export with `--batch-prefill 0`)"
                )
        self._batch_prefill = batch_prefill

        opt_config = ORTOptimizerConfig(
            num_heads=self._config.num_attention_heads,
            hidden_size=self._config.hidden_size,
        )
        opt_configs = {"model": opt_config, "model_prefill": opt_config}
        if self._ptd_rows is not None:
            # The static PTD body bakes the causal mask as "all committed keys
            # visible"; that equals Gemma's local (sliding-window) semantics only
            # while the cache never outgrows the window.
            sliding_window = getattr(self._config, "sliding_window", None)
            if sliding_window is not None and max_gen_tokens > sliding_window:
                raise ValueError(
                    f"`--ptd-rows` requires `--max-gen-tokens` ({max_gen_tokens}) <= "
                    f"sliding_window ({sliding_window}) so the local-attention window "
                    "stays inactive (same dependence as the static decode baseline)"
                )
            opt_configs["model_ptd"] = opt_config

        super().__init__(
            "fp32",
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            dynamic_quantize=dynamic_quantize,
            convert_dtypes=convert_dtypes,
            opt_configs=opt_configs,
            split_weights=split_weights,
        )

        # PTD component file layout mirrors the regular model's.
        self._ptd_export_model_filenames = _GEMMA3_PTD_MODEL_FILENAMES[0 if split_lm_head else 1]

    def _setup_dirs(self) -> list[Path]:
        onnx_dir, export_dir, quantize_dir, convert_dir, torq_dir = [None] * 5
        if self._onnx_source_dir is not None:
            onnx_dir = self._onnx_source_dir
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
        suffix = "static" if self._static_models else "dynamic"
        root = self._models_dir / "export" / model_type / model_topology
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

    def _export_path_for_component(self, component: str) -> Path:
        if component == "model":
            return self._export_dir / self._export_model_filenames[0]
        if component == "model_prefill":
            return self._export_dir / "transformer_prefill.onnx"
        if component == "model_ptd":
            return self._export_dir / self._ptd_export_model_filenames[0]
        if component == "ptd_lm_head":
            return self._export_dir / self._ptd_export_model_filenames[1]
        return super()._export_path_for_component(component)

    def _resolve_source_asset_path(self, asset_name: str) -> Path:
        for asset_dir in self._source_asset_dirs:
            asset_path = asset_dir / asset_name
            if asset_path.exists():
                return asset_path
        if self._onnx_source_dir is not None:
            raise FileNotFoundError(
                f"Expected {asset_name} in ONNX source directory: '{self._onnx_source_dir}'"
            )
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

    def graph_edit_blocks(self) -> dict[str, list[EditSpec]]:
        blocks: dict[str, list[EditSpec]] = {}
        blocks["model.static"] = [
            EditSpec("RemoveRedundantCasts"),
            EditSpec("RemoveIsNaN"),
        ]
        blocks["model.static (KV cache)"] = [
            EditSpec("ReplaceDynamicKVCache", (ctx("cur_len"), ctx("max_tokens"), ctx("chunk_len"))),
            EditSpec(
                "MaskFutureAttentionScores",
                (ctx("cur_len"), ctx("max_tokens"), ctx("export_dtype"), False, ctx("chunk_len")),
            ),
            EditSpec("AddCurrLenInput", (ctx("cur_len"),)),
            EditSpec("ConvertToStaticIndex", (ctx("chunk_len"),)),
        ]
        if self._ptd_rows is not None:
            # PTD body (Gao et al. 2026, arXiv:2607.10661; attribution in
            # torq.inference.ptd): packed anchor + tree rows; per-row
            # positions and a host-supplied visibility matrix replace the
            # consecutive-chunk causal machinery, and tree rows stay scratch
            # (see PTDKVCache).
            # The mask edit runs first: it is the consumer of the fresh
            # tree_vis graph input (per-edit cleanup drops inputs without
            # consumers).
            blocks["model_ptd (KV cache)"] = [
                EditSpec(
                    "PTDTreeAttentionMask",
                    (ctx("cur_len"), ctx("max_tokens"), ctx("export_dtype"), ctx("tree_vis"), ctx("rows")),
                ),
                EditSpec("PTDKVCache", (ctx("cur_len"), ctx("max_tokens"), ctx("rows"))),
                EditSpec("AddCurrLenInput", (ctx("cur_len"),)),
                EditSpec("ConvertToStaticIndex", (ctx("rows"), ctx("positions_input"))),
            ]

        patch = [
            EditSpec("EliminateTranspose"),
            EditSpec("CollapseReshapeChain"),
            EditSpec("CollapseGQABroadcast"),
            EditSpec("FoldScalarMatMul"),
        ]
        if self._broadcast_ops is not None:
            patch.append(EditSpec("BroadcastOpInputs", (self._broadcast_ops,)))
        else:
            patch.append(EditSpec("EliminateExpand", (["Mul", "MatMul", "Where", "Transpose"],)))
        if self._extract_embeddings:
            patch.append(EditSpec(
                "ExtractConstantLUT",
                ((self._vocab_size, self._hidden_size), ctx("save_embeddings_path"), "token_embedding"),
            ))
            patch.append(EditSpec(
                "ComputeDequantizedLUT", (ctx("embeddings_path"), ctx("export_dtype"))
            ))
        blocks["model.patch"] = patch

        if self._trim_vocab:
            blocks["model.patch (trim vocab)"] = [
                EditSpec("TrimLMHeadVocab", (ctx("kept_token_ids"), "logits", ctx("token_id_lut_path"), False))
            ]
        if self._split_lm_head:
            blocks["model.patch (split LM head)"] = [
                EditSpec("SplitLMHead", (ctx("lm_head_path"), "logits", "last_hidden_states"))
            ]
        if self._batch_prefill is not None:
            blocks["model.patch (batch prefill)"] = [EditSpec("TakeLastToken")]
        return blocks

    def _make_model_static(
        self, model: onnx.ModelProto, component: str = "model", chunk_len: int = 1
    ) -> onnx.ModelProto:
        """
        Make model static by replacing dynamic dimensions with fixed values and applying necessary transformations.

        Replaces KV caching and other dynamic operations with static equivalents.

        Args:
            model (onnx.ModelProto): ONNX decoder model to modify
            component (str): Component name (selects the staticization scheme)
            chunk_len (int): Fixed query-row count for this component

        Returns:
            onnx.ModelProto: The modified decoder model with static dimensions and transformations applied

        Raises:
            ValueError: If an unexpected dynamic dimension is found in the model inputs, outputs, or nodes
        """
        graph: gs.Graph = gs.import_onnx(model)
        ptd = component == "model_ptd"
        rows = self._ptd_rows if ptd else chunk_len
        self._logger.debug(
            "Set export data type to %s for model data type %s",
            onnx.helper.tensor_dtype_to_string(self._onnx_export_dtype), self._model_dtype
        )
        
        editor = Gemma3OnnxGraphEditor(
            graph,
            self._onnx_export_dtype,
            **self._editor_dump_kwargs(self._export_path_for_component(component)),
        )
        editor.fix_io(self._max_gen_tokens, chunk_len=rows)

        blocks = self.graph_edit_blocks()
        # Remove redundant Cast + isNaN ops
        editor.apply_specs(blocks["model.static"], self._harness)

        if ptd:
            # PTD: per-row logical positions as a [1, rows] input; the anchor
            # position (first row) drives the KV/mask edits.
            cur_len_2d = gs.Variable("position_ids", dtype=np.int64, shape=[1, rows])
            graph.inputs.append(cur_len_2d)
            positions_1d = graph.layer(
                name="current_len_positions_1d",
                op="Squeeze",
                inputs=[cur_len_2d, [0]],
                outputs=[gs.Variable("position_ids_1d", dtype=np.int64, shape=[rows])],
            )[0]
            cur_len = graph.layer(
                name="current_len_ptd",
                op="Gather",
                inputs=[positions_1d, [0]],
                outputs=[gs.Variable("cur_len_ptd_1d", dtype=np.int64, shape=[1])],
            )[0]
            cur_len_scalar = None
        else:
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
            positions_1d = None

        ctx_values = {
            "cur_len": cur_len,
            "max_tokens": self._max_gen_tokens,
            "export_dtype": self._onnx_export_dtype,
            "chunk_len": rows,
        }
        kv_cache_block = "model.static (KV cache)"
        if ptd:
            tree_vis = gs.Variable(
                "tree_vis", dtype=np.int32, shape=[1, 1, rows, rows - 1]
            )
            graph.inputs.append(tree_vis)
            ctx_values.update({
                "rows": rows,
                "tree_vis": tree_vis,
                "positions_input": positions_1d,
            })
            kv_cache_block = "model_ptd (KV cache)"

        editor.apply_specs(
            blocks[kv_cache_block],
            self._harness,
            ctx_values,
        )

        new_model = editor.to_onnx(override_ir=model.ir_version)
        return new_model

    def _patch_static_model(self, model_path: str | os.PathLike, component: str):
        model = load_onnx_lazy(model_path)
        editor = Gemma3OnnxGraphEditor.from_onnx(
            model,
            self._onnx_export_dtype,
            **self._editor_dump_kwargs(Path(model_path)),
        )

        blocks = self.graph_edit_blocks()
        embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
        editor.apply_specs(
            blocks["model.patch"], self._harness,
            {
                "embeddings_path": embeddings_npy,
                "save_embeddings_path": embeddings_npy if component == "model" else None,
                "export_dtype": self._onnx_export_dtype,
            }
        )

        if self._extract_embeddings:
            editor.reorder_graph_input("token_embedding", 0)

        if not self._keep_individual_kv_io:
            editor.combine_kv_io_tensors([
                1,                                                              # B
                self._config.num_key_value_heads,                               # H
                self._max_gen_tokens,                                           # L
                self._config.head_dim                                           # D
            ])
            if component == "model_ptd":
                editor.combine_kv_io_tensors(
                    [1, self._config.num_key_value_heads, self._ptd_rows - 1, self._config.head_dim],
                    kv_layer_re=r"\.(\d+)\.(key|value)_scratch$",
                    combined_suffix="_scratch",
                )

        if self._trim_vocab:
            spec = self._get_trimmed_vocab_spec()
            token_id_lut_path = Path(model_path).parent / "token_id_lut.npy"
            editor.apply_specs(
                blocks["model.patch (trim vocab)"],
                self._harness,
                {
                    "kept_token_ids": np.array(spec.kept_model_ids, dtype=np.int64),
                    "token_id_lut_path": token_id_lut_path if component == "model" else None,
                },
            )

        editor.reorder_graph_input("position_ids", 1)

        if component == "model_prefill":
            editor.apply_specs(blocks["model.patch (batch prefill)"], self._harness)

        if self._split_lm_head:
            # PTD keeps all rows (no TakeLastToken): its block head projects
            # the full [1, rows, hidden] stack in one call.
            lm_head_component = "ptd_lm_head" if component == "model_ptd" else "lm_head"
            lm_head_path = self._export_path_for_component(lm_head_component)
            editor.apply_specs(
                blocks["model.patch (split LM head)"],
                self._harness,
                {"lm_head_path": lm_head_path},
            )
            lm_head_model = load_onnx_lazy(lm_head_path)
            lm_head_model.ir_version = model.ir_version
            self._save_component_model(self.check_model(lm_head_model), lm_head_path)
            self._export_paths[lm_head_component] = lm_head_path
            self._logger.info("(%s) Saved split LM head to '%s'", lm_head_component, str(lm_head_path))

        new_model = editor.to_onnx(override_ir=model.ir_version)
        self._save_component_model(new_model, model_path)

    def make_static(self):
        source_model = self.check_model(self._components["model"])
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self._make_model_static(source_model)
        if self._batch_prefill is not None:
            self._logger.info(
                "(model_prefill) Making graph static with %d tokens...", self._batch_prefill
            )
            self._components["model_prefill"] = self._make_model_static(
                source_model,
                component="model_prefill",
                chunk_len=self._batch_prefill,
            )
        if self._ptd_rows is not None:
            self._logger.info(
                "(model_ptd) Making graph static with %d PTD rows...", self._ptd_rows
            )
            self._components["model_ptd"] = self._make_model_static(
                source_model,
                component="model_ptd",
                chunk_len=self._ptd_rows,
            )

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
            Gemma3Static,
            Gemma3Dynamic,
            self._export_paths["model"],
            self._onnx_dir / "model.onnx",
            static_models=self._static_models,
            max_gen_tokens=self._max_gen_tokens,
            instruct_model=self._instruct_model,
            repo_id=self._hf_repo,
            n_iters=n_iters,
            lm_head_path=self._export_paths.get("lm_head"),
            prefill_model_path=self._export_paths.get("model_prefill"),
            prefill_size=self._batch_prefill,
        )
        if self._ptd_rows is not None:
            self._validate_ptd_onnx()

    def _validate_ptd_onnx(self):
        from ...model_export.validation_ptd import validate_ptd_onnx

        emb_path = self._export_paths["model"].parent / "token_embeddings.npy"
        token_embeddings = np.load(emb_path) if emb_path.exists() else None
        lut_path = self._export_paths["model"].parent / "token_id_lut.npy"
        validate_ptd_onnx(
            self._logger,
            self._export_paths["model_ptd"],
            self._export_paths["model"],
            ptd_lm_head_path=self._export_paths.get("ptd_lm_head"),
            decode_lm_head_path=self._export_paths.get("lm_head"),
            token_embeddings=token_embeddings,
            token_id_lut=np.load(lut_path) if self._trim_vocab else None,
            vocab_size=self._vocab_size,
            rows=self._ptd_rows,
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
        batch_prefill=args.batch_prefill or None,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        dynamic_quantize=args.dynamic_quantize,
        convert_dtypes=args.convert_dtypes,
        trim_vocab=args.trim_vocab,
        split_lm_head=args.split_lm_head,
        trim_vocab_groups=args.trim_vocab_groups,
        trim_byte_fallback=args.trim_byte_fallback,
        split_weights=args.split_weights,
        ptd_rows=args.ptd_rows or None,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops
    )
    exporter.set_graph_edit_harness(GraphEditHarness.from_args(args))
    if args.view_graph_edits:
        print(render_graph_edit_plan(exporter.describe_graph_edits()))
        return
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
        # The PTD components are an ONNX-only experimental variant: the
        # compiler path for the fixed-row body is unvalidated, so keep it out
        # of the standard compile loop.
        ptd_skip = ["model_ptd", "ptd_lm_head"] if args.ptd_rows else []
        if ptd_skip:
            exporter._logger.info("(gemma3) --ptd-rows: skipping PTD components in the torq compile (ONNX-only experimental)")
        exporter.export_torq(
            torq_compile_args=args.compile_flags or [],
            use_binary=args.use_binary,
            local_compile=args.local_compile,
            compiler_path=args.compiler_path,
            skip=ptd_skip,
        )


def main():
    parser = argparse.ArgumentParser(description="Export Gemma3 to Torq")
    add_gemma3_export_args(parser)
    export_gemma3_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
