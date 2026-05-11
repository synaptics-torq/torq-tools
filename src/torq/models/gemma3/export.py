# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import ml_dtypes
from transformers import AutoConfig
from torq.utils.logging import configure_logging

from . import add_gemma3_export_args
from ._graph import Gemma3OnnxGraphEditor
from ._inference import Gemma3Dynamic, Gemma3Static
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig
from ...model_export.hf import optimum_export_onnx


class Gemma3ModelExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["270m", "1b"] = "270m",
        instruct_model: bool = False,
        extract_embeddings: bool = False,
        keep_individual_kv_io: bool = False,
        static_models: bool = True,
        *,
        max_gen_tokens: int = 256,
        model_dtype: str = "fp32",
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
        self._is_quantized = model_dtype in ("int4", "int8")
        self._model_quant_dtype = model_dtype  # "int4", "int8", or "fp32"
        self._hf_repo = f"google/gemma-3-{model_size}"
        if self._instruct_model:
            self._hf_repo += "-it"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)
        self._replace_int_bf16_cast = edit_args.get("replace_int_bf16_cast", False)
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._dequantize_weights = edit_args.get("dequantize_weights", False)
        self._dequantize_weights_linear = edit_args.get("dequantize_weights_linear", False)
        self._simulate_bf16 = edit_args.get("simulate_bf16", False)
        self._fp8_e5m2 = edit_args.get("fp8_e5m2", False)
        self._fp8_e4m3 = edit_args.get("fp8_e4m3", False)

        super().__init__(
            "fp32",
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs={} if self._is_quantized else {"model": ORTOptimizerConfig(
                num_heads=self._config.num_attention_heads,
                hidden_size=self._config.hidden_size
            )}
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir, export_dir, convert_dir, iree_dir = [None] * 4
        if self._onnx_source_dir:
            onnx_source_dir = Path(self._onnx_source_dir)
            if not onnx_source_dir.exists():
                # Auto-download from HuggingFace based on directory name
                variant = onnx_source_dir.name  # e.g. "int4" or "int8"
                if variant in ("int4", "int8"):
                    self._download_from_hf(onnx_source_dir, variant=variant)
                else:
                    raise FileNotFoundError(
                        f"Source directory not found: '{onnx_source_dir}'"
                    )
            onnx_dir = onnx_source_dir
        elif self._is_quantized:
            onnx_dir = self._models_dir / "source" / self._model_quant_dtype
            if not onnx_dir.exists():
                self._download_from_hf(onnx_dir, variant=self._model_quant_dtype)
        else:
            onnx_dir = self._models_dir / "source" / self._model_dtype
            onnx_dir.mkdir(parents=True, exist_ok=True)
            optimum_export_onnx(
                onnx_dir, self._hf_repo, self._model_dtype, ["model.onnx"], opt_level=None
            )
        dtype_tag = self._model_quant_dtype if self._is_quantized else self._model_dtype
        # Append "_dql" suffix for DequantizeLinear (runtime dequant) exports
        dql_suffix = "_dql" if self._dequantize_weights_linear else ""
        export_dir = (
            self._models_dir / 
            "export" / 
            "onnx" / 
            (dtype_tag + dql_suffix) / 
            ("static" if self._static_models else "dynamic")
        )
        # converted dir: "converted" for int4/fp32, "int8_converted" for int8
        convert_tag = "int8_converted" if self._model_quant_dtype == "int8" else "converted"
        convert_dir = (
            self._models_dir 
            / "export"
            / "onnx"
            / (convert_tag + dql_suffix)
            / ("static" if self._static_models else "dynamic")
        )
        iree_dir = (
            self._models_dir
            / "export"
            / "iree"
            / ((convert_tag if self._convert_dtypes else dtype_tag) + dql_suffix)
            / ("static" if self._static_models else "dynamic")
        )
        return onnx_dir, export_dir, convert_dir, iree_dir

    # HuggingFace repo containing pre-quantized ONNX models
    HF_ONNX_REPO = "onnx-community/gemma-3-270m-it-ONNX"

    def _download_from_hf(self, target_dir: Path, variant: str = "int4"):
        """Download quantized ONNX model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        files_map = {
            "int4": ["model_q4.onnx", "model_q4.onnx_data"],
            "int8": ["model_quantized.onnx", "model_quantized.onnx_data"],
            "fp32": ["model.onnx", "model.onnx_data"],
        }
        if variant not in files_map:
            raise ValueError(f"Unknown variant '{variant}', expected one of {list(files_map.keys())}")

        files = files_map[variant]
        target_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Downloading %s model from '%s'...", variant, self.HF_ONNX_REPO)
        for filename in files:
            dest = target_dir / filename
            if dest.exists():
                self._logger.info("  Already exists: %s", dest)
                continue
            self._logger.info("  Downloading: %s", filename)
            hf_hub_download(
                repo_id=self.HF_ONNX_REPO,
                filename=f"onnx/{filename}",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download places files in subdirs; move to target
            downloaded = target_dir / "onnx" / filename
            if downloaded.exists():
                downloaded.rename(dest)
        # Clean up empty onnx/ subdir if created
        onnx_subdir = target_dir / "onnx"
        if onnx_subdir.exists() and not any(onnx_subdir.iterdir()):
            onnx_subdir.rmdir()
        self._logger.info("Download complete: %s", target_dir)

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        if self._is_quantized:
            return self._load_onnx_quantized()
        model_path = self._onnx_dir /  "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.onnx @ '{self._onnx_dir}'")
        self._val_model_path = model_path
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

    def _load_onnx_quantized(self) -> dict[str, onnx.ModelProto]:
        # Search for quantized model file in order of preference
        candidates = ["model_q4.onnx", "model_quantized.onnx", "model.onnx"]
        model_path = None
        for name in candidates:
            p = self._onnx_dir / name
            if p.exists():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(
                f"No quantized model found in '{self._onnx_dir}'. "
                f"Expected one of: {candidates}"
            )
        self._logger.info("Loading quantized model from '%s'", model_path)
        model = onnx.load(model_path)
        orig_ir = model.ir_version
        graph = gs.import_onnx(model)
        graph.name = "main"

        # Replace ORT custom ops with standard ONNX ops
        editor = Gemma3OnnxGraphEditor(graph, self._onnx_export_dtype)
        self._logger.info("Replacing SimplifiedLayerNormalization ops...")
        editor.replace_simplified_layer_norm()
        self._logger.info("Replacing SkipSimplifiedLayerNormalization ops...")
        editor.replace_skip_simplified_layer_norm()
        if self._dequantize_weights:
            self._logger.info("Replacing MatMulNBits ops (dequantizing to fp32 MatMul)...")
            editor.replace_matmul_nbits()
        elif self._dequantize_weights_linear:
            self._logger.info("Replacing MatMulNBits ops with DequantizeLinear+Reshape+MatMul...")
            editor.replace_matmul_nbits_linear()
        else:
            self._logger.info("Keeping MatMulNBits ops as quantized (use --dequantize-weights to convert)")
        self._logger.info("Replacing GroupQueryAttention ops...")
        editor.replace_group_query_attention(
            num_heads=self._config.num_attention_heads,
            kv_num_heads=self._config.num_key_value_heads,
            head_dim=self._config.head_dim,
        )

        if self._extract_embeddings:
            embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
            embeddings_inp = "token_embedding"
            editor.extract_quantized_embeddings(
                self._hidden_size,
                self._vocab_size,
                embeddings_npy,
                inp_name=embeddings_inp
            )
            editor.reorder_graph_input(embeddings_inp, 0)

        # Remove com.microsoft opset import only if no custom ops remain (e.g. GatherBlockQuantized)
        # Cleanup first to remove disconnected nodes (avoids duplicate-name warnings from gs.export_onnx)
        editor.graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        model = gs.export_onnx(editor.graph)
        model.ir_version = orig_ir

        # DequantizeLinear with block_size requires opset 21; pack uint8 → UINT4 for 4-bit only
        if self._dequantize_weights_linear:
            for opset in model.opset_import:
                if opset.domain == "" and opset.version < 21:
                    opset.version = 21
            if self._model_quant_dtype != "int8":
                from torq.graph_edit.edits import pack_dq_weights_uint4
                pack_dq_weights_uint4(model)

        has_ms_ops = any(n.domain == "com.microsoft" for n in model.graph.node)
        if not has_ms_ops:
            ms_opsets = [opset for opset in model.opset_import if opset.domain == "com.microsoft"]
            for opset in ms_opsets:
                model.opset_import.remove(opset)
        else:
            remaining = set(n.op_type for n in model.graph.node if n.domain == "com.microsoft")
            self._logger.info("Keeping com.microsoft opset for remaining ops: %s", remaining)

        graph = gs.import_onnx(model)
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        model = gs.export_onnx(graph)
        model.ir_version = orig_ir

        # Run shape inference so all intermediate tensors have shape info.
        # GatherBlockQuantized is a com.microsoft custom op that shape inference
        # cannot handle; manually provide its output shape so downstream shapes
        # can propagate.
        del model.graph.value_info[:]
        for node in model.graph.node:
            if node.op_type == "GatherBlockQuantized":
                model.graph.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        node.output[0], onnx.TensorProto.FLOAT,
                        ["batch_size", "sequence_length", self._hidden_size]
                    )
                )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=False, data_prop=True
        )

        # Save converted dynamic model for validation reference
        val_dir = self._models_dir / "source" / "int4_converted"
        val_dir.mkdir(parents=True, exist_ok=True)
        self._val_model_path = val_dir / "model.onnx"
        onnx.save(model, self._val_model_path)
        self._logger.info("Saved converted dynamic model to '%s'", self._val_model_path)

        return {"model": model}

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
        # int4 GQA replacement produces "total_sequence_length" on present KV outputs
        extra_dims = None
        if self._is_quantized:
            from ...graph_edit import DimMatchType, FixedDimMapping
            extra_dims = [FixedDimMapping("total_sequence_length", DimMatchType.EXACT, self._max_gen_tokens)]
        editor.fix_io(self._max_gen_tokens, dims=extra_dims)

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

        # int4: Replace the attention_mask → seqlen_k chain with position_ids.
        # The GQA replacement uses seqlen_k for cos/sin position indexing.
        # In the static model, position_ids (cur_len) provides the same value,
        # so we rewire all consumers and let cleanup remove the attention_mask input.
        if self._is_quantized:
            from ...graph_edit.onnx import rewire_consumers
            # Try both seqlen_k variable names (model.onnx vs model_q4.onnx)
            seqlen_k_candidates = [
                "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
                "/model/attn_mask_reformat/attn_mask_subgraph/Expand/Cast/output_0",
            ]
            tensors = graph.tensors()
            rewired = False
            for seqlen_k_name in seqlen_k_candidates:
                if seqlen_k_name in tensors:
                    seqlen_k_var = tensors[seqlen_k_name]
                    rewire_consumers(
                        list(seqlen_k_var.outputs),
                        seqlen_k_var,
                        cur_len_scalar
                    )
                    rewired = True
                    break
            # Also rewire any Squeeze outputs (seqlen_k_squeezed) from model_q4's (B,1) squeeze
            for name, t in tensors.items():
                if name.endswith("/seqlen_k_squeezed") and hasattr(t, 'outputs'):
                    rewire_consumers(list(t.outputs), t, cur_len_scalar)
                    rewired = True
            if rewired:
                self._logger.info("Replaced seqlen_k chain with position_ids for RoPE indexing")

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

        new_model = editor.to_onnx(override_ir=model.ir_version, strict_mode=not self._is_quantized)
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

        if self._extract_embeddings:
            embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
            embeddings_inp = "token_embedding"
            if self._is_quantized:
                # GatherBlockQuantized was already extracted during dynamic conversion;
                # copy the .npy to the static export directory.
                src_npy = self._onnx_dir / "token_embeddings.npy"
                if src_npy.exists() and src_npy != embeddings_npy:
                    import shutil
                    shutil.copy2(src_npy, embeddings_npy)
            else:
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

        editor.reorder_graph_input("position_ids", 1)

        new_model = editor.to_onnx(override_ir=model.ir_version, strict_mode=not self._is_quantized)
        # Re-run shape inference with stale value_info cleared to fill all shapes
        if self._is_quantized:
            del new_model.graph.value_info[:]
            new_model = onnx.shape_inference.infer_shapes(
                new_model, check_type=True, strict_mode=False, data_prop=True
            )
        onnx.save(new_model, model_path)

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = False) -> onnx.ModelProto:
        # int4 source models may have mismatched value_info metadata (e.g. Constant
        # nodes with shape [] in value_info but actual rank-1 tensor data). Relax
        # strict_mode so shape inference overwrites stale annotations instead of
        # raising InferenceError.
        strict = not self._is_quantized
        if self._is_quantized:
            # Clear all stale value_info entries — the int4 model has many shape
            # annotations that become invalid after graph transformations (GQA
            # replacement, seqlen_k → position_ids rewiring, etc.).  Shape
            # inference with strict_mode=False will re-derive correct shapes.
            del model.graph.value_info[:]
        if model.ir_version > 10:
            self._logger.warning(
                "Warning: Model IR version is > 10 (%d), which might be unsupported by onnxruntime",
                model.ir_version
            )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=strict, data_prop=not skip_data_prop
        )
        onnx.checker.check_model(model, full_check=True)
        return model

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self.check_model(
            self._components["model"], skip_data_prop=self._is_quantized
        )
        self._components["model"] = self._make_model_static(self._components["model"])

    def apply_post_static_patches(self, model_path: str | os.PathLike, _):
        self._patch_static_model(model_path)
        if self._simulate_bf16:
            self._logger.info("(model) Creating bf16-simulated copy...")
            sim_dir = Path(model_path).parent.parent / "int4_bf16_sim" / "static"
            sim_dir.mkdir(parents=True, exist_ok=True)
            sim_path = sim_dir / Path(model_path).name
            import shutil
            shutil.copy2(model_path, sim_path)
            # Copy token_embeddings.npy if present
            emb_src = Path(model_path).parent / "token_embeddings.npy"
            if emb_src.exists():
                shutil.copy2(emb_src, sim_dir / "token_embeddings.npy")
            self._simulate_bf16_precision(sim_path)

    def _skip_static_shape_check(self) -> bool:
        # int4 models have cos/sin constant cache dims that leak into
        # intermediate shape inference, causing false dynamic-shape reports
        return self._is_quantized

    def _simulate_bf16_precision(self, model_path: str | os.PathLike):
        """Simulate bf16 precision loss on both weights and activations.

        Weights: round-trip fp32 constant data through bf16 in-place.
        Activations: for each non-Cast node output insert
            output → Cast(bf16) → Cast(fp32)
        then rewire all downstream consumers to the fp32 copy.
        I/O stays fp32 so onnxruntime can run the model normally.
        """
        import ml_dtypes

        model = onnx.load(model_path)
        graph = gs.import_onnx(model)

        # --- 1. Quantize weights: round-trip fp32 constants through bf16 ---
        n_weights = 0
        for tensor in graph.tensors().values():
            if isinstance(tensor, gs.Constant) and tensor.values is not None:
                if tensor.values.dtype == np.float32:
                    tensor.values = tensor.values.astype(ml_dtypes.bfloat16).astype(np.float32)
                    n_weights += 1

        # --- 2. Quantize activations: Cast sandwich on node outputs ---
        targets: list[tuple[gs.Node, gs.Variable]] = []
        for node in graph.nodes:
            if node.op == "Cast":
                continue
            for out_var in node.outputs:
                if out_var.name and out_var.dtype in (np.float32, np.float64):
                    targets.append((node, out_var))

        n_casts = 0
        for orig_node, out_var in targets:
            bf16_var = gs.Variable(f"{out_var.name}__bf16")
            fp32_var = gs.Variable(f"{out_var.name}__fp32", dtype=np.float32)

            cast_to_bf16 = gs.Node(
                op="Cast",
                name=f"{out_var.name}__cast_bf16",
                inputs=[out_var],
                outputs=[bf16_var],
                attrs={"to": onnx.TensorProto.BFLOAT16},
            )
            cast_to_fp32 = gs.Node(
                op="Cast",
                name=f"{out_var.name}__cast_fp32",
                inputs=[bf16_var],
                outputs=[fp32_var],
                attrs={"to": onnx.TensorProto.FLOAT},
            )
            graph.nodes.append(cast_to_bf16)
            graph.nodes.append(cast_to_fp32)

            for consumer in list(out_var.outputs):
                if consumer is cast_to_bf16:
                    continue
                for idx, inp in enumerate(consumer.inputs):
                    if inp is out_var:
                        consumer.inputs[idx] = fp32_var
            n_casts += 1

        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        onnx.save(model, model_path)
        self._logger.info(
            "Saved bf16-simulated model to '%s' (%d weights quantized, %d activation Cast pairs)",
            model_path, n_weights, n_casts,
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
            self._val_model_path,
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
        external_data = []
        if self._extract_embeddings:
            external_data.append(
                (self._export_paths["model"].parent / "token_embeddings.npy", np.dtype(ml_dtypes.bfloat16))
            )
        return super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            external_data=external_data
        )

def export_gemma3_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = Gemma3ModelExporter(
        args.model_size,
        args.instruct_model,
        args.extract_embeddings,
        args.keep_individual_kv_io,
        not args.dynamic_models,
        max_gen_tokens=args.max_gen_tokens,
        model_dtype=args.model_dtype,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        replace_int_bf16_cast=args.replace_int_bf16_cast,
        broadcast_ops=args.broadcast_ops,
        dequantize_weights=args.dequantize_weights,
        dequantize_weights_linear=args.dequantize_weights_linear,
        simulate_bf16=args.simulate_bf16,
        fp8_e5m2=args.fp8_e5m2,
        fp8_e4m3=args.fp8_e4m3
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if not args.skip_iree:
        exporter.export_iree(iree_compile_args=process_iree_args(args))


def main():
    parser = argparse.ArgumentParser(description="Export Gemma3 to Torq")
    add_gemma3_export_args(parser)
    export_gemma3_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
