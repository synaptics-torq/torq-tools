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
from torq.utils.logging import configure_logging


def _patch_gs_bf16_converter():
    """onnx_graphsurgeon's bf16 / fp8 exporter relies on
    `onnx.helper.float32_to_bfloat16` which was removed in onnx>=1.20.
    When the helper is missing, `_NUMPY_ARRAY_CONVERTERS` is empty and
    saving a fp32 Constant with `export_dtype=BFLOAT16` asserts out.
    Install a small ml_dtypes-backed converter at import time so the
    torq convert_dtype pipeline keeps working.
    """
    try:
        from onnx_graphsurgeon.exporters import onnx_exporter as _gs_export
    except Exception:
        return
    if getattr(onnx, "TensorProto", None) is None:
        return
    if _gs_export._NUMPY_ARRAY_CONVERTERS:
        return

    def _f32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        bf = arr.astype(ml_dtypes.bfloat16)
        return bf.view(np.uint16)

    _gs_export._NUMPY_ARRAY_CONVERTERS = {
        onnx.TensorProto.BFLOAT16: _gs_export.NumpyArrayConverter(
            np.uint16, _f32_to_bf16_uint16
        ),
    }


_patch_gs_bf16_converter()

from . import add_liquid_export_args
from ._graph import LiquidOnnxGraphEditor
from ._inference import LiquidDynamic, LiquidStatic
from ...graph_edit import DimMatchType, FixedDimMapping
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig


# HuggingFace repos containing LFM2.5 model + tokenizer
HF_REPO_BASE: dict[str, str] = {
    "350m": "LiquidAI/LFM2.5-350M",
}
# Source ONNX repos, tried in order. The Synaptics-hosted mirror is primary so
# the pipeline keeps working if the upstream LiquidAI repo is renamed/removed;
# the upstream repo is the fallback. The Synaptics repo is private, so fetching
# from it needs an HF token (HF_TOKEN env or `huggingface-cli login`); without
# one, that repo 401s and we fall through to the public upstream repo.
HF_REPO_ONNX: dict[str, tuple[str, ...]] = {
    "350m": (
        "Synaptics/LiquidAI-LFM2p5-350M-LLM",
        "LiquidAI/LFM2.5-350M-ONNX",
    ),
}

# Full torq-compile flag set LFM2.5 needs — the validated set the legacy
# compile_v1.5.sh hardcoded.  All of these matter for the bf16 model:
#   --torq-convert-dtypes / --torq-convert-io-dtype run the dtype-conversion
#     passes; without them a tensor's element type resolves to `none` and
#     torq-compile asserts out (Kernel.cpp `elementType != DType::none`).
#   --torq-hw=SL2610 / --torq-disable-slicing / --torq-enable-annotate-tied-operands
#     target + lowering options the chip needs.
#   --torq-enable-split-constants-optimization is faster (303 vs 432 ms/step)
#     and lower-heap (each dispatch reads its constant slice from the mmap'd
#     vmfb instead of staging the whole blob into anonymous DRAM).
LIQUID_TORQ_FLAGS: tuple[str, ...] = (
    "--torq-hw=SL2610",
    "--torq-disable-slicing",
    "--torq-enable-transpose-optimization",
    "--torq-convert-dtypes",
    "--torq-enable-annotate-tied-operands",
    "--torq-convert-io-dtype",
    "--torq-enable-split-constants-optimization",
)


class LiquidModelExporter(OnnxModelExporterBase):
    """Exporter for the LiquidAI LFM2.5 hybrid model.

    LFM2.5 is a hybrid conv + attention LM.  The source ONNX uses
    `com.microsoft.GroupQueryAttention` and `SimplifiedLayerNormalization`
    custom ops, plus dynamic KV cache and a `num_logits_to_keep` scalar
    input.  This exporter:
        1. Replaces the custom ops with standard ONNX ops.
        2. Makes KV cache static (fixed `max_gen_tokens` sequence length).
        3. Folds `num_logits_to_keep` to the constant 1.
        4. Optionally extracts the (large) token-embedding LUT.
        5. Optionally converts the static fp32 ONNX to bf16 and compiles to IREE.
    """

    def __init__(
        self,
        model_size: Literal["350m"] = "350m",
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
        self._model_size = model_size
        self._hf_repo = HF_REPO_BASE[model_size]
        self._hf_repo_onnx = HF_REPO_ONNX[model_size]
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._simulate_bf16 = edit_args.get("simulate_bf16", False)
        # Chip-specific graph rewrites. The SL2610's depthwise-conv path
        # crashes torq-compile (Kernel.cpp DType::none assertion) on LFM2.5's
        # short conv, so by default we replace each depthwise Conv1D with a
        # bit-exact batched-MatMul chain.  The 512-chunk lm_head split is no
        # longer needed now that tile-and-fuse is default — the exporter emits
        # a single [1024, 65536] MatMul unless --split-lm-head is passed.
        self._replace_conv1d = not edit_args.get("keep_conv1d", False)
        self._split_lm_head = edit_args.get("split_lm_head", False)

        # Read config so we know architecture params; do this directly from
        # the source dir if a config.json is present, otherwise from HF.
        cfg_path: Path | None = None
        if onnx_source_dir is not None:
            cand = Path(onnx_source_dir) / "config.json"
            if cand.exists():
                cfg_path = cand
        if cfg_path is None:
            from huggingface_hub import hf_hub_download
            cfg_path = Path(hf_hub_download(self._hf_repo, "config.json"))
        import json
        with open(cfg_path) as f:
            self._config_dict = json.load(f)
        self._hidden_size = int(self._config_dict["hidden_size"])
        self._vocab_size = int(self._config_dict["vocab_size"])
        self._num_attention_heads = int(self._config_dict["num_attention_heads"])
        self._num_key_value_heads = int(self._config_dict["num_key_value_heads"])
        self._head_dim = int(self._config_dict.get("head_dim") or (self._hidden_size // self._num_attention_heads))
        self._conv_dim = int(self._config_dict.get("conv_dim", self._hidden_size))
        self._conv_L_cache = int(self._config_dict.get("conv_L_cache", 3))
        self._layer_types = tuple(self._config_dict["layer_types"])
        self._num_hidden_layers = int(self._config_dict["num_hidden_layers"])
        self._cfg_path = cfg_path

        super().__init__(
            "fp32",
            static_models,
            self._config_dict,
            # Nest all artifacts under a per-model subdir (like gemma3 nests
            # under its HF-repo dir). This must match download_source.py's
            # `<models-dir>/liquid-2p5-<size>/` layout so a pre-fetched source
            # is reused instead of re-downloaded, and matches the README paths.
            Path(models_dir) / f"liquid-2p5-{model_size}",
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            # LFM2's custom ops break the ORT bert optimizer; skip it.
            opt_configs={},
        )

    def _setup_dirs(self) -> list[Path]:
        if self._onnx_source_dir is not None:
            onnx_dir = Path(self._onnx_source_dir)
        else:
            onnx_dir = self._models_dir / "source" / "onnx" / self._model_dtype
        if not onnx_dir.exists() or not any(onnx_dir.glob("model.onnx*")):
            self._download_from_hf(onnx_dir)

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
            / "bf16"
            / ("static" if self._static_models else "dynamic")
        )
        iree_dir = (
            self._models_dir
            / "export"
            / "iree"
            / ("bf16" if self._convert_dtypes else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )
        return onnx_dir, export_dir, convert_dir, iree_dir

    def _download_from_hf(self, target_dir: Path):
        from huggingface_hub import hf_hub_download
        import shutil

        target_dir.mkdir(parents=True, exist_ok=True)
        repos = self._hf_repo_onnx
        if isinstance(repos, str):
            repos = (repos,)

        required = ["onnx/model.onnx", "onnx/model.onnx_data"]
        optional = ["config.json", "tokenizer.json", "tokenizer_config.json"]

        last_err: Exception | None = None
        for repo in repos:
            self._logger.info("Downloading source ONNX from '%s'...", repo)
            try:
                for filename in required:
                    p = hf_hub_download(repo, filename)
                    dest = target_dir / Path(filename).name
                    if not dest.exists():
                        shutil.copy(p, dest)
            except Exception as e:
                # Repo missing / renamed / private-without-token: try the next.
                last_err = e
                self._logger.warning(
                    "  source repo '%s' unavailable (%s); trying next", repo, e
                )
                continue
            # Required files came from this repo; pull the optional ones too.
            for filename in optional:
                try:
                    p = hf_hub_download(repo, filename)
                    dest = target_dir / filename
                    if not dest.exists():
                        shutil.copy(p, dest)
                except Exception as e:
                    self._logger.debug("  optional file %s not in %s: %s", filename, repo, e)
            self._logger.info("Download complete from '%s': %s", repo, target_dir)
            return
        raise RuntimeError(
            f"Failed to download source ONNX from any of {list(repos)}: {last_err}"
        )

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        from onnx.external_data_helper import (
            load_external_data_for_model,
            convert_model_to_external_data,
        )

        model_path = self._onnx_dir / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.onnx @ '{self._onnx_dir}'")
        # Reference for validation is the original (unmodified) source model
        self._val_model_path = model_path

        # Load weights inline so subsequent edits / saves don't need external
        # data file resolution.
        model = onnx.load(model_path, load_external_data=True)
        orig_ir = model.ir_version

        graph = gs.import_onnx(model)
        graph.name = "main"

        # Replace ORT custom ops with standard ONNX ops in the source graph.
        editor = LiquidOnnxGraphEditor(graph, self._onnx_export_dtype)
        self._logger.info("Replacing SimplifiedLayerNormalization ops...")
        editor.replace_simplified_layer_norm()
        self._logger.info("Replacing SkipSimplifiedLayerNormalization ops...")
        editor.replace_skip_simplified_layer_norm()
        self._logger.info("Replacing GroupQueryAttention ops...")
        editor.replace_group_query_attention(
            num_heads=self._num_attention_heads,
            kv_num_heads=self._num_key_value_heads,
            head_dim=self._head_dim,
        )

        editor.graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        model = gs.export_onnx(editor.graph)
        model.ir_version = orig_ir

        # Drop the com.microsoft opset import if no custom ops remain.
        has_ms_ops = any(n.domain == "com.microsoft" for n in model.graph.node)
        if not has_ms_ops:
            for opset in list(model.opset_import):
                if opset.domain == "com.microsoft":
                    model.opset_import.remove(opset)
        else:
            remaining = sorted({n.op_type for n in model.graph.node if n.domain == "com.microsoft"})
            self._logger.warning("Keeping com.microsoft opset; remaining ops: %s", remaining)

        # Save the converted dynamic ONNX as a single self-contained file
        # (weights inline, no external .onnx_data) so it can be opened in
        # Netron or other viewers without dragging external data along.
        try:
            import shutil
            from copy import deepcopy
            converted_dir = self._models_dir / "source" / "onnx" / "converted"
            if converted_dir.exists():
                shutil.rmtree(converted_dir, ignore_errors=True)
            converted_dir.mkdir(parents=True, exist_ok=True)
            converted_path = converted_dir / "model.onnx"
            single_model = deepcopy(model)
            # Force every initializer back to inline storage
            for init in single_model.graph.initializer:
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    init.data_location = onnx.TensorProto.DEFAULT
                    del init.external_data[:]
            onnx.save(single_model, converted_path)
            self._logger.info("Saved single-file converted source model to '%s'", converted_path)
        except Exception as e:
            self._logger.warning("Could not save single-file source model: %s", e)

        return {"model": model}

    @staticmethod
    def sanitize_onnx_names(model: onnx.ModelProto) -> onnx.ModelProto:
        """LFM2.5-specific tensor-name sanitizer.

        The base class's sanitizer replaces illegal MLIR identifier
        characters with `_`, which on LFM2.5 collapses three distinct
        source constants — `/model/constants/INT64/[1]`,
        `/model/constants/INT64/[-1]` and `/model/constants/INT64/-1` —
        onto the same name `/model/constants/INT64/_1`.  Only one initializer
        survives, and downstream Slice ops that expected the rank-1
        constants instead bind to the surviving rank-0 scalar, producing
        the rank-mismatch crash inside torq-compile.

        We pre-rename initializers to avoid the collision before the
        general sanitization step is invoked.
        """
        import re

        # Step 1: pre-rename initializers whose sanitized names collide
        # by appending a deterministic encoding of their dims and sign.
        illegal = re.compile(r"[^a-zA-Z0-9_./]+")

        def _basic_sanitize(name: str) -> str:
            return illegal.sub("_", name).strip("_")

        sanitized_to_originals: dict[str, list[str]] = {}
        for init in model.graph.initializer:
            sanitized_to_originals.setdefault(_basic_sanitize(init.name), []).append(init.name)

        rename_map: dict[str, str] = {}
        for sanitized, originals in sanitized_to_originals.items():
            if len(originals) <= 1:
                continue
            # Disambiguate by appending dim info + a content tag.
            init_by_name = {i.name: i for i in model.graph.initializer}
            for orig in originals:
                init = init_by_name[orig]
                dims_tag = "x".join(str(d) for d in init.dims) or "0d"
                # Use the original name's last segment (e.g. "[-1]" → "neg1")
                tail = orig.rsplit("/", 1)[-1]
                if tail.startswith("[") and tail.endswith("]"):
                    val = tail[1:-1]
                    tag = "neg" + val[1:] if val.startswith("-") else val
                    tag = "vec" + tag
                else:
                    tag = "neg" + tail[1:] if tail.startswith("-") else "scl" + tail
                new_name = f"{orig.rsplit('/', 1)[0]}/{tag}_d{dims_tag}"
                rename_map[orig] = new_name

        if rename_map:
            for init in model.graph.initializer:
                if init.name in rename_map:
                    init.name = rename_map[init.name]
            for node in model.graph.node:
                for i, inp in enumerate(node.input):
                    if inp in rename_map:
                        node.input[i] = rename_map[inp]
                for i, out in enumerate(node.output):
                    if out in rename_map:
                        node.output[i] = rename_map[out]
            for vi in model.graph.value_info:
                if vi.name in rename_map:
                    vi.name = rename_map[vi.name]
            for io in list(model.graph.input) + list(model.graph.output):
                if io.name in rename_map:
                    io.name = rename_map[io.name]

        # Step 2: defer to base sanitizer for any remaining illegal chars.
        return OnnxModelExporterBase.sanitize_onnx_names(model)

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = False) -> onnx.ModelProto:
        if model.ir_version > 10:
            self._logger.warning(
                "Model IR version is %d (>10), may be unsupported by onnxruntime",
                model.ir_version,
            )
        # Clear stale value_info entries — graph transformations (GQA, layer
        # norm replacement) invalidate them.  Use lenient shape inference
        # because rewires (num_logits_to_keep fold, RoPE rewiring) can leave
        # rank mismatches that strict checking refuses to overwrite.
        del model.graph.value_info[:]
        try:
            model = onnx.shape_inference.infer_shapes(
                model, check_type=False, strict_mode=False, data_prop=not skip_data_prop
            )
        except Exception as e:
            self._logger.warning("Shape inference reported issues; continuing: %s", e)
        try:
            onnx.checker.check_model(model, full_check=False)
        except Exception as e:
            self._logger.warning("ONNX checker reported issues; continuing: %s", e)
        return model

    def _make_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph: gs.Graph = gs.import_onnx(model)
        editor = LiquidOnnxGraphEditor(graph, self._onnx_export_dtype)

        # Fix all dynamic IO dims first.
        editor.fix_io(self._max_gen_tokens)

        # Fold `num_logits_to_keep` -> constant 1 (autoregressive decode).
        editor.fold_num_logits_to_keep(1)

        # Clean up common artifacts.
        editor.remove_redundant_casts()
        editor.remove_isNaN()

        # Build a 1D `current_len` scalar from the existing seqlen_k tensors
        # that the GQA replacement created.  If absent (no attention layers),
        # fall back to a position-id-based length.
        from ...graph_edit.onnx import rewire_consumers

        # The replaced GroupQueryAttention emits a `seqlen_k` chain derived
        # from `attention_mask`.  For static decode, the current length is
        # simply the position index for the new token.  We add a 2D
        # `position_ids` input and use it as the index.
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

        # Rewire any `seqlen_k` chain produced by GQA replacement
        seqlen_k_candidates = [
            "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
            "/model/attn_mask_reformat/attn_mask_subgraph/Expand/Cast/output_0",
        ]
        tensors = graph.tensors()
        for name in seqlen_k_candidates:
            if name in tensors:
                seqlen_k_var = tensors[name]
                rewire_consumers(list(seqlen_k_var.outputs), seqlen_k_var, cur_len_scalar)
                self._logger.info("Replaced seqlen_k chain '%s' with position_ids", name)
                break
        for name, t in list(tensors.items()):
            if name.endswith("/seqlen_k_squeezed") and hasattr(t, "outputs"):
                rewire_consumers(list(t.outputs), t, cur_len_scalar)

        (
            editor
            .replace_dynamic_kv_cache(cur_len, self._max_gen_tokens)
            .mask_future_attn_scores(cur_len, self._max_gen_tokens)
            .add_curr_len_input(cur_len)
            .convert_to_static_index()
        )

        new_model = editor.to_onnx(override_ir=model.ir_version, strict_mode=False)
        return new_model

    @staticmethod
    def _fold_shape_ops(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
        """Constant-fold ``Shape`` ops (whose input has a fully static shape)
        and the small set of shape-math ops that typically sit between
        ``Shape`` and downstream consumers in transformer graphs: Gather,
        Slice, Squeeze, Unsqueeze, Concat, Cast, Add, Sub, Mul, Div, Mod,
        Range, ReduceProd.  Folding these lets ``onnx.shape_inference`` see
        all the symbolic dims (``unk__N``) as concrete integers, which in
        turn lets torq-compile's Where-broadcast lowering succeed.
        """
        graph = gs.import_onnx(model)
        folded = 0

        def _const(t):
            return isinstance(t, gs.Constant)

        def _all_const(node):
            return node.inputs and all(_const(i) for i in node.inputs)

        def _as_arr(t):
            return np.asarray(t.values)

        progressed = True
        while progressed:
            progressed = False
            for node in list(graph.nodes):
                out = node.outputs[0] if node.outputs else None
                if out is None:
                    continue
                new_val = None
                try:
                    if node.op == "Shape":
                        inp = node.inputs[0]
                        sh = getattr(inp, "shape", None)
                        if sh is None or any(not isinstance(d, int) for d in sh):
                            continue
                        new_val = np.array(sh, dtype=np.int64)
                    elif not _all_const(node):
                        continue
                    elif node.op == "Cast":
                        to = int(node.attrs["to"])
                        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(to)
                        new_val = _as_arr(node.inputs[0]).astype(np_dtype)
                    elif node.op == "Gather":
                        data = _as_arr(node.inputs[0]); idx = _as_arr(node.inputs[1])
                        axis = int(node.attrs.get("axis", 0))
                        new_val = np.take(data, idx, axis=axis)
                    elif node.op == "Slice":
                        data = _as_arr(node.inputs[0])
                        starts = _as_arr(node.inputs[1]); ends = _as_arr(node.inputs[2])
                        axes = _as_arr(node.inputs[3]) if len(node.inputs) > 3 else np.arange(len(starts))
                        steps = _as_arr(node.inputs[4]) if len(node.inputs) > 4 else np.ones_like(starts)
                        sl = [slice(None)] * data.ndim
                        for a, s, e, st in zip(np.atleast_1d(axes), np.atleast_1d(starts), np.atleast_1d(ends), np.atleast_1d(steps)):
                            sl[int(a)] = slice(int(s), int(e), int(st))
                        new_val = data[tuple(sl)]
                    elif node.op == "Squeeze":
                        data = _as_arr(node.inputs[0])
                        axes = _as_arr(node.inputs[1]) if len(node.inputs) > 1 else None
                        new_val = np.squeeze(data, axis=tuple(int(a) for a in np.atleast_1d(axes)) if axes is not None else None)
                    elif node.op == "Unsqueeze":
                        data = _as_arr(node.inputs[0])
                        axes = _as_arr(node.inputs[1])
                        new_val = data
                        for a in sorted(int(x) for x in np.atleast_1d(axes)):
                            new_val = np.expand_dims(new_val, axis=a)
                    elif node.op == "Concat":
                        arrs = [_as_arr(i) for i in node.inputs]
                        axis = int(node.attrs.get("axis", 0))
                        new_val = np.concatenate(arrs, axis=axis)
                    elif node.op in ("Add", "Sub", "Mul", "Div", "Mod"):
                        a = _as_arr(node.inputs[0]); b = _as_arr(node.inputs[1])
                        op_map = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
                                  "Div": np.divide if a.dtype.kind == "f" else np.floor_divide,
                                  "Mod": np.mod}
                        new_val = op_map[node.op](a, b)
                    elif node.op == "Neg":
                        new_val = -_as_arr(node.inputs[0])
                    elif node.op == "Identity":
                        new_val = _as_arr(node.inputs[0]).copy()
                    elif node.op == "Abs":
                        new_val = np.abs(_as_arr(node.inputs[0]))
                    elif node.op == "Reshape":
                        data = _as_arr(node.inputs[0]); shape = _as_arr(node.inputs[1])
                        new_val = data.reshape(tuple(int(s) for s in shape))
                    elif node.op == "Constant":
                        # Already a constant — skip (it's just the value).
                        continue
                    elif node.op == "ConstantOfShape":
                        shape = _as_arr(node.inputs[0])
                        value_attr = node.attrs.get("value")
                        if value_attr is None:
                            new_val = np.zeros(tuple(int(s) for s in shape), dtype=np.float32)
                        else:
                            v = value_attr.values
                            new_val = np.full(tuple(int(s) for s in shape), v.item() if v.size == 1 else v, dtype=v.dtype)
                    elif node.op == "ReduceProd":
                        data = _as_arr(node.inputs[0])
                        axes_t = node.inputs[1] if len(node.inputs) > 1 else None
                        axes = tuple(int(x) for x in np.atleast_1d(_as_arr(axes_t))) if axes_t is not None else None
                        keepdims = bool(node.attrs.get("keepdims", 1))
                        new_val = np.prod(data, axis=axes, keepdims=keepdims)
                    elif node.op == "Range":
                        s = int(_as_arr(node.inputs[0])); e = int(_as_arr(node.inputs[1])); st = int(_as_arr(node.inputs[2]))
                        new_val = np.arange(s, e, st, dtype=_as_arr(node.inputs[0]).dtype)
                except Exception:
                    new_val = None

                if new_val is None:
                    continue

                new_const = gs.Constant(
                    name=out.name + "_folded",
                    values=np.asarray(new_val),
                )
                for consumer in list(out.outputs):
                    for idx, t in enumerate(consumer.inputs):
                        if t is out:
                            consumer.inputs[idx] = new_const
                # Disconnect the node by detaching its inputs/outputs so
                # gs.cleanup removes it.
                node.inputs.clear()
                node.outputs.clear()
                folded += 1
                progressed = True

        # Remove disconnected nodes explicitly (cleanup with remove_unused
        # only removes nodes whose outputs are unused — but our cleared
        # outputs aren't considered "used" so this is redundant insurance).
        graph.nodes = [n for n in graph.nodes if n.outputs or n.inputs]
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        if folded:
            new_model = gs.export_onnx(graph)
            new_model.ir_version = model.ir_version
            return new_model, folded
        return model, 0

    @staticmethod
    def _split_lm_head_matmul(model: onnx.ModelProto, chunk_size: int = 128) -> tuple[onnx.ModelProto, int]:
        """Split the lm_head MatMul along the vocab axis into chunks that fit
        in the SL2610's 512 KB LRAM.

        LFM2.5's lm_head is ``MatMul(hidden[1,1,1024], Transpose(W))`` where
        ``W = model.embed_tokens.weight`` of shape ``[vocab=65536, hidden=1024]``.
        The transposed weight ``[1024, 65536]`` is 134 MB in bf16 — far
        larger than the 512 KB LRAM, so torq-compile cannot allocate it
        and fails ``failed to allocate LRAM addresses``.  Splitting the
        vocab dim into chunks of ``chunk_size`` columns (default 128 →
        ``[1024, 128] = 256 KB`` per chunk, comfortably under 512 KB)
        produces ``vocab/chunk_size`` smaller MatMuls whose results are
        concatenated back into the original ``[1, 1, vocab]`` logits.
        """
        graph = gs.import_onnx(model)
        target = None
        for node in graph.nodes:
            if node.op != "MatMul" or len(node.inputs) < 2:
                continue
            rhs = node.inputs[1]
            if not isinstance(rhs, gs.Variable):
                continue
            producer_inputs = getattr(rhs, "inputs", None) or []
            if not producer_inputs:
                continue
            producer = producer_inputs[0]
            if producer.op != "Transpose":
                continue
            w = producer.inputs[0]
            if not isinstance(w, gs.Constant) or w.values.ndim != 2:
                continue
            # Heuristic: a [vocab, hidden] tensor whose vocab dim is the
            # largest dim in the model — that's the LM head weight.
            if max(w.values.shape) < 16384:
                continue
            target = (node, producer, w)
            break

        if target is None:
            return model, 0

        matmul_node, transpose_node, weight = target
        V, H = int(weight.values.shape[0]), int(weight.values.shape[1])
        lhs_input = matmul_node.inputs[0]
        original_output = matmul_node.outputs[0]
        original_output_name = original_output.name
        original_output_shape = list(original_output.shape) if original_output.shape else None
        original_output_dtype = original_output.dtype

        chunk_outputs: list[gs.Variable] = []
        for ci, start in enumerate(range(0, V, chunk_size)):
            end = min(start + chunk_size, V)
            # Pre-transpose the chunk at export time so the MatMul can
            # consume a [H, chunk_size] constant directly — no Transpose
            # op needed in the graph.
            chunk_t = weight.values[start:end].T.copy()  # [H, end-start]
            chunk_const = gs.Constant(
                name=f"{weight.name}_chunkT_{ci}",
                values=chunk_t,
            )
            m_out = gs.Variable(
                name=f"{matmul_node.name}_chunk_{ci}",
                dtype=weight.values.dtype,
                shape=(original_output_shape[:-1] + [end - start])
                if original_output_shape else None,
            )
            m_node = gs.Node(
                op="MatMul",
                name=f"{matmul_node.name}_chunk_{ci}",
                inputs=[lhs_input, chunk_const],
                outputs=[m_out],
            )
            graph.nodes.append(m_node)
            chunk_outputs.append(m_out)

        concat_axis = (len(original_output_shape) - 1) if original_output_shape else -1
        concat_node = gs.Node(
            op="Concat",
            name=f"{matmul_node.name}_concat",
            inputs=chunk_outputs,
            outputs=[original_output],
            attrs={"axis": concat_axis},
        )
        graph.nodes.append(concat_node)

        # Detach the original MatMul and Transpose so cleanup removes them.
        matmul_node.outputs.clear()
        matmul_node.inputs.clear()
        transpose_node.outputs.clear()
        transpose_node.inputs.clear()

        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_model = gs.export_onnx(graph)
        new_model.ir_version = model.ir_version
        return new_model, len(chunk_outputs)

    @staticmethod
    def _unsplit_lm_head_matmul(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
        """Pre-transpose the lm_head weight host-side and replace
        ``MatMul(hidden, Transpose(W))`` with a single ``MatMul(hidden, W_T)``.

        With ``torq-compile``'s tile-and-fuse enabled by default, the chip
        tiles a single ``[H, V]`` MatMul against the 512 KB LRAM without the
        export-time 512-chunk split.  Skipping the split shrinks the
        compile-time IR (fewer dispatches feed ``SegmentNSSPrograms`` /
        ``ResolveAddresses``) and produces a simpler vmfb.
        """
        graph = gs.import_onnx(model)
        target = None
        for node in graph.nodes:
            if node.op != "MatMul" or len(node.inputs) < 2:
                continue
            rhs = node.inputs[1]
            if not isinstance(rhs, gs.Variable):
                continue
            producer_inputs = getattr(rhs, "inputs", None) or []
            if not producer_inputs:
                continue
            producer = producer_inputs[0]
            if producer.op != "Transpose":
                continue
            w = producer.inputs[0]
            if not isinstance(w, gs.Constant) or w.values.ndim != 2:
                continue
            if max(w.values.shape) < 16384:
                continue
            target = (node, producer, w)
            break

        if target is None:
            return model, 0

        matmul_node, transpose_node, weight = target
        wt_const = gs.Constant(name=f"{weight.name}_T", values=weight.values.T.copy())
        matmul_node.inputs[1] = wt_const
        transpose_node.outputs.clear()
        transpose_node.inputs.clear()
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_model = gs.export_onnx(graph)
        new_model.ir_version = model.ir_version
        return new_model, 1

    @staticmethod
    def _replace_conv1d_with_matmul(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
        """Replace LFM2.5's depthwise Conv1D with a bit-exact MatMul chain.

        The SL2610's depthwise-conv lowering asserts out of torq-compile
        (``Kernel.cpp`` ``DType::none``) on this kernel.  The replacement is
        mathematically identical: slice each output time-step's K-wide
        receptive field out of the input, reshape into a per-channel batched
        MatMul (``[C, 1, K] @ [C, K, 1] -> [C, 1, 1]``), and concat the
        per-step results back along the time axis.

        Matches only depthwise 1D Convs (``groups == out_channels``, weight
        rank 3, single spatial dim, stride/dilation 1, no padding) — anything
        else is left alone for ``_inject_zero_bias_into_conv`` to handle.
        """
        graph = gs.import_onnx(model)
        replaced = 0
        for node in list(graph.nodes):
            if node.op != "Conv" or len(node.inputs) < 2:
                continue
            w_in = node.inputs[1]
            if not isinstance(w_in, gs.Constant):
                continue
            w = w_in.values
            if w.ndim != 3:
                continue  # not 1D conv
            C_out, C_in_per_g, K = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
            groups = int(node.attrs.get("group", 1))
            if groups != C_out or C_in_per_g != 1:
                continue  # not depthwise
            if (list(node.attrs.get("strides", [1])) != [1]
                    or list(node.attrs.get("pads", [0, 0])) != [0, 0]
                    or list(node.attrs.get("dilations", [1])) != [1]):
                continue  # unusual conv geometry — leave alone

            X = node.inputs[0]
            X_shape = list(X.shape) if X.shape else None
            if not X_shape or len(X_shape) != 3:
                continue
            N, C, L_in = X_shape
            if int(C) != C_out:
                continue
            L_out = int(L_in) - K + 1
            if L_out <= 0:
                continue

            base = node.name or f"conv_{replaced}"
            np_dtype = w.dtype

            # Reshape kernel [C, 1, K] -> [C, K, 1] (direct reshape preserves
            # per-channel weight order; no transpose).
            w_b_const = gs.Constant(
                name=f"{w_in.name}_KbyN",
                values=w.reshape(C_out, K, 1).copy(),
            )

            def _i64(name, vals):
                return gs.Constant(name=name, values=np.array(vals, dtype=np.int64))

            window_results: list[gs.Variable] = []
            for t in range(L_out):
                sl_out = gs.Variable(f"{base}/slice_{t}", dtype=np_dtype, shape=[N, C, K])
                graph.nodes.append(gs.Node(
                    op="Slice", name=f"{base}/slice_{t}",
                    inputs=[
                        X,
                        _i64(f"{base}/slice_{t}_starts", [t]),
                        _i64(f"{base}/slice_{t}_ends", [t + K]),
                        _i64(f"{base}/slice_{t}_axes", [2]),
                        _i64(f"{base}/slice_{t}_steps", [1]),
                    ],
                    outputs=[sl_out],
                ))
                rsh_in = gs.Variable(f"{base}/resh_{t}", dtype=np_dtype, shape=[C, 1, K])
                graph.nodes.append(gs.Node(
                    op="Reshape", name=f"{base}/resh_{t}",
                    inputs=[sl_out, _i64(f"{base}/resh_{t}_shape", [C_out, 1, K])],
                    outputs=[rsh_in],
                ))
                mm_out = gs.Variable(f"{base}/matmul_{t}", dtype=np_dtype, shape=[C, 1, 1])
                graph.nodes.append(gs.Node(
                    op="MatMul", name=f"{base}/matmul_{t}",
                    inputs=[rsh_in, w_b_const], outputs=[mm_out],
                ))
                step_out = gs.Variable(f"{base}/out_{t}", dtype=np_dtype, shape=[N, C, 1])
                graph.nodes.append(gs.Node(
                    op="Reshape", name=f"{base}/out_{t}_resh",
                    inputs=[mm_out, _i64(f"{base}/out_{t}_shape", [N, C_out, 1])],
                    outputs=[step_out],
                ))
                window_results.append(step_out)

            original_output = node.outputs[0]
            if L_out == 1:
                only = window_results[0]
                only.name = original_output.name
                for n in graph.nodes:
                    n.inputs = [only if i is original_output else i for i in n.inputs]
                graph.outputs = [only if o is original_output else o for o in graph.outputs]
            else:
                graph.nodes.append(gs.Node(
                    op="Concat", name=f"{base}/concat",
                    inputs=window_results, outputs=[original_output],
                    attrs={"axis": 2},
                ))

            node.outputs.clear()
            node.inputs.clear()
            replaced += 1

        if replaced:
            graph.cleanup(
                remove_unused_graph_inputs=True, remove_unused_node_outputs=True
            ).toposort()
            new_model = gs.export_onnx(graph)
            new_model.ir_version = model.ir_version
            return new_model, replaced
        return model, 0

    @staticmethod
    def _inject_zero_bias_into_conv(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
        """LFM2.5's config has ``conv_bias: false`` so every ONNX Conv op
        ships with no bias input.  torq-compile's depthwise-conv lowering
        pattern (``LinalgToTorqHL`` → ``Conv2dConvert`` → ``computeBias``)
        crashes on ``SmallVector::back()`` of an empty bias list.  Inject a
        zero bias initializer for every Conv op missing one so the torq
        pattern matches.
        """
        graph = gs.import_onnx(model)
        injected = 0
        for node in graph.nodes:
            if node.op != "Conv":
                continue
            if len(node.inputs) >= 3 and node.inputs[2] is not None:
                continue
            weight = node.inputs[1]
            if not isinstance(weight, gs.Constant):
                continue
            # ONNX Conv weight is [M, C/group, k1, k2, ...] — first dim is
            # the number of output channels, which is the bias length.
            out_channels = int(weight.values.shape[0])
            dtype = weight.values.dtype
            bias = gs.Constant(
                name=f"{node.name}_bias_zero",
                values=np.zeros((out_channels,), dtype=dtype),
            )
            if len(node.inputs) == 2:
                node.inputs.append(bias)
            else:
                node.inputs[2] = bias
            injected += 1
        if injected:
            new_model = gs.export_onnx(graph)
            new_model.ir_version = model.ir_version
            return new_model, injected
        return model, 0

    @staticmethod
    def _resolve_negative_slices(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
        """Replace Slice ops whose ``starts``/``ends`` constants are
        negative (or clamped to INT64_MAX) into normalized positive
        indices based on the data tensor's static shape, so downstream
        shape inference can resolve the output to a concrete shape.

        Required because ONNX shape inference does not normalize negative
        Slice indices when the data shape is static, leaving the output
        marked as dynamic.  torq-compile's downstream passes then refuse
        to lower the op.  We rewrite the Slice's starts/ends constants
        in place — equivalent to a no-op for runtime but unlocks full
        static shape propagation.
        """
        graph = gs.import_onnx(model)
        rewritten = 0
        INT_MAX = 9223372036854775807
        for node in graph.nodes:
            if node.op != "Slice":
                continue
            data = node.inputs[0]
            data_shape = getattr(data, "shape", None)
            if data_shape is None or any(not isinstance(d, int) for d in data_shape):
                continue
            if len(node.inputs) < 3:
                continue
            starts = node.inputs[1]
            ends = node.inputs[2]
            axes = node.inputs[3] if len(node.inputs) > 3 else None
            if not (isinstance(starts, gs.Constant) and isinstance(ends, gs.Constant)):
                continue
            if axes is not None and not isinstance(axes, gs.Constant):
                continue
            s_vals = np.atleast_1d(starts.values).astype(np.int64).copy()
            e_vals = np.atleast_1d(ends.values).astype(np.int64).copy()
            a_vals = (
                np.atleast_1d(axes.values).astype(np.int64)
                if axes is not None else np.arange(len(s_vals), dtype=np.int64)
            )
            changed = False
            for i, ax in enumerate(a_vals):
                dim_size = int(data_shape[int(ax)])
                if s_vals[i] < 0:
                    s_vals[i] = max(0, s_vals[i] + dim_size)
                    changed = True
                if e_vals[i] < 0:
                    e_vals[i] = max(0, e_vals[i] + dim_size)
                    changed = True
                if e_vals[i] > dim_size or e_vals[i] >= INT_MAX // 2:
                    e_vals[i] = dim_size
                    changed = True
            if changed:
                # Replace inputs with fresh constants — the original
                # constants are shared across many Slice ops, so we must
                # NOT mutate them in place.
                node.inputs[1] = gs.Constant(
                    name=f"{node.name}_starts_norm",
                    values=s_vals.astype(starts.values.dtype),
                )
                node.inputs[2] = gs.Constant(
                    name=f"{node.name}_ends_norm",
                    values=e_vals.astype(ends.values.dtype),
                )
                rewritten += 1
        if rewritten:
            new_model = gs.export_onnx(graph)
            new_model.ir_version = model.ir_version
            return new_model, rewritten
        return model, 0

    def _propagate_static_shapes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Iteratively fold Shape ops and re-run data-propagating shape
        inference until no more changes occur.  Required for the torq-compile
        Where-lowering broadcast inference to see static dims.
        """
        for it in range(8):
            model, n_folded = self._fold_shape_ops(model)
            model, n_slice = self._resolve_negative_slices(model)
            self._logger.info(
                "(shape-prop) iter %d: folded %d node(s), normalized %d slice op(s)",
                it, n_folded, n_slice,
            )
            del model.graph.value_info[:]
            try:
                model = onnx.shape_inference.infer_shapes(
                    model, check_type=False, strict_mode=False, data_prop=True
                )
            except Exception as e:
                self._logger.warning(
                    "(shape-prop) shape inference reported issues at iter %d: %s",
                    it, e,
                )
            if n_folded == 0 and n_slice == 0:
                self._logger.info("(shape-prop) converged after %d iter(s)", it)
                break
        # Final tally for diagnostics
        n_dyn = 0
        for vi in model.graph.value_info:
            if any(
                (d.HasField("dim_param") and d.dim_param)
                or (not d.HasField("dim_value") and not d.HasField("dim_param"))
                for d in vi.type.tensor_type.shape.dim
            ):
                n_dyn += 1
        self._logger.info(
            "(shape-prop) remaining dynamic value_info: %d / %d",
            n_dyn, len(model.graph.value_info),
        )
        return model

    def _patch_static_model(self, model_path: str | os.PathLike):
        model = onnx.load(model_path)
        editor = LiquidOnnxGraphEditor.from_onnx(model, self._onnx_export_dtype)

        editor.eliminate_transposes()
        editor.collapse_reshape_chains()
        editor.collapse_gqa_broadcast()
        editor.fold_scalar_matmul()
        if self._broadcast_ops is not None:
            editor.broadcast_op_inputs(ops=self._broadcast_ops)

        if self._extract_embeddings:
            embeddings_npy = Path(model_path).parent / "token_embeddings.npy"
            # ExtractConstantLUT requires axis=0 to be explicit on the
            # Gather; ONNX defaults to 0 when absent, but the matcher uses
            # `attrs.get("axis", None) != 0` and refuses None.  Normalize.
            for node in editor.graph.nodes:
                if node.op == "Gather" and "axis" not in node.attrs:
                    node.attrs["axis"] = 0
            editor.extract_token_embeddings(
                self._hidden_size,
                self._vocab_size,
                embeddings_npy,
                inp_name="token_embedding",
            )
            editor.reorder_graph_input("token_embedding", 0)

        if not self._keep_individual_kv_io:
            editor.combine_kv_io_tensors([
                1,                          # B
                self._num_key_value_heads,  # H
                self._max_gen_tokens,       # L
                self._head_dim              # D
            ])

        editor.reorder_graph_input("position_ids", 1)
        new_model = editor.to_onnx(override_ir=model.ir_version, strict_mode=False)
        # Re-run shape inference + iteratively fold residual Shape ops so
        # intermediate dims are concrete integers, not dim_param symbols.
        new_model = self._propagate_static_shapes(new_model)
        # Replace depthwise Conv1D with a bit-exact batched-MatMul chain; the
        # SL2610's depthwise-conv path asserts out of torq-compile on this
        # kernel.  Disabled by --keep-conv1d.
        if self._replace_conv1d:
            new_model, n_conv = self._replace_conv1d_with_matmul(new_model)
            if n_conv:
                self._logger.info(
                    "(conv-rewrite) Replaced %d depthwise Conv1D op(s) with batched MatMul",
                    n_conv,
                )
        # Inject zero bias into any Conv ops we couldn't replace above.
        new_model, n_bias = self._inject_zero_bias_into_conv(new_model)
        if n_bias:
            self._logger.info("(conv-bias) Injected zero bias into %d Conv op(s)", n_bias)
        # lm_head: default to a single [1024, 65536] MatMul (tile-and-fuse
        # handles it); --split-lm-head keeps the legacy 512-chunk split.
        if self._split_lm_head:
            new_model, n_chunks = self._split_lm_head_matmul(new_model, chunk_size=128)
            if n_chunks:
                self._logger.info("(lm-head) Split lm_head MatMul into %d chunks", n_chunks)
        else:
            new_model, n_unsplit = self._unsplit_lm_head_matmul(new_model)
            if n_unsplit:
                self._logger.info(
                    "(lm-head) Folded Transpose into pre-transposed weight; "
                    "single [H, V] MatMul")
        # Re-run shape inference once more so the new chunks have full
        # value_info for downstream consumers / static-shape checks.
        del new_model.graph.value_info[:]
        try:
            new_model = onnx.shape_inference.infer_shapes(
                new_model, check_type=False, strict_mode=False, data_prop=True
            )
        except Exception as e:
            self._logger.warning("(lm-head) shape inference after split: %s", e)
        onnx.save(new_model, model_path)

    def make_static(self):
        self._logger.info("(model) Making graph static...")
        self._components["model"] = self.check_model(
            self._components["model"], skip_data_prop=True
        )
        self._components["model"] = self._make_model_static(self._components["model"])

    def apply_post_static_patches(self, model_path: str | os.PathLike, _):
        self._patch_static_model(model_path)
        if self._simulate_bf16:
            self._logger.info("(model) Creating bf16-simulated copy...")
            sim_dir = Path(model_path).parent.parent / "bf16_sim" / "static"
            sim_dir.mkdir(parents=True, exist_ok=True)
            sim_path = sim_dir / Path(model_path).name
            import shutil
            shutil.copy2(model_path, sim_path)
            emb_src = Path(model_path).parent / "token_embeddings.npy"
            if emb_src.exists():
                shutil.copy2(emb_src, sim_dir / "token_embeddings.npy")
            self._simulate_bf16_precision(sim_path)

    def _simulate_bf16_precision(self, model_path: str | os.PathLike):
        """Round-trip fp32 weights and activations through bf16 in fp32 ops."""
        model = onnx.load(model_path)
        graph = gs.import_onnx(model)

        n_weights = 0
        for tensor in graph.tensors().values():
            if isinstance(tensor, gs.Constant) and tensor.values is not None:
                if tensor.values.dtype == np.float32:
                    tensor.values = tensor.values.astype(ml_dtypes.bfloat16).astype(np.float32)
                    n_weights += 1

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
                op="Cast", name=f"{out_var.name}__cast_bf16",
                inputs=[out_var], outputs=[bf16_var],
                attrs={"to": onnx.TensorProto.BFLOAT16},
            )
            cast_to_fp32 = gs.Node(
                op="Cast", name=f"{out_var.name}__cast_fp32",
                inputs=[bf16_var], outputs=[fp32_var],
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
            "Saved bf16-simulated model to '%s' (%d weights, %d activation cast-pairs)",
            model_path, n_weights, n_casts,
        )

    def validate_onnx(self, n_iters: int = 3):
        """Light-weight validation: compare static fp32 ONNX vs source ONNX.

        bf16 cannot be validated under onnxruntime; the user must use the
        compiled .vmfb via VMFBInferenceRunner for end-to-end bf16 testing.
        """
        prompts = [
            "Hello",
            "The quick brown fox jumps over the lazy dog.",
            "def foo(x): return x * 2",
        ]
        n_threads: int = os.cpu_count()

        # Look for local tokenizer/config alongside the source ONNX
        local_cfg = self._onnx_dir / "config.json"
        local_tok = self._onnx_dir / "tokenizer.json"
        cfg_path = str(local_cfg) if local_cfg.exists() else None
        tok_path = str(local_tok) if local_tok.exists() else None

        if self._static_models:
            runner = LiquidStatic.from_onnx(
                self._export_paths["model"],
                self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo,
                config_path=cfg_path,
                tokenizer_path=tok_path,
            )
        else:
            runner = LiquidDynamic.from_onnx(
                self._export_paths["model"],
                max_gen_tokens=self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo,
                config_path=cfg_path,
                tokenizer_path=tok_path,
            )
        try:
            val_runner = LiquidDynamic.from_onnx(
                self._val_model_path,
                max_gen_tokens=self._max_gen_tokens,
                n_threads=n_threads,
                instruct_model=self._instruct_model,
                repo_id=self._hf_repo,
                config_path=cfg_path,
                tokenizer_path=tok_path,
            )
        except Exception as e:
            self._logger.warning("Could not load reference dynamic model for validation: %s", e)
            val_runner = None

        for i in range(n_iters):
            if i >= len(prompts):
                break
            inp = prompts[i]
            try:
                out = runner.run(inp)
                if val_runner is not None:
                    val_out = val_runner.run(inp)
                    ok = out[: min(len(out), len(val_out))] == val_out[: min(len(out), len(val_out))]
                    res = "OK" if ok else f"MISMATCH\n  ref: {val_out!r}\n  got: {out!r}"
                else:
                    res = f"(no reference) generated: {out!r}"
                self._logger.info("(ONNX-validation) [iter %d, %.3f ms] %s", i, runner.last_infer_time / 1e6, res)
            except Exception as e:
                self._logger.error("(ONNX-validation) [iter %d] failed: %s", i, e)

    def export_torq(
        self,
        torq_export_dir: str | os.PathLike | None = None,
        torq_compile_args: list[str] | None = None,
        use_binary: bool = False,
        skip: list[str] | None = None,
        local_compile: bool = False,
        compiler_path: str | Path | None = None,
    ):
        """Compile the exported (bf16) ONNX to a Torq vmfb.

        Prepends LFM2.5's validated torq-compile flag set (``LIQUID_TORQ_FLAGS``)
        — in particular ``--torq-enable-split-constants-optimization``, which
        we measured to be faster and lower-heap than the default — ahead of any
        user-supplied ``--compile-flags``, then defers to the shared
        ``torq.utils.compile`` driver via the base exporter.
        """
        merged_args = list(LIQUID_TORQ_FLAGS) + list(torq_compile_args or [])
        return super().export_torq(
            torq_export_dir=torq_export_dir,
            torq_compile_args=merged_args,
            use_binary=use_binary,
            skip=skip,
            local_compile=local_compile,
            compiler_path=compiler_path,
        )

    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
    ):
        """Convert exported fp32 ONNX to bf16.

        Skips the int64 → int32 post-step used by other exporters: LFM2.5
        has Shape / Unsqueeze / Split chains that the int32 converter
        misclassifies, leaving the resulting model fundamentally broken.
        The bf16 model alone is what IREE consumes for bf16 compilation.
        """
        import shutil
        from ...tools.convert_dtype.onnx import convert_model as _convert_dtype

        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
            return
        self._convert_dir = Path(convert_dir or self._convert_dir)
        if self._convert_dir.exists():
            shutil.rmtree(self._convert_dir, ignore_errors=True)
        self._convert_dir.mkdir(parents=True, exist_ok=True)

        for comp, model_path in list(self._export_paths.items()):
            self._logger.info("(ONNX-convert) Converting '%s' to bf16...", model_path)
            converted_model_path = self._convert_dir / model_path.name
            _convert_dtype(model_path, converted_model_path, "bf16", convert_io=not preserve_io)
            self._logger.info("(ONNX-convert) Wrote '%s'", converted_model_path)
            self._export_paths[comp] = converted_model_path

        if self._extract_embeddings:
            emb_src = self._export_dir / "token_embeddings.npy"
            if emb_src.exists():
                emb_data = np.load(emb_src).astype(ml_dtypes.bfloat16)
                np.save(self._convert_dir / emb_src.name, emb_data)


def export_liquid_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = LiquidModelExporter(
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
        broadcast_ops=args.broadcast_ops,
        simulate_bf16=args.simulate_bf16,
        keep_conv1d=args.keep_conv1d,
        split_lm_head=args.split_lm_head,
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
    parser = argparse.ArgumentParser(description="Export LFM2.5 (Liquid) to Torq")
    add_liquid_export_args(parser)
    export_liquid_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
