# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Exporter for LFM2-VL-450M (vision-language) to Torq.

Unlike the text-only LFM2.5-350M (a single ``model.onnx``), the HF ONNX
export of LFM2-VL ships **three** separate components:

``embed_tokens.onnx``
    A single ``Gather(weight[V, H], input_ids) -> inputs_embeds`` — the
    token-embedding LUT.  We extract its weight to ``token_embeddings.npy``
    (CPU-side, identical to the 350m ``--extract-embeddings`` flow) and never
    compile it.

``decoder_model_merged.onnx``
    The LFM2 hybrid conv + attention decoder.  It takes ``inputs_embeds``
    directly (i.e. it is the 350m decoder *after* embedding extraction) and is
    architecturally identical to LFM2.5-350M: 16 layers (6 GQA attention + 10
    short-conv), hidden 1024, vocab 65536, 16/8 heads, head_dim 64,
    conv_L_cache 3.  It therefore reuses the entire LFM2.5 chip-rewrite
    pipeline (custom-op replacement, static KV/conv cache, Conv1D→MatMul,
    single ``[H, V]`` lm_head) and is the primary chip target.

``vision_encoder.onnx``
    A SigLIP-style tower (``MultiHeadAttention``, ``Resize``, ``Compress``,
    ``ScatterND``, dynamic ``num_patches``).  It is exported as ONNX so it can
    be run on CPU/ORT, but chip compilation is experimental and **off by
    default** (enable with ``--compile-vision``).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import ml_dtypes
from torq.utils.logging import configure_logging

from .export import LiquidModelExporter, LIQUID_TORQ_FLAGS  # noqa: F401  (import triggers gs bf16 patch)
from ._graph import LiquidOnnxGraphEditor
from ...model_export.onnx import OnnxModelExporterBase


# Component keys — match the source ONNX filenames so the base exporter writes
# ``<comp>.onnx`` back out under the same names.
DECODER = "decoder_model_merged"
VISION = "vision_encoder"
EMBED_FILE = "embed_tokens.onnx"

# HuggingFace repo for config / tokenizer.  The text tower is architecturally
# identical to LFM2.5-350M, so that repo is the architecture-compatible
# fallback when the VL repo (or network) is unavailable.
HF_REPO_VL = "LiquidAI/LFM2-VL-450M"
HF_REPO_TEXT_FALLBACK = "LiquidAI/LFM2.5-350M"

# ONNX source repos for the 3 VL components (under onnx/), tried in order: the
# Synaptics-hosted mirror first (so the pipeline doesn't break if the upstream
# community repo is renamed/removed), then the public upstream export. The
# Synaptics mirror is private, so it needs an HF token (HF_TOKEN env or
# `huggingface-cli login`); without one it 401s and we fall through to the
# public upstream. (The base LiquidAI/LFM2-VL-450M repo ships safetensors only,
# no ONNX, so it is not a source here.)
HF_REPO_ONNX_VL = (
    "Synaptics/liquidAI-LFM2-VLM",
    "onnx-community/LFM2-VL-450M-ONNX",
)


class LiquidVLModelExporter(LiquidModelExporter):
    """Export + compile LFM2-VL-450M for the SL2610.

    Subclasses :class:`LiquidModelExporter` to reuse its full LFM2.5
    graph-rewrite toolbox (custom-op replacement, static KV/conv cache,
    Conv1D→batched-MatMul, lm_head folding) on the decoder, and overrides the
    multi-component plumbing (load / static / patch / convert / compile) so the
    three VL components are each handled appropriately.
    """

    def __init__(
        self,
        *,
        instruct_model: bool = False,
        max_gen_tokens: int = 256,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        compile_vision: bool = False,
        keep_individual_kv_io: bool = False,
        static_models: bool = True,
        **edit_args,
    ):
        # --- LFM2.5 exporter knobs (mirror LiquidModelExporter.__init__) ----
        self._instruct_model = instruct_model
        # The VL decoder takes inputs_embeds directly; there is no Gather to
        # extract.  The token-embedding LUT lives in embed_tokens.onnx and is
        # handled separately (see _extract_embed_lut).
        self._extract_embeddings = False
        self._keep_individual_kv_io = keep_individual_kv_io
        self._max_gen_tokens = max_gen_tokens
        self._onnx_source_dir = onnx_source_dir
        self._model_size = "450m-vl"
        self._hf_repo = HF_REPO_VL
        self._hf_repo_onnx = HF_REPO_VL
        self._compile_vision = compile_vision
        # Build + compile a static single-resolution vision encoder
        # (vision_encoder_<res>.vmfb) instead of the dynamic one. 128 or 256.
        self._vision_res = edit_args.get("vision_res", None)
        # Also build + compile the one-shot image-prefill decoder, split into N
        # layer-boundary parts (decoder_image_<N>part_<A..>.vmfb). 2, 3 or 5.
        self._image_decoder_parts = edit_args.get("image_decoder_parts", None)
        self._dynamic_decoder = None  # captured pre-static for the image build
        self._broadcast_ops = edit_args.get("broadcast_ops", None)
        self._simulate_bf16 = edit_args.get("simulate_bf16", False)
        self._replace_conv1d = not edit_args.get("keep_conv1d", False)
        self._split_lm_head = edit_args.get("split_lm_head", False)
        # Also emit decoder_nolm.vmfb (body) + lm_head.vmfb (the board's
        # lower-TTFT split) alongside the merged decoder.
        self._split_decoder = edit_args.get("split_decoder", False)

        # --- resolve the text-decoder architecture config -------------------
        self._config_dict = self._resolve_text_config(onnx_source_dir, models_dir)
        self._hidden_size = int(self._config_dict["hidden_size"])
        self._vocab_size = int(self._config_dict["vocab_size"])
        self._num_attention_heads = int(self._config_dict["num_attention_heads"])
        self._num_key_value_heads = int(self._config_dict["num_key_value_heads"])
        self._head_dim = int(
            self._config_dict.get("head_dim")
            or (self._hidden_size // self._num_attention_heads)
        )
        self._conv_dim = int(self._config_dict.get("conv_dim", self._hidden_size))
        self._conv_L_cache = int(self._config_dict.get("conv_L_cache", 3))
        self._num_hidden_layers = int(self._config_dict["num_hidden_layers"])
        self._layer_types = tuple(
            self._config_dict.get("layer_types")
            or self._default_layer_types(self._num_hidden_layers)
        )

        # Skip the LFM2.5 __init__ (it hard-codes the single-model 350m config
        # download); go straight to the generic base, which calls our
        # _setup_dirs / _load_onnx overrides.
        OnnxModelExporterBase.__init__(
            self,
            "fp32",
            static_models,
            self._config_dict,
            Path(models_dir),
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs={},  # LFM2 custom ops break the ORT bert optimizer
        )

    # ------------------------------------------------------------------ config
    @staticmethod
    def _default_layer_types(num_layers: int) -> list[str]:
        """LFM2's published conv/attention interleave (used only if a config
        omits ``layer_types``).  Matches LFM2.5-350M / LFM2-VL-450M: full
        attention at layers 2, 5, 8, 10, 12, 14; short conv everywhere else."""
        attn = {2, 5, 8, 10, 12, 14}
        return ["full_attention" if i in attn else "conv" for i in range(num_layers)]

    def _resolve_text_config(
        self,
        onnx_source_dir: str | os.PathLike | None,
        models_dir: str | os.PathLike,
    ) -> dict:
        """Locate the LFM2 text-decoder config for LFM2-VL.

        LFM2-VL's ``config.json`` nests the language-model params under
        ``text_config``.  Resolution order: explicit source dir → VL source
        dir → HF VL repo → local LFM2.5-350m config → HF LFM2.5-350M.  The
        350m text tower is architecturally identical, so it is a safe
        fallback when the VL repo (or network) is unavailable.
        """
        def _read(path: Path) -> dict | None:
            if not path.exists():
                return None
            with open(path) as f:
                cfg = json.load(f)
            return cfg.get("text_config", cfg)

        search: list[Path] = []
        if onnx_source_dir is not None:
            search.append(Path(onnx_source_dir) / "config.json")
        search.append(Path(models_dir) / "source" / "onnx" / "fp32" / "config.json")
        search.append(
            Path(models_dir).parent / "liquid-2p5-350m"
            / "source" / "onnx" / "fp32" / "config.json"
        )
        for cand in search:
            cfg = _read(cand)
            if cfg is not None:
                return cfg

        from huggingface_hub import hf_hub_download
        for repo in (HF_REPO_VL, HF_REPO_TEXT_FALLBACK):
            try:
                cfg = _read(Path(hf_hub_download(repo, "config.json")))
                if cfg is not None:
                    return cfg
            except Exception:
                continue
        raise RuntimeError(
            "Could not resolve an LFM2 text config for LFM2-VL. Place a "
            "config.json next to the source ONNX, or ensure the local "
            "liquid-2p5-350m config is available."
        )

    # -------------------------------------------------------------------- dirs
    def _setup_dirs(self) -> list[Path]:
        if self._onnx_source_dir is not None:
            onnx_dir = Path(self._onnx_source_dir)
        else:
            onnx_dir = self._models_dir / "source" / "onnx" / self._model_dtype
        if not (onnx_dir / f"{DECODER}.onnx").exists():
            self._download_source(onnx_dir)
        if not (onnx_dir / f"{DECODER}.onnx").exists():
            raise FileNotFoundError(
                f"Expected LFM2-VL source components ({DECODER}.onnx, "
                f"{VISION}.onnx, {EMBED_FILE}) under '{onnx_dir}'. Set HF_TOKEN "
                f"to auto-download them from '{HF_REPO_ONNX_VL}', or place them "
                f"there manually."
            )

        suffix = "static" if self._static_models else "dynamic"
        export_dir = self._models_dir / "export" / "onnx" / self._model_dtype / suffix
        convert_dir = self._models_dir / "export" / "onnx" / "bf16" / suffix
        iree_dir = (
            self._models_dir / "export" / "iree"
            / ("bf16" if self._convert_dtypes else self._model_dtype) / suffix
        )
        return onnx_dir, export_dir, convert_dir, iree_dir

    def _download_source(self, target_dir: Path):
        """Fetch the LFM2-VL ONNX components from the mirror repos.

        Tries each repo in :data:`HF_REPO_ONNX_VL` in order (Synaptics mirror
        first, public upstream fallback). The private mirror needs an HF token;
        if none of the repos work we warn and leave ``_setup_dirs`` to raise so
        the user can place the components manually.

        Only the decoder pair is strictly required — ``embed_tokens`` is a
        single self-contained file on the mirror but split into ``.onnx_data``
        upstream, so the extra weight files are downloaded only if present.
        """
        from huggingface_hub import hf_hub_download
        import shutil

        target_dir.mkdir(parents=True, exist_ok=True)
        required = [f"onnx/{DECODER}.onnx", f"onnx/{DECODER}.onnx_data", f"onnx/{EMBED_FILE}"]
        optional = [
            f"onnx/{EMBED_FILE}_data",
            f"onnx/{VISION}.onnx",
            f"onnx/{VISION}.onnx_data",
            "config.json",
            "tokenizer.json",
        ]
        repos = HF_REPO_ONNX_VL if isinstance(HF_REPO_ONNX_VL, (list, tuple)) else (HF_REPO_ONNX_VL,)
        for repo in repos:
            self._logger.info("Downloading LFM2-VL source ONNX from '%s'...", repo)
            try:
                for filename in required:
                    p = hf_hub_download(repo, filename)
                    dest = target_dir / Path(filename).name
                    if not dest.exists():
                        shutil.copy(p, dest)
            except Exception as e:
                self._logger.warning("  source repo '%s' unavailable (%s); trying next", repo, e)
                continue
            for filename in optional:
                try:
                    p = hf_hub_download(repo, filename)
                    dest = target_dir / Path(filename).name
                    if not dest.exists():
                        shutil.copy(p, dest)
                except Exception as e:
                    self._logger.debug("  optional %s not in %s: %s", filename, repo, e)
            self._logger.info("VL source ready from '%s' at %s", repo, target_dir)
            return
        self._logger.warning(
            "  could not fetch VL source from any of %s; set HF_TOKEN or place "
            "the ONNX under '%s' manually", list(repos), target_dir
        )

    # -------------------------------------------------------------------- load
    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        # The token-embedding LUT is pulled out to a sibling .npy now so it can
        # ride along with the decoder vmfb on the board.
        self._extract_embed_lut()

        components: dict[str, onnx.ModelProto] = {DECODER: self._load_decoder()}

        vision_path = self._onnx_dir / f"{VISION}.onnx"
        if vision_path.exists():
            self._logger.info("Loading vision encoder '%s'...", vision_path)
            components[VISION] = onnx.load(vision_path, load_external_data=True)
        else:
            self._logger.warning("No vision_encoder.onnx found; skipping it")
        return components

    def _extract_embed_lut(self):
        """Dump the ``embed_tokens.onnx`` Gather weight to token_embeddings.npy."""
        embed_path = self._onnx_dir / EMBED_FILE
        if not embed_path.exists():
            self._logger.warning("No %s found; skipping LUT extraction", EMBED_FILE)
            return
        model = onnx.load(embed_path, load_external_data=True)
        gather = next((n for n in model.graph.node if n.op_type == "Gather"), None)
        if gather is None:
            self._logger.warning("%s has no Gather; skipping LUT extraction", EMBED_FILE)
            return
        weight_name = gather.input[0]
        init = next((i for i in model.graph.initializer if i.name == weight_name), None)
        if init is None:
            self._logger.warning(
                "Gather weight '%s' is not an initializer; skipping LUT extraction",
                weight_name,
            )
            return
        lut = onnx.numpy_helper.to_array(init)  # [vocab, hidden], fp32
        out = self._export_dir / "token_embeddings.npy"
        np.save(out, lut)
        self._logger.info("Extracted token embedding LUT %s -> '%s'", lut.shape, out)

    def _load_decoder(self) -> onnx.ModelProto:
        """Load decoder_model_merged.onnx and run the LFM2.5 custom-op
        replacement, exactly as LiquidModelExporter._load_onnx does for the
        single 350m model — but rename ``inputs_embeds`` to ``token_embedding``
        so the existing LiquidStatic runner / chip demo feed it unchanged."""
        model_path = self._onnx_dir / f"{DECODER}.onnx"
        self._val_model_path = model_path

        model = onnx.load(model_path, load_external_data=True)
        orig_ir = model.ir_version

        graph = gs.import_onnx(model)
        graph.name = "main"

        for inp in graph.inputs:
            if inp.name == "inputs_embeds":
                inp.name = "token_embedding"

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

        has_ms_ops = any(n.domain == "com.microsoft" for n in model.graph.node)
        if not has_ms_ops:
            for opset in list(model.opset_import):
                if opset.domain == "com.microsoft":
                    model.opset_import.remove(opset)
        else:
            remaining = sorted(
                {n.op_type for n in model.graph.node if n.domain == "com.microsoft"}
            )
            self._logger.warning(
                "Decoder still has com.microsoft ops: %s", remaining
            )
        return model

    # ------------------------------------------------------- shape propagation
    def _propagate_static_shapes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fold Shape ops + normalize Slices to a fixpoint.

        Identical to the base LFM2.5 routine but iterates until convergence
        (capped at 30) instead of a fixed 8.  The LFM2-VL decoder's optimum
        export emits a per-conv-layer causal-trim subgraph
        (``Shape→Gather→Mul→Unsqueeze→Slice(ends=INT64_MAX)``); each layer's
        ``Shape`` only resolves once the previous layer's output shape is
        concrete, so resolution proceeds one layer per iteration and needs
        ~10–12 passes across the 16-layer stack — more than the base cap.
        """
        max_iters = 30
        it = 0
        for it in range(max_iters):
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
        else:
            self._logger.warning(
                "(shape-prop) hit iteration cap (%d) without converging", max_iters
            )
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

    # ------------------------------------------------------------------ static
    def make_static(self):
        """Only the decoder needs static KV/conv-cache treatment; the vision
        encoder is left dynamic (its compile is experimental)."""
        self._logger.info("(%s) Making graph static...", DECODER)
        self._components[DECODER] = self.check_model(
            self._components[DECODER], skip_data_prop=True
        )
        # The image-prefill decoder is built from the custom-op-replaced *dynamic*
        # decoder (different static shape: [1,64,1024], empty past), so capture a
        # copy before it is static-ized (and before later export steps can mutate
        # the shared proto) for single-token decode.
        if self._image_decoder_parts:
            import copy
            self._dynamic_decoder = copy.deepcopy(self._components[DECODER])
        self._components[DECODER] = self._make_model_static(self._components[DECODER])

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        if component != DECODER:
            return
        self._patch_static_model(model_path)
        # The chip runner invokes the vmfb positionally, so pin the decoder's
        # leading inputs: token_embedding (0), position_ids (1).
        self._reorder_decoder_inputs(model_path)
        if self._simulate_bf16:
            self._logger.info("(model) Creating bf16-simulated copy...")
            sim_dir = Path(model_path).parent.parent / "bf16_sim" / "static"
            sim_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            sim_path = sim_dir / Path(model_path).name
            shutil.copy2(model_path, sim_path)
            emb_src = self._export_dir / "token_embeddings.npy"
            if emb_src.exists():
                shutil.copy2(emb_src, sim_dir / "token_embeddings.npy")
            self._simulate_bf16_precision(sim_path)

    @staticmethod
    def _reorder_decoder_inputs(model_path: str | os.PathLike):
        """Pin ``token_embedding`` then ``position_ids`` as the first two graph
        inputs (the rest keep their order)."""
        order = ["token_embedding", "position_ids"]
        model = onnx.load(model_path)
        by_name = {i.name: i for i in model.graph.input}
        front = [by_name[n] for n in order if n in by_name]
        rest = [i for i in model.graph.input if i.name not in set(order)]
        del model.graph.input[:]
        model.graph.input.extend(front + rest)
        onnx.save(model, model_path)

    # ----------------------------------------------------------------- convert
    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
    ):
        """Convert the exported fp32 ONNX (and the embedding LUT) to bf16.

        Skips the int64→int32 post-step (LFM2 Shape/Unsqueeze/Split chains
        break it) like the 350m exporter, and skips the vision encoder unless
        ``--compile-vision`` was requested."""
        import shutil
        from ...tools.convert_dtype.onnx import convert_model as _convert_dtype

        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
            return
        self._convert_dir = Path(convert_dir or self._convert_dir)
        if self._convert_dir.exists():
            shutil.rmtree(self._convert_dir, ignore_errors=True)
        self._convert_dir.mkdir(parents=True, exist_ok=True)

        # The static vision encoder has its own build+bf16 pipeline, so pull the
        # dynamic vision component out of the plain bf16-convert loop.
        vision_src = None
        if self._vision_res:
            vp = self._export_paths.pop(VISION, None)
            vision_src = str(vp) if vp else None

        for comp, model_path in list(self._export_paths.items()):
            if comp == VISION and not self._compile_vision:
                self._logger.info("(ONNX-convert) Skipping vision encoder (fp32 only)")
                continue
            self._logger.info("(ONNX-convert) Converting '%s' to bf16...", model_path)
            converted = self._convert_dir / model_path.name
            _convert_dtype(model_path, converted, "bf16", convert_io=not preserve_io)
            self._export_paths[comp] = converted

        emb_src = self._export_dir / "token_embeddings.npy"
        if emb_src.exists():
            emb = np.load(emb_src).astype(ml_dtypes.bfloat16)
            np.save(self._convert_dir / "token_embeddings.npy", emb)
            self._logger.info("(ONNX-convert) Wrote bf16 token_embeddings.npy")

        if self._vision_res and vision_src:
            self._make_static_vision(vision_src)

        if self._split_decoder:
            self._make_decoder_split()

        if self._image_decoder_parts:
            self._make_image_decoder_parts()

    def _make_image_decoder_parts(self):
        """Build the one-shot image-prefill decoder + split into N layer parts,
        registering each ``decoder_image_<N>part_<label>`` bf16 component so
        export_torq compiles it. See :mod:`._image_prefill`."""
        from ._image_prefill import build_image_decoder, split_image_decoder
        from ...tools.convert_dtype.onnx import convert_model as _convert_dtype

        n = self._image_decoder_parts
        self._logger.info("(image-decoder) building cache-only image decoder + %d-part split...", n)
        full = build_image_decoder(
            self._dynamic_decoder,
            LiquidModelExporter._replace_conv1d_with_matmul,
            self._propagate_static_shapes,
        )
        for label, part in split_image_decoder(full, n):
            name = f"decoder_image_{n}part_{label}"
            fp32 = self._convert_dir / f"{name}.fp32.onnx"
            onnx.save(part, str(fp32), save_as_external_data=True,
                      location=fp32.name + "_data", size_threshold=1024)
            bf16 = self._convert_dir / f"{name}.onnx"
            _convert_dtype(str(fp32), str(bf16), "bf16", convert_io=True)
            # Drop the large fp32 intermediate — only the bf16 part is compiled.
            fp32.unlink(missing_ok=True)
            (self._convert_dir / (fp32.name + "_data")).unlink(missing_ok=True)
            self._export_paths[name] = bf16
            self._logger.info("(image-decoder) part %s: %d outputs -> %s",
                              label, len(part.graph.output), bf16.name)

    def _make_static_vision(self, vision_src: str):
        """Build a static single-resolution vision encoder and register it as a
        ``vision_encoder_<res>`` component so export_torq compiles it. Reuses
        the (validated) transforms in :mod:`._vision_static`."""
        from ._vision_static import build_static_vision_encoder, VISION_RES

        patches, grid = VISION_RES[self._vision_res]
        out_prefix = str(self._convert_dir / f"vision_encoder_{self._vision_res}")
        self._logger.info("(vision) building static %d-res encoder from '%s'...",
                          self._vision_res, vision_src)
        # split_matmul=512: chunk the wide (3072-output) SigLIP FFN/embed
        # matmuls into [768,512] slices. Torq-compile funnels the encoder
        # front-end into one fused dispatch that over-segments into ~100k NSS
        # slice programs, sending SegmentNSSPrograms super-linear; splitting the
        # matmuls cuts the slice count (284k->107k passes) and is bit-exact
        # (validated, max abs diff ~0.002). NOTE: this only mitigates — the
        # 256-res encoder is still compile-heavy on constrained hosts; always
        # keep --torq-disable-slicing (slicing OOMs) and RAM-guard the compile.
        bf16 = build_static_vision_encoder(
            out_prefix, patches=patches, grid=grid, src=vision_src,
            split_matmul=512,
        )
        self._export_paths[f"vision_encoder_{self._vision_res}"] = Path(bf16)

    def _make_decoder_split(self):
        """Split the converted bf16 decoder into ``decoder_nolm`` (body, hidden
        output) + ``lm_head`` (standalone hidden->logits), and register both so
        export_torq compiles ``decoder_nolm.vmfb`` + ``lm_head.vmfb`` — the
        board's lower-TTFT deployment (decode body on NPU, lm_head applied only
        when sampling).

        Structure-based (no hard-coded names): the lm_head is the node that
        produces the vocab-logits graph output; its activation input is the
        split boundary. Works whether the lm_head is a folded single MatMul or
        the 512-chunk split.
        """
        import copy
        from onnx import helper, TensorProto

        src_path = self._export_paths[DECODER]
        model = onnx.load(src_path, load_external_data=True)
        g = model.graph
        bf16 = TensorProto.BFLOAT16

        logits_name = "logits"
        if not any(o.name == logits_name for o in g.output):
            logits_name = next(
                o.name for o in g.output
                if o.type.tensor_type.shape.dim
                and o.type.tensor_type.shape.dim[-1].dim_value == self._vocab_size
            )
        init_names = {i.name for i in g.initializer}
        lm_nodes = [n for n in g.node if any(o == logits_name for o in n.output)
                    or (n.name and "lm_head" in n.name)]
        # The activation feeding the lm_head (the final-norm hidden state): an
        # lm_head input that no other lm_head node produces and isn't a weight.
        lm_produced = {o for n in lm_nodes for o in n.output}
        lm_inputs = {i for n in lm_nodes for i in n.input}
        hidden_name = next(
            i for i in lm_inputs
            if i and i not in init_names and i not in lm_produced
        )
        lm_node_names = {n.name for n in lm_nodes}

        # ---- decoder_nolm: drop lm_head node(s), expose hidden as output -----
        body = onnx.ModelProto()
        body.CopyFrom(model)
        bg = body.graph
        del bg.node[:]
        bg.node.extend([n for n in g.node if n.name not in lm_node_names])
        outs = [helper.make_tensor_value_info(hidden_name, bf16, [1, 1, self._hidden_size])]
        outs += [o for o in g.output if o.name != logits_name]
        del bg.output[:]
        bg.output.extend(outs)
        used = {i for n in bg.node for i in n.input}
        del bg.initializer[:]
        bg.initializer.extend([i for i in g.initializer if i.name in used])
        nolm_path = self._convert_dir / "decoder_nolm.onnx"
        onnx.save(body, str(nolm_path), save_as_external_data=False)

        # ---- lm_head: standalone hidden -> logits ---------------------------
        weights = [copy.deepcopy(i) for i in g.initializer
                   if i.name in (lm_inputs & init_names)]
        lm_graph = helper.make_graph(
            [copy.deepcopy(n) for n in lm_nodes],
            "main",
            [helper.make_tensor_value_info(hidden_name, bf16, [1, 1, self._hidden_size])],
            [helper.make_tensor_value_info(logits_name, bf16, [1, 1, self._vocab_size])],
            weights,
        )
        lm_model = helper.make_model(lm_graph, opset_imports=list(model.opset_import))
        lm_model.ir_version = model.ir_version
        lmh_path = self._convert_dir / "lm_head.onnx"
        onnx.save(lm_model, str(lmh_path), save_as_external_data=False)

        self._export_paths["decoder_nolm"] = nolm_path
        self._export_paths["lm_head"] = lmh_path
        self._logger.info(
            "(split) derived decoder_nolm (%d nodes) + lm_head (%d node(s)) from '%s'",
            len(bg.node), len(lm_nodes), src_path.name,
        )

    # ------------------------------------------------------------------ compile
    def export_torq(self, *args, skip: list[str] | None = None,
                    torq_compile_args: list[str] | None = None, **kwargs):
        """Compile the decoder (and, if requested, the vision encoder / image
        decoder parts)."""
        skip = list(skip or [])
        if not self._compile_vision and VISION not in skip:
            skip.append(VISION)
        extra = list(torq_compile_args or [])
        if (self._image_decoder_parts or self._vision_res) and \
                "--torq-max-nss-programs-size" not in extra:
            # The image-decoder parts and the materialized static vision encoder
            # emit many NSS programs (~195-205 MB); the 8 MB default is far too
            # small (image_prefill.md §3e). Harmless for the other components.
            extra += ["--torq-max-nss-programs-size", "402653184"]
        return super().export_torq(*args, skip=skip, torq_compile_args=extra, **kwargs)

    # ------------------------------------------------------- deployment assets
    def stage_deploy_assets(self):
        """Place the runner's sidecar files next to the decoder vmfb.

        The LiquidStatic runner loads ``token_embeddings.npy``, ``config.json``
        and ``tokenizer.json`` from the vmfb's parent directory. The LFM2-VL
        source ships no config/tokenizer, so we write the resolved text config
        and copy a tokenizer (from the source dir, else the local 350m, else
        HF) into the iree output dir."""
        import shutil

        dest = self._torq_dir  # export/iree/<dtype>/<static|dynamic>
        if not dest.exists():
            return

        # token-embedding LUT (bf16 if converted, else fp32)
        lut = (self._convert_dir / "token_embeddings.npy")
        if not lut.exists():
            lut = self._export_dir / "token_embeddings.npy"
        if lut.exists():
            shutil.copy2(lut, dest / "token_embeddings.npy")

        # resolved (flat) text config
        try:
            with open(dest / "config.json", "w") as f:
                json.dump(self._config_dict, f, indent=2)
        except Exception as e:
            self._logger.warning("Could not write config.json: %s", e)

        # tokenizer.json: source dir → local 350m → HF
        tok_src = self._onnx_dir / "tokenizer.json"
        if not tok_src.exists():
            cand = (
                self._models_dir.parent / "liquid-2p5-350m"
                / "source" / "onnx" / "fp32" / "tokenizer.json"
            )
            tok_src = cand if cand.exists() else None
        if tok_src is None:
            try:
                from huggingface_hub import hf_hub_download
                for repo in (HF_REPO_VL, HF_REPO_TEXT_FALLBACK):
                    try:
                        tok_src = Path(hf_hub_download(repo, "tokenizer.json"))
                        break
                    except Exception:
                        continue
            except Exception:
                tok_src = None
        if tok_src is not None and Path(tok_src).exists():
            shutil.copy2(tok_src, dest / "tokenizer.json")
            self._logger.info("Staged deploy assets (LUT, config, tokenizer) -> '%s'", dest)
        else:
            self._logger.warning(
                "No tokenizer.json found for LFM2-VL; place one next to the vmfb "
                "before running the demo."
            )

    # --------------------------------------------------------------- validation
    def validate_onnx(self, n_iters: int = 3):
        # Full VL validation needs image inputs + the merged vision/text
        # pipeline, which is out of scope for this exporter. The fp32 decoder
        # can still be sanity-checked through onnxruntime by the runner in
        # torq-examples; skip automatic validation here.
        self._logger.info(
            "(validation) Skipping automatic VL validation; validate the "
            "decoder via torq-examples with token_embeddings.npy."
        )


def export_liquid_vl_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = LiquidVLModelExporter(
        instruct_model=args.instruct_model,
        max_gen_tokens=args.max_gen_tokens,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        compile_vision=args.compile_vision,
        keep_individual_kv_io=args.keep_individual_kv_io,
        static_models=not args.dynamic_models,
        broadcast_ops=args.broadcast_ops,
        simulate_bf16=args.simulate_bf16,
        keep_conv1d=args.keep_conv1d,
        split_lm_head=args.split_lm_head,
        split_decoder=args.split_decoder,
        vision_res=args.vision_res,
        image_decoder_parts=args.image_decoder_parts,
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
        exporter.stage_deploy_assets()


def main():
    import argparse as _argparse
    from . import add_liquid_vl_export_args

    parser = _argparse.ArgumentParser(description="Export LFM2-VL (Liquid) to Torq")
    add_liquid_vl_export_args(parser)
    export_liquid_vl_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
