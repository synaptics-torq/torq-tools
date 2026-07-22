# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Static ONNX export for Gemma-4-E2B from a pre-quantized (int4) ONNX source.

Unlike ``export.py`` (which builds a *dynamic* KV-cache graph from
safetensors via `transformers`/`torch.onnx`), this exporter loads an
already-quantized ONNX pair (``embed_tokens`` + ``decoder_model_merged``,
using ``MatMulNBits``/``GatherBlockQuantized`` weight-only int4 quantization)
and makes it *static* (fixed batch=1, seq_len=1, fixed-length KV cache).
No `torch`/`transformers` dependency. See ``STATIC_EXPORT_PLAN.md`` for the
full design rationale and the graph structure this was reverse-engineered
against.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from . import add_gemma4_int4_export_args
from ._graph import Gemma4OnnxGraphEditor
from ...model_export.onnx import OnnxModelExporterBase
from ...utils.logging import configure_logging
from ...utils.onnx import check_dynamic_shapes, propagate_static_shapes

DEFAULT_HF_REPO_INT4: Final[str] = "tss-deposium/gemma-4-E2B-text-only-onnx-int4"
# The int4 repo doesn't ship a chat template; pull it from a sibling ONNX
# export of the same underlying checkpoint that does.
DEFAULT_TEMPLATE_REPO: Final[str] = "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX"

_DECODER_FILENAME: Final[str] = "decoder_model_merged_q4.onnx"
_EMBED_FILENAME: Final[str] = "embed_tokens_q4.onnx"

# GQA-fused (sliding-window) layer config -- confirmed uniform across all 12
# fused GroupQueryAttention nodes in the source graph (see STATIC_EXPORT_PLAN.md).
_GQA_NUM_HEADS: Final[int] = 8
_GQA_KV_NUM_HEADS: Final[int] = 1
_GQA_HEAD_DIM: Final[int] = 256

# The ONLY initializers this exporter's graph edits ever read `.values` on
# (`ReplaceRotaryEmbedding` reads each RotaryEmbedding node's cos/sin cache
# to derive head_dim). Everything else -- notably the ~972 individually
# externalized MatMulNBits/GatherBlockQuantized weight chunks (~1.73GB
# combined; each one is itself small enough that a size-based threshold
# can't distinguish them from these caches, confirmed empirically -- do NOT
# switch this back to a byte-size threshold) -- must stay external/unloaded.
_ROPE_CACHE_NAMES: Final[frozenset[str]] = frozenset(
    {"cos_cache_local", "sin_cache_local", "cos_cache_global", "sin_cache_global"}
)


def _preload_small_external_initializers(model: onnx.ModelProto, base_dir: Path) -> None:
    """Inline the RoPE cos/sin cache initializers so onnx_graphsurgeon's lazy
    ``.values`` accessor can read them.

    `onnx_graphsurgeon`'s ``LazyValues.load()`` calls
    ``onnx.numpy_helper.to_array(tensor)`` with no ``base_dir``, so it
    resolves the tensor's external-data location against the process's
    *current working directory* instead of the model's own directory -- a
    real bug, not a hypothetical. Pre-loading the handful of tensors our
    graph edits actually read sidesteps it entirely, while leaving the large
    weight blobs external and unmaterialized (nothing in this exporter's
    edits reads their values, only their declared shapes -- always available
    from the TensorProto header regardless of load state).
    """
    from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data

    for tensor in model.graph.initializer:
        if tensor.name not in _ROPE_CACHE_NAMES or not uses_external_data(tensor):
            continue
        load_external_data_for_tensor(tensor, str(base_dir))
        tensor.data_location = onnx.TensorProto.DEFAULT
        del tensor.external_data[:]


def _fix_gather_block_quantized_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Correct `SymbolicShapeInference`'s `GatherBlockQuantized` output shape.

    Confirmed empirically (see STATIC_EXPORT_PLAN.md): `SymbolicShapeInference`
    declares this op's output last dim as the *packed* (quantized) weight's
    last dim, not the unpacked (logical) one -- e.g. a `bits=4` weight packs
    2 values/byte, so the real output dim is 2x what gets declared (a
    `[262144, 768]` packed weight really produces a 1536-wide embedding,
    not 768). Onnxruntime tolerates the mismatch at load/run time via a
    "lenient merge" (a warning, not an error) and computes the correct
    value regardless -- this fix is cosmetic (removes log noise) and
    precautionary (a future consumer that trusts declared shapes strictly,
    e.g. a torq-compile attempt, shouldn't have to also tolerate it).
    """
    graph = gs.import_onnx(model)
    fixed = 0
    for node in graph.nodes:
        if node.op != "GatherBlockQuantized":
            continue
        bits = int(node.attrs.get("bits", 4))
        packed_last_dim = int(node.inputs[0].shape[-1])
        unpacked_last_dim = packed_last_dim * (8 // bits)
        out = node.outputs[0]
        if out.shape and int(out.shape[-1]) != unpacked_last_dim:
            out.shape[-1] = unpacked_last_dim
            fixed += 1
    if not fixed:
        return model
    fixed_model = gs.export_onnx(graph)
    fixed_model.ir_version = model.ir_version
    return fixed_model


class Gemma4Int4ModelExporter(OnnxModelExporterBase):
    """Loads Gemma-4-E2B's quantized int4 ONNX export and makes it static.

    Two components: ``embed_tokens`` (input_ids -> inputs_embeds +
    per_layer_inputs) and ``decoder`` (the autoregressive decoder, KV-cache
    I/O). Always static -- this exporter has no dynamic mode (that's
    ``Gemma4ModelExporter`` in ``export.py``, which this leaves untouched).
    """

    def __init__(
        self,
        *,
        hf_repo: str = DEFAULT_HF_REPO_INT4,
        template_repo: str = DEFAULT_TEMPLATE_REPO,
        max_kv_len: int = 256,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
    ):
        self._hf_repo = hf_repo
        self._template_repo = template_repo
        self._max_kv_len = max_kv_len
        # Resolved to absolute up front: `make_static`/`validate_onnx`
        # temporarily `chdir` into `self._onnx_dir` (see their docstrings --
        # needed because onnx's/onnxruntime's external-data loaders resolve
        # relative locations against the process cwd), so every path derived
        # from these must be cwd-independent, not just "happens to be
        # absolute because tests always pass an absolute `models_dir`".
        self._onnx_source_dir = (
            Path(onnx_source_dir).resolve() if onnx_source_dir is not None else None
        )

        super().__init__(
            "fp32",
            True,  # static_models: this exporter only ever produces a static graph
            {},
            Path(models_dir).resolve() / "gemma4-e2b-int4",
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            # MatMulNBits/GatherBlockQuantized/custom attention ops would
            # break the ORT bert optimizer; skip it (mirrors liquid).
            opt_configs={},
            # Lets components be processed/tested one at a time -- each
            # component's ~GB-scale external weight data is only touched
            # while *that* component is being processed (see
            # `apply_post_static_patches`'s JIT data-file copy), so skipping
            # one here roughly halves peak memory during development.
            skip_export=skip_export,
        )

    def _setup_dirs(self) -> list[Path]:
        if self._onnx_source_dir is not None:
            onnx_dir = Path(self._onnx_source_dir)
        else:
            onnx_dir = self._models_dir / "source" / "onnx"
        if not (onnx_dir / _DECODER_FILENAME).exists() or not (onnx_dir / _EMBED_FILENAME).exists():
            self._download_from_hf(onnx_dir)

        export_dir = self._models_dir / "export" / "onnx" / "static"
        convert_dir = self._models_dir / "export" / "onnx" / "bf16" / "static"
        torq_dir = self._models_dir / "export" / "torq" / "static"
        return onnx_dir, export_dir, convert_dir, torq_dir

    def _download_from_hf(self, target_dir: Path):
        from huggingface_hub import hf_hub_download

        target_dir.mkdir(parents=True, exist_ok=True)
        required = [
            f"onnx/{_DECODER_FILENAME}", f"onnx/{_DECODER_FILENAME}_data",
            f"onnx/{_EMBED_FILENAME}", f"onnx/{_EMBED_FILENAME}_data",
        ]
        optional = ["tokenizer.json", "tokenizer_config.json", "config.json"]

        self._logger.info("Downloading source ONNX from '%s'...", self._hf_repo)
        for filename in required:
            p = hf_hub_download(self._hf_repo, filename)
            dest = target_dir / Path(filename).name
            if not dest.exists():
                shutil.copy(p, dest)
        for filename in optional:
            try:
                p = hf_hub_download(self._hf_repo, filename)
                dest = target_dir / filename
                if not dest.exists():
                    shutil.copy(p, dest)
            except Exception as e:
                self._logger.debug("  optional file '%s' not in %s: %s", filename, self._hf_repo, e)

        # chat_template.jinja isn't in the int4 repo -- pull it from the
        # template repo (same underlying Gemma-4 checkpoint / chat format).
        try:
            p = hf_hub_download(self._template_repo, "chat_template.jinja")
            dest = target_dir / "chat_template.jinja"
            if not dest.exists():
                shutil.copy(p, dest)
        except Exception as e:
            self._logger.warning(
                "chat_template.jinja not found in template repo '%s': %s", self._template_repo, e
            )
        self._logger.info("Download complete: %s", target_dir)

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        decoder_path = self._onnx_dir / _DECODER_FILENAME
        embed_path = self._onnx_dir / _EMBED_FILENAME
        if not decoder_path.exists() or not embed_path.exists():
            raise FileNotFoundError(
                f"Expected '{_DECODER_FILENAME}' and '{_EMBED_FILENAME}' @ '{self._onnx_dir}'"
            )
        # Do NOT eagerly materialize external data (`load_external_data=True`)
        # -- the decoder+embed weights are ~3.6GB combined, and none of this
        # exporter's graph edits read the big MatMulNBits/GatherBlockQuantized
        # weight blobs' *values* (only their declared shapes, which are always
        # available from the TensorProto header regardless of load state).
        # Only a handful of small constants (the RoPE cos/sin caches) are
        # actually read by `ReplaceRotaryEmbedding` -- preload just those.
        # Components in `self._skip_export` aren't loaded at all -- lets
        # `--skip-export` (or a test's `skip_export=[...]`) meaningfully
        # reduce peak memory, not just skip the final save.
        components: dict[str, onnx.ModelProto] = {}
        if "embed_tokens" not in self._skip_export:
            embed_tokens = onnx.load(embed_path, load_external_data=False)
            _preload_small_external_initializers(embed_tokens, self._onnx_dir)
            components["embed_tokens"] = embed_tokens
        if "decoder" not in self._skip_export:
            decoder = onnx.load(decoder_path, load_external_data=False)
            _preload_small_external_initializers(decoder, self._onnx_dir)
            components["decoder"] = decoder
        return components

    @staticmethod
    def sanitize_onnx_names(model: onnx.ModelProto) -> onnx.ModelProto:
        """Disambiguate initializer names that collide after the base
        sanitizer strips illegal-for-MLIR characters, then defer to it.

        Confirmed collision in the decoder source graph: `/model/constants/
        INT64/[1]` and `/model/constants/INT64/[-1]` both sanitize to
        `/model/constants/INT64/_1` (the base sanitizer replaces any run of
        illegal chars with a single `_`, so `[1]`'s `[` `]` and `[-1]`'s `[`
        `-` `]` both collapse the same way) -- onnx's checker rejects the
        result outright (duplicate initializer name). Liquid hit the exact
        same collision class for the same reason; unlike its content-aware
        fix, this one just appends a numeric suffix on collision, since
        correctness (not a descriptive name) is all that matters here.
        """
        import re

        illegal = re.compile(r"[^a-zA-Z0-9_./]+")

        def _clean(name: str) -> str:
            return illegal.sub("_", name).strip("_")

        seen: dict[str, int] = {}
        rename_map: dict[str, str] = {}
        for init in model.graph.initializer:
            clean = _clean(init.name)
            n = seen.get(clean, 0)
            seen[clean] = n + 1
            if n > 0:
                rename_map[init.name] = f"{init.name}_dup{n}"

        if rename_map:
            for init in model.graph.initializer:
                if init.name in rename_map:
                    init.name = rename_map[init.name]
            for node in model.graph.node:
                for i, inp in enumerate(node.input):
                    if inp in rename_map:
                        node.input[i] = rename_map[inp]

        return OnnxModelExporterBase.sanitize_onnx_names(model)

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = True) -> onnx.ModelProto:
        # Deliberately skips `onnx.shape_inference.infer_shapes()` and
        # `onnx.checker.check_model()` entirely here, unlike every other
        # exporter in this repo -- both were measured to cost several GB of
        # extra peak RSS on the decoder component (1287 nodes, ~1.9GB of
        # materialized weight data by the time the base class calls this,
        # `onnx.load()` having already run with default `load_external_data
        # =True`), independent of `full_check`/`data_prop` settings, on top
        # of a ~2GB `onnx.load()` that already happened. That cost buys
        # nothing here: this exporter calls `check_model` only on models
        # that already went through `make_static()`'s full
        # `SymbolicShapeInference` resolution + `propagate_static_shapes`,
        # so shapes are already complete and correct (re-running plain
        # `onnx.shape_inference` can't even verify MatMulNBits/
        # GatherBlockQuantized nodes -- it has no schema for them -- and an
        # earlier draft that cleared+re-ran it silently discarded correct
        # shapes for everything downstream of one, see STATIC_EXPORT_PLAN.md).
        # The actual bar for this milestone (`check_dynamic_shapes` returning
        # empty) is independently and cheaply verified by the base class
        # right after this call, and by `validate_onnx()`.
        if model.ir_version > 10:
            self._logger.warning(
                "Model IR version is %d (>10), may be unsupported by onnxruntime", model.ir_version
            )
        return model

    # -- make_static: structural graph surgery (fix shapes, decompose custom
    # attention/rotary ops, replace dynamic KV-cache Concats). Runs once,
    # before the per-component sanitize/check/save loop. --

    def make_static(self):
        # Respect `self._skip_export`: the base class's `export_onnx()` loop
        # already skips saving/checking/patching skipped components, but
        # `make_static()` itself runs once, unconditionally, *before* that
        # loop -- without this check every component's (expensive) graph
        # surgery would run regardless, defeating the point of skipping one
        # for isolated testing.
        if "embed_tokens" not in self._skip_export:
            self._logger.info("(embed_tokens) Making graph static...")
            self._components["embed_tokens"] = self._make_embed_tokens_static(
                self._components["embed_tokens"]
            )
        if "decoder" not in self._skip_export:
            self._logger.info("(decoder) Making graph static...")
            self._components["decoder"] = self._make_decoder_static(
                self._components["decoder"]
            )

    def _make_embed_tokens_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = gs.import_onnx(model)
        editor = Gemma4OnnxGraphEditor(graph, self._onnx_export_dtype)
        editor.fix_io(self._max_kv_len)  # only batch_size/sequence_length appear here
        static_model = editor.to_onnx(
            check_type=False, strict_mode=False, data_prop=False, override_ir=model.ir_version
        )
        # Resolve custom-op (GatherBlockQuantized) shapes here, on the graph
        # continuously held in memory since `gs.import_onnx` -- NOT after a
        # save/reload round-trip. Measured empirically to matter a lot:
        # running this same call after `onnx.save()` + `onnx.load()` cost
        # several GB more peak RSS than running it directly on the in-memory
        # `editor`/`gs` state, for reasons not fully root-caused (suspected
        # onnx_graphsurgeon re-parsing overhead) -- see STATIC_EXPORT_PLAN.md.
        static_model = self._resolve_custom_op_shapes(static_model)
        return _fix_gather_block_quantized_shapes(static_model)

    def _make_decoder_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = gs.import_onnx(model)
        editor = Gemma4OnnxGraphEditor(graph, self._onnx_export_dtype)

        editor.fix_io(self._max_kv_len)
        editor.normalize_kv_concat_axis()
        editor.replace_simplified_layer_norm()
        editor.replace_rotary_embedding()
        editor.replace_group_query_attention(
            num_heads=_GQA_NUM_HEADS, kv_num_heads=_GQA_KV_NUM_HEADS, head_dim=_GQA_HEAD_DIM
        )
        editor.fold_num_logits_to_keep(1)

        # `cur_len` for the static KV-cache/mask edits: derived from the
        # real `position_ids` graph input (shape [1,1] post fix_io), not a
        # synthesized one -- Gemma-4 already has this input, unlike models
        # that only expose a `Shape(past_key_values)->Gather` chain.
        position_ids = next(
            (t for t in editor.graph.inputs if t.name == "position_ids"), None
        )
        if position_ids is None:
            raise RuntimeError("Expected a 'position_ids' graph input on the decoder component")
        cur_len = editor.graph.layer(
            name="current_len_to_1d", op="Squeeze",
            inputs=[position_ids, [0]],
            outputs=[gs.Variable("current_len_1d", dtype=np.int64, shape=[1])],
        )[0]
        cur_len_scalar = editor.graph.layer(
            name="current_len_to_scalar", op="Squeeze",
            inputs=[cur_len, [0]],
            outputs=[gs.Variable("current_len_scalar", dtype=np.int64, shape=[])],
        )[0]

        editor.replace_dynamic_kv_cache(cur_len_scalar, self._max_kv_len)
        editor.mask_future_attn_scores(cur_len_scalar, self._max_kv_len)

        static_model = editor.to_onnx(
            check_type=False, strict_mode=False, data_prop=False, override_ir=model.ir_version
        )
        # Shape-fold (resolves the native mask-construction Shape/Range
        # chains) + custom-op (MatMulNBits) shape resolution, both run here
        # on the graph continuously held in memory -- see the comment in
        # `_make_embed_tokens_static` for why NOT to defer this to a
        # save/reload round-trip in `apply_post_static_patches`.
        static_model = propagate_static_shapes(static_model)
        return self._resolve_custom_op_shapes(static_model)

    # -- apply_post_static_patches: no-op. All structural/shape work already
    # happened in `make_static()`, on the in-memory graph; the external-data
    # file copy happens up front in `export_onnx()`, before any of that
    # heavy processing (see the comment there for why: doing it *after*,
    # under memory pressure from the graph work, was observed to produce a
    # silently-truncated copy at least once -- moving it earlier, when the
    # process is still light, sidesteps that outright rather than explaining
    # it). --

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        pass

    def _resolve_custom_op_shapes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Resolve MatMulNBits/GatherBlockQuantized output shapes via ORT's
        SymbolicShapeInference (registered shape functions for both ops).
        Known to crash on the raw decoder graph before shape-folding (see
        STATIC_EXPORT_PLAN.md); running this after `propagate_static_shapes`
        has already collapsed the mask-construction Shape/Range chains into
        constants is expected to avoid that failure mode.

        Runs with the process cwd temporarily switched to `self._onnx_dir`.
        Most of this model's weight initializers are still external/
        unmaterialized at this point (by design -- see
        `_preload_small_external_initializers`), and `SymbolicShapeInference`
        was observed, at least once, to internally touch one of them (a
        LayerNorm weight, not one of the RoPE caches this exporter
        explicitly preloads) for reasons not fully root-caused -- possibly a
        constant-folding optimization it performs internally. Onnx's
        external-data loader resolves a tensor's relative `location` string
        against the process cwd when no explicit `base_dir` is threaded
        through (the same underlying bug as `onnx_graphsurgeon`'s lazy
        loader, see that docstring) -- so unlike the targeted by-name
        preload (which only covers tensors *this exporter's own edits*
        read), this chdir is the general fix: it makes the cwd correct for
        *any* external tensor SymbolicShapeInference might touch, whichever
        one that turns out to be.
        """
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        except ImportError:
            self._logger.warning("onnxruntime.tools.symbolic_shape_infer unavailable; skipping")
            return model
        prev_cwd = os.getcwd()
        os.chdir(self._onnx_dir)
        try:
            resolved = SymbolicShapeInference.infer_shapes(
                model, auto_merge=True, guess_output_rank=True, verbose=0
            )
        except Exception as e:
            self._logger.error(
                "SymbolicShapeInference failed (%s) -- custom-op (MatMulNBits/"
                "GatherBlockQuantized) output shapes were NOT resolved; "
                "`check_dynamic_shapes` will likely fail downstream", e,
            )
            return model
        finally:
            os.chdir(prev_cwd)
        return resolved

    def export_onnx(self, validate: bool = True):
        # Every saved component still references the *original* external-data
        # filename (e.g. "decoder_model_merged_q4.onnx_data") -- none of this
        # exporter's edits touch the untouched MatMulNBits/GatherBlockQuantized
        # weight initializers, so that reference is never rewritten. Copy the
        # original data file(s) into the export dir now, up front, before any
        # of `make_static()`'s heavy graph processing runs -- a large
        # `shutil.copy2` competing with that processing's memory usage was
        # observed to produce a silently-truncated copy at least once.
        # Only copies files for components that will actually be exported.
        for comp, fname in (("embed_tokens", _EMBED_FILENAME), ("decoder", _DECODER_FILENAME)):
            if comp in self._skip_export:
                continue
            src = self._onnx_dir / f"{fname}_data"
            dest = self._export_dir / f"{fname}_data"
            if src.exists() and not dest.exists():
                self._logger.info("Copying external data '%s' -> '%s' ...", src, dest)
                shutil.copy2(src, dest)
                if dest.stat().st_size != src.stat().st_size:
                    raise RuntimeError(
                        f"Copied external data '{dest}' size {dest.stat().st_size} != "
                        f"source size {src.stat().st_size} -- truncated copy"
                    )

        super().export_onnx(validate=False)
        for fname in ("tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja"):
            src = self._onnx_dir / fname
            if src.exists():
                shutil.copy2(src, self._export_dir / fname)
        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 3):
        """Schema-level validation for this milestone (no PyTorch reference
        exists for this int4 graph): onnx checker + the static-shape gate +
        a single-step onnxruntime smoke test per component, plus a
        multi-token teacher-forced comparison against the original dynamic
        ONNX (see `_validate_generation_matches_dynamic`).

        Uses `full_check=False` deliberately: `full_check=True` forces the
        checker down a path that, on a component with a GB-scale external
        weight tensor materialized, was measured to need ~9GB peak RSS just
        for the ~13-node embed_tokens component -- not tractable on a
        dev machine, and not needed for this milestone's actual bar
        (`check_dynamic_shapes` returning empty). Loads with
        `load_external_data=False` for the same reason -- the checker/shape
        checks here don't need tensor values.

        Runs entirely with the process cwd switched to `self._onnx_dir`:
        every exported component's saved weight initializers are still
        external, referencing filenames relative to that directory (only
        copied there, not into the export dir -- see `export_onnx`), and
        onnx's/onnxruntime's external-data loaders resolve relative
        locations against the process cwd when not given an explicit
        `base_dir` (the same underlying issue documented on
        `_resolve_custom_op_shapes` and `_preload_small_external_initializers`,
        just hit here by the checker/`onnx.load`/`InferenceSession` instead).
        """
        prev_cwd = os.getcwd()
        os.chdir(self._onnx_dir)
        try:
            self._validate_onnx_impl(n_iters)
        finally:
            os.chdir(prev_cwd)

    def _validate_onnx_impl(self, n_iters: int):
        import onnxruntime as ort

        for comp, path in self._export_paths.items():
            model = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(model, full_check=False)
            dynamic = check_dynamic_shapes(model)
            if dynamic:
                raise ValueError(f"(validate) '{comp}' still has dynamic shapes: {dynamic}")
            self._logger.info("(validate) '%s': checker + static-shape gate passed", comp)
            del model

        if "embed_tokens" not in self._export_paths or "decoder" not in self._export_paths:
            self._logger.info(
                "(validate) Skipping the combined one-step smoke test -- both components "
                "must be exported (got: %s)", list(self._export_paths)
            )
            return

        session_kwargs = dict(providers=["CPUExecutionProvider"])
        embed_session = ort.InferenceSession(str(self._export_paths["embed_tokens"]), **session_kwargs)
        decoder_session = ort.InferenceSession(str(self._export_paths["decoder"]), **session_kwargs)

        input_ids = np.array([[2]], dtype=np.int64)  # <bos>
        inputs_embeds, per_layer_inputs = embed_session.run(
            None, {"input_ids": input_ids}
        )
        # Build feeds from whatever inputs the static graph actually has --
        # `attention_mask`/`num_logits_to_keep` are gone (dead-input-eliminated
        # after `replace_dynamic_kv_cache`/`mask_future_attn_scores` replaced
        # their role with a `position_ids`-derived `cur_len`, and
        # `fold_num_logits_to_keep` folded the other to a constant), unlike
        # the pre-edit graph -- don't assume the original I/O signature.
        available = {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}
        feeds = {}
        for i in decoder_session.get_inputs():
            if i.name in available:
                feeds[i.name] = available[i.name]
            elif i.name == "position_ids":
                feeds[i.name] = np.array([[0]], dtype=np.int64)
            elif i.name == "attention_mask":
                feeds[i.name] = np.ones((1, self._max_kv_len), dtype=np.int64)
            elif i.name.startswith("past_key_values"):
                feeds[i.name] = np.zeros(
                    [1 if isinstance(d, str) or d is None else d for d in i.shape], dtype=np.float32
                )
            else:
                raise RuntimeError(f"(validate) Unexpected decoder input '{i.name}' -- update the smoke test")
        logits, *_ = decoder_session.run(None, feeds)
        self._logger.info(
            "(validate) One-step onnxruntime smoke test passed; logits shape=%s", logits.shape
        )

        self._validate_generation_matches_dynamic(n_iters, embed_session)

    def _validate_generation_matches_dynamic(self, n_iters: int, embed_session):
        """Multi-token, teacher-forced comparison: does the static decoder's
        per-step greedy (argmax) prediction match the original, unmodified
        dynamic `decoder_model_merged_q4.onnx` (naturally-growing KV cache
        via Concat) given the *same* input token history at every step?

        This is the strongest correctness check available for this
        milestone -- there's no PyTorch reference for this int4 graph, but
        the original dynamic ONNX (the exact thing this exporter transforms
        into a static graph) is a legitimate one, and both run through the
        same onnxruntime. "Teacher-forced" means each step feeds the
        *reference* decoder's own greedy pick to both decoders, regardless
        of what the static decoder predicted -- so a mismatch at step N is
        attributable to that one step's computation, not to cascading
        divergence from an earlier mismatch.

        Skips gracefully (warns, doesn't fail) if `tokenizer.json` or the
        original dynamic decoder aren't available locally -- both are only
        guaranteed present after a real (non-`--onnx-source-dir`) download.
        """
        tokenizer_path = self._onnx_dir / "tokenizer.json"
        decoder_ref_path = self._onnx_dir / _DECODER_FILENAME
        if not tokenizer_path.exists() or not decoder_ref_path.exists():
            self._logger.warning(
                "(validate) 'tokenizer.json' and/or the original '%s' not found @ '%s' -- "
                "skipping the generation-vs-dynamic-reference check",
                _DECODER_FILENAME, self._onnx_dir,
            )
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        static_decoder = ort.InferenceSession(
            str(self._export_paths["decoder"]), providers=["CPUExecutionProvider"]
        )
        ref_decoder = ort.InferenceSession(str(decoder_ref_path), providers=["CPUExecutionProvider"])

        ref_head_dims = {
            i.name: int(i.shape[-1]) for i in ref_decoder.get_inputs() if i.name.startswith("past_key_values")
        }
        static_out_names = [o.name for o in static_decoder.get_outputs()]
        ref_out_names = [o.name for o in ref_decoder.get_outputs()]

        def _present_to_past(name: str) -> str:
            return "past_key_values" + name[len("present"):]

        prompts = ["The capital of France is", "1, 2, 3, 4,", "Hello, my name is"][: max(1, n_iters)]
        max_new_tokens = 6
        n_compared = 0
        n_mismatches = 0

        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt).ids
            static_kv = {
                i.name: np.zeros(
                    [1 if isinstance(d, str) or d is None else d for d in i.shape], dtype=np.float32
                )
                for i in static_decoder.get_inputs()
                if i.name.startswith("past_key_values")
            }
            ref_kv = {
                name: np.zeros((1, 1, 0, head_dim), dtype=np.float32)
                for name, head_dim in ref_head_dims.items()
            }

            gen_static, gen_ref = [], []
            ref_next = static_next = None
            mismatch_at = None
            n_steps = min(len(prompt_ids) + max_new_tokens, self._max_kv_len)
            for step in range(n_steps):
                if step < len(prompt_ids):
                    tok = prompt_ids[step]
                else:
                    tok = ref_next  # teacher-force with the reference's own greedy pick
                    gen_ref.append(ref_next)
                    gen_static.append(static_next)

                input_ids = np.array([[tok]], dtype=np.int64)
                inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})

                static_feeds = {
                    "inputs_embeds": inputs_embeds,
                    "position_ids": np.array([[step]], dtype=np.int64),
                    "per_layer_inputs": per_layer_inputs,
                    **static_kv,
                }
                static_logits, *static_present = static_decoder.run(None, static_feeds)
                for out_name, val in zip(static_out_names[1:], static_present):
                    static_kv[_present_to_past(out_name)] = val
                static_next = int(static_logits[0, -1].argmax())

                ref_feeds = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": np.ones((1, step + 1), dtype=np.int64),
                    "position_ids": np.array([[step]], dtype=np.int64),
                    "num_logits_to_keep": np.array(1, dtype=np.int64),
                    "per_layer_inputs": per_layer_inputs,
                    **ref_kv,
                }
                ref_logits, *ref_present = ref_decoder.run(None, ref_feeds)
                for out_name, val in zip(ref_out_names[1:], ref_present):
                    ref_kv[_present_to_past(out_name)] = val
                ref_next = int(ref_logits[0, -1].argmax())

                n_compared += 1
                if static_next != ref_next:
                    n_mismatches += 1
                    if mismatch_at is None:
                        mismatch_at = step

            if mismatch_at is None:
                self._logger.info("(validate) generation check [%r]: match", prompt)
            else:
                self._logger.warning(
                    "(validate) generation check [%r]: first mismatch at step %d "
                    "(static gen=%s vs ref gen=%s)",
                    prompt, mismatch_at, gen_static, gen_ref,
                )

        if n_mismatches:
            raise ValueError(
                f"(validate) static decoder's per-step greedy prediction diverged from the "
                f"original dynamic decoder at {n_mismatches}/{n_compared} compared step(s) "
                "across all prompts -- see warnings above for details"
            )
        self._logger.info(
            "(validate) generation-vs-dynamic-reference check passed: %d/%d steps matched "
            "across %d prompt(s)", n_compared, n_compared, len(prompts),
        )


def export_gemma4_int4_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = Gemma4Int4ModelExporter(
        hf_repo=args.hf_repo,
        template_repo=args.template_repo,
        max_kv_len=args.max_kv_len,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
    )
    exporter.export_onnx(validate=not args.skip_validation)


def main():
    parser = argparse.ArgumentParser(
        description="Export Gemma4-E2B (int4 quantized source) to a static ONNX graph"
    )
    add_gemma4_int4_export_args(parser)
    export_gemma4_int4_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
