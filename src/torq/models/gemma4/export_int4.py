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
import json
import os
import shutil
from pathlib import Path
from typing import Final

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

from . import add_gemma4_int4_export_args
from ._graph import Gemma4OnnxGraphEditor
from ...graph_edit.edits import ConstantBroadcastPolicy
from ...model_export.onnx import OnnxModelExporterBase
from ...tools.convert_dtype.onnx import convert_model as _convert_dtype
from ...utils.logging import configure_logging
from ...utils.onnx import check_dynamic_shapes, get_model_opset, propagate_static_shapes


def _patch_gs_bf16_converter():
    """onnx_graphsurgeon's bf16 exporter relies on
    `onnx.helper.float32_to_bfloat16`, removed in onnx>=1.20 -- without this,
    exporting a graph containing a bf16 `gs.Constant` (as
    `DecomposeMatMulNBits`'s dequant scale is) asserts out. Same patch used
    in `src/torq/models/liquid/export.py`; needed here because
    `_decompose_decoder_matmul_nbits` calls `editor.to_onnx()` on a graph
    with bf16 constants for the first time in this file.
    """
    try:
        from onnx_graphsurgeon.exporters import onnx_exporter as _gs_export
    except Exception:
        return
    if _gs_export._NUMPY_ARRAY_CONVERTERS:
        return

    def _f32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return arr.astype(ml_dtypes.bfloat16).view(np.uint16)

    _gs_export._NUMPY_ARRAY_CONVERTERS = {
        onnx.TensorProto.BFLOAT16: _gs_export.NumpyArrayConverter(
            np.uint16, _f32_to_bf16_uint16
        ),
    }


_patch_gs_bf16_converter()

DEFAULT_HF_REPO_INT4: Final[str] = "tss-deposium/gemma-4-E2B-text-only-onnx-int4"
# The int4 repo doesn't ship a chat template; pull it from a sibling ONNX
# export of the same underlying checkpoint that does.
DEFAULT_TEMPLATE_REPO: Final[str] = "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX"

_DECODER_FILENAME: Final[str] = "decoder_model_merged_q4.onnx"
_EMBED_FILENAME: Final[str] = "embed_tokens_q4.onnx"

# Fixed node names for the two `com.microsoft.GatherBlockQuantized` embedding
# lookups in `embed_tokens_q4.onnx` -- verified against the actual source
# graph (see STATIC_EXPORT_PLAN.md). `_make_embed_tokens_static` extracts
# each one individually by name; a single unscoped
# `extract_gather_block_quantized_lut` call would apply the same
# `save_to`/`inp_name` to BOTH nodes, since `OnnxGraphEditor.apply_edit` runs
# one edit instance across every matching node in the graph.
_EMBED_TOKENS_GATHER_NODE: Final[str] = "/model/embed_tokens/Gather_Quant"
_EMBED_TOKENS_PER_LAYER_GATHER_NODE: Final[str] = "/model/embed_tokens_per_layer/Gather_Quant"
# Directories (not single files): each holds `data_quant.npy`/`scales.npy`/
# `zero_points.npy` + `meta.json`, so the big arrays can be memory-mapped.
# A `.npz` cannot be mmap'd (zip container), and these tables are read a
# row at a time -- see `_PackedEmbeddingLUT`.
_TOKEN_EMBEDDINGS_FILENAME: Final[str] = "token_embeddings"
_PER_LAYER_EMBEDDINGS_FILENAME: Final[str] = "per_layer_embeddings"
# Directory holding one `<variant>.npy` per distinct RoPE cos/sin cache
# (`cos_full_local`/`sin_full_local`/`cos_full_global`/`sin_full_global`),
# already gathered/duplicated to full head_dim by `ReplaceRotaryEmbedding`
# (see `custom_ops.py`). Memory-mapped by `_RopeCacheLUT`, whose host-side
# lookups replace the in-graph Gather this exporter no longer emits.
_ROPE_CACHES_FILENAME: Final[str] = "rope_caches"

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


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """[..., n_bytes] uint8, 2 packed 4-bit values/byte (low nibble = even
    index, high nibble = odd index) -> [..., 2*n_bytes] uint8 unpacked.
    Convention verified empirically against a real MatMulNBits node's
    onnxruntime output (see gemma4's op_repros/decompose_matmul_nbits.py);
    the same convention holds for GatherBlockQuantized's packed weight/
    zero_point (confirmed via the bit-exact validation described in
    STATIC_EXPORT_PLAN.md).
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


class _PackedEmbeddingLUT:
    """Holds one `ExtractGatherBlockQuantizedLUT`-produced table (packed
    int4 weights + scales + zero_points + dequant metadata) and dequantizes
    single rows on demand.

    **Memory-mapped, not resident.** Only 1-2 rows are ever read per
    inference step, so keeping the whole table in RAM is pure waste.
    Measured on gemma4 before this was mmap'd: `token_embeddings` cost
    ~314MB and `per_layer_embeddings` ~1435MB of *resident* RSS, ~1.7GB
    combined for data that is touched a few KB at a time. With
    `mmap_mode="r"` the pages are file-backed and evictable, so resident
    cost is roughly the handful of pages actually indexed.

    This is why the on-disk format is a **directory of individual `.npy`
    files** rather than a single `.npz`: numpy cannot mmap a `.npz` at all
    (it is a zip container, so `np.load(..., mmap_mode=...)` silently has no
    effect on the members). Legacy `.npz` files are still accepted for
    backwards compatibility, but load fully resident -- re-export to get the
    mmap benefit.
    """

    _ARRAYS = ("data_quant", "scales", "zero_points")

    def __init__(self, path: str | os.PathLike):
        path = Path(path)
        if path.is_dir():
            self._data_quant, self._scales, self._zero_points = (
                np.load(path / f"{n}.npy", mmap_mode="r") for n in self._ARRAYS
            )
            meta = json.loads((path / "meta.json").read_text())
            self._bits = int(meta["bits"])
            self._block_size = int(meta["block_size"])
            self._per_row_shape = tuple(int(d) for d in meta["per_row_shape"])
        else:
            # Legacy single-`.npz` layout: cannot be memory-mapped.
            with np.load(path) as npz:
                self._data_quant = npz["data_quant"]
                self._scales = npz["scales"]
                self._zero_points = npz["zero_points"]
                self._bits = int(npz["bits"])
                self._block_size = int(npz["block_size"])
                self._per_row_shape = tuple(int(d) for d in npz["per_row_shape"])

    def dequant_row(self, token_id: int) -> np.ndarray:
        """Same block-dequant math as the original `GatherBlockQuantized`
        op (unpack nibbles -> subtract zero_point -> multiply by scale),
        computed for a single row instead of the whole table -- cheap
        (microseconds), since `block_size`-sized chunks are tiny.
        """
        k = self._data_quant.shape[-1] * (8 // self._bits)
        n_blocks = k // self._block_size
        w = _unpack_nibbles(self._data_quant[token_id : token_id + 1])[0, :k].astype(np.float32)
        zp = _unpack_nibbles(self._zero_points[token_id : token_id + 1])[0, :n_blocks].astype(np.float32)
        scale = self._scales[token_id].astype(np.float32)
        dq = (w.reshape(n_blocks, self._block_size) - zp[:, None]) * scale[:, None]
        dq = dq.reshape(k)
        if self._per_row_shape:
            dq = dq.reshape(self._per_row_shape)
        return dq


def _lookup_embeddings(
    token_lut: "_PackedEmbeddingLUT", per_layer_lut: "_PackedEmbeddingLUT", token_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Token ID -> (inputs_embeds, per_layer_inputs), dequantized on the fly
    from the extracted packed `.npz` LUTs (see `_make_embed_tokens_static`)
    instead of running `embed_tokens.onnx` -- post-extraction that graph is
    a 0-node passthrough with no computation left to run for this. Shapes
    match what the decoder's `inputs_embeds`/`per_layer_inputs` inputs
    expect for a single-token step: `[1, 1, hidden]` / `[1, 1, 35, 256]`.
    """
    inputs_embeds = token_lut.dequant_row(token_id)[None, None, :]
    per_layer_inputs = per_layer_lut.dequant_row(token_id)[None, None, ...]
    return inputs_embeds, per_layer_inputs


class _RopeCacheLUT:
    """Holds the `ReplaceRotaryEmbedding`-produced RoPE caches (one array per
    distinct variant this component uses, e.g. `cos_full_local`/
    `sin_full_local`, each already gathered/duplicated to full head_dim) and
    returns the row for a given position, keyed by the same names the
    decoder's graph inputs expect.

    **Memory-mapped, not resident** -- exactly one row per variant is read
    per inference step, but the tables are sized for the source model's full
    131072-token context (~768MB combined for gemma4, measured resident
    before this was mmap'd). Stored as a **directory of `.npy` files** for
    the same reason as `_PackedEmbeddingLUT`: `.npz` cannot be mmap'd.
    Legacy `.npz` files still load (fully resident).

    A no-op (empty `variants()`) if the path doesn't exist, so callers don't
    need to special-case components that have no `RotaryEmbedding` nodes at
    all (e.g. post-`SplitLMHead` components).
    """

    def __init__(self, path: str | os.PathLike):
        self._tables: dict[str, np.ndarray] = {}
        path = Path(path)
        if path.is_dir():
            self._tables = {
                p.stem: np.load(p, mmap_mode="r") for p in sorted(path.glob("*.npy"))
            }
        elif path.exists():
            # Legacy single-`.npz` layout: cannot be memory-mapped.
            with np.load(path) as npz:
                self._tables = {name: npz[name] for name in npz.files}

    def variants(self) -> frozenset[str]:
        return frozenset(self._tables)

    def lookup(self, position_id: int) -> dict[str, np.ndarray]:
        """position_id -> {variant_name: row}, each shaped `[1, 1, head_dim]`
        -- matching the graph input `ReplaceRotaryEmbedding` creates.
        """
        return {name: table[position_id][None, None, :] for name, table in self._tables.items()}


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


def decompose_matmul_nbits_in_file(model_path: str | os.PathLike) -> None:
    """Decompose every `MatMulNBits` node in a single ONNX file into
    standard ops (`DequantizeLinear -> MatMul`, see `DecomposeMatMulNBits`)
    in place, so the saved model has no custom `com.microsoft` ops left at
    all. Standalone (no `Gemma4Int4ModelExporter` instance needed) so it can
    run per split component from `split_decoder.py`, not just from this
    module -- decomposing the *whole* ~3713-node, ~242-MatMulNBits decoder
    in one pass needs to hold all ~3.7GB of newly-unpacked int8 weight data
    simultaneously, which OOM'd repeatedly even under an 8G cap on this dev
    machine; running it per (much smaller, ~500MB-600MB) component instead
    keeps peak memory bounded.

    Runs under `os.chdir(model_path.parent)`: loaded with
    `load_external_data=False`, so `DecomposeMatMulNBits` lazily
    materializes each node's packed weight/scale/zero_point on demand via
    onnx_graphsurgeon's lazy loader, which resolves external-data locations
    against the process cwd (the same bug documented elsewhere in this
    file).

    Before saving, every remaining external tensor is explicitly
    materialized via `onnx.load_external_data_for_model` (mirrors
    `split_decoder.py`'s `split_decoder()`) -- required, not optional:
    tensors `DecomposeMatMulNBits` never touches (e.g. plain LayerNorm
    weights) keep carrying their *original* external-data offsets, which
    onnx's own `save_as_external_data=True` leaves untouched for any tensor
    already marked external (it assumes an existing external reference is
    already valid and doesn't rewrite it). Since the original data file is
    about to be deleted and replaced by a same-named file with a totally
    different layout, those stale offsets would silently point at wrong (or
    out-of-range) bytes in the new file -- confirmed: this exact bug
    produced a `TensorProto ... should contain one and only one value
    field` checker error downstream, traced to an offset exceeding the new
    file's actual size. Materializing first sidesteps this by ensuring
    every tensor's data is fresh, in-memory bytes at save time, not a
    reference into a soon-to-be-replaced file.

    Saved with `save_as_external_data=True`: decomposed weights are
    unpacked from int4 nibbles to one-byte-per-value int8, doubling their
    size (0.5 -> 1 byte/value). The original external data file is removed
    only after the new model's data has been fully materialized in memory.
    """
    model_path = Path(model_path)
    prev_cwd = os.getcwd()
    os.chdir(model_path.parent)
    try:
        model = onnx.load(model_path.name, load_external_data=False)
        graph = gs.import_onnx(model)
        editor = Gemma4OnnxGraphEditor(graph, "fp32")
        editor.decompose_matmul_nbits()
        new_model = editor.to_onnx(
            check_type=False, strict_mode=False, data_prop=False, override_ir=model.ir_version
        )
        onnx.load_external_data_for_model(new_model, ".")
        old_data = Path(f"{model_path.name}_data")
        old_data.unlink(missing_ok=True)
        onnx.save(
            new_model, model_path.name,
            save_as_external_data=True, location=f"{model_path.name}_data", size_threshold=1024,
        )
    finally:
        os.chdir(prev_cwd)


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
        editor.fix_io(self._max_kv_len)  # only batch_size/sequence_length appear here;
        # do this BEFORE extraction so the traced graph outputs (and thus the
        # new passthrough inputs replacing them) already carry static shapes.

        # Both embedding tables are `com.microsoft.GatherBlockQuantized`
        # lookups over a constant int4 table. Mirroring gemma3's
        # `extract_token_embeddings` (which lifts its plain, unquantized
        # embedding table out of the graph entirely so it runs host-side,
        # not on the NPU): fully dequantize each table to a standalone
        # `.npy` and replace its graph output with a plain graph input of
        # the same name. The resulting component ends up with zero compute
        # nodes -- nothing here needs torq-compile at all.
        #
        # Runs under `os.chdir(self._onnx_dir)`: unlike the RoPE caches
        # (preloaded by name in `_load_onnx`), these two tables' `data_quant`/
        # `scales`/`zero_points` initializers are deliberately left external/
        # unmaterialized until now (~1.76GB combined) -- extraction lazily
        # accesses their `.values` for the first time here, and
        # onnx_graphsurgeon's lazy loader resolves external-data locations
        # against the process cwd, not the model's directory (same bug
        # documented on `_preload_small_external_initializers`).
        prev_cwd = os.getcwd()
        os.chdir(self._onnx_dir)
        try:
            editor.extract_gather_block_quantized_lut(
                save_to=self._export_dir / _TOKEN_EMBEDDINGS_FILENAME,
                inp_name="inputs_embeds",
                node_name=_EMBED_TOKENS_GATHER_NODE,
            )
            editor.extract_gather_block_quantized_lut(
                save_to=self._export_dir / _PER_LAYER_EMBEDDINGS_FILENAME,
                inp_name="per_layer_inputs",
                node_name=_EMBED_TOKENS_PER_LAYER_GATHER_NODE,
            )
            static_model = editor.to_onnx(
                check_type=False, strict_mode=False, data_prop=False, override_ir=model.ir_version
            )
        finally:
            os.chdir(prev_cwd)
        # No custom-op shape resolution needed anymore: both
        # GatherBlockQuantized nodes were fully extracted above, so this
        # component is now a trivial passthrough (0 compute nodes, plain
        # float32 I/O) with nothing left for SymbolicShapeInference to do.
        return static_model

    def _make_decoder_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = gs.import_onnx(model)
        editor = Gemma4OnnxGraphEditor(graph, self._onnx_export_dtype)

        editor.fix_io(self._max_kv_len)
        editor.normalize_kv_concat_axis()
        editor.replace_simplified_layer_norm()
        editor.replace_rotary_embedding(save_to=self._export_dir / _ROPE_CACHES_FILENAME)
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
        static_model = self._resolve_custom_op_shapes(static_model)
        return self._broadcast_where_value_operands(static_model)

    @staticmethod
    def _broadcast_where_value_operands(model: onnx.ModelProto) -> onnx.ModelProto:
        """Rank-match the *value* operands (inputs 1/2) of every `Where` to
        the op's output shape, materializing constant operands rather than
        deferring to a runtime `Expand`.

        Required for `torq-compile` to lower these ops at all. The static
        KV-cache/mask edits above (`replace_dynamic_kv_cache`,
        `mask_future_attn_scores`, and the source graph's own shared-donor /
        manual-attention masks) emit `Where`s whose fill value is a rank-0
        scalar while the condition broadcasts against a larger score tensor,
        e.g.:

            cond=[1,1,1,256]  X=[1,8,1,256]  Y=scalar  ->  [1,8,1,256]

        That exact combination aborts the compiler in `SelectPattern`'s
        dimension fusion:

            Kernel.cpp:3370: void mlir::syna::torq::fuse(LData &, int):
            Assertion `fused >= count && "Could not fuse the requested number
            of dimensions"' failed.

        `SelectPattern.cpp:72` fuses both operands to
        `min(input.denseDims(), output.denseDims())`; a stride-0 broadcast
        condition together with a rank-0 operand cannot supply that many
        dense dims. Isolated to a single synthetic `Where` -- **both**
        ingredients are needed, either alone compiles fine -- see
        `models/gemma4-e2b-int4/export/onnx/op_repros/
        select_broadcast_scalar_repro.py` for the 4-variant control matrix.

        Broadcasting the *value* operands (rather than the condition) is
        deliberate: for the mask `Where`s the fill value is a constant, so
        `MATERIALIZE` pre-broadcasts it at export time and adds **no**
        runtime op. Broadcasting the condition instead would need a real
        `Expand` on every forward pass.

        Runs after shape resolution, not immediately after the edits that
        create these nodes: `BroadcastOpInputs` skips any op whose output
        shape isn't fully static, which is only guaranteed once
        `propagate_static_shapes`/`_resolve_custom_op_shapes` have run.

        Verified: with this edit all 9 split components compile to `.vmfb`
        (9/9); without it, 7 of 9 abort with the assertion above. Note it
        also needs the sub-byte tile-size fix on the compiler side (see
        STATIC_EXPORT_PLAN.md) -- the two are independent and both required.
        """
        graph = gs.import_onnx(model)
        editor = Gemma4OnnxGraphEditor(graph, "fp32")
        editor.broadcast_op_inputs(
            ops=["Where"],
            inputs_idx=[1, 2],
            constants_policy=ConstantBroadcastPolicy.MATERIALIZE,
        )
        return editor.to_onnx(
            check_type=False, strict_mode=False, data_prop=False, override_ir=model.ir_version
        )

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

    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
    ):
        """Converts each exported static component to bf16, then int32 --
        the same two-step conversion the base class's `convert_models()`
        does (and gemma3 inherits unchanged), just invoked with
        `large_model=True` throughout (see `convert_dtype.onnx.convert_model`'s
        docstring for what that changes).

        The base class's own default-settings path was measured needing
        >14GB peak RSS for just the *smaller* component's int32 step on
        this model -- more than this dev machine's 15GB total, and it
        writes everything inline into a single file (no external data),
        which independently risks exceeding protobuf's 2GB message limit
        for the decoder. `large_model=True` keeps the untouched GB-scale
        weight tensors external throughout instead of materializing them.
        See STATIC_EXPORT_PLAN.md for the numbers.
        """
        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
            return
        self._convert_dir = Path(convert_dir or self._convert_dir)
        if self._convert_dir.exists():
            shutil.rmtree(self._convert_dir, ignore_errors=True)
        self._convert_dir.mkdir(parents=True, exist_ok=True)

        for comp, model_path in list(self._export_paths.items()):
            # Pin target_opset to the model's own opset so `upgrade_model`
            # inside `convert_model()` is a no-op -- the version converter
            # is untested against this graph and no upgrade is needed here.
            source_opset = get_model_opset(onnx.load(model_path, load_external_data=False))

            self._logger.info("(ONNX-convert) '%s': converting to bf16...", model_path)
            bf16_path = self._convert_dir / model_path.name
            _convert_dtype(
                model_path, bf16_path, "bf16",
                convert_io=not preserve_io, target_opset=source_opset, large_model=True,
            )
            self._logger.info("(ONNX-convert) '%s': converting to int32...", bf16_path)
            _convert_dtype(
                bf16_path, bf16_path, "int32",
                convert_io=not preserve_io, target_opset=source_opset, large_model=True,
            )
            self._export_paths[comp] = bf16_path
            self._logger.info("(ONNX-convert) '%s' -> '%s'", comp, bf16_path)

            # `MatMulNBits` decomposition is deliberately NOT run here
            # anymore: decomposing the whole ~3713-node, ~242-MatMulNBits
            # decoder in one pass needs to hold all ~3.7GB of newly-unpacked
            # int8 weight data simultaneously, which OOM'd repeatedly on
            # this dev machine even under an 8G cap. Split the converted
            # decoder into smaller components first (see
            # `models/gemma4-e2b-int4/export/onnx/split_decoder.py`) and
            # decompose each one individually instead -- same
            # `decompose_matmul_nbits_in_file` helper below, just invoked
            # per component so peak memory stays bounded to one component's
            # weight data (~500MB-600MB) at a time.

    def _decompose_decoder_matmul_nbits(self, model_path: Path) -> None:
        """Instance-method wrapper around `decompose_matmul_nbits_in_file`
        (kept for this class's own internal use); see that function for
        what it actually does and why it's no longer auto-invoked on the
        whole decoder from `convert_models`.
        """
        decompose_matmul_nbits_in_file(model_path)
        self._logger.info("(ONNX-convert) '%s': MatMulNBits decomposition complete", model_path)

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
        # `embed_tokens` is deliberately excluded: `_make_embed_tokens_static`
        # always fully extracts both `GatherBlockQuantized` embedding tables
        # to `.npy`, leaving a 0-initializer graph that never references its
        # original ~1.76GB external data -- copying it would be pure waste.
        for comp, fname in (("decoder", _DECODER_FILENAME),):
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
        decoder_session = ort.InferenceSession(str(self._export_paths["decoder"]), **session_kwargs)
        # `embed_tokens.onnx` is a 0-node passthrough post-extraction (both
        # `GatherBlockQuantized` tables were lifted out, still packed, to
        # `.npz` -- see `_make_embed_tokens_static`/STATIC_EXPORT_PLAN.md),
        # so there's no computation left in it to run via onnxruntime for a
        # token -> embedding lookup; dequantize directly from the packed
        # LUTs instead.
        token_lut = _PackedEmbeddingLUT(self._export_dir / _TOKEN_EMBEDDINGS_FILENAME)
        per_layer_lut = _PackedEmbeddingLUT(self._export_dir / _PER_LAYER_EMBEDDINGS_FILENAME)
        rope_lut = _RopeCacheLUT(self._export_dir / _ROPE_CACHES_FILENAME)

        inputs_embeds, per_layer_inputs = _lookup_embeddings(token_lut, per_layer_lut, 2)  # <bos>
        # Build feeds from whatever inputs the static graph actually has --
        # `attention_mask`/`num_logits_to_keep` are gone (dead-input-eliminated
        # after `replace_dynamic_kv_cache`/`mask_future_attn_scores` replaced
        # their role with a `position_ids`-derived `cur_len`, and
        # `fold_num_logits_to_keep` folded the other to a constant), unlike
        # the pre-edit graph -- don't assume the original I/O signature.
        # `cos_full_*`/`sin_full_*` (see `_RopeCacheLUT`) are likewise looked
        # up host-side rather than computed in-graph.
        available = {
            "inputs_embeds": inputs_embeds,
            "per_layer_inputs": per_layer_inputs,
            **rope_lut.lookup(0),
        }
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

        self._validate_generation_matches_dynamic(n_iters, token_lut, per_layer_lut, rope_lut)

    def _validate_generation_matches_dynamic(self, n_iters: int, token_lut, per_layer_lut, rope_lut):
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

                inputs_embeds, per_layer_inputs = _lookup_embeddings(token_lut, per_layer_lut, tok)

                static_feeds = {
                    "inputs_embeds": inputs_embeds,
                    "position_ids": np.array([[step]], dtype=np.int64),
                    "per_layer_inputs": per_layer_inputs,
                    **rope_lut.lookup(step),
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
        convert_dtypes=args.convert_dtypes,
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)


def main():
    parser = argparse.ArgumentParser(
        description="Export Gemma4-E2B (int4 quantized source) to a static ONNX graph"
    )
    add_gemma4_int4_export_args(parser)
    export_gemma4_int4_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
