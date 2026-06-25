# Moonshine Streaming — Integration Plan

Plan for incorporating the copied `moonshine_streaming_2_split/` model into `torq-tools`,
renaming it to `moonshine_streaming`, deduplicating against the repo's shared graph-edit
framework, and aligning its outputs/dirs with the other model exporters.

## Status at a glance (updated 2026-06-25)

| Phase | Status |
|---|---|
| 1 — Rename & restructure | ✅ done |
| 2 — Deduplicate against shared framework | ✅ done (regression PASS) |
| 3 — New edits stay model-local | ✅ absorbed into Phase 2 |
| 4 — Align dirs / component names / `convert_models` | ✅ done (regression PASS) |
| 5 — Static-only inference + `infer.py` + wav-path fix | ✅ done (real transcripts verified) |
| 6 — Register (export_model/infer_model) + packaging + README | ✅ done |
| 7 — Cleanup & verify | ⬜ next |

Supporting facts: dependency blocker resolved (§1.3), baseline export verified working (§4),
and a golden-baseline byte comparison is the regression oracle for every later phase (§4.1).

---

## 1. Current state of the copied folder

`src/torq/models/moonshine_streaming_2_split/` is a **2-model streaming ASR exporter**
(a fused encoder + a KV-cache decoder). It was developed standalone, so it **vendored its
own private copies** of the repo's graph-edit framework instead of importing them.

| File | Lines | What it is |
|---|---|---|
| `export.py` | 853 | PyTorch wrapper modules + `MoonshineStreaming2SplitExporter(OnnxModelExporterBase)` — genuinely new, model-specific |
| `_inference.py` | 335 | `MoonshineStreaming2Split` runtime runner — new |
| `_graph.py` | 87 | `...2SplitOnnxGraphEditor` subclass + IO fixers — new, thin |
| `_graph_base.py` | 790 | Editor base **+ a re-implemented `DecomposeStridedConv1D`** + 4 inline new edits |
| `edits.py` | 1639 | **~80% duplicates** of `torq.graph_edit.edits` + 4 genuinely-new edits |
| `onnx.py` | 249 | **Stale fork** of `torq.graph_edit.onnx` |
| `__init__.py` | 136 | arg parsers + constants — new |
| `validate.py` | 83 | standalone validator — new |
| 3× `*.md` | — | design notes (to be removed) |

### 1.1 Duplication map (the core cleanup)

**Pure duplicates of repo code — delete + rewire:**

- `onnx.py` → entire file is a stale copy of `src/torq/graph_edit/onnx.py`. The fork is
  missing the newer `requires_shape_inference`, `_infer_shapes`, `freeze_shape_seeds`,
  `apply_fixed_input_shapes`, and RNN handling.
- `edits.py` → these all already exist in `torq.graph_edit.edits`:
  `ReplaceDynamicKVCache`, `MaskFutureAttentionScores`, `AddCurrLenInput`,
  `ConvertToStaticIndex`, `DequantizeProjectionsMatMul`, `RemoveIsNaN`,
  `RemoveRedundantCasts`, `FoldScalarMatMul`, `ReplaceConstantDivWithMul`,
  `ConstantBroadcastPolicy`, `BroadcastOpInputs`, `ExtractConstantLUT`,
  `CombineKVCacheMixin`, `CommonGraphEditsMixin`.
- `_graph_base.py` → its `DecomposeStridedConv1D` duplicates
  `torq.graph_edit.edits.conv.DecomposeStridedConv1D`.

**Genuinely new — must be preserved (kept model-local, see decisions):**

- Edits: `DecomposeLayerNormalization`, `DecomposeLayerNormalizationMulReciprocal`,
  `DecomposeGelu`, `DecomposeBooleanAnd`.
- Inline editor methods: `remove_identity_gather_nd`, `decompose_reduce_sum`,
  `decompose_asinh`, `clear_intermediate_shapes`.

### 1.2 Two things already broken against the current repo

1. `_graph_base.py` does
   `from ..moonshine._graph import (MoveOutputFromConcat, ReplaceInt64FloatCast, ReplacePadWithConcat)`
   — but `moonshine._graph` only defines `MoveOutputFromConcat`; the other two live in
   `torq.graph_edit.edits` (`arithmetic` and `padding`). **This import fails today.**
2. The new LayerNorm edits depend on `editor._dim_map`, which the folder's forked
   `fix_io_dims` sets. The repo's `fix_io_dims` does **not** set `_dim_map`, so rewiring
   needs reconciliation.

### 1.3 Dependency reality check — ✅ RESOLVED

Originally `transformers 4.57.6` shipped `moonshine` but **not** `moonshine_streaming`, so
`_generate_source_onnx()` could not run. This is now fixed:

- A `moonshine-streaming` extra was added to `pyproject.toml` plus a per-model
  `requirements.txt`: `transformers>=5.5.1`, `torch`, `onnxscript`, `soundfile`, `scipy`.
  (`onnxscript` is required by `torch.onnx.export(dynamo=True)`; `moonshine_streaming` landed
  in the Transformers 5.5.x series.)
- Installed and verified in `.venv`: `transformers 5.12.1`, `onnxscript 0.7.0`,
  `torch 2.12.1`. `import transformers.models.moonshine_streaming` succeeds.
- Install command: `pip install -e '.[moonshine-streaming]' --extra-index-url https://download.pytorch.org/whl/cpu`.

---

## 2. Decisions (locked)

1. **Output names:** rename emitted components to `encoder.onnx` / `decoder.onnx`
   (was `fused_encoder.onnx` / `decoder_kv.onnx`).
2. **New edits:** keep the 4 new graph edits **model-local in `_graph.py`** — do NOT promote
   to the shared `graph_edit/edits` package.
3. **Inference:** **drop the dynamic-decoder code path** in `_inference.py` (static-only,
   matching what the exporter actually emits).
4. **Design docs:** **delete** `plan.md`, `onnx_quant_plan.md`, `on_device_optimization_plan.md`.
5. **No sibling-model dependencies:** `moonshine_streaming` must reuse code **only from the
   shared `src/torq` level** (`torq.graph_edit`, `torq.model_export`, `torq.utils`,
   `torq.inference`). It must **not** import from sibling model packages such as
   `torq.models.moonshine`. If a helper is needed by more than one model, it gets promoted to
   the shared `graph_edit/edits` package rather than imported across models.

---

## 3. The Plan

### Phase 1 — Rename & restructure — ✅ DONE
- ✅ Moved `moonshine_streaming_2_split/` → `moonshine_streaming/` (plain `mv`; the folder was
  untracked so `git mv` did not apply).
- ✅ Deleted the 3 design `.md` files and stale `__pycache__`.
- ✅ Dropped `2Split` from class names:
  - `MoonshineStreaming2SplitExporter` → `MoonshineStreamingExporter`
  - `MoonshineStreaming2Split` (runner) → `MoonshineStreaming`
  - `MoonshineStreaming2SplitOnnxGraphEditor` → `MoonshineStreamingOnnxGraphEditor`
- ✅ Collapsed `_graph_base.py` + `_graph.py` into a single `_graph.py` with one editor class
  (moved `fix_fused_encoder_io`, `fix_decoder_kv_io`, `make_decoder_static` into the base
  class), mirroring the `moonshine` one-editor-per-model layout.
- ✅ Removed the cross-model dependency (per decision 5): the only import from a sibling
  package was `from ..moonshine._graph import MoveOutputFromConcat`. It — and the three dead
  wrapper methods it fed (`move_output_from_concat`, `replace_int64_float_cast`,
  `replace_pad_with_concat`, none called anywhere in the streaming pipeline) — were deleted.
  Package now imports **only** from the shared `src/torq` level.
- ✅ Cleaned cosmetic "2-split"/"2-Split" prose in log lines, comments, CLI descriptions, and
  logger names. **Left** the `2split_*` / `export/iree` directory-path literals for Phase 4.
- ✅ Verified: `__init__`, `_graph`, `_inference` import cleanly. `export.py` fails **only** on
  the known `transformers.models.moonshine_streaming` blocker (§4), not on any Phase-1 change.

> Note: the forked `onnx.py` and the duplicate-laden `edits.py` are still present — Phase 2
> removes them. The `_dim_map` reconciliation (Phase 2) and dir/name alignment (Phase 4) are
> still pending.

### Phase 2 — Deduplicate against the repo framework — ✅ DONE (regression PASS)

**Key correction discovered during execution:** the fork's "duplicated" edits are *not* all
pure duplicates. Diffing fork-vs-repo source showed three classes are **streaming-specialised
supersets** that are actually used by the export path and therefore **must stay model-local**:
`ReplaceDynamicKVCache` (handles the dynamo stacked-cache Gather-from-graph-input pattern +
pre-allocated output shape), `MaskFutureAttentionScores` (shape-based self-attn detection), and
`AddCurrLenInput` (accepts `past_self` + Gather/Squeeze). Swapping these for the repo versions
would change the decoder graph. `ExtractConstantLUT` also diverged but is **unused** in
streaming (embeddings handled at the PyTorch level) → dropped.

What was done:
- ✅ **Deleted forked `onnx.py`** (249 lines, a stale copy of `graph_edit/onnx.py`); editor now
  imports `OnnxGraphEditor, FixedDimMapping, DimMatchType` from `...graph_edit` and
  `rewire_consumers` where needed.
- ✅ **Trimmed `edits.py`** 1639 → 826 lines: removed the 8 byte-identical duplicate edits +
  the two mixins + unused `DequantizeProjectionsMatMul`/`ExtractConstantLUT`. Kept only the
  **3 divergent** (`ReplaceDynamicKVCache`, `MaskFutureAttentionScores`, `AddCurrLenInput`) and
  the **4 new** (`DecomposeLayerNormalization`(`MulReciprocal`), `DecomposeGelu`,
  `DecomposeBooleanAnd`) — all importing `OnnxGraphEdit` from the shared package.
- ✅ **Removed the local `DecomposeStridedConv1D`** from `_graph.py` (818 → 461 lines); the
  shared edit is functionally identical (only diff was a `@dataclass` decorator) and is now
  inherited via `CommonGraphEditsMixin.decompose_strided_conv1d`.
- ✅ Editor now extends the **repo** `OnnxGraphEditor` + repo `CommonGraphEditsMixin` /
  `CombineKVCacheMixin`, **overriding** only `replace_dynamic_kv_cache` /
  `mask_future_attn_scores` / `add_curr_len_input` to construct the local divergent edits, and
  adding wrappers for the 4 new decompositions.
- ✅ **`_dim_map` reconciled**: overrode `fix_io_dims` to record the `name -> value` map the
  LayerNorm edits need (repo `fix_io_dims` doesn't expose it).
- ✅ Net: ~1,420 lines of duplicated framework code removed; **zero** modifications to the
  shared `graph_edit` package; package imports only from the shared `src/torq` level.

**Regression (oracle = §4.1):** re-ran the same `-i 8` export (EXIT=0) and re-compared to the
golden baseline — all four `float`/`converted` graphs **byte-identical modulo cosmetic
metadata**, all initializer weight bytes equal. Behavior preserved.

> The previously-noted broken `from ..moonshine._graph import (ReplaceInt64FloatCast,
> ReplacePadWithConcat)` import was already removed in Phase 1, so it was not a Phase 2 task.

### Phase 3 — New edits stay model-local — ✅ ABSORBED INTO PHASE 2
- The 4 new edits (`DecomposeLayerNormalization`, `DecomposeLayerNormalizationMulReciprocal`,
  `DecomposeGelu`, `DecomposeBooleanAnd`) now live in `moonshine_streaming/edits.py`; the
  inline methods (`decompose_asinh`, `decompose_reduce_sum`, `remove_identity_gather_nd`,
  `clear_intermediate_shapes`) live on the editor in `moonshine_streaming/_graph.py`.
- The editor (`MoonshineStreamingOnnxGraphEditor`) extends the **repo** editor + repo mixins
  and adds model-local wrappers for these edits — mirroring how `moonshine/_graph.py` keeps
  `MoveOutputFromConcat` local.
- Pending dead-code prune (defer to Phase 7): `decompose_reduce_sum`, `fix_decoder_kv_io`, and
  the vestigial 5-split `fix_preprocessor_io` / `fix_encoder_io` / `fix_decoder_io` helpers are
  currently unused by the export path; `DecomposeLayerNormalizationMulReciprocal` is kept but
  not yet called.

### Phase 4 — Align exporter to repo conventions — ✅ DONE (regression PASS)
- ✅ Renamed emitted components `fused_encoder→encoder`, `decoder_kv→decoder`:
  `STATIC_MODEL_COMPONENTS`, the source/export ONNX file stems, the editor `component`
  strings, the `_components` dict keys, and the `apply_post_static_patches` branches
  (`if "encoder"/"decoder" in component`). (Tensor-level `output_names` were already
  graph-internal and unchanged. Editor method names `fix_fused_encoder_io` and the runner's
  internal `fused_encoder` naming are internal and left for Phase 5.)
- ✅ Rewrote `_setup_dirs` to the shared convention:
  - `export/onnx/<dtype>/static/`  (was `2split_streaming_static`)
  - `export/onnx/converted/static/`  (was `2split_static`)
  - `export/torq/<dtype|converted>/static/`  (**fixed** the `export/iree/...` mistake)
  - source under `source/onnx/merged/<size>/<dtype>/c{chunk}_t{tokens}/` — a config subdir is
    kept (dropping the `2split_` prefix) because the torch export bakes `chunk_len`/`max_tokens`
    into the source graph, so a per-config cache is required for correctness.
- ✅ Added a `convert_models` override that places the sidecars into `converted/`:
  - The **embedding LUTs** (`decoder_token_embeddings.npy` → decoder `inputs_embeds`,
    `adapter_pos_emb.npy` → encoder `position_embeddings`) are **bf16-converted** (via the base
    converter's `external_data` arg). **Deliberate deviation from the golden:** the golden left
    them float32, but the converted graphs take **BFLOAT16** for those inputs (verified), so a
    float32 LUT would dtype-mismatch at runtime. Under `--preserve-io-dtypes` they are copied
    float32 instead. The `moonshine` model already bf16-converts its embedding LUT the same way.
  - The **metadata** (`tokenizer.json`, `config.json`, `streaming_config.json`) is copied
    verbatim. Constants `EMBEDDING_SIDECARS` / `METADATA_SIDECARS` / `SIDECAR_FILES` drive this.
- ✅ Removed the stale pre-rename dirs from the models tree.

**Regression:** re-ran the `-i 8` export (EXIT=0, source regenerated under the new names) and
compared with path-mapping (`encoder`↔`fused_encoder`, `decoder`↔`decoder_kv`): all four graphs
content-equivalent to golden (weights equal). In `converted/`, the 3 **metadata** sidecars are
bit-identical to golden; the 2 **embedding LUTs** are intentionally **bf16** (golden = float32)
per the dtype fix above — verified to be the byte-exact `astype(bfloat16)` of the float originals.

### Phase 5 — Inference + entry points — ✅ DONE
- ✅ Simplified `_inference.py` to **static-only**: removed the dynamic-decoder branch and the
  `_is_static_decoder` flag (the exporter always emits the static KV-cache decoder; the runner
  now raises if `current_len` is absent).
- ✅ **All dims derived from the ONNX graphs** (no per-size hardcoding): `n_layers`,
  `n_kv_heads`, `head_dim`, `hidden_size`, `max_tokens`, `max_memory_len`, `chunk_len`, `F`,
  `total_lookahead`, `c1_channels`, `enc_left_ctx`, and `extract_embeddings`. `model_size` is now
  optional/informational.
- ✅ **Dtype-aware feeds**: the runner reads the model's I/O dtype from the session and builds
  all zero-buffers / casts feeds to match, so the same runner drives the float **and** bf16
  exports (the host-side LUTs already carry the right dtype per export dir). Added a
  `_ORT_TYPE_TO_NP` map (incl. `bfloat16`).
- ✅ Renamed the runner's public ctor kwargs `encoder_model` / `decoder_model`; added a
  `load_moonshine_streaming(model_dir, ...)` loader (ONNX now; VMFB raises a clear "not yet
  wired" error).
- ✅ Added **`infer.py`** with `infer_moonshine_streaming(args)` + `_transcribe` (loads WAV via
  soundfile, resamples to 16 kHz, decodes with the model dir's `tokenizer.json`).
- ✅ Fixed the self-referential wav path in **both** `export.validate_onnx` and `validate.py`:
  they now pull samples from `hf-internal-testing/librispeech_asr_dummy` (dummy-audio fallback
  when offline).

**Verification (real audio, float export):**
- `validate.py` → `'Mr. Quilter is the apostle of the middle classes…'` (correct LibriSpeech ref).
- `infer.py` on a real WAV → `'Nor is Mr. Quilter's manner less interesting than his matter.'`
- Integrated `export … validate_onnx` → 5/5 LibriSpeech samples transcribed correctly.
- Graph regression: graphs **unchanged** vs golden (Phase 5 touched only inference/validation).

> VMFB/Torq-runtime streaming inference is intentionally **deferred** — it needs richer config
> than the VMFB exposes (the `streaming_config.json` sidecar plus decoder shapes). The bf16 LUT
> dtype work (Phase 4) means the converted models + sidecars are already correct for that runtime.

### Phase 6 — Register & package — ✅ DONE
- ✅ Added a `moonshine_streaming` subparser + dispatch branch in
  `src/torq/models/export_model.py` and `src/torq/models/infer_model.py`. The package
  `__init__` is lightweight (argparse only), and the heavy `export`/`infer` modules are
  imported lazily on dispatch, so registering streaming does **not** pull torch/transformers
  into the other models' CLI paths.
- ✅ `moonshine-streaming` optional-dependency extra in `pyproject.toml`
  (`transformers>=5.5.1`, `torch`, `onnxscript`, `soundfile`, `scipy`), folded into `all`, plus
  a per-model `requirements.txt`. The `torq-export-model` / `torq-infer-model` entry points
  cover it via the registry.
- ✅ Updated `README.md`: extras table, export example (`--chunk-len`), inference example, and
  the CLI-usage block.

**Verification:** `torq-export-model --help` lists `{moonshine,moonshine_streaming,smollm2,gemma3}`;
`torq-export-model moonshine_streaming --skip-torq all --chunk-len 1280 -i 8 --skip-validation`
runs end-to-end and writes `encoder.onnx`/`decoder.onnx`; the infer subparser `--help` resolves.

### Phase 7 — Cleanup & verify
- Run focused graph-edit tests, then broaden as needed.
- Do a `--skip-torq` dry run against pre-exported source ONNX to confirm the static +
  graph-edit pipeline works end-to-end (this venv's transformers cannot regenerate source).

---

## 4. Baseline export — ✅ VERIFIED WORKING (2026-06-25)

The dependency blocker is resolved (§1.3) and a full export was run successfully **after
Phase 1**, before the Phase 2–6 cleanup:

```
python -m torq.models.moonshine_streaming.export \
    --skip-torq all --chunk-len 1280 --extract-embeddings \
    --convert-dtypes -i 8 --export-attention
```

- **EXIT=0.** Pipeline: download `UsefulSensors/moonshine-streaming-tiny` → build PyTorch
  wrappers → dynamo ONNX export → static graph edits → static-shape verification (passed) →
  ONNX validation → bf16 + int64→int32 dtype conversion.
- Detected config: `chunk_len=1280, F=4, total_lookahead=16, warmup_chunks=4, max_tokens=48,
  max_memory_len=400, extract_embeddings=True, export_attention=True`.
- Artifacts under `models/UsefulSensors/moonshine-streaming-tiny/export/`:
  - `onnx/float/2split_streaming_static/` → `fused_encoder.onnx`, `decoder_kv.onnx`,
    `decoder_token_embeddings.npy`, `adapter_pos_emb.npy`, `tokenizer.json`, `config.json`,
    `streaming_config.json`
  - `onnx/converted/2split_static/` → bf16/int32 `fused_encoder.onnx`, `decoder_kv.onnx`

Confirmed observations that map to pending phases:

1. Validation fell back to **dummy audio** ("Test audio not found") — self-referential wav
   path. → **Phase 5**.
2. Dirs still `2split_streaming_static` / `2split_static`; components still
   `fused_encoder` / `decoder_kv`. → **Phase 4** (rename to `encoder`/`decoder`,
   `export/onnx/<dtype>/static`, `export/torq/...`).
3. `convert_models` did **not** copy/convert the `.npy` sidecars into `converted/` (only the
   two `.onnx`). → **Phase 4** (override `convert_models`).
4. Cosmetic only: `[W] colored module is not installed` (ignorable).

### 4.1 Golden-baseline comparison — ✅ FUNCTIONALLY BIT-EQUIVALENT

Compared every generated file against a golden export at `baseline/moonshine-streaming-tiny/`:

- **22 files bit-identical**: all weights (`model.safetensors`), external data (`*.onnx.data`),
  `.npy` sidecars, `tokenizer.json`, all JSON configs.
- **6 `.onnx` graphs differ by cosmetic metadata only** — after stripping `producer_version`
  and per-node `metadata_props`, every graph is **byte-identical** and **all initializer
  weight bytes are equal** (incl. the bf16/int32-converted weights). The two differing fields:
  1. `producer_version` `pytorch 2.12.0+cu130` (golden) vs `2.12.1+cu130` (torch patch bump).
  2. `pkg.torch.onnx.stack_trace` node metadata embedding the absolute export path
     (`/home/yhtet/projects/...` vs our repo path) — the path-length delta × node count is the
     entire ~15 KB source-graph size gap. Pure torch/dynamo debug provenance; no compute impact.
- **9 missing** = `.mlir`/`.vmfb` (we ran `--skip-torq all`) + `converted/` sidecars (Phase 4 gap).

**Conclusion:** the refactored exporter reproduces the golden computation graphs and weights
exactly. Optional future hooks for true *byte* determinism: pin `torch==2.12.0` and/or strip
dynamo `metadata_props` during export. **This comparison is the regression oracle for Phases
2–6: re-export + re-compare must keep the same equivalence (graphs equal modulo that metadata).**

---

## 5. Execution order & progress

1. ✅ Phase 1–2 (rename + dedup) — done; Phase 3 absorbed into Phase 2. Regression PASS.
2. ✅ Phase 4 (align dirs/component names, `convert_models` + bf16 LUTs). Regression PASS.
3. ✅ Phase 5 (static-only inference, dim/dtype derivation, `infer.py`, wav-path) — real
   transcripts verified.
4. ⬜ Phase 6 registration (`export_model.py` / `infer_model.py` subparser + dispatch); packaging done.
5. ⬜ Phase 7 (dead-code prune + verify).

After each remaining phase, re-run the `-i 8` export and re-check against the golden baseline
(§4.1) — graphs must stay equal modulo the cosmetic `producer_version` / dynamo `metadata_props`.
Note Phase 4 renames the `converted/` dir and adds the `.npy` sidecars there, so the comparison
paths shift accordingly (the *graph* equivalence must still hold).
