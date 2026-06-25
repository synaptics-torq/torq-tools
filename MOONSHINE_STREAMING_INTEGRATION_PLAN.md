# Moonshine Streaming — Integration Plan

Plan for incorporating the copied `moonshine_streaming_2_split/` model into `torq-tools`,
renaming it to `moonshine_streaming`, deduplicating against the repo's shared graph-edit
framework, and aligning its outputs/dirs with the other model exporters.

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

### Phase 2 — Deduplicate against the repo framework
- **Delete `onnx.py`**; repoint imports to
  `from ...graph_edit import OnnxGraphEdit, OnnxGraphEditor, FixedDimMapping, DimMatchType, rewire_consumers`.
- **Strip the duplicated edits from `edits.py`** and the local `DecomposeStridedConv1D`;
  import them from `torq.graph_edit.edits` and use the repo's
  `CommonGraphEditsMixin` / `CombineKVCacheMixin`.
- Reconcile the `_dim_map` dependency: pass `dim_map` explicitly to the LayerNorm edits from
  the model editor (since the repo `fix_io_dims` doesn't populate it).

  > The previously-noted broken `from ..moonshine._graph import (ReplaceInt64FloatCast,
  > ReplacePadWithConcat)` import was already removed in Phase 1 (it fed only dead code), so it
  > is no longer a Phase 2 task.

### Phase 3 — New edits stay model-local
- Keep `DecomposeLayerNormalization`, `DecomposeLayerNormalizationMulReciprocal`,
  `DecomposeGelu`, `DecomposeBooleanAnd` and the inline methods (`decompose_asinh`,
  `decompose_reduce_sum`, `remove_identity_gather_nd`, `clear_intermediate_shapes`) inside
  `moonshine_streaming/_graph.py`.
- Net effect: one `_graph.py` containing the `MoonshineStreamingOnnxGraphEditor` (extending
  the repo editor + repo mixins) plus the model-specific edits — mirroring how
  `moonshine/_graph.py` keeps `MoveOutputFromConcat` local.

### Phase 4 — Align exporter to repo conventions
- Rename emitted components `fused_encoder→encoder`, `decoder_kv→decoder` everywhere:
  `STATIC_MODEL_COMPONENTS`, ONNX `output_names`, the runner's shape-detection input keys,
  and the `apply_post_static_patches` / `make_static` branches.
- Rewrite `_setup_dirs` to mirror `moonshine`:
  - `export/onnx/<dtype>/static/`  (drop `2split_streaming_static`)
  - `export/onnx/converted/static/`
  - `export/torq/<dtype|converted>/static/`  (**fix** the current `export/iree/...`)
  - source under `source/onnx/.../<dtype>/`.
- Override `convert_models` (like `moonshine` does) so the external data
  `decoder_token_embeddings.npy` **and** `adapter_pos_emb.npy` get bf16-converted alongside
  the models.

### Phase 5 — Inference + entry points
- Simplify `_inference.py` to **static-only** (remove the dynamic-decoder branch).
- Derive `head_dim` / `hidden_size` from config or ONNX shapes instead of hardcoding per size.
- Add `infer.py` with `infer_moonshine_streaming(args)` (mirror `moonshine/infer.py`'s
  `_transcribe` shape). The folder currently has only `validate.py`.
- Fix the hardcoded validation wav path (`../moonshine_streaming/OSR_us_000_0010_8k.wav`
  becomes self-referential after the rename): use the librispeech dataset approach like
  `moonshine`, or add the wav as a committed test asset.

### Phase 6 — Register & package
- Add a `moonshine_streaming` subparser + dispatch branch in
  `src/torq/models/export_model.py` and `src/torq/models/infer_model.py`.
- ✅ **DONE** — Added a `moonshine-streaming` optional-dependency extra in `pyproject.toml`
  (`transformers>=5.5.1`, `torch`, `onnxscript`, `soundfile`, `scipy`), folded into `all`, plus
  a per-model `src/torq/models/moonshine_streaming/requirements.txt`. The
  `torq-export-model` / `torq-infer-model` entry points already cover it via the registry.
- Update `README.md` usage section.

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

---

## 5. Suggested execution order

1. Phase 1–2 (rename + dedup) — the bulk, lowest-risk, easy to verify by import/load.
2. Phase 3–4 (consolidate `_graph.py`, align dirs/names, `convert_models`).
3. Phase 5–6 (static-only inference, `infer.py`, registration, packaging).
4. Phase 7 (verify).
