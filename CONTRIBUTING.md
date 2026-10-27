# Contributing

Thanks for taking the time to improve `torq-tools`. This project is used by compiler developers, application engineers, AI/ML researchers, and people who are just starting to adapt models for Torq. You do not need to know every part of the repository before contributing; a small, well-tested change is often the most useful kind.

This guide focuses on the most common contribution paths: ONNX graph edits, model export pipelines, command line tools, and shared utilities.

## Development Environment

A virtual environment for Python development is highly recommended. The first step is to install dependencies:

```bash
cd torq-tools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements.txt
```

CI installs only `tests/requirements.txt`. Keep it up to date when tests need additional packages, including Jinja2 for Hugging Face chat templates. Tokenizer and mocked exporter tests should run without PyTorch or model downloads.

Install optional extras only for workflows that need them:

```bash
python -m pip install -e '.[moonshine]' --extra-index-url https://download.pytorch.org/whl/cpu
```

Moonshine Streaming is the exception: it requires `transformers` 5.x, which is incompatible with the `optimum`-based exporters (they cap `transformers` below 4.58). Do **not** layer it onto the base environment: installing it there
produces pip resolver conflicts and breaks the `optimum`-based exporters. Set it up in a dedicated virtual environment (see [README.md](README.md#moonshine-streaming-separate-environment)):

```bash
bash src/torq/models/moonshine_streaming/install_requirements.sh
source .venv-moonshine-streaming/bin/activate
```

The script installs torq-tools in editable mode with `--no-deps` (skipping the base `optimum`/`transformers` 4.x pins) plus the complete pinned requirements.

Generated model artifacts should stay in ignored locations such as `models/`, `.torq/`, or pytest `tmp_path`. Large generated files like `.onnx`, `.tflite`, `.vmfb`, and `.mlir` are usually better to reproduce from commands than to commit.

## Adding Graph Edits

Graph edits live under `src/torq/graph_edit/edits/`. Pick the domain module that best matches the operator family, such as `arithmetic.py`, `shape.py`, `padding.py`, `conv.py`, `rnn.py`, `transformer.py`, or `artifacts.py`. If a helper is useful to more than one edit, `_helpers.py` is the right home for it.

Most edits subclass `OnnxGraphEdit` from `src/torq/graph_edit/onnx.py` and implement three small methods:

- `match(node)`: decides whether this edit applies. Try to keep this narrow so unrelated graphs are left alone.
- `transform(node)`: performs the graph change.
- `finalize(node)`: optional cleanup or metadata work after the transform.

If the edit should be part of the public graph-edit API, export it from `src/torq/graph_edit/edits/__init__.py`. Add a convenience method to `CommonGraphEditsMixin` when model exporters should call the edit through the fluent editor API.

> [!TIP]
> If a graph edit is only useful for one model exporter, keep it close to that model in `src/torq/models/<model>/_graph.py` instead of promoting it to the shared `torq.graph_edit.edits` package. See [Model Export Pipelines](#model-export-pipelines) for how model-specific graph editors fit into the exporter flow.

A few details make graph edits easier to review and safer to reuse:

- Keep the changed subgraph as small as practical.
- Preserve tensor names, shapes, dtypes, and `export_dtype` intentionally.
- Use `requires_shape_inference = True` when later logic needs inferred shapes; shape inference reimports the graph, so it is worth being deliberate.
- Let `OnnxGraphEditor` handle cleanup and topological sorting unless the edit creates an artifact that needs an immediate check.
- Write generated ONNX or NumPy files to `tmp_path` in tests unless the fixture is intentionally stable and shared under `tests/data/graph_edit/`.

### Graph Edit Tests

Every graph edit should have unit coverage under `tests/unit/graph_edit/test_<domain>.py`. The existing tests build tiny synthetic graphs with `onnx_graphsurgeon` and helpers from `tests/support/graph_edit.py`; that pattern is usually easier to understand than using a full exported model. Good unit tests cover the expected rewrite, no-match behavior, metadata, constants, and useful error paths without requiring ONNX Runtime.

Add ONNX Runtime integration coverage under `tests/integration/graph_edit/test_<domain>_accuracy.py` when the edit changes executed graph behavior or is meant to preserve numerical equivalence. The helpers `assert_model_outputs_close`, `run_model`, and `clone_graph` in `tests/support/graph_edit.py` are there to keep these tests compact. Integration modules normally use the domain marker plus `pytest.mark.ort`.

If you add a new graph-edit domain, add a pytest marker in `pyproject.toml`. The path-based markers in `tests/conftest.py` will still add `graph_edit`, `unit`, and `integration` for the existing test layout.

Useful commands:

```bash
python -m pytest tests/unit/graph_edit/test_<domain>.py
python -m pytest tests/integration/graph_edit/test_<domain>_accuracy.py
python -m pytest -m "graph_edit and not ort"
python -m pytest -m graph_edit
```

For bug fixes, it is helpful to add the failing test first. A small graph that reproduces the issue is usually the clearest way to show what changed and why.

## Model Export Pipelines

Model exporters live under `src/torq/models/<model>/` and usually extend `OnnxModelExporterBase` from `src/torq/model_export/onnx.py`. A complete exporter normally handles source model loading, static conversion, post-static graph patches, validation, dtype conversion, and optional Torq export.

The existing model directories use this layout:

- `export.py`: CLI argument handling and the exporter class.
- `_graph.py`: model-specific `OnnxGraphEditor` subclass and IO-shape helpers.
- `_inference.py`: runtime wrappers used for validation and demos.
- `infer.py` or `validate.py`: user-facing inference and validation entry points.

For decoder-only models, implement `validate_onnx()` by calling `validate_decoder_only_onnx()` from `src/torq/model_export/validation.py` with the model's static and dynamic runner classes, rather than writing the prompt loop by hand. It compares the export against the unedited source ONNX and keeps the prompt set and pass/fail reporting consistent across models.

Architecture-compatible checkpoints should share an exporter. FunctionGemma uses `torq-export-model gemma3 --functiongemma`, with optional source overrides, rather than a separate model package. Its validation supplies tool definitions through `runner_kwargs` and a tool prompt through `prompts`; `max_inp_len` covers the complete prompt independently of the prefill chunk size. Never reuse a quantized checkpoint from another source model.

Each exported variant must carry its runtime sidecars so its directory can be staged independently. The shared ONNX exporter copies JSON metadata, Jinja templates, tokenizer model/vocabulary files, and NumPy tables through export, quantization, conversion, and compilation with `_copy_runtime_assets()`. Keep stage-specific `compile_flags.json` snapshots separate. Copy tables before dtype conversion so converted tables win; compilation preserves their dtype unless `--torq-convert-io-dtype` requests runtime dtype conversion.

Static decoders read their KV caches without writing them: after `combine_kv_io_tensors` (and, for Hugging Face GQA exports, `group_query_heads`), call `read_only_kv_cache` with the step's chunk length. Each cache becomes a time-major `[1, L, 2 * kv, D]` input and each `present.*` output holds only the step's new rows, so exporters that support it register `--read-only-kv-cache` (on by default) through `add_onnx_args(read_only_kv_cache=True)`. `DecoderOnlyRunner._update_cache` writes a cache output that is shorter than its input into the cache at the step's position, and replaces the cache with any other output; static runners take cache shapes from the model's inputs (`_kv_cache_zeros`).

Exporters keep weights on disk between pipeline stages, so models far beyond the 2 GiB protobuf limit export without materializing every weight at every stage. Load sources and saved components with `load_onnx_lazy` (`torq/utils/onnx.py`): it tags external weights with their data directory and loads integer initializers inline, because ONNX shape inference and data propagation cannot read external shape/position controls. `gs.import_onnx` wraps those tensors in `LazyValues`, so an edit reads a weight only when it touches `Constant.values`. Prefer `.shape` / `.dtype` in `match()`; `.values` loads the array and keeps it for the editor's lifetime. Save lazy models with `save_onnx` (or `self._save_component_model`), never plain `onnx.save`: it copies untouched weights file-to-file and writes edited or new ones. Run the in-memory onnx checker or an ORT session built from a lazy model's bytes inside `with external_data_cwd(model):`, since both resolve external data from the working directory.

Exporter tests should be fast and deterministic by default. Prefer local synthetic data or mocked downloads for path naming, config handling, asset copying, graph patch decisions, and validation helpers. Full Hugging Face downloads, full ONNX export, and Torq compiler runs are useful manual checks, but they are usually too heavy for normal tests.

For pipeline-level changes, please run a manual end-to-end export when the dependencies are available and record the exact command and result in the PR notes or patch summary. Common checks include:

```bash
python -m torq.models.<model>.export --skip-torq
python -m torq.models.<model>.export --convert-dtype bf16 --skip-torq
torq-export-model <model> --skip-torq
```

If the change affects Torq compilation, also run the exporter without `--skip-torq` with `torq-compiler` installed, or point the run at a compiler with `TORQ_COMPILER_PATH` / `--compiler-path`.

### Graph Edit Harness

Exporters declare their default graph edits *declaratively* rather than calling editor methods imperatively. Each exporter overrides `graph_edit_blocks()` (on `OnnxModelExporterBase`) to return an ordered `dict[str, list[EditSpec]]`, where each block is a contiguous run of edits applied to one component/graph. An `EditSpec` is just an edit's registered name (its `OnnxGraphEdit` class name) plus positional constructor args. Args that are only known at export time (e.g. a `cur_len` tensor) are written as `ctx("name")` placeholders and resolved from a context mapping passed to `OnnxGraphEditor.apply_specs(specs, harness, context)`.

Inside `make_static` / post-static patches, replace imperative edit calls with `editor.apply_specs(self.graph_edit_blocks()[block], self._harness, context)`. Non-edit editor operations (IO fixing, KV combining, input reordering, `to_onnx`) stay imperative and simply bracket the `apply_specs` calls.

This wires up the shared CLI flags automatically once you call `add_graph_edit_harness_args(parser)` and, in `export_<model>_from_args`, `exporter.set_graph_edit_harness(GraphEditHarness.from_args(args))` plus the `--view-graph-edits` short-circuit:

```python
exporter.set_graph_edit_harness(GraphEditHarness.from_args(args))
if args.view_graph_edits:
    print(render_graph_edit_plan(exporter.describe_graph_edits()))
    return
```

Users then get `--apply-graph-edit NAME[:ARGS]` (repeatable, YAML args), `--apply-graph-edits-from-file FILE`, their `--exclude-graph-edit` / `--exclude-graph-edits-from-file` counterparts, and `--view-graph-edits`. Precedence when merging is flags > file > defaults: an overriding edit keeps the default's position, brand-new edits are appended (once per editor instance), and excluded edits are removed last.

The harness core and registry live in `src/torq/graph_edit/harness.py`; add unit coverage under `tests/unit/graph_edit/test_harness.py`.

`OnnxGraphEditor` supports per-edit intermediate dumps for manual graph inspection: pass `dump_path` and `dump_after_edit` (``"all"`` or a comma-separated edit-name list), and the editor writes the current graph after each matching edit as `<export-dir>/intermediates/0001_<EditName>_<component>.onnx`. Each editor continues from the highest index already written for its component, so snapshots stay ordered across static and post-static editor instances. Dump weights are always externalized before their `.data` sidecar is discarded, leaving lightweight structural snapshots regardless of `--split-weights`. The shared CLI flags `--dump-after-edit[=EDITS]` (graph-edit harness args) and `--split-weights` (ONNX args) remain orthogonal: `GraphEditHarness` stores the dump trigger (validated like the other edit names), while `OnnxModelExporterBase` stores `split_weights` for final component exports only. An exporter opts in by (a) passing `split_weights=args.split_weights` through to its `super().__init__`, (b) adding `**self._editor_dump_kwargs(final_path)` at each `OnnxGraphEditor` construction site (the kwargs stay empty when the flag is off), and (c) routing any post-export component re-saves through `self._save_component_model`. It saves in-progress components split (tensor data above 1024 bytes goes to the adjacent `<model>.onnx.data`; see `save_onnx` in `torq/utils/onnx.py`), and `export_onnx` writes the final layout: a single file for components that fit the 2 GiB limit unless `--split-weights` is set.


## Tools and Utilities

Standalone tools live next to the domain they serve: ONNX cleanup under `src/torq/model_export/`, model benchmarking under `src/torq/utils/benchmark/`. Shared helpers live under `src/torq/utils/`. Generic model-workflow tools (ONNX dtype conversion, ONNX quantization, TFLite static-shape conversion) ship in the `torq-compiler` package (`torq.lab`).

For new or updated tools, it is easiest to keep the command line wrapper thin and put the behavior in importable functions. Then tests can call those functions directly, with CLI-level tests reserved for argument parsing, file wiring, or user-visible behavior.

For ONNX tools, tiny synthetic models in the test are preferred over checked-in binaries. Please update `README.md` when a command, option, output artifact, or supported workflow changes.

## Before Review

Run the focused tests for the code you touched, then broaden to the relevant suite when the change affects shared behavior. For Python changes, please remove unused imports, variables, classes, and functions before handing off.

## Native decoder families

Export and size-probing instructions live in the [native batch-prefill guide](src/torq/model_export/doc/batch_prefill.md); each model README documents its own supported configurations. Compiler documentation covers the supplied multi-entry image and runtime contract. Native VMFB execution belongs in `torq-examples`, using `torq.runtime`; torq-tools validates the exported ONNX graphs with ONNX Runtime.

Finalize native decoder weights after all graph transforms using `share_native_weights`, and repeat this step after Gemma weight quantization. Final decoder ONNX files share `transformer.weights.data` regardless of `--split-weights`; integer controls stay inline for ONNX shape/data propagation. Deduplicate by tensor type, shape and byte content, preserving packed INT4 and BF16 storage. Keep all family graphs beside the shared data file when copying exports. Intermediate saves and independent head exports retain their existing storage policy.

The experimental Gemma3 270M/FunctionGemma and LFM2.5 230M path is opt-in via `--native-batch-prefill`, which is mutually exclusive with `--batch-prefill`. Optional `--batch-prefill-sizes` pins ascending sizes; its default of `None` probes the normal exported decoder after graph edits, optimization and weight quantization, then builds only the selected prefill graphs. Model graph construction belongs in each exporter. The shared boundary is `torq.model_export.prefill`: import named static ONNX graphs serially into one MLIR context, check common state signatures and parameter identities, encode recognized controls as typed literals, and compile with supplied-family support. The importer fingerprints packed weights, scales and zero points across entries, reuses parameter resources and supplies byte-aligned i4 storage plus typed BF16 literals to MLIR; fixed causal masks remain typed controls. The family ABI permits B returned KV rows while keeping full cache inputs shared. Unknown or mixed constant roles fail rather than relying on tensor byte sizes. Do not forward compiler generation options or add an architecture recognizer here.

Run `tests/.test/bin/python -m pytest tests/unit/model_export/test_native_prefill.py tests/unit/model_export/test_split_weights.py` for the small reproducible gates. An ordinary native export without `--skip-validation` also compares every hidden/state output to sequential FP32 ONNX decode at zero, populated and final-fit positions, before optional weight quantization. Read-only families assemble the sequential rows and compare all returned rows, bounding both relative L2 error and peak-scaled element error at 1e-4 to account for FP32 reduction-order differences. Legacy full-cache comparisons retain their elementwise tolerance. BF16/INT4 device generation and performance require separate board measurements; ORT CPU does not execute BF16 MatMul. Use `--dump-after-edit` to inspect intermediate ONNX graphs and `compiled/transformer.mlirbc` to inspect the exact imported family. The adjacent `transformer.family.json` records the ABI and actual decoder flags; use it to inspect entry signatures and compilation policy.

Decoder generation and its model-specific tests belong to torq-tools. The compiler stack consists of core and runtime support for supplied families; it does not need the former frontend branch. Keep compiler regressions for family validation, shared constants, layout and runtime dispatch.

The optional size probe is `python -m torq.model_export.prefill_sizes transformer.onnx --tool /path/to/torq-compile --max-batch-size 128 --context-length 256 -- --torq-convert-dtypes --torq-convert-io-dtype --torq-enable-split-constants-optimization --torq-prefetch-algorithm=stream-schedule`. It prints exporter `--native-batch-prefill --batch-prefill-sizes ...` options and records its tiling evidence in `prefill_sizes.json`. Probe the actual exported weight format with the same dtype-conversion and scheduling flags as the decoder compile: the probe extracts the projection and its constant producers, including packed INT4 weights, scales, zero points and dequantization, instead of synthesizing BF16 weights. Cache identities include the extracted graph and real constants; equal shapes with different graph/type structures remain separate. Direct BF16 constants retain BF16 split-grid snapping; quantized and other representations use observed widths. Context inference handles both legacy and time-major read-only KV I/O. The sweep is advisory: surrounding activation producers and attention can change full-model placement. Run `tests/unit/model_export/test_prefill_sizes.py` for extraction, dtype/quantization preservation, parser, cache and candidate-selection regressions.

The full Gemma3 retirement gate uses an FP32 Gemma3-270M IT export at context 256 with sizes 4,12,29,51,92,106,124. Export with `torq-export-model gemma3 --instruct-model --weight-quantization none --extract-embeddings --split-lm-head --no-read-only-kv-cache --native-batch-prefill --batch-prefill-sizes 4,12,29,51,92,106,124 --max-gen-tokens 256 --skip-torq`. Set `TORQ_NATIVE_GEMMA_EXPORT` to that native ONNX directory and `TORQ_COMPILER_PATH` to an independently built core/runtime compiler, then run `tests/.test/bin/python -m pytest -q tests/integration/model_export/test_native_prefill_compiler.py`. Use matching compiler/importer/runtime bindings. The check rejects a compiler exposing generation-size options, imports the FP32 family, compiles it with split constants and constant-load prefetch under the 32 MiB per-entry NSS limit, verifies one device image, and checks every exported entry.

Production readiness still requires checkpoint-specific accuracy, simulator/device numerics, entry-switch state, repeated sequences, TTFT, decode latency and memory measurements. Dynamic shapes, sliding-window execution and vision exports remain separate work.

The [batch-prefill size probe README](src/torq/model_export/doc/prefill_sizes.md) explains projection extraction, the recommendation flow, loop-order weight reuse, plateau selection, split-constant snapping and caching with Mermaid diagrams.
