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

Install optional extras only for workflows that need them:

```bash
python -m pip install -e '.[moonshine]' --extra-index-url https://download.pytorch.org/whl/cpu
```

Moonshine Streaming is the exception: it requires `transformers` 5.x, which is
incompatible with the `optimum`-based exporters (they cap `transformers` below
4.58). Install it in a **separate** virtual environment from its own pinned
requirements file:

```bash
python3 -m venv .venv-moonshine-streaming
source .venv-moonshine-streaming/bin/activate
python -m pip install -r src/torq/models/moonshine_streaming/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

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


## Tools and Utilities

Tools live under `src/torq/tools/`, and shared helpers live under `src/torq/utils/`. For new or updated tools, it is easiest to keep the command line wrapper thin and put the behavior in importable functions. Then tests can call those functions directly, with CLI-level tests reserved for argument parsing, file wiring, or user-visible behavior.

For ONNX tools, tiny synthetic models in the test are preferred over checked-in binaries. For TFLite tools, generated fixtures in `tmp_path` are usually enough. Please update `README.md` when a command, option, output artifact, or supported workflow changes.

## Before Review

Run the focused tests for the code you touched, then broaden to the relevant suite when the change affects shared behavior. For Python changes, please remove unused imports, variables, classes, and functions before handing off.
