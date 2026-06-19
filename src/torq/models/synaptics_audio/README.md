<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.
-->

# `torq.models.synaptics_audio`

Prepare Synaptics audio models (AEC-VAD, NNNR, SED, Voice-Filter, and the
Voice-Filter speaker embedder) for the Torq compiler with a single fixed
pipeline:

```
FP32 ONNX  ->  simplified FP32 ONNX  ->  (verify equivalence)  ->  BF16 ONNX
```

There is one entry point (`prepare`), one fixed sequence of value-preserving
rewrites built on top of `torq.graph_edit.OnnxGraphEditor`, and one declarative
description per recipe (`recipes.*`). Recipes may declare multiple HF source
filenames, but the preparation pipeline stays the same.

## Quickstart

```bash
# Auto-fetch source ONNX file(s) from HuggingFace using the recipe's repo_id.
# Passing a directory keeps each source filename stem and appends _torq_bf16.
python -m torq.models.synaptics_audio voice_filter .

# Convert the Voice-Filter speaker embedder. This recipe pins feats to
# [1, 100, 40] because the source ONNX declares dynamic B/T dimensions.
python -m torq.models.synaptics_audio voice_filter_speaker_embedder .

# Or supply your own local source FP32 ONNX:
python -m torq.models.synaptics_audio voice_filter . \
    --src /path/to/local_fp32.onnx
```

```python
# Programmatic
from pathlib import Path
from torq.models.synaptics_audio import BY_KEY
from torq.models.synaptics_audio.prepare import prepare

# Auto-fetch from HF, writing one output per declared source filename:
prepare(BY_KEY["voice_filter"], Path("."))

# With an explicit source path, writing ./local_fp32_torq_bf16.onnx:
prepare(
    BY_KEY["voice_filter"],
    Path("."),
    src=Path("local_fp32.onnx"),
)
```

Input shapes are **auto-discovered** from `model.graph.input`. Output shapes
are derived by `shape_inference` once inputs are fixed. The recipe declares
only the recipe identity (HF `repo_id`, source filenames inside the repo) plus
an optional dynamic-input safety hatch.

## Pipeline

The simplification sequence lives in
`prepare._SynapticsAudioGraphEditor.run_audio_pipeline`. Every step is
**idempotent** and **self-skipping** -- if its pattern doesn't appear in the
model, it leaves the graph unchanged:

| # | Step | Source | Purpose |
|--:|------|--------|---------|
| 1 | `apply_fixed_input_shapes`              | `OnnxGraphEditor`            | Stamp explicit `Recipe.input_shape_overrides` onto `graph.input` (no-op when overrides are empty -- the common case). |
| 2 | `freeze_shape_seeds`                    | `OnnxGraphEditor`            | Constant-fold shape-computation subgraphs feeding `Reshape` / `Expand` / `Slice` / `Pad` controls. |
| 3 | `EliminateTranspose`                    | `graph_edit.edits.shape`     | Remove `Transpose` nodes that don't physically rearrange data (handles singleton-axis Transposes). |
| 4 | `DecomposeBidirectionalRnn`             | `graph_edit.edits.rnn`       | Split bidirectional `GRU` / `LSTM` / `RNN` into two unidirectional forward layers (works around torch-mlir). |
| 5 | `EliminateRank0Gather`                  | `graph_edit.edits.shape`     | Rewrite `Gather` ops producing rank-0 scalars + `Unsqueeze[0]` into a rank-1 path (works around an IREE codegen bug). |
| 6 | `RewriteNegativePads`                   | `graph_edit.edits.padding`   | Rewrite `Pad` ops with negative (crop) paddings into `Pad(positive) + Slice`. |
| 7 | `AbsorbPadding`                         | `graph_edit.edits.padding`   | Fuse non-negative `Pad` layers into the following `Conv`'s `pads` attribute. |
| 8 | `EliminateSingletonGatherUnsqueeze`     | `graph_edit.edits.shape`     | Remove singleton-axis `Gather -> unary -> Unsqueeze` rank shims. |
| 9 | `WidenStridedDepthwiseConv`             | `graph_edit.edits.conv`      | Widen narrow strided-depthwise `Conv`s so the Torq compiler avoids the DEDR scatter-gather codegen path. |
| 10 | `finalize_torq_ready_onnx`             | `torq.utils.onnx`            | Symbolic shape inference, `value_info` cleanup, IR-version cap (run after the editor exports). |

Order matters: shape-resolving steps first (1-2), shape-aware rewrites in the
middle (3-9), finalization last (10). The runner has no per-model branching --
self-skipping rewrites are how recipe-specific behavior is achieved.

## Verification

After all passes have run, `torq.utils.onnx_verify.verify_equivalence` runs
both the source FP32 model and the simplified FP32 model on the same random
inputs (seeded, shapes from the recipe) via `onnxruntime` CPU and asserts the
outputs match within tight FP32 tolerances (`atol=1e-5`, `rtol=1e-4`).

This is the **safety net** that catches any pass that is not value-preserving.
BF16 numeric loss is a separate concern and is **not** verified here -- it is
the responsibility of `torq.tools.convert_dtype.onnx`, which is invoked
afterwards to do the FP32 -> BF16 conversion.

## Recipes

A recipe is a frozen dataclass that declares one conversion target. See
`recipes/base.py`:

```python
@dataclass(frozen=True, slots=True)
class Recipe:
    key: str                                                  # CLI-facing identifier
    repo_id: str                                              # HuggingFace repo identifier
    source_filename: str | tuple[str, ...] | None = None      # source ONNX path(s) inside the HF repo
    input_shape_overrides: Mapping[str, Sequence[int]] = {}   # rare safety hatch
```

Recipes carry only the recipe identity and source locations. Input shapes come
from the selected source ONNX's `graph.input`; output shapes come from
`shape_inference` once inputs are pinned. The optional
`input_shape_overrides` field is only needed if the source ONNX has dynamic
input dim_params.

| key | HF repo | source ONNX inside repo |
|-----|---------|-------------------------|
| `voice_filter` | `Synaptics/Voice-Filter` | `baseline125K/VF_0290_19.onnx` |
| `voice_filter` | `Synaptics/Voice-Filter` | `baseline125KReLU6/VF_0075_11.8389.onnx` |
| `voice_filter` | `Synaptics/Voice-Filter` | `baseline423K/VF_0126_19.0517.onnx` |
| `voice_filter_speaker_embedder` | `Synaptics/Voice-Filter` | `baseline125K/model_epoch_0290_19.0540_speaker_embedder.onnx` |
| `aec_vad` | `Synaptics/AI-VAD` | `standalone_2ch_DT_VAD_python_0909/2025-02-11_01-38-13_aec_vad_exp12_d4_model_epoch_t710.onnx` |
| `sed` | `Synaptics/SED` | `Vivint_GB_SED_v3.57.902.150_e365_0.941/model[Vivint_GB_SED_v3.57.902.150].onnx` |
| `nnnr` | `Synaptics/NNNR3` | `NNNR3_0079_0.0960.onnx` |

## Fetching from HuggingFace

`prepare(...)` will auto-fetch the source ONNX when `--src` (CLI) or `src=`
(programmatic) is omitted:

```python
from torq.models.synaptics_audio import BY_KEY, fetch_source, fetch_sources

# Just download (no preparation):
local_path = fetch_source(BY_KEY["voice_filter"])
print(local_path)  # ~/.cache/huggingface/hub/.../VF_0290_19.onnx

local_paths = fetch_sources(BY_KEY["voice_filter"])
print(local_paths)  # all declared Voice Filter source ONNXs
```

Files are cached by `huggingface_hub` under `$HF_HOME` (default
`~/.cache/huggingface/hub/`); repeated calls are a no-op on cache hit. Pass
`cache_dir=...` to override the location. Private repos require an HF token
(`huggingface-cli login` or `$HUGGING_FACE_HUB_TOKEN`).

### Adding a new recipe

1. Drop a new `recipes/<model>.py`:

   ```python
   from .base import Recipe

   MY_MODEL = Recipe(
       key="my_model",
       repo_id="Synaptics/My-Model",
       source_filename=(
           "path/inside/repo/model.onnx",
           "path/inside/repo/larger_model.onnx",
       ),
   )
   ```

2. Register it in `recipes/__init__.py`: import the constant and add it to the
   `ALL` tuple. `BY_KEY` and the CLI recipe choices update automatically.

That's it -- no code changes elsewhere. The pipeline is the same for every
recipe.

### Source ONNX has dynamic inputs?

`prepare` will raise:

```
ValueError: source ONNX has dynamic input dims that cannot be auto-resolved:
foo=[1, '?', 256]. Provide Recipe.input_shape_overrides for the affected input(s).
```

Then add an override:

```python
MY_MODEL = Recipe(
    key="my_model",
    repo_id="Synaptics/My-Model",
    source_filename="model.onnx",
    input_shape_overrides={"foo": (1, 16, 256)},
)
```

For the Voice-Filter speaker embedder, the source input is `feats=[B, T, 40]`.
The checked-in recipe fixes it to `feats=(1, 100, 40)`, producing
`embs=[1, 256]`. Change that override if the runtime uses a different fixed
feature-frame count.

### Adding a new rewrite

If the rewrite is generic (could plausibly be useful to another exporter), add
it as an `OnnxGraphEdit` subclass in the appropriate `torq.graph_edit.edits`
submodule -- for example `shape`, `padding`, `conv`, or `rnn`. Then call it
from `run_audio_pipeline` at the appropriate position (shape-resolving first,
structural rewrites in the middle).

If the rewrite is whole-graph (not a per-node pattern -- e.g. a constant-fold
that needs an ORT round-trip), add it as a method on `OnnxGraphEditor` next to
`apply_fixed_input_shapes` / `freeze_shape_seeds`.

## Layout

```
synaptics_audio/
├── __init__.py            # public API (recipes registry, Recipe, fetch_source)
├── __main__.py            # CLI entry point
├── prepare.py             # source fetch, simplify, verify, BF16 conversion
├── fetch.py               # HuggingFace Hub source-ONNX downloader
└── recipes/
    ├── __init__.py        # ALL / BY_KEY
    ├── base.py            # Recipe dataclass
    └── *.py               # one recipe per file
```
