# LFM2.5 (Liquid) export and compile

End-to-end recipe to take an `LiquidAI/LFM2.5-*-ONNX` source model — **350M**
(default) or **230M** — from HuggingFace, produce a Torq-ready bf16 ONNX, and
compile it to a vmfb for the SL2610. Pass `--model-size 230m` for the smaller
model; everything else is identical (the exporter reads the architecture —
layer count, hidden size, layer types — from the model's `config.json`).

This page covers **export** and **compile**, plus deploying the resulting vmfb
to the board and a basic on-board run (see the end).

> For the vision-language model **LFM2-VL-450M** (three ONNX components:
> decoder + vision encoder + embed_tokens), see [`README_VL.md`](README_VL.md).
> Its `torq-export-model liquid-vl` subcommand reuses this decoder pipeline.

---

## 0. Source model

The source is the LFM2.5-350M ONNX export. The exporter downloads it
automatically on first run, so you do not normally need to fetch it yourself.
It tries the Synaptics-hosted mirror
[`Synaptics/LiquidAI-LFM2p5-350M-LLM`](https://huggingface.co/Synaptics/LiquidAI-LFM2p5-350M-LLM)
first — so the pipeline does not break if the upstream repo is renamed or
removed — then falls back to the upstream `LiquidAI/LFM2.5-350M-ONNX`. The
Synaptics repo is **private**, so set `HF_TOKEN` (or run `huggingface-cli
login`) to use it; without a token the exporter falls through to the public
upstream repo.

If you want it on disk ahead of time (offline export, inspection, or to
avoid a re-download), place it at the canonical source location the exporter
reads from:

```
models/liquid-2p5-350m/source/onnx/fp32/
├── model.onnx           (~140 KB — graph only)
├── model.onnx_data      (~1.45 GB — external weights)
├── config.json
├── tokenizer.json
└── tokenizer_config.json
```

Download it there with the helper script:

```sh
source /home/kshanmug/torq/.venv-torq-tools/bin/activate
cd /home/kshanmug/torq/torq-tools-dev

python src/torq/models/liquid/download_source.py \
  --models-dir /home/kshanmug/torq/torq-tools-dev/models
```

(`download_source.py` is self-contained — no compiler-toolchain imports — and
skips any files already present, so it's safe to re-run.)

Or, if you already have the model cached elsewhere (e.g. under
`~/.cache/huggingface/`), point the exporter straight at it instead of the
canonical location with `--onnx-source-dir <dir-containing-model.onnx>` (see
the flag table below).

---

## 1. Export: HF source → bf16 ONNX

The exporter is the `liquid` subcommand of `torq-export-model` (installed by
this package). It downloads the HF model on first run (or reuses the source
from Section 0), applies all the LFM2.5 graph fix-ups required for the
SL2610, and emits both fp32 and bf16 static ONNX models.

Graph transforms applied automatically (each one used to be a separate
host-side script; they now live as static methods on
[`LiquidModelExporter`](export.py)):

- collision-aware tensor-name sanitization
- iterative shape propagation + negative-Slice resolution
- **Conv1D → batched-MatMul** replacement (the SL2610's depthwise-conv path
  takes an IOCTL timeout; bit-exact `Slice + Reshape + MatMul + Concat`
  chain replaces it)
- zero-bias injection into bias-less Conv ops (only those we couldn't
  replace above)
- **lm_head**: a single `[1024, 65536]` MatMul against a pre-transposed
  weight constant (tile-and-fuse handles it; no export-time chunking)
- optional `token_embedding` extraction with `--extract-embeddings`
- custom-op replacement for `GroupQueryAttention` /
  `SimplifiedLayerNormalization`

```sh
source /home/kshanmug/torq/.venv-torq-tools/bin/activate
cd /home/kshanmug/torq/torq-tools-dev

torq-export-model liquid \
  --models-dir /home/kshanmug/torq/torq-tools-dev/models \
  --instruct-model \
  --convert-dtypes \
  --extract-embeddings
```

Like the other models (gemma3, smollm2), this single command **exports and
compiles**: it produces the bf16 ONNX and then compiles it to a vmfb under
`export/iree/bf16/static/`. Pass `--skip-torq` to stop at the ONNX.

Flag breakdown:

| flag | meaning |
|---|---|
| `liquid` | subcommand (registered in [`models/export_model.py`](../export_model.py)) |
| `-s, --model-size {350m,230m}` | which LFM2.5 size to export (default: `350m`). `230m` pulls from `LiquidAI/LFM2.5-230M-ONNX` and reads/writes under `<dir>/liquid-2p5-230m/` |
| `--models-dir` | base dir; the exporter reads from `<dir>/liquid-2p5-<size>/source/` and writes to `<dir>/liquid-2p5-<size>/export/` |
| `--onnx-source-dir` | use an existing source ONNX directory instead of the canonical `<models-dir>/.../source/onnx/fp32/` (e.g. a HF cache snapshot dir). Skips the auto-download. |
| `--instruct-model` | use the instruction-tuned variant (this is what enables ChatML at inference) |
| `--convert-dtypes` | emit bf16 alongside fp32 (driven by the `convert_dtypes=["bf16","fp16"]` list passed to `add_onnx_args` in `__init__.py`) |
| `--extract-embeddings` | replace the embedding `Gather` with a `token_embedding` graph input and dump `token_embeddings.npy` (CPU-side LUT). Required for the demo runner. |
| `--skip-torq` | stop after the ONNX export; do not compile to a vmfb |
| `--compile-flags …` | extra flags forwarded to `torq-compile` (must be last). The liquid export already adds `--torq-enable-transpose-optimization --torq-enable-split-constants-optimization`. |

Two opt-out flags for the chip-specific rewrites:

| flag | meaning |
|---|---|
| `--keep-conv1d` | leave the original depthwise Conv1D in place (useful for CPU/ORT targets) |
| `--split-lm-head` | revert to the legacy 512-chunk lm_head split (only needed for torq without tile-and-fuse) |

Output on disk after a successful run:

```
models/liquid-2p5-350m/
├── source/onnx/fp32/model.onnx        (~1.4 GB — original HF safetensors, converted)
└── export/onnx/
    ├── fp32/static/
    │   ├── model.onnx                (~1.4 GB)
    │   ├── token_embeddings.npy      (~128 MB)
    │   ├── config.json
    │   └── tokenizer.json
    └── bf16/static/
        ├── model.onnx                (~700 MB)
        ├── token_embeddings.npy      (~128 MB)
        ├── config.json
        └── tokenizer.json
```

> Note: the fp32 export is the right artifact to validate end-to-end through
> onnxruntime ("What is the capital of France?" → "The capital of France is
> Paris."). bf16 cannot be validated through ORT because the CPU MatMul
> kernel has no bf16 path; compare bf16 against fp32 via the host casting
> tools if you need a quality check.

---

## 2. Compile: bf16 ONNX → Torq vmfb

Same as gemma3/smollm2: the Section 1 export command already compiles (unless
you pass `--skip-torq`), writing `model.vmfb` to
`export/iree/bf16/static/`. Compilation goes through the shared
`torq.utils.compile` driver (ONNX → MLIR via `iree-import-onnx`, then
MLIR → vmfb via `torq-compile`), and the liquid export adds
`--torq-enable-transpose-optimization --torq-enable-split-constants-optimization`
on top of `torq-compile`'s SL2610 defaults.

To compile a standalone ONNX/MLIR later (e.g. a diagnostic variant), use the
same driver directly (the output directory is created automatically):

```sh
export TORQ_COMPILER_PATH=/home/kshanmug/torq/iree-build/third_party/iree/tools/torq-compile
python -m torq.utils.compile \
  models/export/onnx/bf16/static/model.onnx \
  -o models/export/iree/bf16/static/model.vmfb \
  --compile-flags --torq-enable-transpose-optimization --torq-enable-split-constants-optimization
```

Notes:
- **`--torq-enable-split-constants-optimization`** is kept on for liquid: we
  measured it faster (303 ms vs 432 ms/step) and lower-heap because each
  dispatch reads its constant slice straight from the mmap'd vmfb instead of
  staging the whole blob into anonymous DRAM.
- For per-pass timing on a slow/hung compile, append `--torq-show-pass-progress`
  to `--compile-flags`.
- The compiler binary is resolved via `TORQ_COMPILER_PATH` (default:
  `torq-compile` on `PATH`).

Output: `model.vmfb` (~712 MB for the full bf16 build, ~553 MB without
lm_head, ~258 MB without FFN).

---