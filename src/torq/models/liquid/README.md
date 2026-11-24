# LFM2.5 (Liquid) export and compile

See the [model export guide](../README.md) for setup, common flags, compiler configuration, output variants and cache behavior.

End-to-end recipe to take an `LiquidAI/LFM2.5-*-ONNX` source model — **350M** (default) or **230M** — from HuggingFace, produce a Torq-ready bf16 ONNX, and compile it to a vmfb for the SL2610. Pass `--model-size 230m` for the smaller model.

> For the vision-language model **LFM2-VL-450M** (three ONNX components: decoder + vision encoder + embed_tokens), see [`README_VL.md`](README_VL.md). Its `torq-export-model liquid-vl` subcommand reuses this decoder pipeline.

---

## 0. Source model

The source is the LFM2.5-350M ONNX export. The exporter downloads it automatically on first run, so you do not normally need to fetch it yourself. It tries the Synaptics-hosted mirror [`Synaptics/LiquidAI-LFM2p5-350M-LLM`](https://huggingface.co/Synaptics/LiquidAI-LFM2p5-350M-LLM) first, then falls back to the upstream `LiquidAI/LFM2.5-350M-ONNX`. The mirror requires [authentication](../README.md#setup-and-pipeline); unauthenticated exports use the public upstream repository.

If you want it on disk ahead of time (offline export, inspection, or to avoid a re-download), place it at the canonical source location the exporter reads from:

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
source .venv/bin/activate
cd torq-tools-dev

python src/torq/models/liquid/download_source.py \
  --models-dir models
```

The download helper reuses files already present.

Or, if you already have the model cached elsewhere (e.g. under `~/.cache/huggingface/`), point the exporter straight at it instead of the canonical location with `--onnx-source-dir <dir-containing-model.onnx>` (see the [export flags](../README.md#common-export-flags)).

---

## 1. Export and compile

Export and compile the default 350M model:

```sh
source .venv/bin/activate
cd torq-tools-dev

torq-export-model liquid --models-dir models
```

The command produces the static split-lm-head ONNX set (embedding extraction and a 64-token batched prefill on) and then compiles it to a vmfb under `export/split_lm_head/fp32/static/compiled/`. Pass `--convert-dtypes` to compile the converted bf16 variant instead (the VMFB is dtype-converted by the compiler either way). Pass `--skip-torq` to stop at the ONNX.

Model-specific flags:

| flag | meaning |
|---|---|
| `-s, --model-size {350m,230m}` | which LFM2.5 size to export (default: `350m`). `230m` pulls from `LiquidAI/LFM2.5-230M-ONNX` and reads/writes under `<dir>/liquid-2p5-230m/` |

Opt-out flags for the chip-specific rewrites:

| flag | meaning |
|---|---|
| `--keep-conv1d` | Leave the original depthwise Conv1D in place for CPU/ORT targets |
| `--chunk-lm-head` | Use the legacy 512-chunk LM-head split for compilers without tile-and-fuse |

Split and unified exports use separate output directories (`export/unified/…` vs `export/split_lm_head/…`):

```
models/liquid-2p5-350m/
├── source/onnx/fp32/model.onnx
└── export/<unified|split_lm_head>/    (default: split_lm_head; --no-split-lm-head picks the former)
    ├── fp32/static/
    │   ├── model.onnx                (~1.4 GB; unified export only)
    │   ├── transformer.onnx          (~1.4 GB; split export body with last_hidden_states output)
    │   ├── lm_head.onnx              (split export, on by default; ~268 MB fp32)
    │   ├── transformer_prefill.onnx  (default: 64-token prefill; --batch-prefill 0 disables; stays fused)
    │   ├── token_embeddings.npy      (~128 MB, fp32; for host inference)
    │   ├── config.json
    │   ├── tokenizer.json
    │   └── compiled/                 (only without --skip-torq; one .vmfb + .mlir per component)
    ├── quantized/static/             (only with --dynamic-quantize; int8 copy of every component)
    └── bf16/static/                  (only with --convert-dtypes)
```

See [output variants and runtime assets](../README.md#output-variants-and-runtime-assets) and [validation](../README.md#validation) for deployment file requirements and dtype restrictions.

## 2. Compile: bf16 ONNX → Torq vmfb

The export command already compiles the selected variant. See [compiler setup and standalone compilation](../README.md#compiler-setup-and-flags).

[`compile_flags.json`](compile_flags.json) provides the validated compiler settings, including split-constants optimization for weight access.

---

## 3. Batched prefill

See the [batched prefill guide](../README.md#batched-prefill). Liquid keeps the prefill LM head fused and handles short-conv state alongside the attention caches.
