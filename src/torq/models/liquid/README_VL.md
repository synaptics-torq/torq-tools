# LFM2-VL-450M (Liquid vision-language) export and compile

End-to-end recipe to take the three-part `LiquidAI/LFM2-VL-450M` ONNX export
and produce Torq-ready bf16 ONNX + a compiled vmfb for the SL2610.

This is the vision-language sibling of the text-only LFM2.5-350M flow (see
[`README.md`](README.md)). It reuses the entire LFM2.5 chip-rewrite toolbox for
the language decoder; the differences are all in the multi-component layout.

---

## 0. Source model

The three ONNX components are auto-downloaded on first run: the exporter tries
the Synaptics-hosted mirror
[`Synaptics/liquidAI-LFM2-VLM`](https://huggingface.co/Synaptics/liquidAI-LFM2-VLM)
first — so the pipeline doesn't break if the upstream is renamed or removed —
then falls back to the public community export
[`onnx-community/LFM2-VL-450M-ONNX`](https://huggingface.co/onnx-community/LFM2-VL-450M-ONNX).
The Synaptics mirror is **private**, so set `HF_TOKEN` (or run `huggingface-cli
login`) to use it; without a token the exporter falls through to the public
upstream. config / tokenizer / chat template come from the base repo
[`LiquidAI/LFM2-VL-450M`](https://huggingface.co/LiquidAI/LFM2-VL-450M) (or the
local 350m config fallback).

Unlike LFM2.5-350M (a single `model.onnx`), LFM2-VL ships **three** components.
They are fetched to (or can be placed manually at) the canonical source
location:

```
models/liquid-2p5-450M-VL/source/onnx/fp32/
├── decoder_model_merged.onnx        (~140 KB graph)
├── decoder_model_merged.onnx_data   (~1.45 GB external weights)
├── vision_encoder.onnx              (~120 KB graph)
├── vision_encoder.onnx_data         (~360 MB external weights)
└── embed_tokens.onnx                (~257 MB, single Gather)
```

| component | role |
|---|---|
| `embed_tokens.onnx` | `Gather(weight[65536, 1024], input_ids) -> inputs_embeds`. The token-embedding LUT. Extracted to `token_embeddings.npy` (CPU-side, like the 350m `--extract-embeddings` flow); **never compiled**. |
| `decoder_model_merged.onnx` | The LFM2 hybrid conv + attention decoder. Takes `inputs_embeds` directly — i.e. it is the 350m decoder *after* embedding extraction. Architecturally identical to LFM2.5-350M (16 layers: 6 GQA attention + 10 short-conv, hidden 1024, vocab 65536, 16/8 heads, head_dim 64, conv_L_cache 3). **Primary chip target.** |
| `vision_encoder.onnx` | SigLIP-style tower (`MultiHeadAttention`, `Resize`, `Compress`, `ScatterND`, dynamic `num_patches`). Exported as ONNX for CPU/ORT; **chip compile is experimental and off by default** (`--compile-vision`). |

> The source repo has no `config.json` / `tokenizer.json` next to the ONNX.
> The exporter resolves the text-decoder architecture from the VL repo's
> `text_config` if available, otherwise falls back to the local
> `liquid-2p5-350m` config (the text tower is identical). For deployment it
> also stages a tokenizer from the source dir → local 350m → HF.

---

## 1. Export + compile

The `liquid-vl` subcommand of `torq-export-model` exports all three
components, extracts the embedding LUT, makes the decoder static, applies the
LFM2.5 chip rewrites, converts to bf16, and compiles the decoder to a vmfb —
all in one command:

```sh
source /home/kshanmug/torq/.venv-torq-tools/bin/activate
cd /home/kshanmug/torq/torq-tools-dev
export TORQ_COMPILER_PATH=/home/kshanmug/torq/iree-build/third_party/iree/tools/torq-compile

torq-export-model liquid-vl \
  --models-dir models/liquid-2p5-450M-VL \
  --instruct-model \
  --convert-dtypes \
  --skip-validation
```

Add `--skip-torq` to stop at the ONNX, or `--compile-vision` to also attempt
the vision-encoder vmfb (experimental).

Graph transforms applied to the **decoder** (reused verbatim from the 350m
exporter via subclassing): `SimplifiedLayerNormalization` /
`SkipSimplifiedLayerNormalization` / `GroupQueryAttention` custom-op
replacement, static KV + conv cache, `position_ids` input, Conv1D→batched-MatMul,
single `[1024, 65536]` lm_head. The decoder's `inputs_embeds` input is renamed
`token_embedding` and pinned as input 0 (with `position_ids` at 1) so the
existing `LiquidStatic` runner / chip demo feed it unchanged.

VL-specific flags (everything else matches `torq-export-model liquid`):

| flag | meaning |
|---|---|
| `--models-dir` | base dir; reads `<dir>/source/onnx/fp32/` and writes `<dir>/export/` |
| `--compile-vision` | also compile the SigLIP vision encoder (experimental; dynamic shapes + exotic ops) |
| `--skip-torq` | stop after ONNX export |
| `--onnx-source-dir` | use an existing source dir instead of the canonical one |

> Note on shape propagation: the LFM2-VL decoder's optimum export emits a
> per-conv-layer causal-trim subgraph (`Shape→Gather→Mul→Unsqueeze→Slice`).
> Each layer's `Shape` only resolves once the previous layer's output shape is
> concrete, so static-shape resolution proceeds one layer per iteration and
> needs ~10–12 passes across the 16-layer stack. The VL exporter raises the
> shape-fold iteration cap accordingly (the 350m path is untouched).

Output on disk after a successful run:

```
models/liquid-2p5-450M-VL/export/
├── onnx/
│   ├── fp32/static/
│   │   ├── decoder_model_merged.onnx   (~1.4 GB)
│   │   ├── vision_encoder.onnx         (~360 MB, fp32)
│   │   └── token_embeddings.npy        (~257 MB, fp32)
│   └── bf16/static/
│       ├── decoder_model_merged.onnx   (~677 MB)
│       └── token_embeddings.npy        (~129 MB, bf16)
└── iree/bf16/static/
    ├── decoder_model_merged.vmfb       (~679 MB)   ← chip artifact
    ├── token_embeddings.npy            (~129 MB)   ← staged for the runner
    ├── config.json                                 ← staged (flat text config)
    └── tokenizer.json                              ← staged
```

bf16 cannot be validated through onnxruntime (no CPU bf16 MatMul kernel), so
the exporter skips automatic validation; sanity-check the fp32 decoder via the
runner in `torq-examples` with `token_embeddings.npy`.

### Artifacts (what each flag produces)

The board's deployment bundle (downloaded from `Synaptics/liquidAI-LFM2-VLM`)
is reproducible from this command, one flag per artifact:

| board vmfb / file | produced by |
|---|---|
| `decoder_model_merged.vmfb` (a.k.a. `decoder_main.vmfb`) | default (the merged decoder + lm_head) |
| `decoder_nolm.vmfb` + `lm_head.vmfb` | `--split-decoder` (lower-TTFT body/lm_head split) |
| `vision_encoder_256.vmfb` (64 tokens) / `vision_encoder.vmfb` (16 tokens) | `--vision-res 256` / `--vision-res 128` (static SigLIP encoder; compile is heavy but succeeds) |
| `token_embeddings.npy`, `config.json`, `tokenizer.json` | staged automatically |

Not reproducible from a flag: the one-shot image-prefill decoders
(`decoder_image_*`) — see the last section.

---

## 2. Deploy to the board (text decoder)

The `iree/bf16/static/` dir is self-contained for the LiquidStatic runner
(vmfb + LUT + config + tokenizer all staged):

```sh
M=/home/kshanmug/torq/torq-tools-dev/models/liquid-2p5-450M-VL/export/iree/bf16/static
scp \
  $M/decoder_model_merged.vmfb \
  $M/token_embeddings.npy \
  $M/config.json \
  $M/tokenizer.json \
  root@10.3.10.55:/home/root/torq-examples/models/Synaptics/LFM2-VL-450M-torq/
```

Then run the text decoder on the board with the LFM2.5 runner, pointing `-m`
at the VL decoder vmfb (the runner feeds `token_embedding` from the staged
LUT, so the text path works as-is):

```sh
cd ~/torq-examples/liquid
python src/infer.py \
  -m ../models/Synaptics/LFM2-VL-450M-torq/decoder_model_merged.vmfb \
  --instruct-model
```

The image path (vision encoder → image features → merge into the embedding
stream at the image-token positions) is **not** wired into the chip runner;
the vision encoder is exported as ONNX only. Compiling and integrating it is
follow-up work.

---

## Sanity-checking in onnxruntime

Two host-side scripts validate the exported components on a real image (no
board needed). Both reimplement the official `Lfm2VlImageProcessorFast`
preprocessing in numpy/PIL — native-resolution resize within 512×512, tiling
+ thumbnail for larger images, no upscaling — because the HF fast processor
needs torchvision, which is ABI-incompatible with this venv's custom torch
build. (Requires `pillow`.)

```sh
# vision encoder only: stats + content-sensitivity
python scripts/run_vision_encoder.py img1.jpg img2.jpg

# full pipeline: image -> caption (vision -> embed-splice -> decoder decode)
python scripts/run_vl_e2e.py img.jpg --prompt "What is in this image?"
```

Notes:
- The exported `vision_encoder` must be run **one sub-image at a time**
  (batch=1): the position-embedding interpolation can't broadcast a batch of
  tiles with different `spatial_shapes`. The scripts loop per sub-image and
  concatenate the feature tokens.
- The number of `<image>` (id 396) placeholders per sub-image equals the
  encoder's per-sub-image output token count: `ceil(h/2)*ceil(w/2)` for a
  `[h, w]`-patch sub-image (the 2× connector downsample). The merge splices
  the concatenated features into those positions in order (tiles row-major,
  then thumbnail).
- `run_vl_e2e.py` uses the **dynamic source decoder** (`source/.../decoder_model_merged.onnx`,
  which ORT runs via its `com.microsoft` contrib ops) so a real ~250-image-token
  prompt fits. The exported static decoder is the same computation but has a
  compile-time-fixed KV cache (256 by default — raise `--max-gen-tokens` to fit
  image prompts before compiling for the chip).

Observed output (greedy): the COCO two-cats image →
*"two cats sleeping on a pink blanket … two remote controls"*; a stock dog
photo (split into 6 tiles + thumbnail) → *"a dog on the grass … plants in the
background"*. This confirms the vision-encoder export, the embedding-LUT
extraction, and the feature-splice are correct end to end.

## Hybrid run: vision encoder (host ORT) + decoder (Torq board)

The vision encoder and decoder are run as **separate processes** with the
result saved in between — running onnxruntime (the ~360 MB vision encoder) and
the ~700 MB decoder vmfb together would OOM the ~1.9 GB board.

```sh
# 1. host: vision encoder in ORT -> prompt embeds (image features spliced in)
python scripts/prep_vl_decoder_inputs.py img.jpg --out /tmp/vl_board \
    --prompt "What is in this image?"
# 2. scp the staged vmfb / token_embeddings.npy / tokenizer.json plus
#    /tmp/vl_board/{prompt_embeds.npy,meta.json} and scripts/board_vl_decode.py
#    to a dir on the board, then:
# 3. board: decoder on Torq (no ORT here)
python -u board_vl_decode.py .
```

`board_vl_decode.py` mirrors the LFM2.5-350M demo's memory model exactly:
- `ManagedSelfAttnCacheRunner` with the **default `preload`** load (preload is
  what exposes the I/O reflection metadata the runner needs to size the caches;
  `mmap` does not expose it). The 16 per-layer conv+KV caches are kept
  **on the torq device** (`device_outputs=True`) and reused each step — the host
  never re-uploads them.
- the 134 MB token-embedding LUT is **mmap'd** (`np.load(mmap_mode="r")`), not
  loaded into RAM. This is the one thing to get right: loading the LUT into RAM
  on top of the preload'd vmfb pushes the board over and it OOMs (SSH hangs at
  banner exchange while it still pings). With the LUT mmap'd, load leaves
  ~350 MB free and per-step memory is flat.
- the static decoder runs **one token per invocation**, so a single image
  inference = `prompt_len + (generated − 1)` decoder calls (e.g. 259 prefill +
  37 decode = 296 for the COCO image). `position_ids` is **int32** in the vmfb
  (after `--torq-convert-io-dtype`); take dtypes from the vmfb metadata.

Observed (COCO image, 512 KV cache, single board): the Torq decoder generates
the **same caption** as the host ORT reference — *"two cats sleeping on a pink
blanket … two remote controls"* — at ~0.54 s/decoder-call (TTFT ≈ 140 s for the
259-token image prompt, decode ≈ 1.8 tok/s).

## What this exporter does *not* do (yet)

- **Image-prefill decoders** (`decoder_image_{2,3,5}part_*.vmfb`,
  `decoder_image_full.vmfb`). The board's one-shot image-prefill decoders have
  no in-repo builder, and the deployed ones are numerically broken on the NPU
  (NaN/overflow from layer 0 — see `torq-examples/RUN_ON_BOARD.md`). Reproducing
  them needs dedicated work on the layer-split build plus a numerical fix; no
  flag produces them.
- **Image/text merge in the runner.** Running the full VL model end-to-end
  (image → features → splice into the text embeds) needs runner changes in
  `torq-examples`.

> The dynamic vision encoder is *not* chip-compilable, but `--vision-res
> {128,256}` builds a **static** single-resolution encoder that is (see the
> Artifacts table above). Its compile is heavy/slow but succeeds.
