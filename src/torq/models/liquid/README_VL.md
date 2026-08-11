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
| `vision_encoder.onnx` | SigLIP-style tower (`MultiHeadAttention`, `Resize`, `Compress`, `ScatterND`, dynamic `num_patches`). Exported as fp32 ONNX for CPU/ORT and left **dynamic on purpose** — `Resize`/`ScatterND` over a dynamic patch count is not chip-compilable, so it is the one component exempt from the static-shape verification (`_allows_dynamic_shapes`). The chip artifact comes from `--vision-res {128,256}`, which builds a *separate*, fully static `vision_encoder_<res>` component. (`--compile-vision` compiles this dynamic graph as-is; experimental.) |

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
source .venv/bin/activate
cd torq-tools-dev
export TORQ_COMPILER_PATH=/path/to/iree-build/third_party/iree/tools/torq-compile

torq-export-model liquid-vl \
  --models-dir models/liquid-2p5-450M-VL \
  --instruct-model \
  --convert-dtypes \
  --skip-validation
```

That produces the text decoder only. To produce the **whole board bundle** in
one run — single-token decoder, its lower-TTFT split, the static vision encoder
and the image-prefill parts — add the three artifact flags (see
[Artifacts](#artifacts-what-each-flag-produces)):

```sh
torq-export-model liquid-vl \
  --models-dir models/liquid-2p5-450M-VL \
  --instruct-model \
  --convert-dtypes \
  --skip-validation \
  --vision-res 256 \
  --split-decoder \
  --image-decoder-parts
```

Add `--skip-torq` to stop at the ONNX.

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
| `--vision-res {128,256}` | build + compile the **static** SigLIP encoder (`vision_encoder_<res>.vmfb`); 256 → 64 image tokens, 128 → 16 |
| `--split-decoder` | also emit `decoder_nolm.vmfb` + `lm_head.vmfb` (lower-TTFT split) |
| `--image-decoder-parts [N]` | build + compile the one-shot image-prefill decoder, split into N layer parts (bare = 2) |
| `--compile-vision` | compile the *dynamic* encoder as-is (experimental; dynamic shapes + exotic ops — prefer `--vision-res`) |
| `--skip-torq` | stop after ONNX export |
| `--onnx-source-dir` | use an existing source dir instead of the canonical one |

> Note on shape propagation: the LFM2-VL decoder's optimum export emits a
> per-conv-layer causal-trim subgraph (`Shape→Gather→Mul→Unsqueeze→Slice`).
> Each layer's `Shape` only resolves once the previous layer's output shape is
> concrete, so static-shape resolution proceeds one layer per iteration and
> needs ~10–12 passes across the 16-layer stack. The VL exporter raises the
> shape-fold iteration cap accordingly (the 350m path is untouched).

Output on disk after the **full-bundle** run above (a default run produces the
same tree minus the vision / split / image-part entries):

```
models/liquid-2p5-450M-VL/export/
├── onnx/
│   ├── fp32/static/
│   │   ├── decoder_model_merged.onnx      (~1.4 GB)
│   │   ├── vision_encoder.onnx            (~363 MB, fp32, dynamic — for ORT)
│   │   └── token_embeddings.npy           (~256 MB, fp32)
│   └── bf16/static/                        ← convert dir; also the build scratch
│       ├── decoder_model_merged.onnx      (~676 MB)
│       ├── decoder_nolm.onnx              (~548 MB)   --split-decoder
│       ├── lm_head.onnx                   (~128 MB)   --split-decoder
│       ├── vision_encoder_256.onnx        (~182 MB)   --vision-res 256 (compile input)
│       ├── vision_encoder_256.fp32.onnx   (~363 MB)   fp32 static build (ORT reference)
│       ├── decoder_image_2part_A.onnx     (~277 MB)   --image-decoder-parts
│       ├── decoder_image_2part_B.onnx     (~244 MB)   --image-decoder-parts
│       └── token_embeddings.npy           (~128 MB, bf16)
└── iree/bf16/static/                       ← board bundle (+ one .mlir per vmfb)
    ├── decoder_model_merged.vmfb          (~679 MB)   single-token decoder
    ├── decoder_nolm.vmfb                  (~550 MB)
    ├── lm_head.vmfb                       (~128 MB)
    ├── vision_encoder_256.vmfb            (~1.77 GB!)  static SigLIP encoder
    ├── decoder_image_2part_A.vmfb          (~336 MB)   one-shot image prefill
    ├── decoder_image_2part_B.vmfb          (~296 MB)
    ├── token_embeddings.npy               (~128 MB)   ← staged for the runner
    ├── config.json                                    ← staged (flat text config)
    └── tokenizer.json                                 ← staged
```

> **Open issue — vision vmfb size.** The 256-res encoder compiles to ~1.77 GB
> from a 182 MB bf16 input, while every other component lands ~1:1 with its
> ONNX. The bundle on HuggingFace ships it at roughly 200 MB, and the board has
> only ~1.9 GB of RAM, so this build is too big to deploy as-is. Prime suspect is
> `--torq-max-nss-programs-size 402653184`: `export_torq` adds it as soon as
> `--vision-res` *or* `--image-decoder-parts` is set, and the base exporter
> passes one flag list to *every* component, so the encoder gets a 384 MB program
> budget it may simply be filling. If confirmed, scope the flag to the
> image-decoder parts only. Compile the ONNX by hand without the flag to compare
> before trusting a `--vision-res` vmfb on the board.

Each compile also leaves its imported `<component>.mlir` next to the vmfb
(collectively ~3.7 GB for the full bundle); they are not needed on the board.
Compiles run **serially**, one component at a time — the full-bundle run above
took ~2 h 35 min on a 32-core host, ~2 h of which was the two 8-layer image
parts (see `image_prefill.md` §3 for the layers-per-part cost curve).

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
| `vision_encoder_256.vmfb` (64 tokens) / `vision_encoder_128.vmfb` (16 tokens) | `--vision-res 256` / `--vision-res 128` (static SigLIP encoder; compile is heavy but succeeds). One res per run. The bundle ships the 128 build under the legacy name `vision_encoder.vmfb`; rename after export if the runner is pointed at that name. |
| `decoder_image_2part_A/B.vmfb` (one-shot image prefill) | `--image-decoder-parts` (bare = 2-part, the shipping split; `3` / `5` are alternates) |
| `token_embeddings.npy`, `config.json`, `tokenizer.json` | staged automatically |

Every board artifact is reproducible from one `torq-export-model liquid-vl`
invocation — the three flags compose, so the full-bundle command in §1 emits all
six vmfbs in a single run. Compile is heavy for the vision encoder and the
image-decoder parts (`--torq-max-nss-programs-size` is raised automatically for
both).

> The vision encoder, the `decoder_nolm`/`lm_head` split and the image-decoder
> parts are all built during `convert_models`, i.e. *after* `export_onnx`'s
> static-shape verification loop has run, so each is checked by
> `_verify_static_build` as it is registered: graph I/O dims must be concrete and
> no tensor may carry a symbolic dim. It logs the verified I/O per component —
> a quick way to confirm e.g. that the 256-res encoder really is
> `pixel_values [1, 256, 768] -> [64, 1024]` (64 image tokens).

---

## 2. Deploy to the board (text decoder)

The `iree/bf16/static/` dir is self-contained for the LiquidStatic runner
(vmfb + LUT + config + tokenizer all staged):

```sh
M=models/liquid-2p5-450M-VL/export/iree/bf16/static
scp \
  $M/decoder_model_merged.vmfb \
  $M/token_embeddings.npy \
  $M/config.json \
  $M/tokenizer.json \
  <board-user>@<board-host>:/path/to/torq-examples/models/Synaptics/LFM2-VL-450M-torq/
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

- **Image/text merge in the runner.** Running the full VL model end-to-end
  (image → features → splice into the text embeds, image-first cache relocation)
  lives in the board runner (`torq-examples/liquidAI-VLM`), not this exporter.

> The dynamic vision encoder is *not* chip-compilable, but `--vision-res
> {128,256}` builds a **static** single-resolution encoder that is (see the
> Artifacts table above). Its compile is heavy/slow but succeeds.
>
> The image-prefill decoder (`--image-decoder-parts`) is built in **3D**
> (`[1,64,1024]`, cache-only, lm_head dropped) and split at layer boundaries.
> The "numerically broken" note in `torq-examples/RUN_ON_BOARD.md` describes the
> abandoned **rank-2** build; the 3D build this produces is numerically clean
> (cos ≥ 0.9999 vs ORT). See `torq-compiler-dev/image_prefill.md` for the full
> rationale.
