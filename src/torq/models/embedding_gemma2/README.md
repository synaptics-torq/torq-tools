# EmbeddingGemma-2 export for Torq

`torq-export-model embeddinggemma2` builds static Torq graphs for
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (text, image,
video and audio embeddings) and compiles them to VMFBs. The source is the
[onnx-community ONNX export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX),
downloaded from the unchanged copy in
[Synaptics/Google-EmbeddingGemma-2](https://huggingface.co/Synaptics/Google-EmbeddingGemma-2)
(`onnx/`). Files missing there are taken from onnx-community.

The compiled files are what the
[torq-examples EmbeddingGemma-2 demo](https://github.com/synaptics-torq/torq-examples)
downloads and runs.

## Quick start

```sh
pip install -e '.[embeddinggemma2]'      # onnxsim and Pillow

torq-export-model embeddinggemma2 --weights fp32   # bf16 weights   -> models/embeddinggemma-2/export/
torq-export-model embeddinggemma2 --weights q4     # int4 block-32  -> models/embeddinggemma-2/export-q4/
torq-export-model embeddinggemma2 --weights q8     # int8 block-32  -> models/embeddinggemma-2/export-q8/
```

Each run builds every network, checks it against its source graph in ONNX Runtime, and
compiles it. A full run takes several hours. `--no-compile` stops after the bf16 ONNX
graphs, and `--skip-validate` skips the ONNX Runtime check. Compiling the q8 vision encoder
takes the longest (about 40 to 50 minutes).

| Option | Default | Meaning |
|---|---|---|
| `--weights {fp32,q4,q8}` | `fp32` | Source weights. `fp32` gives bf16 weights on Torq. `q4` and `q8` start from the onnx-community `*_q4` and `*_quantized` graphs and keep their block-32 weights as `DequantizeLinear(INT4/INT8) -> MatMul`. Activations are bf16 in all three. |
| `--seq-lens` | `128,512` | Static text lengths |
| `--image-sizes` | `384x384` | Static vision input sizes (multiples of 48 px) |
| `--audio-frames` | `1120,560,280,140` | Static audio windows in 10 ms mel frames (11.2 / 5.6 / 2.8 / 1.4 s) |
| `--no-audio` | | Skip the audio encoders |
| `--onnx-source-dir DIR` | | Use already downloaded source files instead of downloading them |
| `--models-dir DIR` | `models` | Base directory of the source and export trees |
| `--compile-flags ...` | | Extra torq-compile flags, appended to the exporter's own (must come last) |

## Networks

| Graph | Input -> output |
|---|---|
| `text_body_s128`, `text_body_s512` | `inputs_embeds [1,S,512]`, `attention_bias [1,1,1,S]` -> `last_hidden_state [1,S,768]` |
| `vision_384x384` | `pixel_patches [1,576,768]` -> `image_embeds [1,64,512]` (images and video frames) |
| `audio_140`, `audio_280`, `audio_560`, `audio_1120` | log-mel `input_features [1,T,128]` -> soft tokens `[1,T/4,512]` |

The host runs the token-embedding lookup (`token_embeddings.npy`), image patching, video
frame sampling, the audio log-mel front end, the placement of image, video and audio soft
tokens in the token sequence, mean pooling and L2 normalisation. [`preprocess.py`](preprocess.py)
holds the host code; the demo uses a copy of it.

## Output

```
models/embeddinggemma-2/
├── source/                              downloaded source graphs and tokenizer
└── export[-q4|-q8]/
    ├── onnx/{fp32|q4|q8}/static/        static graphs before bf16 conversion (validated)
    ├── onnx/bf16/static/                graphs that torq-compile builds
    ├── onnx/{q4|q8}_ort/static/         MatMulNBits graphs for ONNX Runtime (q4 and q8 only)
    ├── iree/bf16/static/*.vmfb          compiled networks
    └── assets/
        ├── token_embeddings.npy         host token-embedding table (bf16)
        ├── mel_filters.npy, mel_window.npy
        ├── tokenizer.json, tokenizer_config.json, config.json, config_sentence_transformers.json
        └── embeddinggemma2_manifest.json
```

The demo model directory holds the seven VMFBs from `iree/bf16/static/` and these
assets: `token_embeddings.npy`, `mel_filters.npy`, `mel_window.npy`, `tokenizer.json`,
`config.json` and `embeddinggemma2_manifest.json`.

The manifest records the static shapes (`seq_lens`, `image_sizes`, `audio.windows`), the
weight format, the token ids, prompts and Matryoshka dimensions. After compiling, it also
records `xram_span_mb`, the NPU address span of each VMFB. The runner uses it to choose
which networks fit in the NPU mapping at the same time. The value is read with
`torq-run-module --dump_dispatches`; without that tool (on `PATH` or under
`$IREE_BUILD_DIR/runtime/tools`), the manifest has no `xram_span_mb` and the runner loads
every requested network.

## Compile flags

The exporter compiles every network NSS-only with the flags in `TORQ_FLAGS`
([`export.py`](export.py)): slicing, host and CSS offload and constant prefetch are off.
`--torq-max-nss-programs-size` is set per network and weight format (`NSS_PROGRAMS_MB`):
the NSS program budget is reserved in each loaded network's XRAM map, so the budgets are
kept close to what each network needs.

## Graph changes for Torq

The static builders ([`_static.py`](_static.py)) fold the dynamic parts of the source
graphs into constants: the multimodal merge, position and rotary tables, the vision
pooler and sliding-window masks. In the audio encoder, stride-2 convs drop end padding
that no window reads (bit-identical output).

For q4 and q8 ([`q4.py`](q4.py)):
- The weights are dequantized exactly for the static builders and restored as
  `DequantizeLinear(INT4/INT8) -> MatMul` afterwards.
- Rank-2 int4 matmuls run as rank 3.
- Matmuls over more than 128 rows (text S=512) run as 128-row chunks.
- q8, text S<=128: int8 weights wider than 512 columns run as 512-column chunks, so the
  dequantized tile fits local memory.
- q8: block zero points of -128 become -127 (exactly as `q + 1` where possible, otherwise
  requantized with the scale widened by 255/254), because torq-compile negates the zero
  point in int8.
