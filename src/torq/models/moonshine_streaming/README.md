# Moonshine Streaming

Tools for exporting and running the Moonshine Streaming ASR family
(`UsefulSensors/moonshine-streaming-{tiny,small}`) to ONNX.

The older non-streaming `moonshine-tiny` / `moonshine-base` models live under
[src/torq/models/moonshine/](../moonshine/) and use a different exporter; the
two are not interchangeable.

To compile the exported ONNX to VMFB for on-device deployment, see
[torq-compiler-dev](https://github.com/synaptics-torq/torq-compiler-dev).

## Contents

| File | Purpose |
|---|---|
| [export.py](export.py) | PyTorch → 4 dynamic ONNX models → static-shape ONNX. Wired into `torq-export-model moonshine-streaming`. |
| [_wrappers.py](_wrappers.py) | `nn.Module` wrappers that split the HF model into `Preprocessor` / `TransformerEncoder` / `Decoder` / `DecoderWithPast` for ONNX export. |
| [_graph.py](_graph.py) | `MoonshineStreamingOnnxGraphEditor` — graph surgery for the dynamo output (Softmax renaming, KV-cache resize, causal mask injection). |
| [__init__.py](__init__.py) | Argparse helper shared with the top-level CLI dispatcher. |
| [infer_test.py](infer_test.py) | Mic-streaming demo. Supports local ONNX, HF, and side-by-side `--both` modes. |
| [test_chunk_size.py](test_chunk_size.py) | Sweep encoder chunk sizes to find the smallest window that matches the full-utterance baseline via pad-and-mask cross-attention. |
| [test_incremental_static.py](test_incremental_static.py) | End-to-end check of overlap-and-save incremental encoding with static ONNX models. |

The export produces four ONNX models plus tokenizer / embedding tensors:

```
preprocessor.onnx               input_values [B,A] + attention_mask [B,A]
                              → input_features [B,S,H_enc] + padding_mask [B,S]
encoder.onnx                    input_features [B,S,H_enc] + attention_mask [B,S]
                              → last_hidden_state  [B,S,H_enc]
decoder.onnx                    decoder_input_ids [B,1]
                              + encoder_hidden_states + encoder_attention_mask
                              → last_hidden_state + 4·L KV outputs (self+cross)
decoder_with_past.onnx          decoder_input_ids + encoder_hidden_states
                              + encoder_attention_mask + current_len + 4·L past KV
                              → last_hidden_state + 4·L present KV
decoder_token_embeddings.npy    vocab embedding matrix (vocab_size × H_dec)
tokenizer.json                  SentencePiece tokenizer (32768 tokens)
```

The decoders emit `last_hidden_state`, not logits. Compute logits externally:
`logits = last_hidden_state @ decoder_token_embeddings.T`. The LM head is
tied to the input embeddings, so a single matrix serves both.

## Setup

Python ≥ 3.10. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The mic demo additionally requires `sounddevice`:

```bash
pip install sounddevice
```

On Linux, `sounddevice` needs the system PortAudio library
(`apt install portaudio19-dev` on Debian/Ubuntu).

## Quick start

### Export

```bash
torq-export-model moonshine-streaming -s tiny
```

Downloads `UsefulSensors/moonshine-streaming-tiny`, exports four dynamic
ONNX models, and converts them to static shapes sized for 5 s of audio (the
default `--input-seconds`). Output:

```
models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/
├── dynamic/    # variable shapes — used for validation and the mic demo
└── static/     # fixed shapes for downstream compilation
```

### Mic demo

```bash
python src/torq/models/moonshine_streaming/infer_test.py \
    --model-dir models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic
```

Speak into the mic; transcription updates every `--update-interval` seconds
(default 0.5 s) and finalizes after `--silence-sec` of silence (default
1.5 s). Ctrl+C exits and prints a session summary.

### Side-by-side ONNX vs HuggingFace

```bash
python src/torq/models/moonshine_streaming/infer_test.py --both
```

Two-line live display tagged `ONNX>` / `HF  >` with per-update MATCH/DIFF
markers and an overall match rate in the session summary.

## Export CLI reference

```
torq-export-model moonshine-streaming [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `-s, --model-size {tiny,small}` | `tiny` | Moonshine streaming variant. |
| `-i, --input-seconds INT` | `5` | Audio window the static models are sized for. Determines encoder seq-len and the decoder cross-attention buffer. |
| `--chunk-seconds FLOAT` | unset | Enable bounded-window incremental encoding. The encoder static shape is sized for `overlap + chunk + finalization_delay` instead of the full input; the decoder still cross-attends to the full `--input-seconds` buffer. |
| `-t, --tokens-per-sec INT` | `6` | Sizes the decoder KV cache (`max_tokens = input_seconds × tokens_per_sec`). |
| `--hf-repo STR` | `UsefulSensors/moonshine-streaming-{size}` | Override the HF source. |
| `--models-dir DIR` | `models` | Output root. |
| `--dynamic-models` | off | Skip the static-shape conversion step. |
| `--skip-validation` | off | Skip the post-export PyTorch-vs-ORT numerical check. |

### Static export (non-incremental)

```bash
torq-export-model moonshine-streaming -s small -i 10
```

Encoder sized for 10 s of audio (500 post-CNN frames), decoder cross-attn
for 500 frames, KV cache for 60 tokens.

### Chunked encoder (incremental TTFT)

```bash
torq-export-model moonshine-streaming -s tiny -i 5 --chunk-seconds 1.0
```

Encoder sized for a single 1 s update window; decoder cross-attention buffer
kept at the full 5 s (250 frames). Use with the overlap-and-save pattern in
[test_incremental_static.py](test_incremental_static.py) or
[infer_test.py](infer_test.py).

| Variant | Overlap (left) | Finalization (right) | Chunk @ 1 s | Encoder window |
|---|---|---|---|---|
| Tiny  | 99 frames  | 16 frames | 50 frames | 165 frames (3.3 s) |
| Small | 163 frames | 16 frames | 50 frames | 229 frames (4.6 s) |

Overlap = `Σ left_window + 3` (CNN receptive field); finalization =
`Σ right_window`. Both come from `config.encoder_config.sliding_windows`.

## Inference CLI reference

```
python src/torq/models/moonshine_streaming/infer_test.py [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir DIR` | `models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic` | Directory containing the four ONNX models, `decoder_token_embeddings.npy`, and `tokenizer.json`. |
| `--update-interval FLOAT` | `0.5` | Seconds between re-inference updates while speech is active. |
| `--silence-sec FLOAT` | `1.5` | Seconds of silence required to finalize a line. |
| `--device INT` | system default | Audio input device index. |
| `--list-devices` | — | Print device list and exit. |
| `--hf` | off | Use the HuggingFace model from the Hub instead of local ONNX. |
| `--both` | off | Run ONNX and HF side-by-side with MATCH/DIFF tagging. |

`--hf` and `--both` are mutually exclusive. Both load `transformers` and
`MoonshineStreamingForConditionalGeneration` lazily.

The inference loop uses continuous audio accumulation with overlap-and-save
incremental encoding: the encoder cache is reused across updates, only the
trailing region (overlap + finalization + new audio) is re-encoded, and the
decoder always re-decodes from BOS against the full concatenated encoder
output. Encoder cost per update is O(1) in the total buffer length.

## Validation scripts

### `test_incremental_static.py`

Validates that static ONNX + overlap-and-save matches the HF full-encode
baseline end-to-end.

```bash
python src/torq/models/moonshine_streaming/test_incremental_static.py \
    --model-dir moonshine_streaming_tiny \
    --static-dir models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/static \
    --step-seconds 1.0 \
    --n-samples 5
```

For each LibriSpeech sample, audio is grown in `--step-seconds` chunks, the
static ONNX encoder runs over each window with overlap-and-save state, the
finalized frames are placed in the decoder's cross-attention buffer with an
attention mask, and the decoded text is compared against
`MoonshineStreamingForConditionalGeneration.generate(...)`. Prints a
match/total summary.

### `test_chunk_size.py`

Sweeps encoder chunk sizes (`[0.25, 0.5, … 5.0]` seconds) to find the
smallest window that still produces the same tokens as the full-utterance
baseline using pad-and-mask cross-attention. Use this to choose
`--chunk-seconds` for export.

```bash
python src/torq/models/moonshine_streaming/test_chunk_size.py \
    --model-dir moonshine_streaming_tiny \
    --n-samples 5
```

Output ends with `★ Recommended --encoder-seconds: …s` when a chunk size
matches the baseline on every sample.

## Design notes

**Split encoder.** The CNN embedder is exported as its own
`preprocessor.onnx`, isolating the per-frame stateless CMVN / asinh / linear
/ Conv1d frontend from the sliding-window transformer. This lets backends
use different code paths or data types for each.

**Encoder requires `attention_mask`.** The streaming encoder produces
materially different output (max-logit delta ~12.5) without an attention
mask, even when all-ones. Both `preprocessor.onnx` and `encoder.onnx` take
`attention_mask` as a first-class input. The internal mask-creation
functions are monkey-patched to `None` during export; decoder causal masking
is reintroduced by the `mask_future_attn_scores` graph edit.

**Decoupled encoder and decoder shapes.** With `--chunk-seconds`, the
encoder is sized for a small window but the decoder cross-attention buffer
is sized for the full `--input-seconds`. The runtime pads finalized encoder
frames into the buffer and passes `encoder_attention_mask` so the decoder
ignores empty positions. This is what makes constant-TTFT streaming
compatible with static shapes.

**Static decoder KV cache.** `decoder_with_past.onnx` is rewritten so the
self-attention KV cache is a fixed `[1, num_heads, max_tokens, head_dim]`
buffer. The dynamo-exported `Concat(axis=-2)` becomes
`Where(Equal(time_ids, current_len), new_kv, past_kv)`, and a causal mask
`Where(LessOrEqual(time_axis, current_len), 0.0, -1e9)` is added before
each self-attention Softmax. The `current_len` input drives both.

**Dynamo-baked RoPE constants.** `torch.onnx.export(dynamo=True)` traces
the decoder at a single position, so RoPE `cos` / `sin` end up as baked
initializers named `repeat_interleave` / `repeat_interleave_1`. The graph
editor replaces them with `Gather` lookups from a precomputed table indexed
by `current_len`.

## Notes

- The standalone scripts ([infer_test.py](infer_test.py),
  [test_chunk_size.py](test_chunk_size.py),
  [test_incremental_static.py](test_incremental_static.py)) are invoked as
  files. `python -m torq.models.moonshine_streaming.infer_test` works once
  the package is installed.
- The validation script in [test_incremental_static.py](test_incremental_static.py)
  uses `MoonshineStreamingForConditionalGeneration.generate(...)` as the
  baseline, which includes the encoder `attention_mask`. Per-step ONNX-vs-HF
  decoder comparisons that share the same encoder output can hide errors
  introduced upstream of the decoder; always validate end-to-end against
  `generate(...)`.
- After resizing KV tensors, the graph editor clears intermediate shapes
  before `to_onnx()` so ONNX's strict shape inference re-derives them
  correctly. Custom edits should preserve that invariant.
- INT8 dynamic quantization sometimes produces a larger file on small
  models because the added `DequantizeLinear` nodes and scale / zero-point
  metadata exceed the weight savings.
