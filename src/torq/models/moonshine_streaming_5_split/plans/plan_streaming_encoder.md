# Implementation Plan: Stateful Sliding-Window Streaming Encoder

## Goal

Replace the current stateless encoder (re-encodes a growing window of features from scratch on
every call) with a stateful streaming encoder that maintains per-layer hidden-state buffers,
processing only new frames on each call. Focus on dynamic ONNX export first.

---

## Background: Why the Current Encoder Is Inefficient for Streaming

The exported `encoder.onnx` has a single input `features [1, T, 320]` and a single output
`encoded [1, T, 320]`. The sliding window mask is recomputed inside the graph on every call
over the full sequence. The demo compensates by re-encoding a 224-frame left-context window
every encode cycle, re-processing already-seen frames repeatedly.

The tiny model's per-layer sliding windows (from `config.sliding_windows`) are:

```
Layer 0: left=16, right=4
Layer 1: left=16, right=4
Layer 2: left=16, right=0
Layer 3: left=16, right=0
Layer 4: left=16, right=4
Layer 5: left=16, right=4
```

Max left context per layer = **16 feature frames**.
Max right context (lookahead) per layer = **4 feature frames**.
`total_lookahead = 16` feature frames before a stable frame can be emitted (already tracked
by the demo and `_inference.py`).

---

## Proposed Design: `StatefulEncoderWrapper`

### Core idea

For each encoder layer, cache the last `left_ctx` (=16) frames of that layer's input hidden
states. On each call, concatenate `[layer_buffer, new_frames]`, run the layer's attention
over this window, and take only the `new_frames` portion of the output. Update the buffer to
the last 16 frames of `[layer_buffer, new_frames]`.

Right-context (lookahead) stability is handled externally exactly as today: the inference
caller delays emitting the last `total_lookahead` (=16) frames of encoder output. This is
unchanged.

### Per-layer operation (one layer, left_ctx=16)

```
layer_input = concat([layer_buf, new_features], axis=1)  # [1, 16+T_new, 320]
mask = sliding_window_mask(layer_input, left=16, right=right_ctx[layer_idx])
layer_out, _ = encoder_layer(layer_input, attention_mask=mask)  # [1, 16+T_new, 320]
new_hidden = layer_out[:, 16:, :]                               # [1, T_new, 320]
layer_buf_out = layer_input[:, -16:, :]                         # [1, 16, 320]
```

### Module interface

```python
class StatefulEncoderWrapper(torch.nn.Module):
    def forward(
        self,
        new_features,     # [1, T_new, 320]
        buf_0, buf_1, buf_2, buf_3, buf_4, buf_5,  # each [1, 16, 320]
    ) -> tuple:
        # Returns: encoded_new [1, T_new, 320],
        #          buf_0_out ... buf_5_out  (each [1, 16, 320])
```

Buffers are passed as explicit positional inputs/outputs (not a list) so ONNX export is clean.

---

## Proposed Changes

### [NEW] `StatefulEncoderWrapper` in `export.py`

Add a new wrapper class `StatefulEncoderWrapper` alongside the existing `EncoderWrapper`.

- Constructor takes `encoder` (the `EncoderWrapper`'s underlying module) and `sliding_windows`.
- `forward` concatenates each layer's buffer with new hidden states, runs the layer with the
  appropriate sliding window mask (using `create_bidirectional_mask` +
  `sliding_window_mask_function` from transformers), slices the new portion, and updates buffers.
- Use `torch.cat([buf, hidden], dim=1)` then slice — avoids in-place ops that break ONNX export.

### [MODIFY] `export.py` — `_generate_source_onnx`

Add export of `encoder_streaming.onnx` alongside `encoder.onnx`:

```python
streaming_encoder = StatefulEncoderWrapper(model, cfg.sliding_windows).eval()
dummy_new = torch.randn(1, 10, enc_hidden)
dummy_bufs = [torch.zeros(1, 16, enc_hidden) for _ in range(n_enc_layers)]

# Dynamic export: T_new is dynamic, buffers are fixed [1, 16, 320]
t_new = torch.export.Dim("t_new", min=1)
torch.onnx.export(
    streaming_encoder,
    (dummy_new, *dummy_bufs),
    str(self._onnx_dir / "encoder_streaming.onnx"),
    dynamo=True,
    input_names=["new_features", "buf_0", "buf_1", "buf_2", "buf_3", "buf_4", "buf_5"],
    output_names=["encoded_new", "buf_0_out", "buf_1_out", "buf_2_out",
                  "buf_3_out", "buf_4_out", "buf_5_out"],
    dynamic_shapes={
        "new_features": {0: 1, 1: t_new},
    },
)
```

The per-layer buffers are static shape `[1, 16, 320]` — no dynamic axes needed on them.

### [MODIFY] `_graph.py` — `fix_streaming_encoder_io`

Add a new method to `MoonshineStreaming5SplitOnnxGraphEditor` for the static export path (future):

```python
def fix_streaming_encoder_io(self, t_new: int, left_ctx: int = 16):
    to_fix = [
        FixedDimMapping("batch", DimMatchType.EXACT, 1),
        FixedDimMapping("t_new", DimMatchType.EXACT, t_new),
        # Buffer dims are already fixed at left_ctx by construction
    ]
    self.fix_io_dims(to_fix)
```

This is a stub for static model work; not needed for dynamic export.

### [MODIFY] `_inference.py` — `MoonshineStreaming5Split`

Update the inference class to optionally use `encoder_streaming.onnx` when present.

**State change in `run()`:**

```python
# Detect if streaming encoder is loaded
self._streaming_encoder: InferenceRunner | None = ...  # None if not found

# Initialize per-layer buffers if using streaming encoder
enc_layer_bufs = [
    np.zeros((1, 16, self._hidden_size), dtype=np.float32)
    for _ in range(self._n_enc_layers)
]
```

**In the feature accumulation loop:**

```python
if self._streaming_encoder is not None:
    # Only process new frames — no window accumulation needed
    res = self._streaming_encoder.infer({
        "new_features": features,     # [1, T_new, 320] from frontend
        "buf_0": enc_layer_bufs[0],
        ...
        "buf_5": enc_layer_bufs[5],
    })
    new_encoded = res[0]              # [1, T_new, 320]
    enc_layer_bufs = list(res[1:])   # updated buffers
    accum_encoded.append(new_encoded)
else:
    accum_features.append(features)  # existing path

# After audio loop:
if self._streaming_encoder is not None:
    # Concatenate accumulated encoded frames, trim lookahead
    all_encoded = np.concatenate(accum_encoded, axis=1)
    stable_encoded = all_encoded[:, :-total_lookahead or None, :]
else:
    all_features = np.concatenate(accum_features, axis=1)
    stable_encoded = self._encoder.infer({"features": all_features})[0]
```

Keep the adapter and subsequent components unchanged — they still receive `encoded [1, enc_seq, 320]`.

**Loading:**

```python
@classmethod
def from_onnx(cls, ...) -> "MoonshineStreaming5Split":
    streaming_enc_path = Path(decoder_model).parent / "encoder_streaming.onnx"
    return cls(
        ORTInferenceRunner(frontend_model),
        ORTInferenceRunner(encoder_model),
        ORTInferenceRunner(streaming_enc_path) if streaming_enc_path.exists() else None,
        ...
    )
```

---

## Validation Plan

### 1. Parity test — stateful vs stateless encoder

The core correctness check: given the same audio, stateful streaming encoder output should
match stateless batch encoder output for all stable frames (those beyond `total_lookahead`
from the end of each window).

Add a test function to `debug_parity.py` (or a new `debug_streaming_encoder.py`):

```python
def validate_streaming_encoder_parity(model_dir, audio_path, chunk_frames=10):
    """
    Feeds audio through the streaming encoder (chunk_frames at a time) and
    compares accumulated stable output against the batch encoder on the full sequence.
    """
    # 1. Load both encoder ONNX sessions
    batch_sess = ort.InferenceSession("encoder.onnx")
    streaming_sess = ort.InferenceSession("encoder_streaming.onnx")

    # 2. Get features from frontend (full audio)
    features = run_frontend(audio_path)  # [1, T_total, 320]
    T = features.shape[1]
    total_lookahead = 16

    # 3. Batch encoder reference
    ref_encoded = batch_sess.run(None, {"features": features})[0]  # [1, T, 320]

    # 4. Streaming encoder — chunk by chunk
    bufs = {f"buf_{i}": np.zeros((1, 16, 320), dtype=np.float32) for i in range(6)}
    accum = []
    for start in range(0, T, chunk_frames):
        chunk = features[:, start:start + chunk_frames, :]
        inputs = {"new_features": chunk, **bufs}
        outs = streaming_sess.run(None, inputs)
        accum.append(outs[0])
        bufs = {f"buf_{i}": outs[i + 1] for i in range(6)}

    streaming_encoded = np.concatenate(accum, axis=1)  # [1, T, 320]

    # 5. Compare stable region only (drop last total_lookahead frames)
    stable = slice(None, -total_lookahead)
    max_diff = np.abs(ref_encoded[:, stable, :] - streaming_encoded[:, stable, :]).max()
    mean_diff = np.abs(ref_encoded[:, stable, :] - streaming_encoded[:, stable, :]).mean()
    print(f"Max diff (stable region): {max_diff:.6f}")
    print(f"Mean diff (stable region): {mean_diff:.6f}")
    assert max_diff < 1e-4, f"Parity failure: max_diff={max_diff}"
```

Test at multiple chunk sizes: 1, 5, 10, and the full sequence (should all match).

### 2. End-to-end transcription parity

Run `MoonshineStreaming5Split.run()` with streaming encoder enabled vs disabled on the same
wav file. Transcriptions should be identical (or differ only by trailing words due to
lookahead truncation, which is expected).

### 3. Buffer continuity test

Process the same audio in two ways:
- A. One call with 50 frames
- B. Five calls with 10 frames each

Both should produce identical `encoded_new` outputs for frames in the stable region and
identical buffer states at the end.

---

## Order of Work

1. Implement `StatefulEncoderWrapper` in `export.py` — write and smoke-test in PyTorch eager.
2. Add `encoder_streaming.onnx` export to `_generate_source_onnx`.
3. Run parity test at PyTorch level (before ONNX export) to validate the wrapper logic.
4. Export to ONNX and re-run parity test against `encoder.onnx` batch outputs.
5. Update `_inference.py` to optionally use the streaming encoder.
6. Run end-to-end transcription parity on a test wav.
7. (Future) Add `fix_streaming_encoder_io` and static export path.
