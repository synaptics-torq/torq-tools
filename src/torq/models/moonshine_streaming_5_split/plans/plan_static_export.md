# Static Export Plan for Hardware Deployment

## Context

The 5-split Moonshine Streaming pipeline runs as a tightly coupled per-chunk loop:

```
[Every chunk_len samples]
  frontend (chunk_len → F) → encoder (F → F) → adapter (F → F) → cross_kv (F → F KV slices)

[Every N chunks / speech_end]
  decoder (pre-allocated cross-KV, pre-allocated self-KV)
```

**F = 4** (confirmed by profiling: 4 feature frames per 1280-sample / 80ms chunk).

With right_ctx=16 and F=4: warmup = ceil(16/4) = **4 chunks** (320ms) before the first
encoder call. From chunk 5 onward every call receives exactly F=4 stable frames.

The incremental cross_kv design means cross_kv always receives exactly F frames of new
memory and produces exactly F frames of new K/V slices — fixed shapes throughout.

Nothing is hardcoded. F is derived at export time by running the frontend model with a
dummy chunk_len-sample input. chunk_len itself is a new CLI argument.

---

## Phase 1 — Automatic F Derivation at Export Time

F is derived once inside the exporter before any static export begins. No manual
measurement step is needed.

**Where it happens:** `MoonshineStreaming5SplitExporter._derive_feature_stride()`

```python
def _derive_feature_stride(self, model, chunk_len: int) -> int:
    """Run the frontend wrapper once to find F (feature frames per chunk)."""
    preprocessor = StatefulPreprocessorWrapper(model).eval()
    dummy_audio = torch.zeros(1, chunk_len)
    dummy_sample_buf = torch.zeros(1, 79)
    dummy_sample_len = torch.zeros(1, dtype=torch.int64)
    # conv buffer sizes read from the model directly
    c1 = preprocessor.conv1.weight.shape[0]
    c2 = preprocessor.conv2.weight.shape[0]
    dummy_conv1 = torch.zeros(1, c1, 4)
    dummy_conv2 = torch.zeros(1, c2, 4)
    dummy_frame_count = torch.zeros(1, dtype=torch.int64)
    with torch.no_grad():
        features, *_ = preprocessor(
            dummy_audio, dummy_sample_buf, dummy_sample_len,
            dummy_conv1, dummy_conv2, dummy_frame_count
        )
    F = features.shape[1]
    self._logger.info("Derived feature stride F=%d for chunk_len=%d", F, chunk_len)
    return F
```

This is called at the start of the static streaming export path, before exporting encoder,
adapter, or cross_kv. The returned F is passed into each subsequent export call.

---

## Phase 2 — Static Frontend + Encoder + Adapter + Cross KV

### 2a. New CLI argument (`__init__.py`)

Add `--chunk-len` to `add_moonshine_streaming_export_args()`:

```python
parser.add_argument(
    "--chunk-len",
    type=int,
    default=None,
    metavar="N",
    help="Audio chunk size in samples for static streaming export. "
         "When set, exports frontend/encoder/adapter/cross_kv as a fixed per-chunk "
         "pipeline instead of the full-utterance static export. (e.g. 1280 = 80ms @ 16kHz)"
)
```

When `--chunk-len N` is provided and `--dynamic-models` is NOT set, the exporter
switches to the streaming static path for all four per-chunk components.

### 2b. Export path changes (`export.py`)

Add a branch inside `MoonshineStreaming5SplitExporter.export_onnx()` (or equivalent):

```python
if self._chunk_len is not None and not self._dynamic_models:
    F = self._derive_feature_stride(model, self._chunk_len)
    self._export_streaming_static(model, F, self._chunk_len)
```

**`_export_streaming_static(model, F, chunk_len)`** exports four components:

**Frontend** — no model change, just fix chunk_len:
```python
preprocessor = StatefulPreprocessorWrapper(model).eval()
dummy_audio = torch.zeros(1, chunk_len)
# ... other dummy buffers (sizes from model, not hardcoded) ...
torch.onnx.export(preprocessor, (...), "frontend.onnx", dynamo=True,
    input_names=["audio_chunk", "sample_buffer", "sample_len",
                 "conv1_buffer", "conv2_buffer", "frame_count"],
    output_names=["features", "sample_buffer_out", "sample_len_out",
                  "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"])
# No dynamic_shapes — all dims are fixed by fixed chunk_len input.
```

**Encoder (streaming static)** — use `StatefulEncoderWrapper`, fix `stable_features` to F:
```python
streaming_enc = StatefulEncoderWrapper(model).eval()
total_la = streaming_enc._total_lookahead   # 16 for tiny
left_ctxs = streaming_enc._left_ctx
n_layers = len(left_ctxs)
enc_hidden = streaming_enc.config.hidden_size

dummy_stable   = torch.zeros(1, F, enc_hidden)       # fixed F
dummy_right    = torch.zeros(1, total_la, enc_hidden) # fixed 16
dummy_bufs     = [torch.zeros(1, lc, enc_hidden) for lc in left_ctxs]  # fixed per layer

torch.onnx.export(streaming_enc,
    (dummy_stable, dummy_right, *dummy_bufs),
    "encoder.onnx", dynamo=True,
    input_names=["stable_features", "right_ctx"] + [f"buf_{i}" for i in range(n_layers)],
    output_names=["encoded_stable"] + [f"buf_{i}_out" for i in range(n_layers)])
# No dynamic_shapes — dummy_stable has fixed F, all other inputs already fixed.
```

**Adapter** — fix encoded sequence length to F:
```python
dummy_encoded     = torch.zeros(1, F, enc_hidden)
dummy_pos_offset  = torch.zeros(1, dtype=torch.int64)
torch.onnx.export(adapter, (dummy_encoded, dummy_pos_offset), "adapter.onnx",
    dynamo=True,
    input_names=["encoded", "pos_offset"],
    output_names=["memory"])
```

**Cross KV** — fix memory sequence length to F:
```python
dummy_memory = torch.zeros(1, F, memory_dim)
torch.onnx.export(cross_kv, (dummy_memory,), "cross_kv.onnx",
    dynamo=True,
    input_names=["memory"],
    output_names=["out_k_cross", "out_v_cross"])
```

### 2c. Graph editor changes (`_graph.py`)

The existing `fix_adapter_io(seq_len=F)` and `fix_cross_kv_io(seq_len=F)` already handle
adapter and cross_kv. Add one new method for the streaming encoder:

```python
def fix_streaming_encoder_io(
    self,
    stable_len: int,
    batch_dim: str = "batch",
    stable_dim: str = "t_stable",
):
    """Fix the only dynamic dim on the streaming encoder: stable_features sequence length."""
    to_fix = [
        FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
        FixedDimMapping(stable_dim, DimMatchType.EXACT, stable_len),
    ]
    # right_ctx and buf_* dims are already concrete integers in the exported graph.
    self.fix_io_dims(to_fix)
```

The exporter calls these after `torch.onnx.export` (same pattern as the existing
optimize-and-fix pipeline):

```python
editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx("encoder.onnx", "encoder")
editor.fix_streaming_encoder_io(stable_len=F)
editor.save(...)

editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx("adapter.onnx", "adapter")
editor.fix_adapter_io(seq_len=F)
editor.save(...)

editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx("cross_kv.onnx", "cross_kv")
editor.fix_cross_kv_io(seq_len=F)
editor.save(...)
```

### 2d. Warmup handling

With F=4 and right_ctx=16: the first 4 chunks produce 0 stable frames. `encode()`
already returns early when `new_frames <= 0`, so no hardware dispatch happens during
warmup. No model-level change needed — the orchestration layer handles it.

### 2e. Output directory

Static streaming exports go to a separate subdirectory to avoid colliding with the
existing full-utterance static export or the dynamic export:

```
export/onnx/float/streaming_static/
  frontend.onnx
  encoder.onnx
  adapter.onnx
  cross_kv.onnx
  decoder_kv.onnx    ← copied from dynamic for now (Phase 3 replaces this)
  tokenizer.json
  config.json
```

The `--chunk-len` value and derived F are written to a `streaming_config.json` in this
directory so downstream tooling (demo, validator, hardware runtime) can read them:

```json
{ "chunk_len": 1280, "feature_stride": 4, "right_ctx": 16, "warmup_chunks": 4 }
```

### 2f. Validation

After static streaming export, run chunk-by-chunk parity against the dynamic models:

1. Load both dynamic and static streaming sessions for all four components.
2. Feed identical audio through both pipelines chunk by chunk.
3. After each chunk, compare `k_cross` / `v_cross` accumulated state.
4. After `speech_end`, compare final transcription.
5. Skip the first `n_enc_layers × max_left_ctx = 96` feature frames from parity checks
   (zero-buffer warmup contamination — same as `debug_streaming_encoder.py`).
6. Tolerance: max abs diff < 1e-4 for KV states, identical token sequences.

---

## Phase 3 — Static Decoder

Uses the repository-proven **Pre-allocated Buffer + Where blend** strategy, already
applied to standard Moonshine, Whisper, Gemma3, and SmolLM2. Three graph edits handle
the self-KV cache; cross-KV masking requires one additional wrapper change.

### 3a. Strategy overview

**Self-KV (Where blend):**
- Export `DecoderKVWrapper` with `k_self`/`v_self` pre-allocated to
  `[n_layers, 1, n_kv_heads, max_tokens, head_dim]` (fixed shape).
- The dynamic graph contains `Concat(past_kv, new_kv_slice, axis=-2)` for each layer.
- `ReplaceDynamicKVCache` replaces each Concat with:
  `Where(time_ids == cur_len, new_kv_slice_broadcast, past_kv)`
  where `time_ids = [0..max_tokens-1]` is a baked-in constant and `cur_len` is a new
  scalar model input.
- `MaskFutureAttentionScores` injects an additive mask before each self-attention
  Softmax: positions where `arange(max_tokens) > cur_len` receive `-65504` (fp16) or
  `-1e9` (fp32), enforcing causality.
- `AddCurrLenInput` replaces `Shape(past_kv) → Gather(axis=2)` (the dynamic
  sequence-length read inside the model) with the `cur_len` input directly.

**Cross-KV (encoder_attention_mask):**
- Cross-KV inputs are pre-allocated to `max_memory_len` and written F frames at a time
  by the per-chunk cross_kv ONNX (Phase 2).
- The model already accepts `encoder_attention_mask [1, enc_seq]` which flows through
  `create_bidirectional_mask` as an additive mask on cross-attention logits.
- `DecoderKVWrapper.forward` is extended with a `cross_kv_valid_len` input that
  computes this mask at the PyTorch level before export — no new graph edit needed.

### 3b. PyTorch wrapper changes (`export.py`)

Extend `DecoderKVWrapper.forward` to accept and use `cross_kv_valid_len`:

```python
def forward(self, token, k_self, v_self, out_k_cross, out_v_cross, cross_kv_valid_len):
    self_cache = DynamicCache()
    cross_cache = DynamicCache()
    for i in range(self.n_layers):
        self_cache.layers.append(_layer_from_kv(k_self[i], v_self[i]))
        cross_cache.layers.append(_layer_from_kv(out_k_cross[i], out_v_cross[i]))

    pkv = EncoderDecoderCache(self_cache, cross_cache)
    for i in range(self.n_layers):
        pkv.is_updated[i] = True

    max_memory_len = out_k_cross.shape[3]
    dummy_encoder_hidden = torch.zeros(
        1, max_memory_len, self.decoder.config.hidden_size,
        dtype=k_self.dtype, device=k_self.device
    )
    # Mask padded cross-KV positions using the existing encoder_attention_mask path
    encoder_attention_mask = (
        torch.arange(max_memory_len, device=out_k_cross.device) < cross_kv_valid_len
    ).unsqueeze(0)  # [1, max_memory_len]

    dec_out = self.decoder(
        input_ids=token,
        past_key_values=pkv,
        use_cache=True,
        encoder_hidden_states=dummy_encoder_hidden,
        encoder_attention_mask=encoder_attention_mask,
    )

    logits = self.proj_out(dec_out.last_hidden_state)
    updated_k_self = torch.stack([l.keys for l in pkv.self_attention_cache.layers], dim=0)
    updated_v_self = torch.stack([l.values for l in pkv.self_attention_cache.layers], dim=0)
    return logits, updated_k_self, updated_v_self, out_k_cross, out_v_cross
```

### 3c. Export changes (`export.py`)

Choose `max_tokens` and `max_memory_len` from config / CLI:
```
max_tokens     = max_audio_seconds * max_tok_per_sec   # e.g. 30 * 8 = 240
max_memory_len = max_audio_seconds * (16000 // 320)    # e.g. 30 * 50 = 1500
                                                        # (adjust to actual F * max_chunks)
```

Export with fixed-size dummies so all dims are concrete:
```python
decoder_kv = DecoderKVWrapper(model).eval()
dummy_token       = torch.ones(1, 1, dtype=torch.long)
dummy_k_self      = torch.zeros(n_layers, 1, n_kv_heads, max_tokens, head_dim)
dummy_v_self      = torch.zeros(n_layers, 1, n_kv_heads, max_tokens, head_dim)
dummy_k_cross     = torch.zeros(n_layers, 1, n_kv_heads, max_memory_len, head_dim)
dummy_v_cross     = torch.zeros(n_layers, 1, n_kv_heads, max_memory_len, head_dim)
dummy_cross_valid = torch.tensor([1], dtype=torch.long)

torch.onnx.export(
    decoder_kv,
    (dummy_token, dummy_k_self, dummy_v_self, dummy_k_cross, dummy_v_cross, dummy_cross_valid),
    "decoder_kv.onnx", dynamo=True,
    input_names=["token", "k_self", "v_self", "out_k_cross", "out_v_cross", "cross_kv_valid_len"],
    output_names=["logits", "out_k_self", "out_v_self", "out_k_cross_out", "out_v_cross_out"],
    # No dynamic_shapes — all dims are fixed by fixed-size dummies.
)
```

### 3d. Graph editor changes (`_graph.py`)

Add a `make_decoder_static` method that applies the three proven edits in sequence:

```python
def make_decoder_static(self, max_tokens: int):
    cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
    self._graph.inputs.append(cur_len_2d)
    cur_len = self._graph.layer(
        name="current_len_to_1d",
        op="Squeeze",
        inputs=[cur_len_2d, gs.Constant("squeeze_axes", np.array([0], dtype=np.int64))],
        outputs=[gs.Variable("current_len_squeezed", dtype=np.int64, shape=[1])],
    )[0]
    (
        self
        .replace_dynamic_kv_cache(cur_len, max_tokens)
        .mask_future_attn_scores(cur_len, max_tokens)
        .add_curr_len_input(cur_len)
    )
```

The exporter calls this after the ONNX export:
```python
editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx("decoder_kv.onnx", "decoder_kv")
editor.make_decoder_static(max_tokens=max_tokens)
editor.save(...)
```

### 3e. Verification step before applying edits

`MaskFutureAttentionScores` and `AddCurrLenInput` match by ONNX node name patterns.
Before relying on them, do a test export and confirm the patterns exist in the graph:

```python
# After torch.onnx.export, before graph edits:
model_proto = onnx.load("decoder_kv.onnx")
graph = gs.import_onnx(model_proto)

softmax_names = [n.name for n in graph.nodes if n.op == "Softmax"]
shape_names   = [n.inputs[0].name for n in graph.nodes if n.op == "Shape"]

print("Softmax nodes:", softmax_names)   # must contain "self_attn/Softmax" suffix
print("Shape inputs:", shape_names)      # must contain "past_key_values"
```

If names differ, update the `match()` methods in `_graph.py` for the 5-split component
before proceeding. This is a one-time check.

### 3f. Demo changes (`MoonshineStreamingModel`)

**State changes** (`MoonshineStreamingState.reset()`):
```python
# Replace growing k_self/v_self with pre-allocated fixed buffers
self.k_self = np.zeros((depth, 1, heads, max_tokens, head_dim), dtype=np.float32)
self.v_self = np.zeros((depth, 1, heads, max_tokens, head_dim), dtype=np.float32)
self.cur_len = np.array([[0]], dtype=np.int64)  # [1,1] shape for current_len input

# Cross-KV pre-allocated buffer (written F frames at a time by Phase 2 cross_kv)
self.k_cross_buf = np.zeros((depth, 1, heads, max_memory_len, head_dim), dtype=np.float32)
self.v_cross_buf = np.zeros((depth, 1, heads, max_memory_len, head_dim), dtype=np.float32)
self.cross_kv_pos = 0
```

**`encode()` update** — write cross_kv output into pre-allocated buffer:
```python
kv_outs = self.cross_kv.run(None, {"memory": new_memory})
n = kv_outs[0].shape[3]  # = F
state.k_cross_buf[:, :, :, state.cross_kv_pos:state.cross_kv_pos + n, :] = kv_outs[0]
state.v_cross_buf[:, :, :, state.cross_kv_pos:state.cross_kv_pos + n, :] = kv_outs[1]
state.cross_kv_pos += n
```

**`decode()` update** — pass fixed-shape buffers and scalar counters:
```python
# cur_len resets to 0 at the start of each decode() call (matches existing reset behaviour)
state.cur_len = np.array([[0]], dtype=np.int64)
cross_kv_valid_len = np.array([state.cross_kv_pos], dtype=np.int64)

# BOS step
outputs = self.decoder.run(None, {
    "token": np.array([[1]], dtype=np.int64),
    "k_self": state.k_self,
    "v_self": state.v_self,
    "out_k_cross": state.k_cross_buf,
    "out_v_cross": state.v_cross_buf,
    "current_len": state.cur_len,
    "cross_kv_valid_len": cross_kv_valid_len,
})
logits, state.k_self, state.v_self = outputs[0], outputs[1], outputs[2]
state.cur_len = np.array([[1]], dtype=np.int64)

# Autoregressive loop increments cur_len each step
# ...
state.cur_len = np.array([[state.cur_len[0, 0] + 1]], dtype=np.int64)
```

Note: `k_self`/`v_self` buffers do NOT need to be zeroed between `decode()` calls.
Positions beyond `cur_len` are blocked by `MaskFutureAttentionScores`, so stale data
from the previous utterance is never attended to.

### 3g. Speculative decoding

Speculative decoding passes `token: [1, N]` — a variable-length batch. Static decoder
fixes `token: [1, 1]`. These are incompatible shapes.

**Decision: drop speculative decoding for the static path.** The static demo uses
autoregressive-only decoding. If speculative decoding is needed later, export a second
decoder variant with `token: [1, max_spec_tokens]` fixed, with its own `cur_len` logic.

### 3h. Validation

1. Export static decoder, load both dynamic and static sessions.
2. Feed identical `k_cross_buf`, `v_cross_buf`, `cross_kv_valid_len` to both.
3. Run autoregressive decode for N steps, compare logits and token sequences at each step.
4. Confirm `k_self`/`v_self` buffer state after each step matches dynamic output.
5. Test with `cross_kv_valid_len` from 1 to `max_memory_len` to verify masking is correct
   (tokens should not attend to zero-padded cross-KV slots).
6. Tolerance: logits max abs diff < 1e-4, identical token sequences.

---

## Shape Summary

| Component   | Key input shapes (static)                                                          | Key output shapes (static)                               | Cadence        |
|-------------|------------------------------------------------------------------------------------|----------------------------------------------------------|----------------|
| Frontend    | audio [1, 1280], fixed conv/sample buffers                                         | features [1, 4, feat_dim], updated buffers               | Every chunk    |
| Encoder     | stable [1, 4, feat_dim], right_ctx [1, 16, feat_dim], buf_0..N                    | encoded [1, 4, feat_dim], buf_0_out..N_out               | Every chunk*   |
| Adapter     | encoded [1, 4, feat_dim], pos_offset [1]                                           | memory [1, 4, mem_dim]                                   | Every chunk*   |
| Cross KV    | memory [1, 4, mem_dim]                                                             | k/v [depth, 1, heads, 4, head_dim] each                  | Every chunk*   |
| Decoder     | token [1,1], k_self [depth,1,heads,max_tok,hd], current_len [1,1], k_cross_buf [depth,1,heads,max_mem,hd], cross_kv_valid_len [1] | logits [1,1,vocab], k_self_out, v_self_out | Every N chunks |

*skipped during 4-chunk warmup at utterance start (F=4, right_ctx=16)

All shapes derived from F=4 (measured) and chunk_len=1280 (CLI arg). `max_tokens` and
`max_memory_len` come from `--input-seconds` and `--tokens-per-sec` CLI args.

---

## Implementation Order

**Phase 2:**
1. Add `--chunk-len` arg to `__init__.py`.
2. Add `_derive_feature_stride()` to exporter.
3. Add `_export_streaming_static()` to exporter (frontend, encoder, adapter, cross_kv).
4. Add `fix_streaming_encoder_io()` to `_graph.py`.
5. Write `streaming_config.json`.
6. Run export with `--chunk-len 1280 --skip-torq all` and verify all four models load.
7. Run chunk-by-chunk parity validation (§2f).

**Phase 3:**
8. Extend `DecoderKVWrapper.forward` with `cross_kv_valid_len` and `encoder_attention_mask`.
9. Export static decoder with fixed `max_tokens` and `max_memory_len` dummies.
10. Do test export, inspect Softmax and Shape node names (§3e verification step).
11. Add `make_decoder_static()` to `_graph.py`, adjust match patterns if needed.
12. Update demo state (`MoonshineStreamingState`) and `decode()` method.
13. Run decoder parity validation (§3h).
