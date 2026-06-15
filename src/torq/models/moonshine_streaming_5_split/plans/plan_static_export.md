# Static Export Plan for Hardware Deployment

## Context

The 5-split Moonshine Streaming pipeline runs as a tightly coupled per-chunk loop:

```
[Every 1280 samples]
  frontend (1280 → F) → encoder (F → F) → adapter (F → F) → cross_kv (F → F KV slices)

[Every 10 chunks / speech_end]
  decoder (pre-allocated cross-KV, pre-allocated self-KV)
```

Where F = number of feature frames the frontend emits per 1280-sample audio chunk.
With right_ctx=16 and F≈8, encoder is skipped for the first 2 chunks (warmup), then
called with exactly F stable frames every subsequent chunk.

The incremental cross_kv design (introduced in the updated demo) means cross_kv always
receives exactly F frames of new memory, producing exactly F frames of new K/V slices.
This eliminates the O(N²) recompute problem and gives all 4 per-chunk components fixed
shapes in steady state.

---

## Phase 1 — Measure F

Before any export changes, instrument the demo or export pipeline to determine F exactly.

**Method:**
```python
# After frontend.run():
print("Features per 1280-sample chunk:", features.shape[1])
```

Run with the dynamic models on a few audio chunks and confirm F is constant across calls
(it should be, given a fixed chunk size and fixed model weights).

F determines every downstream static shape:
- Encoder: `stable_features` = [1, F, features_dim], `right_ctx` = [1, 16, features_dim]
- Adapter: `encoded` = [1, F, features_dim]
- Cross KV: `memory` = [1, F, memory_dim], outputs = [depth, 1, heads, F, head_dim]

---

## Phase 2 — Static Frontend + Encoder + Adapter + Cross KV

All four per-chunk components share the same fixed F-frame shape and can be exported
and validated together.

### 2a. Export changes (`export.py`)

**Frontend**
- Already exported with a fixed `chunk_len` for static models.
- Confirm `chunk_len=1280` matches the demo's `chunk_size = 1280`.
- Use existing `fix_frontend_io(chunk_len=1280)` in `_graph.py`.

**Encoder (streaming)**
- `stable_features`: [1, F, features_dim] — fix the `t_stable` dynamic dim to F.
- `right_ctx`: [1, right_ctx_len, features_dim] — already fixed (right_ctx_len=16).
- `buf_0`…`buf_N`: already fixed (left_ctx per layer, always 16).
- Outputs: `encoded_stable` [1, F, features_dim], `buf_0_out`…`buf_N_out` fixed.
- Add `fix_streaming_encoder_io(stable_len=F)` to `_graph.py`.

**Adapter**
- `encoded`: [1, F, features_dim] — fix to F.
- `pos_offset`: [1] — scalar, already fixed.
- Output `memory`: [1, F, memory_dim] — follows from fixed input.
- Existing `fix_adapter_io(seq_len=F)` covers this.

**Cross KV**
- `memory`: [1, F, memory_dim] — fix to F (now always F, not growing).
- Outputs `out_k_cross`, `out_v_cross`: [depth, 1, heads, F, head_dim] — fixed.
- Existing `fix_cross_kv_io(seq_len=F)` covers this.

### 2b. Graph editor changes (`_graph.py`)

Add a method for the streaming encoder's new I/O signature:

```python
def fix_streaming_encoder_io(
    self,
    stable_len: int,
    right_ctx_len: int,
    batch_dim: str = "batch",
    stable_dim: str = "t_stable",
):
    to_fix = [
        FixedDimMapping(batch_dim, DimMatchType.EXACT, 1),
        FixedDimMapping(stable_dim, DimMatchType.EXACT, stable_len),
    ]
    # right_ctx and buf_* dims are already fixed integers in the ONNX graph
    self.fix_io_dims(to_fix)
```

### 2c. Warmup handling

The first `ceil(right_ctx / F)` chunks produce no stable frames. The demo's `encode()`
already returns early when `new_frames <= 0`. For hardware dispatch, the caller must
simply not send a job to hardware during those warmup chunks — no shape change is needed
in the model.

### 2d. Validation

After static export, run a parity check:
- Feed the same audio through dynamic models and static models chunk-by-chunk.
- Compare `k_cross` and `v_cross` accumulated state after N chunks.
- Compare final transcription on `speech_end`.
- Tolerate warmup contamination for the first `n_enc_layers × max_left_ctx = 96` feature
  frames (same as streaming encoder parity test).

---

## Phase 3 — Static Decoder

The decoder runs every 10 chunks. Its two dynamic dimensions are:
1. Cross-KV input length: grows by F each encode call.
2. Self-KV length: grows token-by-token during the autoregressive loop within one decode call.

### 3a. Pre-allocated cross-KV buffer

Choose `max_memory_len` based on the maximum expected utterance length:
```
max_audio_seconds = 30
max_memory_len = max_audio_seconds * 100  # 100 feature frames/sec ≈ 3000
```

State holds a pre-allocated buffer written incrementally:
```python
# In MoonshineStreamingState.reset():
self.k_cross_buf = np.zeros((depth, 1, heads, max_memory_len, head_dim), dtype=np.float32)
self.v_cross_buf = np.zeros((depth, 1, heads, max_memory_len, head_dim), dtype=np.float32)
self.cross_kv_pos = 0

# In encode(), after cross_kv.run():
new_k, new_v = kv_outs
n = new_k.shape[3]  # = F
self.k_cross_buf[:, :, :, cross_kv_pos:cross_kv_pos + n, :] = new_k
self.v_cross_buf[:, :, :, cross_kv_pos:cross_kv_pos + n, :] = new_v
self.cross_kv_pos += n
```

Decoder receives the full pre-allocated buffer [depth, 1, heads, max_memory_len, head_dim]
plus a scalar `cross_kv_valid_len` (= `cross_kv_pos`) so it can mask out the unwritten tail.

**Decoder model change:** add `cross_kv_valid_len` as an input and apply a causal mask
on the cross-attention dimension beyond that index. If the decoder model already uses
attention masks, extend the existing mask input. If not, this requires modifying the
decoder's cross-attention forward pass or post-processing the attention logits.

### 3b. Pre-allocated self-KV buffer

Choose `max_tokens` based on maximum expected transcript length:
```
max_tokens = max_audio_seconds * 8  # ~8 tokens/sec ≈ 240 for 30s
```

Replace the growing `k_self`/`v_self` with a pre-allocated static buffer and a `seq_pos`
counter:
```python
# In reset():
self.k_self_buf = np.zeros((depth, 1, heads, max_tokens, head_dim), dtype=np.float32)
self.v_self_buf = np.zeros((depth, 1, heads, max_tokens, head_dim), dtype=np.float32)
self.self_kv_pos = 0

# Each decoder step writes at self_kv_pos and increments.
# Decoder takes (token, k_self_buf, v_self_buf, seq_pos) and returns updated buffers.
```

The decoder model reads K/V from positions `0..seq_pos-1` (masked beyond that) and
writes the new token's K/V at position `seq_pos`.

**Decoder model change:** similar to cross-KV — add `seq_pos` as input, apply self-
attention mask to ignore slots beyond `seq_pos`. The self-KV output is the full
pre-allocated buffer with the new token's K/V written in.

### 3c. Decoder export changes (`export.py`)

In the static export path, `DecoderKVWrapper` needs to accept:
- `token`: [1, 1] — unchanged
- `k_self`: [depth, 1, heads, max_tokens, head_dim] — pre-allocated
- `v_self`: [depth, 1, heads, max_tokens, head_dim] — pre-allocated
- `seq_pos`: [1] — current position counter
- `out_k_cross`: [depth, 1, heads, max_memory_len, head_dim] — pre-allocated
- `out_v_cross`: [depth, 1, heads, max_memory_len, head_dim] — pre-allocated
- `cross_kv_valid_len`: [1] — valid frames in cross-KV buffer

Add `fix_decoder_static_io(max_tokens, max_memory_len)` to `_graph.py`.

### 3d. Demo changes for static decoder

The demo's `decode()` method needs to use the pre-allocated buffers and pass `seq_pos`
and `cross_kv_valid_len` to the static decoder session. The autoregressive loop
increments `seq_pos` instead of growing tensors. On each `decode()` call, `seq_pos`
resets to 0 (since the demo already resets `k_self`/`v_self` to empty at the start of
each decode).

### 3e. Validation

- Run static decoder against dynamic decoder on the same cross-KV inputs.
- Confirm identical token sequence and logits within fp32 tolerance.
- Test with audio lengths from 1s to `max_audio_seconds` to verify no overflow.

---

## Shape Summary

| Component    | Input shapes (static)                                        | Output shapes (static)                                         | Runs           |
|--------------|--------------------------------------------------------------|----------------------------------------------------------------|----------------|
| Frontend     | audio [1,1280], fixed buffers                                | features [1,F,feat_dim], updated buffers                       | Every chunk    |
| Encoder      | stable [1,F,feat_dim], right_ctx [1,16,feat_dim], buf_0..N   | encoded [1,F,feat_dim], buf_0_out..N_out                       | Every chunk*   |
| Adapter      | encoded [1,F,feat_dim], pos_offset [1]                       | memory [1,F,mem_dim]                                           | Every chunk*   |
| Cross KV     | memory [1,F,mem_dim]                                         | k [depth,1,heads,F,head_dim], v [depth,1,heads,F,head_dim]    | Every chunk*   |
| Decoder      | token [1,1], k_self [depth,1,heads,max_tok,head_dim], ...    | logits [1,1,vocab], updated k_self, v_self                     | Every 10 chunks|

*skipped during 2-chunk warmup at utterance start

---

## Implementation Order

1. Measure F from the running dynamic demo.
2. Export static frontend + encoder + adapter + cross_kv with fixed shapes.
3. Validate Phase 2 parity.
4. Modify decoder model wrapper and export static decoder.
5. Update demo's `decode()` to use pre-allocated buffers.
6. Validate Phase 3 parity end-to-end.
