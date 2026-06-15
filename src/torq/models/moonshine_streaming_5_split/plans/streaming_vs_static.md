# 5-Split Component I/O and Static vs Streaming Compatibility

## Component I/O

```
Audio chunks
    │
    ▼
┌─────────────┐
│  FRONTEND   │  IN:  audio_chunk [1, chunk_len]
│             │       sample_buffer [1, 79]
│             │       sample_len [1]
│             │       conv1_buffer [1, hidden, 4]
│             │       conv2_buffer [1, hidden*2, 4]
│             │       frame_count [1]
│             │  OUT: features [1, T_feat, hidden]  ← grows per chunk
│             │       + all 5 state buffers updated
└──────┬──────┘
       │ features (accumulated externally)
       ▼
┌─────────────┐
│   ENCODER   │  IN:  features [1, seq_len, hidden]
│             │  OUT: encoded  [1, seq_len, hidden]
│             │       (pure bidirectional transformer, no internal state)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   ADAPTER   │  IN:  encoded     [1, seq_len, hidden]
│             │       pos_offset  [1]
│             │  OUT: memory      [1, seq_len, memory_dim]
│             │       (adds positional embeddings + projects to decoder space)
└──────┬──────┘
       │ memory (accumulated externally in streaming demo)
       ▼
┌─────────────┐
│  CROSS_KV   │  IN:  memory  [1, enc_seq, memory_dim]
│             │  OUT: k_cross [depth, 1, heads, enc_seq, head_dim]
│             │       v_cross [depth, 1, heads, enc_seq, head_dim]
│             │       (runs k_proj + v_proj for all decoder layers at once)
└──────┬──────┘
       │ k_cross, v_cross
       ▼
┌─────────────┐
│  DECODER_KV │  IN:  token     [1, 1]
│             │       k_self    [depth, 1, heads, past_seq, head_dim]
│             │       v_self    [depth, 1, heads, past_seq, head_dim]
│             │       k_cross   [depth, 1, heads, enc_seq, head_dim]
│             │       v_cross   [depth, 1, heads, enc_seq, head_dim]
│             │  OUT: logits    [1, 1, vocab_size]
│             │       k_self    [depth, 1, heads, past_seq+1, head_dim]  ← grows
│             │       v_self    [depth, 1, heads, past_seq+1, head_dim]  ← grows
│             │       k_cross   (passthrough, unchanged)
│             │       v_cross   (passthrough, unchanged)
└─────────────┘
```

---

## The Streaming Demo's Incremental Memory Problem

In the demo's `encode()`, memory is accumulated chunk by chunk:
```python
state.memory = np.concatenate([state.memory, memory], axis=1)  # grows
state.cross_kv_valid = False  # invalidate → recompute full cross KV next decode
```

So `compute_cross_kv()` reruns the entire `cross_kv` model over the **full growing memory**
on every encode cycle. The shape of `k_cross`/`v_cross` fed to the decoder therefore changes
at every encode step (enc_seq = N, then N+M, then N+M+P, ...).

This makes three dimensions simultaneously dynamic in the streaming design:
1. `memory` input to `cross_kv`: growing `enc_seq`
2. `k_cross`/`v_cross` output from `cross_kv`: growing `enc_seq`
3. `out_k_cross`/`out_v_cross` input to `decoder_kv`: growing `enc_seq`

---

## Issues Using Static Models in the Streaming Demo

### Fixable (demo code changes only)

| Issue | Fix |
|---|---|
| `k_self`/`v_self` init with `seq_len=0` | Read `max_tokens` from `k_self_shape[3]`, pre-allocate full `[depth, 1, heads, max_tokens, head_dim]` buffer in `reset()` |
| Missing `current_len` decoder input | Add step counter to state, pass `np.array([[step_idx]], dtype=np.int64)` each decoder call |
| No `max_tokens` detection | Read from `k_self_shape[3]` at model init |
| Frontend `chunk_len` mismatch (demo hardcodes 1280) | Read `frontend.get_inputs()[0].shape[-1]` and use that as `chunk_size` |

### Fundamental (incompatible with static model design)

**Speculative decoding** — requires passing a variable-length token batch `[1, N]` for parallel
verification. Static models fix the token input to `[1, 1]`. You cannot multi-token verify
without a dynamic sequence dimension. Disabling it (which `supports_speculative_decoding` would
do) is the only option; you cannot recover it without a separate speculative-window export.

**Growing cross KV cache** — this is the deeper fundamental issue. The streaming demo is designed
around incremental memory: accumulate audio → encode new chunk → append to memory → recompute
cross KV → decode. Static models fix `enc_seq = num_samples // 320` (e.g., 250 for 5s audio).
You cannot pass partially-filled or growing memory; the shape must be exactly `enc_seq_len` on
every call.

The static 5-split design is inherently **batch mode**: accumulate all audio → run encoder once
on full sequence → compute cross KV once → autoregressively decode. The `_inference.py`
`run()` method reflects this: it accumulates all features first, then calls encoder and cross_kv
exactly once. The streaming demo's incremental-memory design and static models are architecturally
incompatible across the entire encoder → cross_kv → decoder path.
