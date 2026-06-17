# Commit Note: SL2610 torq-compile compatibility fixes

## Summary

Three independent torq-compile failures on SL2610 hardware, all sharing the same root
cause: `torq_hl.elementwisebinary` and the tile-and-fuse pass require matching tensor
shapes and strides — no broadcasting, no Gather on large tables. Each fix moves the
problematic computation to the host CPU so the ONNX graph only sees same-shaped,
contiguous tensor operations.

---

## Fix 1: Per-layer unstacked KV (self + cross) in DecoderKVWrapper

**Problem — LRAM OOM + DTCM OOM**

`DecoderKVWrapper` previously passed stacked KV tensors:
- Cross-KV: `[6, 1, 8, 750, 40]` → `Gather([6,...], layer_idx, axis=0)` per layer
- Self-KV: `[6, 1, 8, 90, 40]` → same pattern

The tile-and-fuse pass cannot tile a Gather on axis 0 of a stacked tensor — it has to
materialise the entire tensor before the gather can execute. Results:
- `[6,8,750,40]` = 2.75 MB → LRAM OOM
- `[6,8,90,40]` = 345 KB → DTCM OOM (DTCM limit: 32 KB)

**Fix**

Replace both stacked inputs with 12 individual per-layer tensors. Zero Gather nodes on
the layer dimension are generated. The `models_base/decoder.mlir` reference (which
already used per-layer KV and compiled cleanly) validated this approach.

**Interface after fix**

```
Inputs:  token_or_embed
         k_self_0..k_self_5, v_self_0..v_self_5    [1, 8, 90, 40]  each
         k_cross_0..k_cross_5, v_cross_0..v_cross_5 [1, 8, 750, 40] each
         cross_attn_bias                             [1, 8, 1, 750]
         position_ids                                [1, 1]

Outputs: logits                                     [1, 1, 32768]
         out_k_self_0..out_k_self_5, out_v_self_0..out_v_self_5
         out_k_cross_0..out_k_cross_5, out_v_cross_0..out_v_cross_5
```

**Files changed:** `export.py` (`DecoderKVWrapper.forward`, both export call sites),
`_inference.py` (static and dynamic decode loops),
`/home/yhtet/Documents/work/demos_x86/moonshine_original_demo/orchestrated_demo_static_export.py`.

---

## Fix 2: Replace cross_kv_valid_len with precomputed cross_attn_bias

**Problem — onnx.Less stride mismatch + downstream Add broadcast**

The decoder used `cross_kv_valid_len: [1]` (int32) to build a cross-attention padding
mask inside the ONNX graph:

```python
encoder_attention_mask = (torch.arange(enc_seq_len) < cross_kv_valid_len.squeeze()).unsqueeze(0)
```

Two failures on SL2610:
1. `onnx.Less([750], [])` — scalar `memref<i32>` has no strides, `memref<750xi32>` has
   stride 1; elementwisebinary requires matching strides.
2. Even if Less were fixed, HuggingFace would produce a `[1,1,1,750]` float bias and add
   it to attention scores `[1,8,1,750]` → broadcasting Add → same stride failure.

**Fix**

Move ALL mask computation to the host CPU. Pass a fully-materialised
`[1, n_heads, 1, enc_seq_len]` float additive bias (0.0 for valid, −1e9 for padding)
directly as `cross_attn_bias`.

Key insight: `_preprocess_mask_arguments` in HuggingFace has an early-exit for 4D
tensors — it returns them as-is without processing. So `[1, 8, 1, 750]` passes through
to `eager_attention_forward` where:

```python
attn_weights = attn_weights + attention_mask
# [1, 8, 1, 750] + [1, 8, 1, 750]  ← matching shapes, no broadcast
```

No `Less` node. No broadcasting `Add`. Zero softmax distortion (−1e9 → exp(−1e9) ≈ 0).

**Host-side computation (per decode call):**
```python
cross_attn_bias = np.zeros((1, n_kv_heads, 1, max_memory_len), dtype=np.float32)
cross_attn_bias[:, :, :, valid_len:] = -1e9
```

**Files changed:** `export.py` (`DecoderKVWrapper.forward`, streaming-static export),
`_inference.py` (static decode loop), demo.

---

## Fix 3: Move adapter position embedding lookup to host

**Problem — Gather([4096, 320]) LRAM OOM**

`AdapterWrapper.forward` previously took `pos_offset: [1]` int64, computed
`indices = pos_offset + arange(F)` internally, and called `self.pos_emb(indices)` — an
`nn.Embedding` lookup that maps to:

```
onnx.Gather(weight=[4096, 320], indices=[4])
```

The tile-and-fuse pass cannot tile this Gather ("no more domains to tile"). The entire
`[4096, 320]` weight table (2.5 MB) must be loaded into LRAM before the lookup.
LRAM OOM.

**Fix**

Remove the embedding lookup from the ONNX graph entirely. Save
`model.model.decoder.pos_emb.weight` as `adapter_pos_emb.npy` at export time. On the
host, index into this table before each adapter call and pass the resulting
`[1, F, hidden_size]` float tensor directly as `position_embeddings`.

```python
# Host (per chunk):
pos_emb = pos_emb_weights[pos_offset : pos_offset + F].reshape(1, F, -1)
adapter.infer({"encoded": encoded, "position_embeddings": pos_emb})
```

The adapter ONNX becomes: `Add([1,F,320], [1,F,320]) → proj` — two elementwise ops on
same-shaped tensors, trivially compilable.

**Files changed:** `export.py` (`AdapterWrapper.forward`, both export call sites, weight
save, copy lists in `apply_post_static_patches` and `export_onnx`),
`_inference.py` (`_find_pos_emb`, `_pos_emb` load, all 5 adapter call sites), demo.

---

## Root cause pattern

All three failures share the same SL2610 constraint:

| Op | Failure mode |
|----|-------------|
| `Gather(large_stacked_tensor, scalar_idx, axis=0)` | Tile-and-fuse skips it; entire tensor loaded into LRAM/DTCM |
| `elementwisebinary(tensor_N, scalar_or_[1])` | Input strides don't match (rank-0 vs rank-1 memref) |
| `elementwisebinary(tensor_A, tensor_B_broadcast)` | Input strides don't match (stride-0 broadcast view vs stride-1 contiguous) |

The fix pattern in each case: move the problematic computation to the host CPU and pass
only same-shaped, contiguous tensors into the ONNX graph.

---

## New files

- `plans/plan_cross_attn_bias.md` — design doc for Fix 2
- `plans/plan_unstacked_cross_kv.md` — design doc for Fix 1 (copied from repo root)
- `commit_note.md` — this file
