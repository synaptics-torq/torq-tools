# Plan: Unstacked Per-Layer Cross-KV Decoder Interface

## Problem

The current decoder accepts cross-KV as two stacked tensors:

```
out_k_cross  [6, 1, 8, max_memory_len, 40]
out_v_cross  [6, 1, 8, max_memory_len, 40]
```

Inside `DecoderKVWrapper.forward()`, each layer is extracted via:

```python
cross_cache.layers.append(_layer_from_kv(out_k_cross[i], out_v_cross[i]))
```

`out_k_cross[i]` becomes `Gather(out_k_cross, i, axis=0)` in ONNX — one Gather per layer,
12 total (6 k + 6 v).

The torq-compile tile-and-fuse pass cannot tile these Gather ops (the index is a fixed scalar
constant — no iterable loop domain). This forces the compiler to materialise the full stacked
tensor `[6, 8, max_memory_len, 40]` in LRAM before any Gather can execute:

- 15 s export (max_memory_len=750):  6×8×750×40×2 B ≈ **2.75 MB** → LRAM OOM, fatal error
- 5 s export  (max_memory_len=250):  6×8×250×40×2 B ≈ **0.94 MB** → may still fail depending on LRAM size

Option A (enabling slicing) did not resolve the issue.

## Solution: Unstacked Per-Layer Inputs

Remove the stacked dimension entirely.  Pass 12 separate inputs — one per layer per key/value:

```
k_cross_0  [1, 8, max_memory_len, 40]
v_cross_0  [1, 8, max_memory_len, 40]
k_cross_1  [1, 8, max_memory_len, 40]
v_cross_1  [1, 8, max_memory_len, 40]
...
k_cross_5  [1, 8, max_memory_len, 40]
v_cross_5  [1, 8, max_memory_len, 40]
```

No `Gather` on the layer dimension is needed.  Each slice is passed directly to
`_layer_from_kv(k_cross_i, v_cross_i)`.  The compiler sees 12 independent `[1, 8, N, 40]`
tensors and can tile/stream each one against the attention computation without ever needing
the full stacked buffer in LRAM.

---

## Files to Change

### 1. `export.py` — `DecoderKVWrapper`

**`__init__`**: no structural change needed.

**`forward` signature**: replace `out_k_cross, out_v_cross` with 12 named parameters:

```python
def forward(self, token_or_embed,
            k_self, v_self,
            k_cross_0, v_cross_0,
            k_cross_1, v_cross_1,
            k_cross_2, v_cross_2,
            k_cross_3, v_cross_3,
            k_cross_4, v_cross_4,
            k_cross_5, v_cross_5,
            cross_kv_valid_len=None,
            position_ids=None):
```

**Body**: build cross cache directly from the per-layer args:

```python
k_crosses = [k_cross_0, k_cross_1, k_cross_2, k_cross_3, k_cross_4, k_cross_5]
v_crosses = [v_cross_0, v_cross_1, v_cross_2, v_cross_3, v_cross_4, v_cross_5]
for kc, vc in zip(k_crosses, v_crosses):
    cross_cache.layers.append(_layer_from_kv(kc, vc))
```

The `encoder_attention_mask` uses `k_cross_0.shape[-2]` for `enc_seq_len` instead of
`out_k_cross.shape[3]`.

**Return**: the cross-KV pass-through outputs become 12 individual tensors instead of two
stacked ones.  Output names:
`["logits", "out_k_self", "out_v_self",
  "out_k_cross_0", "out_v_cross_0", ..., "out_k_cross_5", "out_v_cross_5"]`

Note: cross-KV outputs are identity pass-throughs (the decoder never writes them); they
exist solely so the VMFB interface is self-contained.

**`n_layers` hardcoding**: the 12 parameter signature hard-codes `n_layers=6`.  Add a
runtime assertion in `__init__` that `len(decoder.layers) == 6` so the assumption is
explicit.

---

### 2. `export.py` — streaming-static export dummy inputs & input_names

In `_export_streaming_static_source_onnx`, replace the two stacked cross-KV dummies with
12 per-layer dummies and update `input_names` / `output_names` accordingly.

Dummy shape per layer: `[1, n_kv_heads, max_memory_len, head_dim]`
i.e. `[1, 8, max_memory_len, 40]`

Input names (ordering matches `forward` positional args):
```
[first_input_name,
 "k_self", "v_self",
 "k_cross_0", "v_cross_0",
 "k_cross_1", "v_cross_1",
 "k_cross_2", "v_cross_2",
 "k_cross_3", "v_cross_3",
 "k_cross_4", "v_cross_4",
 "k_cross_5", "v_cross_5",
 "cross_kv_valid_len", "position_ids"]
```

Output names:
```
["logits", "out_k_self", "out_v_self",
 "out_k_cross_0", "out_v_cross_0",
 "out_k_cross_1", "out_v_cross_1",
 "out_k_cross_2", "out_v_cross_2",
 "out_k_cross_3", "out_v_cross_3",
 "out_k_cross_4", "out_v_cross_4",
 "out_k_cross_5", "out_v_cross_5"]
```

Same pattern applies to the non-streaming-static (batch/dynamic) decoder export paths.

---

### 3. `_inference.py` — static decoder detection & feed dict

**Detection**: currently the code detects `_is_static_decoder` by checking for `"current_len"`
in input names.  This still works unchanged.

**Feed dict construction** (both static and dynamic loops): replace the two stacked arrays
with per-layer slices.  The internal buffers `k_cross_buf` and `v_cross_buf` can keep their
current stacked shape `[n_layers, 1, n_kv_heads, max_memory_len, head_dim]` — just slice
when building the feed dict:

```python
dec_feed = {
    **first_token_feed,          # "token" or "inputs_embeds"
    "k_self": k_self,
    "v_self": v_self,
    "cross_kv_valid_len": cross_kv_valid,
    "current_len": current_len,
    "position_ids": current_len,
}
for i in range(self._n_layers):
    dec_feed[f"k_cross_{i}"] = out_k_cross[i]   # shape [1, n_kv_heads, N, head_dim]
    dec_feed[f"v_cross_{i}"] = out_v_cross[i]
```

**Output handling**: `logits, k_self, v_self = res[0], res[1], res[2]`.  Outputs 3..14 are
the 12 cross-KV pass-throughs — ignore them (cross-KV is never written by the decoder).

---

### 4. `orchestrated_demo_static_export.py` — decode loop

**State buffers**: `state.k_cross` and `state.v_cross` keep their current stacked shape
`[depth, 1, heads, max_memory_len, head_dim]` — no change to buffer allocation.

**decode() feed dict**: same slice pattern as `_inference.py`:

```python
for i in range(self.depth):
    first_feed[f"k_cross_{i}"] = state.k_cross[i]
    first_feed[f"v_cross_{i}"] = state.v_cross[i]
```

Remove `"out_k_cross"` / `"out_v_cross"` keys.

**Output handling**: `logits, k_self, v_self = dec_out[0], dec_out[1], dec_out[2]`.
Remaining outputs are pass-throughs, discard.

The in-place probe at startup also needs updating to use the 12-key interface.

---

## What Does NOT Change

- Self-KV (`k_self`, `v_self`) interface — remains stacked `[6, 1, 8, max_tokens, 40]`
  because the Where-based in-place update already works correctly with the tiler.
- `cross_kv.onnx` — cross-KV generation is unaffected; it still outputs `k_cross [depth,1,8,F,40]`
  and `v_cross [depth,1,8,F,40]`.  The slicing into the pre-allocated buffer in inference/demo
  code already writes per-layer at `[i, :, :, fill:end, :]` — no change needed there.
- `frontend.onnx`, `encoder.onnx`, `adapter.onnx` — untouched.
- `streaming_config.json` — no new keys required.
- `edits.py` graph edits — the `ReplaceDynamicKVCache` / `MaskFutureAttentionScores` /
  `AddCurrLenInput` edits target self-KV and are unaffected by cross-KV interface changes.

---

## Expected Outcome

The 12 `Gather([6,1,8,N,40], i, axis=0)` nodes disappear from the ONNX graph entirely.
The compiler sees 12 independent `[1, 8, max_memory_len, 40]` inputs that can each be tiled
along the `max_memory_len` and `head_dim` dimensions and fused with the corresponding
cross-attention matmul — eliminating the LRAM OOM.
