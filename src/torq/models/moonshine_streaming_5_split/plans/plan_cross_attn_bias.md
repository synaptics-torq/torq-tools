# Plan: Replace cross_kv_valid_len with precomputed cross_attn_bias

## Problem

The streaming-static decoder ONNX currently takes `cross_kv_valid_len: [1]` (int32) and
computes the cross-attention padding mask INSIDE the graph:

```python
encoder_attention_mask = (
    torch.arange(enc_seq_len) < cross_kv_valid_len.squeeze()
).unsqueeze(0)  # [1, 750] bool
```

This fails torq-compile on SL2610 in two places:

1. **`onnx.Less([750], [])`** — `torq_hl.elementwisebinary` requires all inputs to have
   matching shapes/strides. A scalar `memref<i32>` (rank-0, no strides) cannot be paired
   with `memref<750xi32>` (rank-1, stride 1). Neither `[]` (squeeze) nor `[1]` (unsqueeze)
   avoids this — they both fail, just with different error messages.

2. **Downstream `Add([1,8,1,750], [1,1,1,750])`** — HuggingFace's `create_bidirectional_mask`
   converts the `[1, 750]` bool to a `[1, 1, 1, 750]` float additive bias. That bias is then
   added to attention scores `[1, 8, 1, 750]` with broadcasting on dim 1. Same
   elementwisebinary stride-mismatch failure.

Root cause: SL2610 `elementwisebinary` has NO support for broadcasting. Both inputs must
have identical shapes and strides.

## Solution

Move all mask computation to the **host CPU** and pass in the fully-materialized
`[1, n_heads, 1, enc_seq_len]` float additive bias as a decoder input.

### Why this works end-to-end

HuggingFace's `_preprocess_mask_arguments` has an early-exit:
```python
if isinstance(attention_mask, ...) and len(attention_mask.shape) == 4:
    return True, attention_mask, ...   # returned as-is, no further processing
```

So passing `encoder_attention_mask` as shape `[1, n_heads, 1, enc_seq_len]` bypasses
`create_bidirectional_mask` entirely. The 4D tensor lands directly in eager_attention_forward:

```python
# modeling_moonshine.py line 195
attn_weights = attn_weights + attention_mask
# [1, 8, 1, 750] + [1, 8, 1, 750]  ← matching shapes, no broadcast, works on SL2610
```

No `Less` op. No broadcasting. Correct masking behaviour (-1e9 → exp(-1e9) ≈ 0 softmax weight).

## Changes

### 1. `export.py` — `DecoderKVWrapper.forward`

**Remove** `cross_kv_valid_len: torch.Tensor | None = None` parameter.  
**Add** `cross_attn_bias: torch.Tensor | None = None` parameter (float, shape `[1, n_heads, 1, enc_seq_len]`).

Remove the mask-computation block:
```python
# REMOVE THIS:
encoder_attention_mask = None
if cross_kv_valid_len is not None:
    encoder_attention_mask = (
        torch.arange(enc_seq_len, device=k_cross_0.device) < cross_kv_valid_len.squeeze()
    ).unsqueeze(0)
```

Replace with:
```python
encoder_attention_mask = cross_attn_bias  # already [1, n_heads, 1, enc_seq_len] or None
```

### 2. `export.py` — streaming-static export call

Replace the dummy and name:
```python
# BEFORE:
dummy_cross_valid = torch.zeros(1, dtype=torch.int32)
# input_names: [..., "cross_kv_valid_len", "position_ids"]
# args: (..., dummy_cross_valid, dummy_position_ids)

# AFTER:
dummy_cross_attn_bias = torch.zeros(
    1, self._num_kv_heads, 1, self._max_memory_len, dtype=torch.bfloat16
)
# input_names: [..., "cross_attn_bias", "position_ids"]
# args: (..., dummy_cross_attn_bias, dummy_position_ids)
```

The non-streaming dynamic/static export paths do NOT include `cross_attn_bias`
(they pass `cross_attn_bias=None`, triggering the `if cross_attn_bias is not None` guard).

### 3. `_inference.py` — static decoder loop

Compute on the host before each decode step:
```python
cross_attn_bias = np.zeros(
    (1, self._n_kv_heads, 1, self._max_memory_len), dtype=np.float32
)
cross_attn_bias[:, :, :, cross_kv_valid:] = -1e9
```

Replace `"cross_kv_valid_len": cross_kv_valid` with `"cross_attn_bias": cross_attn_bias`
in `dec_feed_kv`.

The dynamic decoder path does not use this (no static KV buffer, no padding issue).

### 4. `orchestrated_demo_static_export.py` — decode()

Same pattern as _inference.py:
```python
cross_attn_bias = np.zeros(
    (1, self.heads, 1, self.max_memory_len), dtype=np.float32
)
cross_attn_bias[:, :, :, valid_len:] = -1e9
```

Replace `"cross_kv_valid_len": cross_kv_valid` with `"cross_attn_bias": cross_attn_bias`.

Also update the in-place probe: probe no longer needs a `cross_kv_valid_len` slot; add
`"cross_attn_bias": np.zeros((1, self.heads, 1, self.max_memory_len), dtype=np.float32)`.

Update the `__init__` shape-detection to look for `cross_attn_bias` instead of
`cross_kv_valid_len` when checking model inputs.

## dtype note

`cross_attn_bias` is added to `attn_weights` which are bf16 in the model. The dummy for
export should be `torch.bfloat16`. On the host (numpy/OnnxRuntime), use `float32` — ORT
will cast as needed. For the compiled VMFB on SL2610, it arrives as bf16 naturally.
-1e9 is representable in bf16 (within the ~±3.39e38 range) and is large enough that
`exp(-1e9 / sqrt(40))` ≈ 0 in any floating-point arithmetic.

## What this does NOT change

- The KV interface (per-layer self + cross, 24 inputs, 25 outputs) is unchanged.
- The `position_ids` input is unchanged.
- The non-streaming (dynamic) decoder export is unchanged.
- Inference quality: padded cross-KV slots get ≈0 attention weight (correct), same as
  explicit bool masking would give.
