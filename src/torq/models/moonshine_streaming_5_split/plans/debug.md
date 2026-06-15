# Debugging Report: 5-Split Moonshine Streaming Parity

This document outlines the findings, errors encountered, and the approaches tried during the implementation and validation of the `moonshine_streaming_5_split` ONNX export pipeline.

---

## 1. Key Findings

### Finding A: Exporter Parameter Key Mismatch (`UserError`)
* **Problem**: The Torch Dynamo-based exporter (`dynamo=True`) threw a `UserError` because the top-level keys in the `dynamic_shapes` dictionary did not match the exact parameter names of the wrapper modules' `forward` signatures.
  - Encoder wrapper input parameter was named `input_features`, but the `dynamic_shapes` dict used `"features"`.
  - Adapter wrapper input parameter `pos_offset` was not specified in the `dynamic_shapes` dict.
* **Solution**: Updated the dynamic shape definitions in `export.py` to match the wrapper parameters exactly (changed `"features"` to `"input_features"` and mapped `"pos_offset": None` to declare it as a static dimension).

### Finding B: Downsampling Stride and Chunk Size Alignment
* **Problem**: Moonshine downsamples raw audio frames by a factor of 4 using two stride-2 causal convolutions in the preprocessor. When the input audio was processed in chunks of 80 samples (5ms, 1 raw frame), the preprocessor generated 1 feature frame at *every* step due to padding. This broke the stride-4 downsampling, resulting in 4x too many features (6,600 instead of 1,650) and causing the positional embedding index to exceed the maximum bound of 4096.
* **Solution**: The chunk size must be a multiple of 320 samples (20ms, 4 raw frames) to preserve the downsampling stride alignment. Using a chunk size of 640 samples (40ms, 8 raw frames) outputs 2 features per step and keeps the states and strides aligned perfectly.

### Finding C: Encoder State and Chunking Limitations
* **Problem**: Unlike the decoder, the Moonshine transformer encoder does not have a past key-value cache. If the encoder is run chunk-by-chunk on individual 40ms features, the self-attention mechanism is restricted to that chunk and cannot see past frames, breaking the sliding-window causal attention.
* **Solution**: The preprocessor must run step-by-step to accumulate the features. Once all features are collected, the encoder, adapter, and cross-attention key-value generator are run *once* on the entire concatenated sequence.

### Finding D: Perfect Numerical Parity (ONNX vs PyTorch Eager)
* **Approach**: Created a parallel verification test comparing the intermediate outputs of:
  1. PyTorch Eager model execution.
  2. Our custom PyTorch wrapper execution.
  3. Our exported ONNX models executed using ONNX Runtime.
* **Results**: Once the stride alignment, feature accumulation, and asinh monkey patch were corrected, we achieved near-perfect numerical parity across all 5 models:
  - **Preprocessor**: Max Diff = $9 \times 10^{-6}$
  - **Encoder**: Max Diff = $7.8 \times 10^{-5}$
  - **Adapter**: Max Diff = $7.8 \times 10^{-5}$
  - **Cross KV Generator**: Max Diff = $2.3 \times 10^{-4}$
  - **Decoder Logits**: Max Diff = $1.9 \times 10^{-5}$

### Finding E: Missing `log_k` Parameter in `asinh` Monkey Patch
* **Problem**: During custom preprocessor export, the `asinh` function in `MoonshineStreamingAsinhCompression` was monkey-patched to bypass an ONNX export limitation. However, the patched function omitted the learnable parameter `self.log_k`, computing `log(x + sqrt(x^2 + 1))` instead of `log(exp(log_k)*x + sqrt((exp(log_k)*x)^2 + 1))`. This caused a large numeric discrepancy (up to 2.89) between baseline preprocessor outputs and the 5-split preprocessor, leading to incorrect first token predictions ("What" instead of "The").
* **Solution**: Updated `_patched_asinh_forward` in `export.py` to correctly scale inputs by `torch.exp(self.log_k)`. After purging cached source ONNX models under the `models/` subdirectory and re-running the export pipeline, the exported preprocessor outputs matched the baseline exactly.

---

## 2. Status & Resolutions

### Logits and Token Discrepancy (e.g., "What's up?" vs "The birch...")
* **Status**: **RESOLVED**. The token generation mismatch was entirely caused by the numerical discrepancy in the stateful preprocessor. With the corrected `asinh` patch, the first generated token ID is correctly predicted as `450` ("The"), matching the batch baseline. The full validation transcription on `OSR_us_000_0010_8k.wav` now completes successfully and produces the expected full paragraph transcription:
  > "The birch cannines split on the smooth planks. Glue the sheet to the dark blue background..."

