# Implementation Plan: 5-Split Static ONNX Model Export and Inference

This plan details the proposed approach to implement the static shape export and inference pipeline for the `moonshine_streaming_5_split` model.

## Goal Description
Support exporting the 5-split Moonshine Streaming model components with fixed (static) input and output shapes. This is necessary for compiling the models for accelerators/hardware targets (such as Torq/Leda) that require fixed tensor dimensions. The static shapes are configurable via CLI arguments.

---

## Proposed Dims & CLI Arguments

We will fix the dynamic shapes of the 5 component models using three primary parameters:
1. **`--input-seconds`**: Maximum input audio duration (default: `5`).
2. **`--tokens-per-sec`**: Maximum tokens decoded per second (default: `6`).
3. **`--chunk-len`**: The frame length of the audio chunk fed to the preprocessor (default: `80` samples / 5ms).

These translate to the following static dimensions:
* `num_samples` = `input_seconds * 16000` (default: `80000`)
* `enc_seq_len` = `num_samples // 320` (default: `250`)
* `max_tokens` = `input_seconds * tokens_per_sec` (default: `30`)

---

## Proposed Changes

### Component: Exporter (`src/torq/models/moonshine_streaming_5_split`)

#### [MODIFY] [export.py](file:///home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/src/torq/models/moonshine_streaming_5_split/export.py)
* Add three optional CLI arguments to the argument parser:
  * `--input-seconds` (type `int`, default `5`)
  * `--tokens-per-sec` (type `int`, default `6`)
  * `--chunk-len` (type `int`, default `80`)
* Use the parsed values directly to set `input_seconds`, `tokens_per_sec`, and `chunk_len`.
* Implement `_make_decoder_kv_static` using ONNX Graph Surgeon:
  * Append `current_len` as a `[1, 1]` graph input and squeeze it to 1D.
  * Invoke the graph editor's `.replace_dynamic_kv_cache(cur_len, self._max_tokens)` to replace dynamic KV concatenation with fixed-size buffer updates.
  * Mask future attention scores with `.mask_future_attn_scores(cur_len, self._max_tokens)`.
  * Inject the length input with `.add_curr_len_input(cur_len)`.
  * Convert dynamic index ranges to static index gathers with `.convert_to_static_index()`.
* Update `_make_frontend_static` to support custom `chunk_len` from the parsed args.

#### [MODIFY] [_graph.py](file:///home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/src/torq/models/moonshine_streaming_5_split/_graph.py)
* Update `fix_decoder_kv_io` to support replacing the dynamic `past_seq` dimension of the self-attention KV caches (`k_self`/`v_self`) and its outputs with the fixed `max_tokens` dimension.

#### [MODIFY] [_inference.py](file:///home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/src/torq/models/moonshine_streaming_5_split/_inference.py)
* Update `MoonshineStreaming5Split` to dynamically detect if the loaded decoder is a static model (by checking if `"current_len"` is an input in the ONNX session).
* If static mode is detected:
  * Initialize the self-attention cache (`k_self`/`v_self`) at the fixed shape `[6, 1, 8, max_tokens, 40]` instead of starting with sequence length `0`.
  * Feed `"current_len"` as a `[1, 1]` array of the step index during decoding.
  * In-place update `k_self`/`v_self` with the output caches at each step.

---

## Verification Plan

### Automated Tests
* Validate the exported models:
  ```bash
  # 1. Export static models using default shapes
  PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_5_split.export --skip-torq all

  # 2. Export static models using custom shapes
  PYTHONPATH=src ../venv/bin/python -m torq.models.moonshine_streaming_5_split.export --skip-torq all --input-seconds 10 --tokens-per-sec 8 --chunk-len 160

  # 3. Run debug_parity.py on the static models to verify parity vs eager PyTorch
  PYTHONPATH=src ../venv/bin/python src/torq/models/moonshine_streaming_5_split/debug_parity.py
  ```

### Manual Verification
* Verify that passing different `--input-seconds`, `--tokens-per-sec`, and `--chunk-len` values correctly re-scales all exported models' static input/output shapes.
