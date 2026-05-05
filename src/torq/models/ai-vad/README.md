# ai-vad

Export, inference and validation tools for the Synaptics **AEC-VAD** (Acoustic Echo
Cancellation + Voice Activity Detection) ONNX model, plus a few standalone ONNX
graph-cleanup utilities that make VAD-style graphs compatible with the Torq
compiler / IREE runtime.

The package plugs into the `torq-tools` framework: it lives under
`torq.models.ai-vad` and reuses the Moonshine inference runners in
`_inference.py` as its ONNX/VMFB execution backends.

## Layout

| File | Purpose |
| ---- | ------- |
| `__init__.py` | Argparse builders: `add_aivad_export_args`, `add_moonshine_infer_args`, and the allowed dtype lists (`ONNX_DTYPES`, `OPTIMUM_DTYPES`). |
| `export.py` | `AiVadModelExporter` — static ONNX export pipeline for the AEC-VAD model (`aec_vad_exp12_d4_model_epoch_t710.onnx`), plus optional dtype conversion and IREE/VMFB compilation. CLI entry point: `main()`. |
| `_graph.py` | ONNX GraphSurgeon editor (`AiVadOnnxGraphEditor`) and shared graph edits (Pad/Concat rewires, int64→bf16 cast LUT, KV-tensor combining, embeddings extraction, scalar MatMul folding, …). Used by `export.py`. |
| `_inference.py` | Moonshine-style inference runners (`MoonshineBase`, `MoonshineDynamic`, `MoonshineStatic`, `load_moonshine`) used by both `infer.py` and `validate.py`. |
| `infer.py` | CLI demo that transcribes one or more WAV files through the Moonshine runner (`python -m torq.models.ai-vad.infer ...`). |
| `inference.py` | Standalone smoke test: loads `aec_vad_exp12_d5_quantized_t39.onnx` with `onnxruntime` and prints the outputs for a synthetic batch. |
| `validate.py` | Accuracy/WER harness that compares ONNX-Runtime FP32, IREE-Runtime FP32 and (optionally) IREE-Runtime BF16 on the LibriSpeech `test.clean` split; writes a summary plus an `.xlsx` report via `openpyxl`. |
| `fold_reshape_shape.py` | Standalone CLI: folds `Shape → ... → Reshape` subgraphs into constant shape initializers. |
| `fold_pad_constants.py` | Standalone CLI: folds `ConstantOfShape` / shape-only subgraphs feeding `Pad` into concrete constant tensors (after optional static input/output shape pinning). |
| `requirements.txt` | Extra Python deps required by this package (`soundfile`, `torchcodec`, `whisper-openai`, `jiwer`, `openpyxl`). |

`_graph.py`, `_inference.py` and `__init__.py` are internal helpers — the stable
entry points are `export.py`, `infer.py`, `validate.py`, `fold_reshape_shape.py`
and `fold_pad_constants.py`.

The narrow strided-depthwise Conv widening previously documented as a
standalone CLI is now applied automatically by
`AiVadModelExporter._make_aivad_model_static`. The underlying graph edit lives
in `_graph.py` as `WidenSmallStridedDepthwiseConv` (exposed via
`AiVadOnnxGraphEditor.widen_small_strided_depthwise_conv`).

## Installation

From the root of the `torq-tools` repository (see the top-level
[`README.md`](../../../../README.md) for the full install guide):

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r src/torq/models/ai-vad/requirements.txt
```

You also need a working Torq compiler environment (`iree-compile`,
`iree-run-module`, matching Python bindings). Any of the standard installs
documented in the
[Torq getting-started guide](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html)
work.

## Typical workflow

1. **Export** the source AEC-VAD ONNX model to a static, Torq-friendly ONNX
   graph and (optionally) compile it to a VMFB:

   ```bash
   python -m torq.models.ai-vad.export \
       --onnx-model path/to/aec_vad_exp12_d4_model_epoch_t710.onnx \
       --dtype float \
       --models-dir models
   ```

   Useful flags (all defined in `add_aivad_export_args`):

   - `--dtype {float,quantized,quantized_4bit,fp32,fp16,bf16}` — source ONNX
     dtype (`ONNX_DTYPES`) or an Optimum-style dtype (`OPTIMUM_DTYPES`) used
     when `--use-optimum` is set.
   - `--convert-dtypes` — additionally emit a converted-dtype copy (e.g. for
     BF16 Torq runs).
   - `--dynamic-models` — keep dynamic shapes (skips the static rewrite).
   - `--extract-embeddings` — pull large embedding LUTs into external `.npy`
     files next to the model.
   - `--replace-int-bf16-cast` — replace `int64 → bf16` casts with a look-up
     table (needed for Torq bf16 export).
   - `--broadcast-ops [OP ...]` — broadcast op inputs up to their output shape
     (pass with no arguments to apply globally).
   - `--skip-export {encoder,decoder,decoder_with_past,decoder_merged}` — skip
     specific sub-components.
   - `--skip-iree` — stop after ONNX export and skip IREE compilation.
   - Logging/IREE flags are contributed by `add_logging_args` and
     `add_iree_args`.

   Outputs are written to
   `models/export/onnx/<dtype>/{static,dynamic}/`,
   `models/export/onnx/converted/...` (when `--convert-dtypes` is set) and
   `models/export/iree/.../` (unless `--skip-iree`).

2. **Run inference** on one or more WAV files via the Moonshine runner:

   ```bash
   python -m torq.models.ai-vad.infer \
       sample.wav \
       -m models/export/iree/float/static/ \
       -s tiny \
       --max-inp-len 80000 \
       --max-dec-len 30
   ```

   `--max-inp-len` / `--max-dec-len` are required whenever the target model is
   a static VMFB build.

3. **Validate accuracy** against the LibriSpeech `test.clean` split and emit a
   WER comparison spreadsheet:

   ```bash
   python -m torq.models.ai-vad.validate \
       --onnx-models-dir models/export/onnx/float/static/ \
       --vmfb-models-dir models/export/iree/float/static/ \
       --bf16-vmfb-models-dir models/export/iree/bf16/static/ \
       --model-size tiny \
       --max-inp-len 300000 \
       --max-dec-len 96 \
       --max-samples 250
   ```

   Only ONNX *or* VMFB models may live in a given `--*-models-dir`. The script
   sets `IREE_RUN_MODULE_FLAGS=--task_topology_group_count=<N>` (tunable via
   `--task-topology-group-count`) before the IREE runs.

4. **Quick ONNX smoke test** (no Torq compiler or dataset required):

   ```bash
   python torq-tools-dev/src/torq/models/ai-vad/inference.py
   ```

   This script is hard-coded to load `aec_vad_exp12_d5_quantized_t39.onnx`
   from the current working directory and print the output tensors for a
   random 1×2×1×256 `in_frame_mag` and zero `input_state`. Use it as a minimal
   sanity check for a freshly-produced ONNX model.

## Standalone graph-cleanup CLIs

Both scripts are pure ONNX utilities and can be used outside the export
pipeline. They have no dependency on Torq itself.

### `fold_reshape_shape.py`

Folds `Shape → Slice/Gather/Concat/... → Reshape` subgraphs into constant
`Reshape` shape inputs. Helpful when a source graph reconstructs a static
shape at runtime; after folding, the `Reshape` consumes a constant and the
original producer chain becomes dead code.

```bash
python fold_reshape_shape.py input.onnx output.onnx
```

### `fold_pad_constants.py`

Folds `ConstantOfShape` / shape-only producers feeding a `Pad` node into a
single constant `pads` (or value) tensor, and optionally pins input/output
shapes before folding. Useful for models where `Pad` amounts are derived from
runtime `Shape` math.

```bash
python fold_pad_constants.py input.onnx output.onnx \
    --input-shape in_frame_mag=1,2,1,256 \
    --output-shape vad=1,1 \
    --output-shape hidden_state=1,16,64
```

- `--input-shape NAME=D0,D1,...` and `--output-shape NAME=D0,D1,...` may be
  repeated.
- `--skip-cleanup` leaves dead nodes/initializers in place.
- The script runs `onnx.shape_inference.infer_shapes` before and after
  folding, and reports the number of `Pad` inputs successfully replaced.

## Notes and caveats

- The Moonshine-derived inference layer assumes `--model-size {base,tiny}`;
  pass the size that matches the decoder KV-head configuration of the model
  you exported.
- `validate.py` downloads `hf-audio/esb-datasets-test-only-sorted` first and
  falls back to `torchaudio`’s `LIBRISPEECH(test-clean)` if the primary
  dataset is unavailable.
- `inference.py` expects a specific filename (`aec_vad_exp12_d5_quantized_t39.onnx`)
  in the working directory; edit the path at the top of the file if your copy
  lives elsewhere.
- License: Apache-2.0 (see SPDX headers in every source file).
