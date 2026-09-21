# torq-tools
Collection of tools for the development of Torq models

## Installation

Clone the repository:
```bash
git clone https://github.com/synaptics-torq/torq-tools.git
torq_tools_dir=$(readlink -f torq-tools)
```

### Compiler dependency

`torq-tools` depends on the Torq compiler Python package (`torq-compiler>=2.2.0`): the model exporters use it for ONNX dtype conversion, dynamic quantization, and the `.onnx`/`.tflite` -> MLIR -> VMFB compilation flow, and the generic model tools (`torq-convert-dtype`, `torq-quantize-model`, `torq-convert-static`) ship in that package. A pip install of `torq-tools` brings it in automatically; If you use torq-tools from a source checkout or submodule without installing it, install `torq-compiler` into the same environment.

> [!NOTE]
> **Temporary (until torq-compiler v2.2.1):** `torq-compiler` is not published on PyPI yet. Install the v2.2.0 release wheel directly from GitHub **with the `onnx` extra**, and do this **before** installing `torq-tools` or any of the `requirements.txt` files:
>
> ```bash
> pip install "torq-compiler[onnx] @ https://github.com/synaptics-torq/torq-compiler/releases/download/v2.2.0/torq_compiler-2.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
> ```

Please see the [documentation](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html#quickstart) on installing the compiler Python package.

### Option 1: Install with pip
Installing via pip makes `torq-tools` available system-wide (or within your virtual environment).
A virtual environment is **strongly recommended**, as this project depends on several large packages.

```bash
cd your_project
source .venv/bin/activate
pip install $torq_tools_dir --extra-index-url https://download.pytorch.org/whl/cpu
```

> [!TIP]
> For development, install in editable mode:
> ```bash
> pip install -e $torq_tools_dir
> ```
> This allows changes in the source tree to take effect immediately without reinstalling.

Pip installation also registers several CLI entry points.

### Option 2: Include as a Git submodule
Include torq-tools as a submodule in your project:
```bash
cd your_project
git submodule add https://github.com/synaptics-torq/torq-tools.git external/torq-tools
git submodule update --init --recursive
```
Then install requirements:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

You can make the submodule importable under the `torq` namespace using either of the following techniques:

**Technique A: Add to site-packages**
Add the submodule path permanently to the Python environment by creating a .pth file in your site-packages directory:
```bash
echo "$torq_tools_dir/src" >> $(python3 -c 'import site; print(site.getsitepackages()[0])')/torq.pth
```

**Technique B: Add to sys.path at runtime**
Append the `src/` directory from the submodule to the Python path, e.g. in your project’s initialization code:
```python
import sys
sys.path.append("external/torq-tools/src")
```

### Install extras
The project ships with optional extras for specific export and validation workflows:

| extra | purpose |
| :---: | ------------ |
| "moonshine" | Install dependencies for Moonshine export and validation |
| "all" | Install dependencies for all extras |

You can install these extras alongside the base package:
```bash
pip install $torq_tools_dir[moonshine]
```
Or manually via their requirements file:
```
pip install -r "$torq_tools_dir/src/torq/models/moonshine/requirements.txt"
```

## Usage
`torq-tools` can be used directly from the command line or imported into application code via the `torq` namespace.

### Available tools
#### Clean up ONNX exporter artifacts
Model-agnostic cleanup for ONNX graphs that don't go through a full `torq.models` exporter.
Composes the `CollapseUnrolledConcat` and `FoldConvBatchNorm` graph edits with ORT-backed constant folding:
it collapses per-element unrolled stack/unbind Concats back into their source tensor,
evaluates all-constant subgraphs (positional-embedding builders, constant weight prep, ...) into initializers,
and folds exported eval-mode BatchNorm (`Conv -> Mul -> Add`) into the conv weights.
Run it on fp32 graphs, before dtype conversion.
```bash
torq-cleanup-model onnx model_fp32.onnx -o model_clean.onnx --verify
```
`--verify` re-runs both models under onnxruntime on random inputs and asserts the outputs still match.
Individual passes can be disabled with `--skip {collapse-concat,fold-constants,fold-conv-bn}`,
and the graph edits are also usable on their own via `--apply-graph-edit` on any model exporter.
Constants larger than `--fold-size-threshold` bytes (default 16 MiB) are not folded, so folding
can't blow up a model by materializing e.g. a transposed lm_head matrix.
The pipeline is idempotent and fails safe, so every model exporter also runs it on each exported
component by default (before dtype conversion); opt out with `--no-onnx-cleanup`.
#### Benchmark quantized models
Run a quantized (VMFB or ONNX) Gemma3 model over a standard question set and compare two runs (throughput, time-to-first-token, answers):
```bash
python -m src.torq.utils.benchmark run -m /path/to/model.vmfb --instruct-model -o results.json
python -m src.torq.utils.benchmark compare -a results_int8.json -b results_mixed.json --name-a int8 --name-b mixed -o comparison.md
```
See `src/torq/utils/benchmark/README.md` for options.
#### Export supported ONNX models to static graphs
Model export pipelines generate static graphs in the model’s original runtime.
These pipelines also apply a range of graph edits to make models more compatible and efficient for the Torq runtime.
```bash
python3 -m src.torq.models.<model>.export
```
For example, to export a static bf16 Moonshine model:
```bash
python3 -m src.torq.models.moonshine.export --convert-dtype bf16
```
To export a static Moonshine Streaming model (fixed audio chunk, with dtype conversion):
```bash
python3 -m src.torq.models.moonshine_streaming.export --chunk-len 1280 --convert-dtypes
```
> [!NOTE]
> Moonshine Streaming requires the `moonshine-streaming` extra (`transformers>=5.5.1`). The
> exporter always produces static models and needs `--chunk-len` (audio samples per chunk,
> e.g. `1280` = 80 ms @ 16 kHz). It emits `encoder.onnx` + `decoder.onnx` plus host-side
> `*.npy` LUTs and a `streaming_config.json`.

> [!WARNING]
> This tool assumes the model's default `shape` values are valid and mutually consistent. If any tensor has an invalid
> default shape (e.g., `0` or `-1`), the exported model will have incorrect static shapes.

#### Run inference
You can run inference directly using helper scripts that support multiple runtimes.
```
python -m src.torq.models.<model>.infer ...
```
Example: run Moonshine inference with ONNX and VMFB backends:
```bash
python -m src.torq.models.moonshine.infer apostle.wav -m models/moonshine_tiny_onnx/ -s tiny
python -m src.torq.models.moonshine.infer apostle.wav -m models/moonshine_iree_onnx/ -s tiny --max-inp-len 80000 --max-dec-len 30
```
Example: run Moonshine Streaming inference (point `-m` at an export dir containing
`encoder.onnx`, `decoder.onnx`, the `*.npy` LUTs and `tokenizer.json`):
```bash
python -m src.torq.models.moonshine_streaming.infer apostle.wav \
  -m models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/static -s tiny
```

### CLI usage
If `torq-tools` was installed as a Python package, its tools are exposed as CLI commands; the generic model tools (`torq-convert-dtype`, `torq-quantize-model`, `torq-convert-static`) are provided by the `torq-compiler` dependency and are available in the same environment.
```bash
# clean up exported artifacts
torq-cleanup-model onnx model_fp32.onnx -o model_clean.onnx --verify

# convert to bf16 (torq-compiler)
torq-convert-dtype onnx -d bf16 -i model_fp32.onnx -o model_bf16.onnx

# quantize weights (torq-compiler)
torq-quantize-model weights analyze -i model_fp32.onnx -o sensitivity.json --config-output quant_config.json --embeddings token_embeddings.npy
torq-quantize-model weights quantize -i model_fp32.onnx -o model_int8.onnx --bits 8
torq-quantize-model weights quantize -i model_fp32.onnx -o model_mixed.onnx --config quant_config.json --dequantize-weights

# export models
torq-export-model moonshine --convert-dtype bf16
torq-export-model moonshine_streaming --chunk-len 1280 --convert-dtypes

# run inference
torq-infer-model moonshine apostle.wav -m models/moonshine_tiny_onnx/ -s tiny
torq-infer-model moonshine apostle.wav -m models/moonshine_iree_onnx/ -s tiny --max-inp-len 80000 --max-dec-len 30
torq-infer-model moonshine_streaming apostle.wav -m models/moonshine_streaming_static/ -s tiny
```

### Using in code
You can import and use the same tools programmatically through the torq namespace:
```python
>>> from torq.lab.model_tools.dtype_conversion.onnx import convert_model
>>> from torq.models.moonshine.export import MoonshineModelExporter
>>> exporter = MoonshineModelExporter(...)
>>> exporter.export_onnx()
>>> convert_model(...)
```