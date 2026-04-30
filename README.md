# torq-tools
Collection of tools for the development of Torq models

## Installation
**The Torq compiler is required to run Torq tools.** 
<br>Please see the [documentation](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html#quickstart) on installing the compiler as a release package or a Docker.


Once the compiler is available, this repository can be installed either as a pip package or as a Git submodule within another project.
First, clone the repository:
```bash
git clone https://github.com/synaptics-torq/torq-tools.git
torq_tools_dir=$(readlink -f torq-tools)
```

### Option 1: Install with pip
Installing via pip makes `torq-tools` available system-wide (or within your virtual environment).
A virtual environment is **strongly recommended**, as this project depends on several large packages. 

- **If using the compiler release package:**
  Activate the same virtual environment that was used to set it up.
- **If using Docker:**
  You can use the system Python environment, as it already operates within an isolated environment.
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

You can make the submodule importable under the `torq.tools` namespace using either of the following techniques:

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
#### Convert ONNX model dtype
Convert fp32 ONNX models to lower-precision formats such as bf16 or fp16.
Particularly useful for getting bf16 models, which have native hardware acceleration in the Torq runtime.
```bash
python3 -m src.torq.tools.convert_dtype -d bf16 -i model_fp32.onnx -o model_bf16.onnx
```
This tool can also downcast int64 tensors to int32 or smaller integer data types like int16 and int8.
```bash
python3 -m src.torq.tools.convert_dtype -d int32 -i model.onnx -o model_int32.onnx
```
> [!WARNING]
> Some operator inputs/outputs cannot be downcasted to int32 due to ONNX spec constraints and are preserved as int64. 
> Additionally, downcasting to small integers like int8 can have a detrimental effect on inference accuracy.

#### Export supported models to static graphs
Model export pipelines generate static graphs in the model’s original runtime.
These pipelines also apply a range of graph edits to make models more compatible and efficient for the Torq runtime.
```bash
python3 -m src.torq.models.<model>.export
```
For example, to export a static bf16 Moonshine model:
```bash
python3 -m src.torq.models.moonshine.export --convert-dtype bf16
``` 

#### Compile models
A helper utility is provided for compiling ONNX or MLIR models into VMFB binaries.
```bash
python -m src.torq.utils.compile model_bf16.onnx -t llvm-cpu
python -m src.torq.utils.compile model_bf16.mlir -t llvm-cpu
```

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
> [!Note]
><details>
><summary>Notes on using the Torq compiler docker for compilation and inference</summary>
> The iree-compile and iree-run-module binaries used depend on your environment:
> 
> - Inside the Torq compiler Docker:
> Uses the binaries bundled in the image, ensuring full compatibility with the Torq runtime.
> 
> - Outside the Docker (e.g., in a local venv):
> Uses binaries installed from PyPI. These are fine for testing or validation but not guaranteed to match Torq runtime behavior exactly.
>
></details>

### CLI usage
If `torq-tools` was installed as a Python package, all major tools are also exposed as CLI commands.
```bash
# convert to bf16
torq-convert-dtype onnx -d bf16 -i model_fp32.onnx -o model_bf16.onnx

# export models
torq-export-model moonshine --convert-dtype bf16

# compile models
torq-compile-model model_bf16.onnx -t llvm-cpu
torq-compile-model model_bf16.mlir -t llvm-cpu

# run inference
torq-infer-model moonshine apostle.wav -m models/moonshine_tiny_onnx/ -s tiny
torq-infer-model moonshine apostle.wav -m models/moonshine_iree_onnx/ -s tiny --max-inp-len 80000 --max-dec-len 30
```

### Using in code
You can import and use the same tools programmatically through the torq namespace:
```python
>>> from torq.tools.convert_dtype.onnx import convert_model
>>> from torq.models.moonshine.export import MoonshineModelExporter
>>> exporter = MoonshineModelExporter(...)
>>> exporter.export_onnx()
>>> convert_model(...)
```

---

## Gemma-3 270M-IT LLM Export

Export pipeline for the [gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) model, converting quantized ONNX models to static bf16 VMFBs for the Torq board.

### Source Models

Pre-quantized ONNX models are downloaded automatically from:
[onnx-community/gemma-3-270m-it-ONNX](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/tree/main/onnx)

| File | Description | Size |
|------|-------------|------|
| `model_q4.onnx` + `_data` | int4 quantized (MatMulNBits, bits=4, block_size=32) | ~323 MB |
| `model_quantized.onnx` + `_data` | int8 quantized (MatMulNBits, bits=8, block_size=32) | ~545 MB |
| `model.onnx` + `_data` | fp32 original | ~1.14 GB |

### Pipeline Overview

```
Source (HuggingFace quantized ONNX)
  → Dequantize weights (MatMulNBits → fp32 MatMul, using bf16-precision scales)
  → Make static (fixed KV cache, seq_len=256)
  → Post-static patches (fuse ops, merge KV I/O)
  → Convert to bf16
  → Compile to VMFB (IREE + Torq backend)
```

> **Note:** Quantization scales are rounded to bf16 precision *before* dequantization by default.
> This ensures the scale precision matches the final bf16 compute precision, producing
> consistent weights without precision mismatch artifacts.

### Quick Start: Export int4 model

```bash
cd torq-tools-dev
source .venv/bin/activate
PYTHONPATH=src python -m torq.models.gemma3.export \
    --model-dtype int4 \
    --instruct-model \
    --extract-embeddings \
    --dequantize-weights \
    --convert-dtypes
```

The int4 source model is downloaded automatically from HuggingFace on first run.

### Export int8 model

```bash
PYTHONPATH=src python -m torq.models.gemma3.export \
    --model-dtype int8 \
    --instruct-model \
    --extract-embeddings \
    --dequantize-weights \
    --convert-dtypes
```

The int8 source model is downloaded automatically from HuggingFace on first run.
Output: `models/google/gemma-3-270m-it/export/onnx/int8_converted/static/model.onnx`

### bf16 Scale Dequantization

The export pipeline uses bf16-precision scales by default during weight dequantization.
This is integrated into the `--dequantize-weights` flag — no separate step is needed.

The standalone `convert_bf16_scales.py` script can regenerate bf16-scale models from
existing static models if needed:

```bash
# Regenerate int8_converted_bf16_scales and int4_converted_bf16_scales from source
PYTHONPATH=src python convert_bf16_scales.py --source-type both
```

### Layer Sensitivity Analysis (layer_sensitivity.py)

Runs teacher-forced perplexity analysis to determine which layers benefit most from int8:

```bash
PYTHONPATH=src python layer_sensitivity.py --n-prompts 10 --n-tokens 30
```

Creates a mixed int8/int4 model keeping only the most sensitive layers at int8 precision.

### Compile to VMFB

```bash
cd /path/to/torq-compiler-dev
source ../venv/bin/activate
./compile_v1.5.sh ../torq-tools-dev/models/google/gemma-3-270m-it/export/onnx/converted/static/model.onnx \
    --torq-enable-transpose-optimization \
    --torq-enable-torq-hl-tiling
```

### Key Export Flags

| Flag | Description |
|------|-------------|
| `--model-dtype int4\|int8` | Quantization type (auto-downloads source from HuggingFace) |
| `--instruct-model` | Use the instruct-tuned (-it) variant |
| `--extract-embeddings` | Extract token embeddings to `.npy` (input becomes embedding vector) |
| `--dequantize-weights` | Dequantize MatMulNBits → fp32 MatMul (bf16 scales by default) |
| `--convert-dtypes` | Convert fp32 → bf16 for IREE compile |
| `--skip-iree` | Skip VMFB compilation step |
| `--onnx-source-dir DIR` | Override source model directory (skips auto-download) |
| `--max-gen-tokens N` | Max sequence length (default: 256) |
