# torq-tools
Collection of tools for the development of Torq models

## Installation

Clone the repository:
```bash
git clone https://github.com/synaptics-torq/torq-tools.git
torq_tools_dir=$(readlink -f torq-tools)
```

### Compiler dependency

The Torq compiler Python package (`torq-compiler`) is **required when exporting models** (i.e. when running model exporters or compiling `.onnx`/`.tflite` files). These workflows need the compiler's Python bindings to convert ONNX/TFLite -> MLIR -> VMFB.

The compiler package is **not required** if you only need to compile pre-exported `.mlir` files and already have a `torq-compile` binary available on your `PATH` (or pointed to via `--compiler-path` / `TORQ_COMPILER_PATH`).

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
> By default these tensors are kept as int64 in place (no extra casts). Pass `--enforce-io-casts` to instead insert `Cast` nodes at these edges, which is needed for models where a spec-mandated int64 input is produced by a generic integer op (otherwise the resulting graph has mixed-type producers).
> Additionally, downcasting to small integers like int8 can have a detrimental effect on inference accuracy.
#### Quantize ONNX model weights
Quantize MatMul weights in fp32 ONNX models to int4/int8 with optional per-layer sensitivity analysis.
Supports two output modes: DequantizeLinear (DQL) nodes for runtime dequantization, or pre-dequantized bf16 for direct IREE compilation.

**Sensitivity analysis** — determines optimal per-layer bit-width:
```bash
python3 -m torq.tools.quantization.weight_quantization analyze \
    -i model_fp32.onnx -o sensitivity.json --config-output quant_config.json \
    --embeddings token_embeddings.npy --tokenizer tokenizer.json \
    --bits 4 8 --num-tokens 15
```

**Quantize with per-layer config** (DQL output for sharing/further compilation):
```bash
python3 -m torq.tools.quantization.weight_quantization quantize \
    -i model_fp32.onnx -o model_int8_int4_dql.onnx --config quant_config.json
```

**Quantize with pre-dequantized bf16** (ready for IREE compilation):
```bash
python3 -m torq.tools.quantization.weight_quantization quantize \
    -i model_fp32.onnx -o model_bf16.onnx --config quant_config.json --dequantize-weights
```

**Uniform quantization** (all layers same bit-width):
```bash
python3 -m torq.tools.quantization.weight_quantization quantize \
    -i model_fp32.onnx -o model_int8.onnx --bits 8
```

> [!NOTE]
> For reduced-vocab models, pass `--token-lut token_id_lut.npy` to the analyze command
> to map reduced vocab indices back to full vocab IDs during evaluation.
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

#### Convert TFLite models to static shapes
Converts dynamic TFLite models to static by removing `shapeSignature` metadata from tensors, forcing the runtime to use the concrete dimensions already present in the `shape` field. This works for most dynamic models whose default shapes are valid.
```bash
python3 -m src.torq.tools.convert_static tflite \
  -i path/model.tflite \
  -o path/model_static.tflite
```

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

### CLI usage
If `torq-tools` was installed as a Python package, all major tools are also exposed as CLI commands.
```bash
# convert to bf16
torq-convert-dtype onnx -d bf16 -i model_fp32.onnx -o model_bf16.onnx

# quantize weights
torq-quantize-model analyze -i model_fp32.onnx -o sensitivity.json --config-output quant_config.json --embeddings token_embeddings.npy
torq-quantize-model quantize -i model_fp32.onnx -o model_int8.onnx --bits 8
torq-quantize-model quantize -i model_fp32.onnx -o model_mixed.onnx --config quant_config.json --dequantize-weights

# export models
torq-export-model moonshine --convert-dtype bf16

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