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
#### Convert ONNX fp32 models to bf16 or fp16
Convert fp32 ONNX models to lower-precision formats such as bf16 or fp16.
Particularly useful for getting bf16 models, which have native hardware acceleration in the Torq runtime.
```bash
python3 -m src.torq.tools.convert_fp32 -e bf16 -i model_fp32.onnx -o model_bf16.onnx
```

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
torq-convert-fp32 onnx -e bf16 -i model_fp32.onnx -o model_bf16.onnx

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
