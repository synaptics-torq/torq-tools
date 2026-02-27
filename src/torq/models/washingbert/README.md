# WashingBERT

Japanese NLP model for laundry-related intent and multi-label classification, fine-tuned from [LINE DistilBERT Japanese](https://huggingface.co/line-corporation/line-distilbert-base-japanese).

- **HuggingFace:** [Synaptics/WashingBERT](https://huggingface.co/Synaptics/WashingBERT)
- **Precision:** FP16
- **Max sequence length:** 128 tokens

## Model Architecture

Single encoder-only DistilBERT with three classification heads:

| Output | Type | Classes | Description |
|--------|------|---------|-------------|
| `intent_output` | Single-label (softmax) | 4 | `general_info`, `maintenance`, `none`, `wash` |
| `types_output` | Multi-label (sigmoid) | 14 | Cloth/wash type — `bedding`, `care`, `clean_tub`, `delicates`, `dry_only`, `everyday`, `light_rinse`, `low_clean_tub`, `none`, `remove_moisture`, `sportswear`, `towels`, `tub_mold`, `whites` |
| `second_types_output` | Multi-label (sigmoid) | 25 | Secondary attributes — `auto`, `back`, `blanket`, `dewrinkle_deodorize`, `fluffy_soft`, `gentle_wash`, `in_a_hurry`, `light_refresh_overall`, `mud_food_spills_strong_odor`, `no`, `none`, `none_in_particular`, `not_in_a_hurry`, `pollen_sanitize_deodorize`, `preferred_energy`, `preferred_night`, `preferred_powerful`, `remove_yellowing`, `restore_water_repellency`, `sanitize`, `sebum_sweat`, `target_sebum_sweat`, `warm_water_thorough`, `yellowing`, `yes` |

### ONNX Model I/O

**Inputs:**
| Name | Type | Shape |
|------|------|-------|
| `input_ids` | INT64 | `[1, seq_len]` |
| `attention_mask` | INT64 | `[1, seq_len]` |

**Outputs:**
| Name | Type | Shape |
|------|------|-------|
| `intent_output` | FLOAT16 | `[1, 4]` |
| `types_output` | FLOAT16 | `[1, 14]` |
| `second_types_output` | FLOAT16 | `[1, 25]` |

## Installation

Install the WashingBERT dependencies:
```bash
pip install -r src/torq/models/washingbert/requirements.txt
```

Or via the pip extra:
```bash
pip install torq-tools[washingbert]
```

### Dependencies
| Package | Purpose |
|---------|---------|
| `transformers` | `AutoTokenizer` for loading the LINE DistilBERT tokenizer |
| `fugashi` | MeCab morphological analyzer (Japanese word segmentation) |
| `mecab-python3` | MeCab Python bindings |
| `unidic-lite` | Lightweight UniDic dictionary for MeCab |
| `sentencepiece` | SentencePiece subword tokenizer |

## Usage

All commands below assume you are in the `torq-tools-dev` directory.

### Demo (recommended starting point)

The demo script auto-downloads the model from HuggingFace on first run.

```bash
# Run with built-in sample sentences
python3 -m src.torq.models.washingbert.demo

# Classify custom Japanese text
python3 -m src.torq.models.washingbert.demo --text "白いシャツの黄ばみを落としたい"

# Multiple sentences
python3 -m src.torq.models.washingbert.demo \
    --text "タオルをふわふわにしたい" "カビ取りをしたい" "布団を洗いたい"

# Use a local model (skip HF download)
python3 -m src.torq.models.washingbert.demo \
    --model models/WashingBERT/source/onnx/best_multi_task_model_fp16.onnx

# Run without tokenizer deps (uses pre-tokenized sample_inputs.json)
python3 -m src.torq.models.washingbert.demo --use-samples
```

#### Demo options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | *(auto-download)* | Path to local ONNX model file |
| `--models-dir DIR` | `models/WashingBERT/source/onnx` | Directory to cache downloaded model |
| `--text TEXT [TEXT ...]` | *(built-in samples)* | Japanese text(s) to classify |
| `--use-samples` | `false` | Use pre-tokenized `sample_inputs.json` (no tokenizer needed) |
| `--max-seq-len N` | `128` | Maximum input sequence length |
| `--threads N` | *(all cores)* | Number of CPU threads for ONNX Runtime |

### Export

Export prepares the model for the Torq NPU: downloads from HF, makes shapes static, optimizes the graph, optionally converts dtypes, and compiles to IREE.

```bash
# Export with static shapes (auto-downloads from HF)
python3 -m src.torq.models.washingbert.export --skip-iree

# Export from a local source directory
python3 -m src.torq.models.washingbert.export \
    --onnx-source-dir /path/to/local/onnx --skip-iree

# Export + convert to bf16
python3 -m src.torq.models.washingbert.export --convert-dtypes --skip-iree

# Export + compile to IREE VMFB
python3 -m src.torq.models.washingbert.export

# Show model info after export
python3 -m src.torq.models.washingbert.export --skip-iree --show-model-info

# Skip ONNX Runtime optimization
python3 -m src.torq.models.washingbert.export --skip-iree --no-optimize
```

#### Export options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-seq-len N` | `128` | Static sequence length for input tensors |
| `-d`, `--dtype DTYPE` | `float` | Model data type: `float`, `fp32`, `fp16`, `bf16` |
| `--onnx-source-dir DIR` | *(auto-download)* | Local directory containing source ONNX model |
| `--models-dir DIR` | `models` | Base directory for source and exported models |
| `--show-model-info` | `false` | Print model I/O and op summary after export |
| `--no-optimize` | `false` | Skip ONNX Runtime graph optimization |
| `--convert-dtypes` | `false` | Convert exported model to bf16 + int32 |
| `--skip-validation` | `false` | Skip output validation against source model |
| `--skip-iree` | `false` | Skip IREE compilation step |
| `--logging LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

#### IREE compilation options (when not using `--skip-iree`)

| Option | Default | Description |
|--------|---------|-------------|
| `--opset N` | `22` | ONNX opset version (older models are upgraded) |
| `--ic-arg ARG` | — | Pass additional args to the IREE compiler |
| `--cross-compile` | `false` | Cross-compile for aarch64 |
| `--use-iree-cli` | `false` | Use the `torq-compile` binary instead of Python API |

### Inference CLI

```bash
# Classify Japanese text
python3 -m src.torq.models.washingbert.infer \
    "タオルをふわふわにしたい" "カビ取りをしたい" \
    -m models/WashingBERT/source/onnx/best_multi_task_model_fp16.onnx
```

#### Inference options

| Option | Default | Description |
|--------|---------|-------------|
| `TEXT [TEXT ...]` | *(required)* | Japanese text input(s) to classify |
| `-m`, `--model-dir PATH` | *(required)* | Path to ONNX model file or directory |
| `--max-seq-len N` | `128` | Maximum input sequence length |
| `-j`, `--threads N` | *(all cores)* | Number of CPU threads |
| `--logging LEVEL` | `INFO` | Logging verbosity |

### CLI entry points (pip-installed)

If `torq-tools` is installed as a package:
```bash
torq-export-model washingbert --skip-iree
torq-infer-model washingbert "タオルをふわふわにしたい" \
    -m models/WashingBERT/source/onnx/best_multi_task_model_fp16.onnx
```

## Python API

### Inference
```python
from transformers import AutoTokenizer
from torq.models.washingbert._inference import WashingBERTRunner

# Load model (auto-discovers label JSON files from model directory)
runner = WashingBERTRunner.from_onnx(
    "models/WashingBERT/source/onnx/best_multi_task_model_fp16.onnx",
    max_seq_len=128,
)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(
    "line-corporation/line-distilbert-base-japanese",
    trust_remote_code=True,
)
encoded = tokenizer("タオルをふわふわにしたい", max_length=128,
                     padding="max_length", truncation=True, return_tensors="np")

# Run
result = runner.run(encoded["input_ids"], encoded["attention_mask"])
print(result)
# Intent: wash (0.996) | Type1: [towels(0.999)] | Type2: [fluffy_soft(0.926)]

# Access fields
print(result.intent)             # "wash"
print(result.intent_confidence)  # 0.996
print(result.type1_labels)       # ["bedding", "dry_only", "towels"]
print(result.type2_labels)       # ["fluffy_soft", "in_a_hurry", ...]
```

### Export
```python
from torq.models.washingbert.export import WashingBERTModelExporter

exporter = WashingBERTModelExporter(
    model_dtype="float",
    max_seq_len=128,
)
exporter.export_onnx()         # Static shapes + optimize + validate
exporter.convert_models()      # Optional: bf16 + int32 conversion
exporter.export_iree()         # Optional: compile to VMFB
```

### Label map
```python
from torq.models.washingbert._inference import LabelMap

# Auto-load from model directory
labels = LabelMap.from_dir("models/WashingBERT/source/onnx/")
print(labels.intents)  # ["general_info", "maintenance", "none", "wash"]
print(labels.type1)    # ["bedding", "care", "clean_tub", ...]
print(labels.type2)    # ["auto", "back", "blanket", ...]
```

## File Structure

```
src/torq/models/washingbert/
├── __init__.py        # Constants, HF repo IDs, CLI argument parsers
├── export.py          # WashingBERTModelExporter (static shapes, optimize, IREE)
├── _inference.py      # WashingBERTRunner, WashingBERTResult, LabelMap
├── infer.py           # CLI inference wrapper (tokenizer + runner)
├── demo.py            # Standalone demo with HF auto-download
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Model Data

Label classes are stored as JSON files alongside the ONNX model and are auto-discovered at runtime:

| File | Content |
|------|---------|
| `intent_classes.json` | Intent label names (4 classes) |
| `types_classes.json` | Type1 / cloth type label names (14 classes) |
| `sec_types_classes.json` | Type2 / secondary attribute label names (25 classes) |
| `sample_inputs.json` | Pre-tokenized sample inputs for tokenizer-free inference |
