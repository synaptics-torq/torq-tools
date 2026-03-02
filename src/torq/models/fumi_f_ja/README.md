# Tsuki TTS (Fumi – Japanese)

Usage instructions for exporting and running the Japanese **Text-to-Speech (TTS)** model.

> ⚠️ Note: This model contains **dynamic shapes** by design.

---

## Model Structure

This TTS model operates with **dynamic input and output dimensions**, meaning several tensor shapes vary at runtime depending on the input text and predicted audio duration.

### Input

**Name:** `texts`  
**Shape:** `int64[1, s26]`

- `1` → Batch size (fixed)
- `s26` → **Dynamic text length** (number of tokens)

The model supports variable-length input text sequences.

---

### Output

**Name:** `audio`  
**Shape:** `float32[300*u0]`

- `u0` → **Dynamic duration variable**
- `300` → Fixed scaling factor (typically related to hop length or vocoder upsampling factor)

The output is a 1D `float32` tensor representing the generated waveform.

Since `u0` depends on the predicted speech duration, the final waveform length varies proportionally to the input text.

---

## Dynamic Shape Propagation

Several intermediate tensors depend on:

- `s26` (input token length)
- `u0` (predicted frame count / duration)

Because of this dependency, the graph contains symbolic dimensions throughout.

However, it is possible to export a **static version** of the model by fixing:

- Input text length (`s26`)
- Output audio duration (`u0`)

---

# Exporting a Static Model

You can generate a static ONNX model by specifying fixed values for the dynamic dimensions.

## Prerequisites

- Activate your virtual environment
- Run commands from the `torq-tools` root directory

---

## Export Command
- **text-len:**
  Configurable input text length.
- **audio-len:**
  Configurable output audio duration.
- **skip-iree:**
  To export a static onnx without generating an executable .vmfb
- **models-dir:**
  You can set any directory to store the static model, but below is a suggested path.
```bash
python3 -m src.torq.models.fumi_f_ja.export \
  --onnx-model src/torq/models/fumi_f_ja/fumi_f_ja.onnx \
  --text-len 128 \
  --audio-len 100 \
  --models-dir src/torq/models/output \
  --skip-iree \
  --skip-validation
```

