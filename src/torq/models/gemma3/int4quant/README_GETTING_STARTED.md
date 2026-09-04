# How to generate the GPTQ INT4 in-feature ONNX

This document covers how to generate the gptq int4 infeature onnx.

---

## 1. Generate the base ONNX

Run this from the repository root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
huggingface-cli login        

torq-export-model gemma3 \
    --instruct-model --extract-embeddings --convert-dtypes --skip-torq
```

This produces
`models/google/gemma-3-270m-it/export/full/unified/onnx/converted/static/model.onnx`.

---

## 2. Set up the quantization environment

```bash
cd src/torq/models/gemma3/int4quant
python3 -m venv .venv && source .venv/bin/activate
pip install torch                    
pip install -r requirements.txt
```

---

## 3. GPTQ quantization

Put the base ONNX in place first.

```bash
mkdir -p onnx
cp ../../../../../models/google/gemma-3-270m-it/export/full/unified/onnx/converted/static/model.onnx onnx/
```

```bash
python select_best_gptq.py \
    --num-runs 5 --gpus 0 \
    --base onnx/model.onnx \
    --template onnx/model_q4k.onnx \
    --grouping in \
    --out onnx/model-qdq.onnx
```

This runs GPTQ five times, converts each candidate to a QDQ ONNX, and copies the
one with the highest cos_sim against the base to `--out`. Calibration uses 512
`HuggingFaceH4/ultrachat_200k` samples. GPTQ has large run-to-run variance, so it
is better not to rely on a single run. With several GPUs, pass them as
`--gpus 0,1,2,3` to run in parallel.

If the template `onnx/model_q4k.onnx` does not exist, it is generated from
`--base`.

---

## 4. Deployment post-process

```bash
python torq_deploy_postprocess.py onnx/model-qdq.onnx --out onnx/model-torq.onnx
```

This applies the trim and removes the unnecessary ops. In order: extract the
lm_head embeddings (`token_embeddings.npy`), cast the scales to bf16 and drop the
Cast nodes, pack uint8 into INT4, trim the lm_head vocab (262144 -> 162578),
remove the GQA Expand nodes, and drop unreferenced initializers.

The outputs are `onnx/model-torq.onnx` and `onnx/token_embeddings.npy`.

---

## 5. Compile

The remaining compile commands are the same as for the int4 model.

```bash
python -m iree.compiler.tools.import_onnx onnx/model-torq.onnx \
    -o model-torq.mlir --data-prop

torq-compile model-torq.mlir -o model-torq.vmfb \
    --torq-convert-dtypes \
    --torq-enable-split-constants-optimization \
    --torq-enable-annotate-tied-operands \
    --torq-enable-transpose-optimization \
    --torq-disable-slicing --torq-hw=SL2610 \
    --iree-flow-inline-constants-max-byte-length=300000000
```
