# Tsuki Source Models

These models are not committed (too large for git). Copy from torq-compiler-dev:

```bash
MODELS=tests/testdata/onnx_models/tsuki_static_new_fp32_split_stft_final_s50_4s

cp /home/breidy/iree-local-dev/torq-compiler-dev/$MODELS/part_a_pre_stft_4s.onnx model/
cp /home/breidy/iree-local-dev/torq-compiler-dev/$MODELS/part_b_post_stft_4s.onnx model/
```
