import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("aec_vad_exp12_d5_quantized_t39.onnx", providers=["CPUExecutionProvider"])

inputs = {
    "in_frame_mag": np.random.randn(1, 2, 1, 256).astype(np.float32),
    "input_state": np.zeros((1, 16, 64), dtype=np.float32),
}

outputs = session.run(None, inputs)

for meta, out in zip(session.get_outputs(), outputs):
    print(meta.name, out.shape)
    print(out)