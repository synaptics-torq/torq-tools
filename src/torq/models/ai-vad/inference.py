import onnxruntime as ort
import numpy as np

# Example input
dummy_input = {
    "in_frame_mag": np.random.randn(1, 2, 1, 256).astype(np.float32),
    "input_state": np.random.randn(1, 16, 64).astype(np.float32),
}

sess = ort.InferenceSession("aec_vad_exp12_d4_model_epoch_t710.onnx")
outputs = sess.run(None, dummy_input)

print("Outputs:")
for name, out in zip([o.name for o in sess.get_outputs()], outputs):
    print(name, out.shape)
