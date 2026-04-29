import copy
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, shape_inference

MODEL_PATH = "model_epoch_0812_15_folded.onnx"
PROBE_PATH = "/tmp/probe_shapes.onnx"

feeds = {
    "in_frame_mag": np.zeros((1, 2, 1, 256), dtype=np.float32),
    "embedding": np.zeros((1, 256), dtype=np.float32),
    "input_state": np.zeros((1, 16, 256), dtype=np.float32),
}

target_tensors = [
    "/model/Where_output_0",
    "/model/Where_1_output_0",
    "/model/Concat_2_output_0",
]

model = onnx.load(MODEL_PATH)
model = shape_inference.infer_shapes(model)
probe = copy.deepcopy(model)

# Keep original outputs if you want, or clear them
del probe.graph.output[:]

# Build Identity outputs with fresh names
probe_output_names = []

for i, tname in enumerate(target_tensors):
    out_name = f"__probe_out_{i}"
    probe_output_names.append(out_name)

    # Add Identity node
    probe.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[tname],
            outputs=[out_name],
            name=f"ProbeIdentity_{i}",
        )
    )

    # Expose that identity output
    probe.graph.output.append(
        helper.make_tensor_value_info(
            out_name,
            TensorProto.INT64,   # shape tensors are usually int64
            None,
        )
    )

onnx.save(probe, PROBE_PATH)

# Optional sanity check
loaded = onnx.load(PROBE_PATH)
print("Probe graph outputs:")
for o in loaded.graph.output:
    print(" ", o.name)

sess = ort.InferenceSession(PROBE_PATH, providers=["CPUExecutionProvider"])
vals = sess.run(probe_output_names, feeds)

for orig_name, out_name, val in zip(target_tensors, probe_output_names, vals):
    print(f"\n{orig_name} -> {out_name}")
    print("shape:", val.shape, "dtype:", val.dtype)
    print(val)