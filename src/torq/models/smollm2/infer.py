from ._inference import SmolLMDynamic

smollm = SmolLMDynamic.from_onnx("/home/spal-synaptics/synap_repos/torq-tools-test/models/SmolLM-Instruct/source/model.onnx")

inps = ["What is 2 + 2?", "What is 3 + 3?"]
for inp in inps:
    out = smollm.run(inp, 64)
    print(out)
