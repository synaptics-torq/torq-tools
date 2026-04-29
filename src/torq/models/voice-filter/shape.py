import onnx

model = onnx.load("model_static.onnx")

producer = {}
for node in model.graph.node:
    for out in node.output:
        producer[out] = node

target_nodes = {"/model/Expand", "/model/Reshape_2", "/model/Expand_1", "/model/Reshape_4"}

for node in model.graph.node:
    if node.name in target_nodes:
        print(f"\nNODE: {node.name} ({node.op_type})")
        print("  inputs :", list(node.input))
        print("  outputs:", list(node.output))
        if len(node.input) > 1:
            shape_tensor = node.input[1]
            print("  shape/control tensor:", shape_tensor)
            p = producer.get(shape_tensor)
            if p:
                print("  produced by:", p.name, p.op_type)
                print("    inputs :", list(p.input))
                print("    outputs:", list(p.output))