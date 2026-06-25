import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def tensor_type_and_shape(value_info):
    tensor_type = value_info.type.tensor_type
    return tensor_type.elem_type, [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in tensor_type.shape.dim
    ]


def collect_shapes(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.type.HasField("tensor_type"):
            shapes[value.name] = tensor_type_and_shape(value)
    for init in inferred.graph.initializer:
        shapes.setdefault(init.name, (init.data_type, list(init.dims)))
    return shapes


def make_vi(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def const_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def const_f32(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)


def attr_map(node):
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def find_target_node(model, output_name):
    for index, node in enumerate(model.graph.node):
        if node.op_type == "ConvTranspose" and output_name in node.output:
            return index, node
    raise ValueError(f"Could not find ConvTranspose producing {output_name!r}")


def build_phase_nodes(node, shapes, initializers):
    attrs = attr_map(node)
    strides = list(attrs.get("strides", [1]))
    pads = list(attrs.get("pads", [0, 0]))
    dilations = list(attrs.get("dilations", [1]))
    group = int(attrs.get("group", 1))
    output_padding = list(attrs.get("output_padding", [0]))
    if len(strides) != 1 or len(pads) != 2 or len(dilations) != 1:
        raise ValueError("Only ConvTranspose1D is supported")
    if pads != [0, 0] or dilations != [1] or output_padding != [0] or group != 1:
        raise ValueError(
            f"Unsupported ConvTranspose attrs: strides={strides}, pads={pads}, "
            f"dilations={dilations}, output_padding={output_padding}, group={group}"
        )

    stride = strides[0]
    input_name, weight_name, bias_name = node.input[:3]
    output_name = node.output[0]
    elem_type, input_shape = shapes[input_name]
    _, output_shape = shapes[output_name]
    weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
    bias = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
    cin, cout, kernel = weight.shape

    if input_shape[0] != 1 or input_shape[1] != cin:
        raise ValueError(f"Unexpected input shape {input_shape} for weight {weight.shape}")
    if output_shape[0] != input_shape[0] or output_shape[1] != cout:
        raise ValueError(f"Unexpected output shape {output_shape} for weight {weight.shape}")
    if kernel % stride != 0:
        raise ValueError(f"Kernel {kernel} must be divisible by stride {stride}")

    input_length = input_shape[2]
    phase_length = input_length + (kernel // stride - 1)
    expected_length = phase_length * stride
    if expected_length != output_shape[2]:
        raise ValueError(
            f"Expected interleaved length {expected_length}, got {output_shape[2]}"
        )

    prefix = f"{output_name}_phase_matmul"
    nodes = [
        helper.make_node(
            "Transpose",
            [input_name],
            [f"{prefix}_x_nlc"],
            name=f"{prefix}_x_to_nlc",
            perm=[0, 2, 1],
        )
    ]
    new_initializers = [
        const_i64(f"{prefix}_unsqueeze_axis", [2]),
        const_i64(f"{prefix}_output_shape", [input_shape[0], output_shape[2], cout]),
        const_f32(f"{prefix}_bias", bias),
    ]
    value_infos = [
        make_vi(f"{prefix}_x_nlc", elem_type, [input_shape[0], input_length, cin])
    ]
    phase_outputs = []
    matmul_count = 0

    for phase in range(stride):
        padded_terms = []
        for tap_group in range(kernel // stride):
            tap_index = phase + tap_group * stride
            weight_tensor = weight[:, :, tap_index]
            weight_init_name = f"{prefix}_phase{phase}_tap{tap_index}_weight"
            mm_name = f"{prefix}_phase{phase}_tap{tap_index}_mm"
            padded_name = f"{prefix}_phase{phase}_tap{tap_index}_padded"
            new_initializers.append(const_f32(weight_init_name, weight_tensor))
            nodes.append(
                helper.make_node(
                    "MatMul",
                    [f"{prefix}_x_nlc", weight_init_name],
                    [mm_name],
                    name=mm_name,
                )
            )
            value_infos.append(make_vi(mm_name, elem_type, [input_shape[0], input_length, cout]))

            pad_head = tap_group
            pad_tail = phase_length - input_length - pad_head
            concat_inputs = []
            if pad_head:
                pad_name = f"{prefix}_phase{phase}_tap{tap_index}_head_zero"
                new_initializers.append(
                    const_f32(pad_name, np.zeros((input_shape[0], pad_head, cout), dtype=np.float32))
                )
                concat_inputs.append(pad_name)
            concat_inputs.append(mm_name)
            if pad_tail:
                pad_name = f"{prefix}_phase{phase}_tap{tap_index}_tail_zero"
                new_initializers.append(
                    const_f32(pad_name, np.zeros((input_shape[0], pad_tail, cout), dtype=np.float32))
                )
                concat_inputs.append(pad_name)

            if len(concat_inputs) == 1:
                padded_terms.append(concat_inputs[0])
            else:
                nodes.append(
                    helper.make_node(
                        "Concat",
                        concat_inputs,
                        [padded_name],
                        name=padded_name,
                        axis=1,
                    )
                )
                value_infos.append(
                    make_vi(padded_name, elem_type, [input_shape[0], phase_length, cout])
                )
                padded_terms.append(padded_name)
            matmul_count += 1

        running = padded_terms[0]
        for term_index, term in enumerate(padded_terms[1:], start=1):
            summed = f"{prefix}_phase{phase}_sum{term_index}"
            nodes.append(helper.make_node("Add", [running, term], [summed], name=summed))
            value_infos.append(make_vi(summed, elem_type, [input_shape[0], phase_length, cout]))
            running = summed

        biased = f"{prefix}_phase{phase}_biased"
        unsqueezed = f"{prefix}_phase{phase}_unsqueezed"
        nodes.append(helper.make_node("Add", [running, f"{prefix}_bias"], [biased], name=biased))
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [biased, f"{prefix}_unsqueeze_axis"],
                [unsqueezed],
                name=unsqueezed,
            )
        )
        value_infos.append(make_vi(biased, elem_type, [input_shape[0], phase_length, cout]))
        value_infos.append(make_vi(unsqueezed, elem_type, [input_shape[0], phase_length, 1, cout]))
        phase_outputs.append(unsqueezed)

    nodes.extend(
        [
            helper.make_node(
                "Concat",
                phase_outputs,
                [f"{prefix}_stacked"],
                name=f"{prefix}_stacked",
                axis=2,
            ),
            helper.make_node(
                "Reshape",
                [f"{prefix}_stacked", f"{prefix}_output_shape"],
                [f"{prefix}_nlc"],
                name=f"{prefix}_interleave",
            ),
            helper.make_node(
                "Transpose",
                [f"{prefix}_nlc"],
                [output_name],
                name=f"{prefix}_to_ncl",
                perm=[0, 2, 1],
            ),
        ]
    )
    value_infos.extend(
        [
            make_vi(f"{prefix}_stacked", elem_type, [input_shape[0], phase_length, stride, cout]),
            make_vi(f"{prefix}_nlc", elem_type, [input_shape[0], output_shape[2], cout]),
        ]
    )

    return nodes, new_initializers, value_infos, matmul_count


def remove_initializer_if_unused(graph, names):
    used = set()
    for node in graph.node:
        used.update(name for name in node.input if name)
    used.update(value.name for value in graph.input)
    keep = []
    removed = []
    for init in graph.initializer:
        if init.name in names and init.name not in used:
            removed.append(init.name)
        else:
            keep.append(init)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return removed


def compare_io_contract(before, after):
    def sig(graph_values):
        out = []
        for value in graph_values:
            elem_type, shape = tensor_type_and_shape(value)
            out.append((value.name, elem_type, shape))
        return out

    return {
        "inputs_match": sig(before.graph.input) == sig(after.graph.input),
        "outputs_match": sig(before.graph.output) == sig(after.graph.output),
        "before_inputs": sig(before.graph.input),
        "after_inputs": sig(after.graph.input),
        "before_outputs": sig(before.graph.output),
        "after_outputs": sig(after.graph.output),
    }


def random_inputs_for_model(model, seed):
    rng = np.random.default_rng(seed)
    feeds = {}
    for value in model.graph.input:
        elem_type, shape = tensor_type_and_shape(value)
        if elem_type == onnx.TensorProto.FLOAT:
            feeds[value.name] = rng.standard_normal(shape).astype(np.float32)
        elif elem_type == onnx.TensorProto.INT64:
            feeds[value.name] = np.zeros(shape, dtype=np.int64)
        elif elem_type == onnx.TensorProto.INT32:
            feeds[value.name] = np.zeros(shape, dtype=np.int32)
        elif elem_type == onnx.TensorProto.BOOL:
            feeds[value.name] = np.zeros(shape, dtype=bool)
        else:
            raise ValueError(f"Unsupported random input elem_type {elem_type} for {value.name}")
    return feeds


def verify_random(before_path, after_path, seed):
    before_model = onnx.load(before_path)
    feeds = random_inputs_for_model(before_model, seed)
    before_session = ort.InferenceSession(str(before_path), providers=["CPUExecutionProvider"])
    after_session = ort.InferenceSession(str(after_path), providers=["CPUExecutionProvider"])
    before_outputs = before_session.run(None, feeds)
    after_outputs = after_session.run(None, feeds)
    metrics = []
    for before_value, after_value in zip(before_outputs, after_outputs):
        diff = np.abs(before_value - after_value)
        metrics.append(
            {
                "shape": list(before_value.shape),
                "max_abs": float(diff.max()) if diff.size else 0.0,
                "mean_abs": float(diff.mean()) if diff.size else 0.0,
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=False, type=Path)
    parser.add_argument("--target-output", default="convolution_1")
    parser.add_argument("--verify-random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    original = onnx.load(args.input)
    model = onnx.ModelProto()
    model.CopyFrom(original)
    shapes = collect_shapes(model)
    initializers = {init.name: init for init in model.graph.initializer}
    node_index, target = find_target_node(model, args.target_output)
    replacement_nodes, replacement_initializers, replacement_value_infos, matmul_count = build_phase_nodes(
        target, shapes, initializers
    )

    nodes = list(model.graph.node)
    nodes[node_index : node_index + 1] = replacement_nodes
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(replacement_initializers)

    existing_value_info = {value.name for value in model.graph.value_info}
    for value_info in replacement_value_infos:
        if value_info.name not in existing_value_info:
            model.graph.value_info.append(value_info)
            existing_value_info.add(value_info.name)

    removed_initializers = remove_initializer_if_unused(model.graph, target.input[1:3])
    onnx.checker.check_model(model)
    contract = compare_io_contract(original, model)
    if not contract["inputs_match"] or not contract["outputs_match"]:
        raise RuntimeError(f"Graph IO contract changed: {contract}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_node_index": node_index,
        "target_node_name": target.name,
        "target_output": args.target_output,
        "replacement_node_count": len(replacement_nodes),
        "replacement_matmul_count": matmul_count,
        "removed_initializers": removed_initializers,
        "io_contract": contract,
        "onnx_checker": "passed",
    }
    if args.verify_random:
        report["random_verification"] = verify_random(args.input, args.output, args.seed)
    
    print(f"Replaced {target.name} ({target.op_type}) at node index {node_index}")
    print(f"Replacement nodes: {len(replacement_nodes)}, MatMuls: {matmul_count}")
    print(f"ONNX checker passed")
    print(f"Wrote model: {args.output}")
    
    if args.verify_random:
        print(f"Random verification: {report['random_verification']}")
        
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
