#!/usr/bin/env python3
"""Apply ONNX-domain compiler workarounds and patch the executor map.

Usage:
  python3 compiler_patches.py <model.onnx> <patched.onnx>

By default this also looks for executor_assignments_<model-stem>.json and writes
executor_assignments_<patched-stem>.json next to the patched ONNX. The JSON patch
imports the patched ONNX to temporary MLIR so mlir_location values match the new
graph.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


CONVTRANSPOSE_NODE_NAME = "node_convolution_1"
CONVTRANSPOSE_OUTPUT = "convolution_1"
MLIR_ONNX_RE = re.compile(r'^(\s*)%[\w#]+(?::\d+)? = torch\.operator "onnx\.([^"]+)"')

ATAN2_HOST_LAYER_IDS = {
    "node_pow_123_pow2_mul",
    "Mul_pow_123",
    "node_add_18262",
    "Add_add_18262",
    "full_static_tail_gate_1161",
    "Mul_full_static_tail_gate_1161_add_18262",
    "node_sqrt",
    "Sqrt_sqrt",
    "node_div_119",
    "Div_div_119",
    "full_static_tail_gate_1163",
    "Mul_full_static_tail_gate_1163_div_119",
    "node_div_120",
    "Div_div_120",
    "full_static_tail_gate_1164",
    "Mul_full_static_tail_gate_1164_div_120",
    "node_Div_13065",
    "Div_val_13109",
    "node_Atan_13066",
    "Atan_val_13110",
    "node_Greater_13069",
    "Greater_val_13113",
    "node_Add_13070",
    "Add_val_13114",
    "node_Sub_13071",
    "Sub_val_13115",
    "node_Where_13072",
    "Where_val_13116",
    "node_Less_13073",
    "Less_val_13117",
    "node_Where_13074",
    "Where_val_13118",
    "full_static_tail_gate_1170",
    "Mul_full_static_tail_gate_1170_val_13118",
    "node_IsNaN_13075_equal_self",
    "Equal_val_13119_equal_self",
    "node_IsNaN_13075_select_not_nan",
    "Where_atan2",
}

STFT_MAG_PHASE_BLOCK_NODES = {
    "node_pow_122_pow2_mul",
    "node_pow_123_pow2_mul",
    "node_add_18262",
    "full_static_tail_gate_1161",
    "node_sqrt",
    "node_div_119",
    "full_static_tail_gate_1163",
    "node_div_120",
    "full_static_tail_gate_1164",
    "node_Div_13065",
    "node_Atan_13066",
    "node_Greater_13069",
    "node_Add_13070",
    "node_Sub_13071",
    "node_Where_13072",
    "node_Less_13073",
    "node_Where_13074",
    "full_static_tail_gate_1170",
    "node_IsNaN_13075",
    "node_atan2",
    "node_IsNaN_13075_equal_self",
    "node_IsNaN_13075_select_not_nan",
    "node_cat_206",
}


def attr_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return list(default)


def attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in graph.initializer}


def unique_name(graph: onnx.GraphProto, base: str) -> str:
    used: set[str] = set()
    for node in graph.node:
        used.update(node.input)
        used.update(node.output)
        if node.name:
            used.add(node.name)
    used.update(init.name for init in graph.initializer)
    used.update(value.name for value in graph.input)
    used.update(value.name for value in graph.output)
    used.update(value.name for value in graph.value_info)

    if base not in used:
        return base
    suffix = 1
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def add_initializer(graph: onnx.GraphProto, name: str, array: np.ndarray) -> str:
    graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def remove_initializer_if_unused(graph: onnx.GraphProto, name: str) -> None:
    if not name:
        return
    for node in graph.node:
        if name in node.input:
            return
    for graph_input in graph.input:
        if graph_input.name == name:
            return
    kept = [init for init in graph.initializer if init.name != name]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def layer_ids(node: onnx.NodeProto) -> list[str]:
    ids = []
    if node.name:
        ids.append(node.name)
    if node.output:
        ids.append(f"{node.op_type}_{node.output[0]}")
    return ids


def value_elem_types(graph: onnx.GraphProto) -> dict[str, int]:
    elem_types: dict[str, int] = {}
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            elem_types[value.name] = tensor_type.elem_type
    for init in graph.initializer:
        elem_types[init.name] = init.data_type
    return elem_types


def value_shapes(graph: onnx.GraphProto) -> dict[str, list[int | None]]:
    shapes: dict[str, list[int | None]] = {}
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | None] = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else None)
        shapes[value.name] = dims
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def infer_and_check(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model, data_prop=True)
    onnx.checker.check_model(model)
    return model


def remove_value_infos(graph: onnx.GraphProto, names: set[str]) -> None:
    kept = [value for value in graph.value_info if value.name not in names]
    del graph.value_info[:]
    graph.value_info.extend(kept)


def replace_stft_real_imag_with_mag_phase_input(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    if any(input_value.name == "cat_206" for input_value in graph.input):
        return model, 0

    target = next((node for node in graph.node if node.name == "node_cat_206"), None)
    if target is None:
        return model, 0
    if target.op_type != "Concat" or list(target.output) != ["cat_206"]:
        raise ValueError("node_cat_206 is not the expected Concat producing cat_206")

    shapes = value_shapes(graph)
    elem_types = value_elem_types(graph)
    cat_shape = shapes.get("cat_206", [1, 62, 12801])
    cat_elem_type = elem_types.get("cat_206", TensorProto.FLOAT)

    removed_outputs: set[str] = set()
    kept_nodes = []
    removed = 0
    for node in graph.node:
        if node.name in STFT_MAG_PHASE_BLOCK_NODES:
            removed_outputs.update(node.output)
            removed += 1
        else:
            kept_nodes.append(node)
    if removed == 0:
        return model, 0

    del graph.node[:]
    graph.node.extend(kept_nodes)

    consumed = {input_name for node in graph.node for input_name in node.input}
    kept_inputs = [
        value
        for value in graph.input
        if value.name not in {"stft_real", "stft_imag"} or value.name in consumed
    ]
    del graph.input[:]
    graph.input.extend(kept_inputs)
    graph.input.append(
        helper.make_tensor_value_info("cat_206", cat_elem_type, cat_shape)
    )
    remove_value_infos(graph, removed_outputs | {"cat_206"})
    return infer_and_check(model), removed


def decompose_convtranspose(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    target_idx = None
    target = None
    for idx, node in enumerate(graph.node):
        if node.op_type == "ConvTranspose" and (
            node.name == CONVTRANSPOSE_NODE_NAME or CONVTRANSPOSE_OUTPUT in node.output
        ):
            target_idx = idx
            target = node
            break
    if target is None or target_idx is None:
        return model, 0

    if len(target.input) not in (2, 3) or len(target.output) != 1:
        raise ValueError(f"{target.name} has unexpected ConvTranspose signature")

    x_name, weight_name = target.input[0], target.input[1]
    bias_name = target.input[2] if len(target.input) == 3 else ""
    output_name = target.output[0]

    inits = initializer_map(graph)
    if weight_name not in inits:
        raise ValueError(f"missing ConvTranspose weight initializer {weight_name!r}")
    w_deconv = numpy_helper.to_array(inits[weight_name])

    strides = attr_ints(target, "strides", [1])
    pads = attr_ints(target, "pads", [0, 0])
    dilations = attr_ints(target, "dilations", [1])
    output_padding = attr_ints(target, "output_padding", [0])
    group = attr_int(target, "group", 1)
    if (
        w_deconv.shape != (128, 64, 10)
        or strides != [5]
        or pads != [0, 0]
        or dilations != [1]
        or output_padding != [0]
        or group != 1
    ):
        raise ValueError(
            "unexpected ConvTranspose parameters: "
            f"weight={w_deconv.shape}, strides={strides}, pads={pads}, "
            f"dilations={dilations}, output_padding={output_padding}, group={group}"
        )

    prefix = unique_name(graph, f"{target.name or 'convtranspose'}_decomposed")
    axes_last = add_initializer(graph, f"{prefix}_axes_last", np.array([3], np.int64))
    reshape_shape = add_initializer(
        graph, f"{prefix}_reshape_shape", np.array([1, 64, 12805], np.int64)
    )

    new_nodes = []
    residue_unsqueezed: list[str] = []
    for residue in range(5):
        w_pair = np.stack(
            [w_deconv[:, :, residue + 5], w_deconv[:, :, residue]], axis=2
        )
        w_pair = np.transpose(w_pair, (1, 0, 2)).astype(np.float32)
        weight = add_initializer(graph, f"{prefix}_residue_{residue}_weight", w_pair)
        conv_out = f"{prefix}_residue_{residue}"
        unsqueeze_out = f"{prefix}_residue_{residue}_unsqueezed"
        residue_unsqueezed.append(unsqueeze_out)
        new_nodes.append(
            helper.make_node(
                "Conv",
                [x_name, weight] + ([bias_name] if bias_name else []),
                [conv_out],
                name=f"{prefix}_residue_{residue}_conv",
                dilations=[1],
                group=1,
                kernel_shape=[2],
                pads=[1, 1],
                strides=[1],
            )
        )
        new_nodes.append(
            helper.make_node(
                "Unsqueeze",
                [conv_out, axes_last],
                [unsqueeze_out],
                name=f"{prefix}_residue_{residue}_unsqueeze",
            )
        )

    interleaved = f"{prefix}_interleaved"
    new_nodes.append(
        helper.make_node(
            "Concat",
            residue_unsqueezed,
            [interleaved],
            name=f"{prefix}_concat_residues",
            axis=3,
        )
    )
    new_nodes.append(
        helper.make_node(
            "Reshape",
            [interleaved, reshape_shape],
            [output_name],
            name=f"{prefix}_reshape_interleaved",
        )
    )

    nodes = list(graph.node)
    nodes[target_idx : target_idx + 1] = new_nodes
    del graph.node[:]
    graph.node.extend(nodes)
    remove_initializer_if_unused(graph, weight_name)
    return infer_and_check(model), 1


def cast_layernorm_reducemeans_to_f32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    elem_types = value_elem_types(graph)
    new_nodes = []
    rewritten = 0
    for node in graph.node:
        if not (
            node.op_type == "ReduceMean"
            and node.name.startswith("ReduceMean__ln_decomp_layer_norm_")
        ):
            new_nodes.append(node)
            continue
        if len(node.output) != 1:
            raise ValueError(f"{node.name} expected one ReduceMean output")

        original_output = node.output[0]
        output_dtype = elem_types.get(original_output, TensorProto.BFLOAT16)
        cast_input_output = unique_name(graph, f"{original_output}_input_f32")
        reduce_f32_output = unique_name(graph, f"{original_output}_f32")
        new_nodes.append(
            helper.make_node(
                "Cast",
                [node.input[0]],
                [cast_input_output],
                name=f"{node.name}_cast_input_f32",
                to=TensorProto.FLOAT,
            )
        )
        node.input[0] = cast_input_output
        node.output[0] = reduce_f32_output
        new_nodes.append(node)
        if output_dtype == TensorProto.FLOAT:
            identity_node = helper.make_node(
                "Identity",
                [reduce_f32_output],
                [original_output],
                name=f"{node.name}_keep_output_f32",
            )
            new_nodes.append(identity_node)
        else:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [reduce_f32_output],
                    [original_output],
                    name=f"{node.name}_cast_output_original",
                    to=output_dtype,
                )
            )
        rewritten += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    return infer_and_check(model), rewritten


def rewrite_bf16_conv_island(graph: onnx.GraphProto, conv_node_name: str) -> None:
    inits = initializer_map(graph)
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    conv = next((node for node in graph.node if node.name == conv_node_name), None)
    if conv is None:
        raise ValueError(f"could not find Conv node {conv_node_name!r}")
    input_cast = producers.get(conv.input[0])
    if input_cast is None or input_cast.op_type != "Cast":
        raise ValueError(f"{conv_node_name} input is not produced by Cast")
    if attr_int(input_cast, "to", -1) != TensorProto.BFLOAT16:
        raise ValueError(f"{input_cast.name} does not cast to BFLOAT16")

    output_consumers = consumers.get(conv.output[0], [])
    if len(output_consumers) != 1 or output_consumers[0].op_type != "Cast":
        raise ValueError(f"{conv_node_name} output is not consumed by one Cast")
    output_cast = output_consumers[0]
    if attr_int(output_cast, "to", -1) != TensorProto.FLOAT:
        raise ValueError(f"{output_cast.name} does not cast to FLOAT")

    conv.input[0] = input_cast.input[0]
    conv.output[0] = output_cast.output[0]
    for input_idx, initializer_name in enumerate(list(conv.input[1:]), start=1):
        tensor = inits.get(initializer_name)
        if tensor is None or tensor.data_type != TensorProto.BFLOAT16:
            continue
        array = numpy_helper.to_array(tensor).astype(np.float32)
        f32_name = unique_name(graph, f"{initializer_name}_f32")
        graph.initializer.append(numpy_helper.from_array(array, name=f32_name))
        conv.input[input_idx] = f32_name
        remove_initializer_if_unused(graph, initializer_name)

    kept_nodes = [
        node for node in graph.node if node is not input_cast and node is not output_cast
    ]
    del graph.node[:]
    graph.node.extend(kept_nodes)


def keep_all_bf16_conv_islands_in_f32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    targets = [
        node.name
        for node in list(graph.node)
        if node.op_type == "Conv"
        and node.name
        and node.output
        and node.output[0].endswith("_bf16")
    ]
    rewritten = 0
    skipped = []
    for node_name in targets:
        try:
            rewrite_bf16_conv_island(graph, node_name)
            rewritten += 1
        except ValueError as exc:
            skipped.append(f"{node_name}: {exc}")
    if skipped:
        print("warning: skipped bf16 Conv island(s):", file=sys.stderr)
        for detail in skipped[:10]:
            print(f"  {detail}", file=sys.stderr)
    return infer_and_check(model), rewritten


def fold_isnan_where(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    new_nodes = []
    skip_ids: set[int] = set()
    rewritten = 0
    skipped = 0
    for node in graph.node:
        if id(node) in skip_ids:
            continue
        if node.op_type != "IsNaN":
            new_nodes.append(node)
            continue
        output_consumers = consumers.get(node.output[0], [])
        if (
            len(node.input) != 1
            or len(node.output) != 1
            or len(output_consumers) != 1
            or output_consumers[0].op_type != "Where"
            or output_consumers[0].input[0] != node.output[0]
        ):
            new_nodes.append(node)
            skipped += 1
            continue
        where = output_consumers[0]
        equal_output = unique_name(graph, f"{node.output[0]}_equal_self")
        base_name = node.name or f"IsNaN_{node.output[0]}"
        new_nodes.append(
            helper.make_node(
                "Equal",
                [node.input[0], node.input[0]],
                [equal_output],
                name=f"{base_name}_equal_self",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Where",
                [equal_output, where.input[2], where.input[1]],
                list(where.output),
                name=f"{base_name}_select_not_nan",
            )
        )
        skip_ids.add(id(where))
        rewritten += 1

    if skipped:
        print(f"warning: skipped {skipped} unsupported IsNaN node(s)", file=sys.stderr)
    del graph.node[:]
    graph.node.extend(new_nodes)
    return infer_and_check(model), rewritten


def split_large_softmax_heads(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    shapes = value_shapes(graph)
    new_nodes = []
    rewritten = 0
    for node in graph.node:
        if node.op_type != "Softmax" or len(node.input) != 1 or len(node.output) != 1:
            new_nodes.append(node)
            continue
        if shapes.get(node.input[0]) != [1, 8, 2560, 2560]:
            new_nodes.append(node)
            continue
        axis = attr_int(node, "axis", -1)
        if axis not in (-1, 3):
            new_nodes.append(node)
            continue

        base_name = node.name or f"Softmax_{node.output[0]}"
        split_name = add_initializer(
            graph,
            unique_name(graph, f"{node.output[0]}_head_splits"),
            np.ones(8, dtype=np.int64),
        )
        split_outputs = [f"{node.output[0]}_head_{idx}_input" for idx in range(8)]
        softmax_outputs = [f"{node.output[0]}_head_{idx}" for idx in range(8)]
        new_nodes.append(
            helper.make_node(
                "Split",
                [node.input[0], split_name],
                split_outputs,
                name=f"{base_name}_split_heads",
                axis=1,
            )
        )
        for idx, (split_output, softmax_output) in enumerate(
            zip(split_outputs, softmax_outputs)
        ):
            new_nodes.append(
                helper.make_node(
                    "Softmax",
                    [split_output],
                    [softmax_output],
                    name=f"{base_name}_head_{idx}",
                    axis=axis,
                )
            )
        new_nodes.append(
            helper.make_node(
                "Concat",
                softmax_outputs,
                list(node.output),
                name=f"{base_name}_concat_heads",
                axis=1,
            )
        )
        rewritten += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    return infer_and_check(model), rewritten


def patch_model(input_model: Path, output_model: Path) -> dict[str, int]:
    model = onnx.load(input_model, load_external_data=False)
    model = infer_and_check(model)

    stats: dict[str, int] = {}
    model, stats["stft_real_imag_to_cat_206_input"] = (
        replace_stft_real_imag_with_mag_phase_input(model)
    )
    model, stats["convtranspose_decomposed"] = decompose_convtranspose(model)
    model, stats["layernorm_reducemeans_f32"] = cast_layernorm_reducemeans_to_f32(model)
    model, stats["bf16_conv_islands_f32"] = keep_all_bf16_conv_islands_in_f32(model)
    model, stats["isnan_where_folded"] = fold_isnan_where(model)
    model, stats["large_softmax_split"] = split_large_softmax_heads(model)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_model)
    return stats


def find_executor_map(input_model: Path) -> Path | None:
    candidates = [
        Path.cwd() / f"executor_assignments_{input_model.stem}.json",
        input_model.parent / f"executor_assignments_{input_model.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def import_onnx_to_mlir(onnx_path: Path, mlir_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "iree.compiler.tools.import_onnx",
        str(onnx_path),
        "-o",
        str(mlir_path),
    ]
    subprocess.run(cmd, check=True)


def mlir_onnx_ops(mlir_path: Path) -> list[tuple[str, str]]:
    ops = []
    with mlir_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            match = MLIR_ONNX_RE.match(line)
            if not match:
                continue
            op_type = match.group(2)
            if op_type == "Constant":
                continue
            col = line.index("torch.operator") + 1
            ops.append((op_type, f"{line_no}:{col}"))
    return ops


def rewritten_locations(onnx_path: Path, mlir_path: Path) -> dict[str, str]:
    model = onnx.load(onnx_path, load_external_data=False)
    nodes = [node for node in model.graph.node if node.op_type != "Constant"]
    mlir_ops = mlir_onnx_ops(mlir_path)
    if len(nodes) != len(mlir_ops):
        raise ValueError(f"ONNX/MLIR op count mismatch: onnx={len(nodes)} mlir={len(mlir_ops)}")

    locations: dict[str, str] = {}
    mismatches = []
    for idx, (node, (mlir_op_type, location)) in enumerate(zip(nodes, mlir_ops)):
        if node.op_type != mlir_op_type:
            mismatches.append((idx, node.name, node.op_type, mlir_op_type, location))
            if len(mismatches) >= 10:
                break
        for layer_id in layer_ids(node):
            locations[layer_id] = location
    if mismatches:
        detail = "\n".join(
            f"{idx}: {name} onnx={onnx_op} mlir={mlir_op} loc={loc}"
            for idx, name, onnx_op, mlir_op, loc in mismatches
        )
        raise ValueError(f"ONNX/MLIR order mismatch:\n{detail}")
    return locations


def synthetic_softmax_layer_ids(onnx_path: Path) -> tuple[set[str], set[str]]:
    model = onnx.load(onnx_path, load_external_data=False)
    glue_nss: set[str] = set()
    head_host: set[str] = set()
    for node in model.graph.node:
        ids = set(layer_ids(node))
        if "_split_heads" in node.name or "_concat_heads" in node.name:
            glue_nss.update(ids)
        if re.search(r"_head_\d+$", node.name):
            head_host.update(ids)
    return glue_nss, head_host


def force_executor_choices(
    ops: dict[str, Any],
    locations: dict[str, str],
    softmax_glue_nss: set[str],
    softmax_head_host: set[str],
) -> None:
    for layer_id, location in locations.items():
        if layer_id.startswith("ReduceMean__ln_decomp_layer_norm_"):
            ops[layer_id] = {
                **ops.get(layer_id, {}),
                "mlir_location": location,
                "recommended_executor": "nss",
                "recommend_convert_dtypes": False,
            }
        if (
            layer_id.startswith("ReduceMean__ln_decomp_layer_norm_")
            and (
                layer_id.endswith("_cast_input_f32")
                or layer_id.endswith("_cast_output_original")
                or layer_id.endswith("_keep_output_f32")
            )
        ):
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "nss",
                "recommend_convert_dtypes": False,
            }
        if layer_id.startswith("node_convolution_1_decomposed_"):
            executor = "host" if layer_id.endswith("_conv") else "nss"
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": executor,
                "recommend_convert_dtypes": False,
            }
        if re.fullmatch(r"node_.*_conv2d", layer_id) or re.fullmatch(
            r"Conv_.*_output_4d", layer_id
        ):
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "host",
                "recommend_convert_dtypes": False,
            }
        if layer_id in ATAN2_HOST_LAYER_IDS:
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "host",
                "recommend_convert_dtypes": False,
            }
        if layer_id.endswith("_equal_self") or layer_id.endswith("_select_not_nan"):
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "host",
                "recommend_convert_dtypes": False,
            }
        if layer_id in softmax_glue_nss:
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "nss",
                "recommend_convert_dtypes": False,
            }
        if layer_id in softmax_head_host:
            ops[layer_id] = {
                "mlir_location": location,
                "recommended_executor": "host",
                "recommend_convert_dtypes": False,
            }


def patch_executor_json(
    input_json: Path, patched_onnx: Path, output_json: Path
) -> tuple[int, int]:
    with input_json.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    with tempfile.TemporaryDirectory(prefix="compiler_patches_") as tmpdir:
        mlir_path = Path(tmpdir) / "patched.mlir"
        import_onnx_to_mlir(patched_onnx, mlir_path)
        locations = rewritten_locations(patched_onnx, mlir_path)
    softmax_glue_nss, softmax_head_host = synthetic_softmax_layer_ids(patched_onnx)

    ops = data.get("ops", {})
    ops.pop("ConvTranspose_convolution_1", None)
    missing = 0
    for layer_id, info in list(ops.items()):
        location = locations.get(layer_id)
        if location is None:
            missing += 1
            ops.pop(layer_id)
            continue
        info["mlir_location"] = location

    before_force = len(ops)
    force_executor_choices(ops, locations, softmax_glue_nss, softmax_head_host)
    added = len(ops) - before_force
    data["ops"] = ops
    data["model_name"] = patched_onnx.stem
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return missing, added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--json-in", type=Path, help="Original executor assignment JSON.")
    parser.add_argument("--json-out", type=Path, help="Patched executor assignment JSON.")
    parser.add_argument("--skip-json", action="store_true")
    args = parser.parse_args()

    stats = patch_model(args.model, args.out)
    print(f"wrote {args.out}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if args.skip_json:
        return

    json_in = args.json_in or find_executor_map(args.model)
    if json_in is None:
        print("warning: no matching executor JSON found; skipped JSON patch", file=sys.stderr)
        return
    json_out = args.json_out or (args.out.parent / f"executor_assignments_{args.out.stem}.json")
    try:
        missing, added = patch_executor_json(json_in, args.out, json_out)
    except Exception as exc:
        print(f"warning: failed to patch executor JSON: {exc}", file=sys.stderr)
        return
    print(f"wrote {json_out}")
    print(f"json_removed_missing_ops: {missing}")
    print(f"json_added_or_rewritten_ops: {added}")


if __name__ == "__main__":
    main()
