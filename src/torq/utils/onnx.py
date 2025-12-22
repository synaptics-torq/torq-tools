# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import hashlib
import logging
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from typing import Union

import onnx
import onnx_graphsurgeon as gs
import numpy as np


logger = logging.getLogger(__name__)


__all__ = [
    # CLI helpers
    "add_onnx_args",

    # model inspection
    "get_model_opset",
    "get_model_ops_count",
    "check_dynamic_shapes",
    "print_onnx_model_inputs_outputs_info",

    # subgraph extraction
    "extract_boundary_tensors",
    "extract_subgraphs",

    # DType utilities
    "DTypeLike",
    "is_same_dtype",

    # Transformations
    "upgrade_model",
]


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def add_onnx_args(
    parser: argparse.ArgumentParser,
    *,
    model_dtypes: list[str] | None = None,
    convert_dtypes: list[str] | None = None,
    allow_no_opt: bool = True,
):
    group = parser.add_argument_group("ONNX args")
    if model_dtypes:
        group.add_argument(
            "-d", "--dtype",
            type=str,
            metavar="DTYPE",
            choices=model_dtypes,
            default=model_dtypes[0],
            help="Model data type (default: %(default)s, choices: %(choices)s)"
        )
    group.add_argument(
        "--onnx-source-dir",
        type=str,
        metavar="DIR",
        help="Directory containing source ONNX models (default: %(default)s)",
    )
    group.add_argument(
        "--show-model-info",
        action="store_true",
        default=False,
        help="Show ONNX model inputs/outputs and ops information",
    )
    group.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip validation for edited ONNX models"
    )
    if allow_no_opt:
        group.add_argument(
            "--no-optimize",
            action="store_true",
            default=False,
            help="Do no optimize exported ONNX models via onnxruntime",
        )
    if convert_dtypes:
        group.add_argument(
            "--convert-dtype",
            type=str,
            metavar="DTYPE",
            choices=convert_dtypes,
            help="Convert FP32 model to 16-bit float dtype (choices: %(choices)s)"
        )


# -----------------------------------------------------------------------------
# Model inspection utilities
# -----------------------------------------------------------------------------

def get_model_opset(
    model: str | os.PathLike | onnx.ModelProto,
    opset_domains: list[str] = ["ai.onnx", ""],
    default_opset: int = 17
) -> int:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    for opset_id in model.opset_import:
        if opset_id.domain in opset_domains:
            return int(opset_id.version)
    logger.warning("Cannot determine opset for model, defaulting to %d", default_opset)
    return default_opset


def get_model_ops_count(model: onnx.ModelProto) -> dict[str, int]:
    op_counts = {}
    for node in model.graph.node:
        if op_counts.get(node.op_type) is None:
            op_counts[node.op_type] = 0
        op_counts[node.op_type] += 1

    op_counts = dict(sorted(op_counts.items(), key=lambda item: item[1], reverse=True))
    return op_counts


def check_dynamic_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:

    def _is_static_shape(shape: list[int | str] | None) -> bool:
        return shape is not None and all(isinstance(d, int) and d >= 0 for d in shape)

    dynamic_shapes: dict[str, list[int | str]] = {}
    graph = gs.import_onnx(model)
    for tensor in graph.inputs + graph.outputs:
        if not _is_static_shape(tensor.shape):
            print(
                f"Static model check failed: I/O tensor '{tensor.name}' has non-static shape {tensor.shape}"
            )
            dynamic_shapes[tensor.name] = tensor.shape
    for tensor_name, tensor in graph.tensors().items():
        if not _is_static_shape(tensor.shape):
            print(
                f"Static model check failed: Graph tensor '{tensor_name}' has non-static shape {tensor.shape}"
            )
            dynamic_shapes[tensor_name] = tensor.shape
    return dynamic_shapes


def print_onnx_model_inputs_outputs_info(model: onnx.ModelProto | str | os.PathLike):
    if isinstance(model, (str, os.PathLike)):
        model = onnx.load(model)

    model_gs = gs.import_onnx(model)

    input_consumers = defaultdict(list)
    graph_input_names = {i.name: (i.shape, i.dtype) for i in model_gs.inputs}

    for node in model_gs.nodes:
        for input in node.inputs:
            name = input.name
            if name in graph_input_names:
                input_consumers[name].append(node)

    print(f"\n\nModel inputs info:\n")
    for name in sorted(graph_input_names):
        shape, dtype = graph_input_names[name]
        consumers = input_consumers.get(name, [])
        if consumers:
            consumers = "\n\t".join([f"'{node.name}'" for node in consumers])
            print(f"Input '{name}' ({dtype}{shape}) consumed by:\n\t{consumers}")
        else:
            print(f"Input '{name}' ({dtype}{shape}) is not consumed by any node")

    output_names = {o.name: (o.shape, o.dtype) for o in model_gs.outputs}
    output_to_node = {out.name: node for node in model_gs.nodes for out in node.outputs}

    print(f"\n\nModel outputs info:\n")
    for name, (shape, dtype) in output_names.items():
        node = output_to_node.get(name)
        if node:
            print(f"Output '{name}' ({dtype}{shape}) produced by:\n\t'{node.name}'")
        elif name in {i.name for i in model_gs.graph.input}:
            print(f"Output '{name}' is a passthrough from graph input")
        elif name in {init.name for init in model_gs.graph.initializer}:
            print(f"Output '{name}' is from initializer")
        else:
            print(f"Output '{name}' has no known producer (invalid?)")


# -----------------------------------------------------------------------------
# Subgraph extraction
# -----------------------------------------------------------------------------

def extract_boundary_tensors(
    model: onnx.ModelProto,
    ops_chain: list[str]
) -> list[dict[str, list | str]]:

    def _unique_subgraph_id(inputs: list[str], outputs: list[str], hash_length: int = 8) -> str:
        id_str = "|".join(inputs) + ">>" + "|".join(outputs) + ">>" + "|".join(ops_chain)
        return hashlib.sha256(id_str.encode()).hexdigest()[:hash_length]

    def _filter_tensors(tensors: list[gs.Constant | gs.Variable]) -> list[str]:
        tensor_names: list[str] = []
        for t in tensors:
            if isinstance(t, gs.Variable) and t.name:
                tensor_names.append(t.name)
        return tensor_names

    def _find_matches(curr: gs.Node, top: gs.Node, remaining: list[str]):
        if not remaining:
            inputs: list[str]  = _filter_tensors(top.inputs)
            outputs: list[str] = _filter_tensors(curr.outputs)
            if not inputs or not outputs:
                return
            if (subgraph_id := _unique_subgraph_id(inputs, outputs)) not in found_subgraph_ids:
                boundary_tensors.append(
                    {
                        "subgraph_id": subgraph_id,
                        "ops_chain": ops_chain,
                        "inputs": inputs,
                        "outputs": outputs
                    }
                )
                found_subgraph_ids.add(subgraph_id)
            return

        for out_t in curr.outputs:
            for consumer in out_t.outputs:
                if consumer.op == remaining[0]:
                    _find_matches(consumer, top, remaining[1:])

    if not ops_chain:
        raise ValueError("`ops` must contain at least one op type")
    boundary_tensors = []
    found_subgraph_ids: set[str] = set()
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == ops_chain[0]:
            _find_matches(node, node, ops_chain[1:])
    return boundary_tensors


def extract_subgraphs(
    model_path: str | os.PathLike,
    ops_chains: list[list[str]],
    save_dir: str | os.PathLike,
    limit: int | None = None
) -> list[Path]:
    model = onnx.load(model_path)
    subgraphs_dirs: list[Path] = []
    for ops_chain in ops_chains:
        chain_name = "-".join(ops_chain)
        subgraphs_dir = Path(save_dir) / chain_name
        subgraphs_dir.mkdir(exist_ok=True, parents=True)
        for f in subgraphs_dir.iterdir():
            if f.is_file() and f.suffix == ".onnx" and chain_name in f.name:
                f.unlink()
            if f.is_dir() and chain_name in f.name:
                rmtree(f, ignore_errors=True)
        matches = extract_boundary_tensors(model, ops_chain)
        for i, match in enumerate(matches):
            if isinstance(limit, int) and i >= limit:
                break
            output_path = subgraphs_dir / f"{chain_name}_{i + 1}.onnx"
            onnx.utils.extract_model(model_path, output_path, match["inputs"], match["outputs"])
            graph = gs.import_onnx(onnx.load(output_path))
            graph.name = "main"
            graph = graph.cleanup(
                remove_unused_graph_inputs=True,
                remove_unused_node_outputs=True
            ).toposort()
            extracted = gs.export_onnx(graph)
            extracted = onnx.shape_inference.infer_shapes(extracted, check_type=True, strict_mode=True)
            onnx.checker.check_model(extracted, full_check=True)
            onnx.save(extracted, output_path)
        if matches:
            subgraphs_dirs.append(subgraphs_dir)
    return subgraphs_dirs


def normalize_layer_name(
    name: str,
    *,
    replacement: str = "_",
    collapse: bool = True,
    strip: bool = True,
    lowercase: bool = False,
) -> str:
    """Normalize an ONNX layer name into a safe version for use in model I/O."""

    _VALID_CHARS = re.compile(r"[^0-9a-zA-Z_]")

    if not name:
        return "unnamed"

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.replace("\\", replacement).replace("/", replacement)
    name = _VALID_CHARS.sub(replacement, name)

    if collapse:
        name = re.sub(rf"{re.escape(replacement)}+", replacement, name)
    if strip:
        name = name.strip(replacement)
    if lowercase:
        name = name.lower()

    return name or "unnamed"


# -----------------------------------------------------------------------------
# DType utilities
# -----------------------------------------------------------------------------

DTypeLike = Union[int, np.dtype, type, str, None]

def is_same_dtype(typ1: DTypeLike, typ2: DTypeLike) -> bool:
    if typ1 is typ2:
        return True
    if typ1 == typ2:
        return True

    def _to_np_dtype(typ: DTypeLike) -> np.dtype | None:
        if typ is None:
            return None
        if isinstance(typ, np.dtype):
            return typ
        if isinstance(typ, int):
            try:
                return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(typ))
            except (TypeError, ValueError, KeyError):
                return None
        try:
            return np.dtype(typ)
        except TypeError:
            return None

    dt1 = _to_np_dtype(typ1)
    dt2 = _to_np_dtype(typ2)
    return dt1 is not None and dt2 is not None and dt1 == dt2


# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------

def upgrade_model(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    if (curr_opset := get_model_opset(model)) >= target_opset:
        logger.info("Model already at opset %d >= %d, skipping upgrade", curr_opset, target_opset)
        return model
    upgraded = onnx.version_converter.convert_version(model, target_opset)
    logger.info("Upgraded model opset to %d", target_opset)
    return upgraded


if __name__ == "__main__":
    pass
