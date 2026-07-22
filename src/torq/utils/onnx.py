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
from onnx import shape_inference


logger = logging.getLogger(__name__)


__all__ = [
    # CLI helpers
    "add_onnx_args",

    # model inspection
    "get_model_opset",
    "get_model_ops_count",
    "check_dynamic_shapes",
    "print_onnx_model_inputs_outputs_info",
    "fold_shape_ops",
    "resolve_negative_slices",
    "propagate_static_shapes",

    # subgraph extraction
    "extract_boundary_tensors",
    "extract_subgraphs",

    # DType utilities
    "DTypeLike",
    "is_same_dtype",

    # Transformations
    "drop_empty_name_value_info",
    "upgrade_model",
    "finalize_torq_ready_onnx",
]


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def add_onnx_args(
    parser: argparse.ArgumentParser,
    *,
    model_dtypes: list[str] | None = None,
    convert_dtypes: bool = False,
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
            "--convert-dtypes",
            action="store_true",
            default=False,
            help="Convert model to supported dtypes"
        )
        group.add_argument(
            "--preserve-io-dtypes",
            action="store_true",
            default=False,
            help="Preserve model input/output dtypes by adding runtime casts"
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


def fold_shape_ops(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Constant-fold ``Shape`` ops (whose input has a fully static shape)
    and the small set of shape-math ops that typically sit between
    ``Shape`` and downstream consumers in transformer graphs: Gather,
    Slice, Squeeze, Unsqueeze, Concat, Cast, Add, Sub, Mul, Div, Mod,
    Range, ReduceProd. Folding these lets ``onnx.shape_inference`` see
    all the symbolic dims (``unk__N``) as concrete integers, which in
    turn lets torq-compile's Where-broadcast lowering succeed.
    """
    graph = gs.import_onnx(model)
    folded = 0

    def _const(t):
        return isinstance(t, gs.Constant)

    def _all_const(node):
        return node.inputs and all(_const(i) for i in node.inputs)

    def _as_arr(t):
        return np.asarray(t.values)

    progressed = True
    while progressed:
        progressed = False
        for node in list(graph.nodes):
            out = node.outputs[0] if node.outputs else None
            if out is None:
                continue
            new_val = None
            try:
                if node.op == "Shape":
                    inp = node.inputs[0]
                    sh = getattr(inp, "shape", None)
                    if sh is None or any(not isinstance(d, int) for d in sh):
                        continue
                    new_val = np.array(sh, dtype=np.int64)
                elif not _all_const(node):
                    continue
                elif node.op == "Cast":
                    to = int(node.attrs["to"])
                    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(to)
                    new_val = _as_arr(node.inputs[0]).astype(np_dtype)
                elif node.op == "Gather":
                    data = _as_arr(node.inputs[0]); idx = _as_arr(node.inputs[1])
                    axis = int(node.attrs.get("axis", 0))
                    new_val = np.take(data, idx, axis=axis)
                elif node.op == "Slice":
                    data = _as_arr(node.inputs[0])
                    starts = _as_arr(node.inputs[1]); ends = _as_arr(node.inputs[2])
                    axes = _as_arr(node.inputs[3]) if len(node.inputs) > 3 else np.arange(len(starts))
                    steps = _as_arr(node.inputs[4]) if len(node.inputs) > 4 else np.ones_like(starts)
                    sl = [slice(None)] * data.ndim
                    for a, s, e, st in zip(np.atleast_1d(axes), np.atleast_1d(starts), np.atleast_1d(ends), np.atleast_1d(steps)):
                        sl[int(a)] = slice(int(s), int(e), int(st))
                    new_val = data[tuple(sl)]
                elif node.op == "Squeeze":
                    data = _as_arr(node.inputs[0])
                    axes = _as_arr(node.inputs[1]) if len(node.inputs) > 1 else None
                    new_val = np.squeeze(data, axis=tuple(int(a) for a in np.atleast_1d(axes)) if axes is not None else None)
                elif node.op == "Unsqueeze":
                    data = _as_arr(node.inputs[0])
                    axes = _as_arr(node.inputs[1])
                    new_val = data
                    for a in sorted(int(x) for x in np.atleast_1d(axes)):
                        new_val = np.expand_dims(new_val, axis=a)
                elif node.op == "Concat":
                    arrs = [_as_arr(i) for i in node.inputs]
                    axis = int(node.attrs.get("axis", 0))
                    new_val = np.concatenate(arrs, axis=axis)
                elif node.op in ("Add", "Sub", "Mul", "Div", "Mod"):
                    a = _as_arr(node.inputs[0]); b = _as_arr(node.inputs[1])
                    op_map = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
                              "Div": np.divide if a.dtype.kind == "f" else np.floor_divide,
                              "Mod": np.mod}
                    new_val = op_map[node.op](a, b)
                elif node.op == "Neg":
                    new_val = -_as_arr(node.inputs[0])
                elif node.op == "Identity":
                    new_val = _as_arr(node.inputs[0]).copy()
                elif node.op == "Abs":
                    new_val = np.abs(_as_arr(node.inputs[0]))
                elif node.op == "Reshape":
                    data = _as_arr(node.inputs[0]); shape = _as_arr(node.inputs[1])
                    new_val = data.reshape(tuple(int(s) for s in shape))
                elif node.op == "Constant":
                    # Already a constant — skip (it's just the value).
                    continue
                elif node.op == "ConstantOfShape":
                    shape = _as_arr(node.inputs[0])
                    value_attr = node.attrs.get("value")
                    if value_attr is None:
                        new_val = np.zeros(tuple(int(s) for s in shape), dtype=np.float32)
                    else:
                        v = value_attr.values
                        new_val = np.full(tuple(int(s) for s in shape), v.item() if v.size == 1 else v, dtype=v.dtype)
                elif node.op == "ReduceProd":
                    data = _as_arr(node.inputs[0])
                    axes_t = node.inputs[1] if len(node.inputs) > 1 else None
                    axes = tuple(int(x) for x in np.atleast_1d(_as_arr(axes_t))) if axes_t is not None else None
                    keepdims = bool(node.attrs.get("keepdims", 1))
                    new_val = np.prod(data, axis=axes, keepdims=keepdims)
                elif node.op == "Range":
                    s = int(_as_arr(node.inputs[0])); e = int(_as_arr(node.inputs[1])); st = int(_as_arr(node.inputs[2]))
                    new_val = np.arange(s, e, st, dtype=_as_arr(node.inputs[0]).dtype)
            except Exception:
                new_val = None

            if new_val is None:
                continue

            new_const = gs.Constant(
                name=out.name + "_folded",
                values=np.asarray(new_val),
            )
            for consumer in list(out.outputs):
                for idx, t in enumerate(consumer.inputs):
                    if t is out:
                        consumer.inputs[idx] = new_const
            # Disconnect the node by detaching its inputs/outputs so
            # gs.cleanup removes it.
            node.inputs.clear()
            node.outputs.clear()
            folded += 1
            progressed = True

    # Remove disconnected nodes explicitly (cleanup with remove_unused
    # only removes nodes whose outputs are unused — but our cleared
    # outputs aren't considered "used" so this is redundant insurance).
    graph.nodes = [n for n in graph.nodes if n.outputs or n.inputs]
    graph.cleanup(
        remove_unused_graph_inputs=True, remove_unused_node_outputs=True
    ).toposort()
    if folded:
        new_model = gs.export_onnx(graph)
        new_model.ir_version = model.ir_version
        return new_model, folded
    return model, 0


def resolve_negative_slices(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace Slice ops whose ``starts``/``ends`` constants are
    negative (or clamped to INT64_MAX) into normalized positive
    indices based on the data tensor's static shape, so downstream
    shape inference can resolve the output to a concrete shape.

    Required because ONNX shape inference does not normalize negative
    Slice indices when the data shape is static, leaving the output
    marked as dynamic. We rewrite the Slice's starts/ends constants
    in place — equivalent to a no-op for runtime but unlocks full
    static shape propagation.
    """
    graph = gs.import_onnx(model)
    rewritten = 0
    INT_MAX = 9223372036854775807
    for node in graph.nodes:
        if node.op != "Slice":
            continue
        data = node.inputs[0]
        data_shape = getattr(data, "shape", None)
        if data_shape is None or any(not isinstance(d, int) for d in data_shape):
            continue
        if len(node.inputs) < 3:
            continue
        starts = node.inputs[1]
        ends = node.inputs[2]
        axes = node.inputs[3] if len(node.inputs) > 3 else None
        if not (isinstance(starts, gs.Constant) and isinstance(ends, gs.Constant)):
            continue
        if axes is not None and not isinstance(axes, gs.Constant):
            continue
        s_vals = np.atleast_1d(starts.values).astype(np.int64).copy()
        e_vals = np.atleast_1d(ends.values).astype(np.int64).copy()
        a_vals = (
            np.atleast_1d(axes.values).astype(np.int64)
            if axes is not None else np.arange(len(s_vals), dtype=np.int64)
        )
        changed = False
        for i, ax in enumerate(a_vals):
            dim_size = int(data_shape[int(ax)])
            if s_vals[i] < 0:
                s_vals[i] = max(0, s_vals[i] + dim_size)
                changed = True
            if e_vals[i] < 0:
                e_vals[i] = max(0, e_vals[i] + dim_size)
                changed = True
            if e_vals[i] > dim_size or e_vals[i] >= INT_MAX // 2:
                e_vals[i] = dim_size
                changed = True
        if changed:
            # Replace inputs with fresh constants — the original
            # constants are shared across many Slice ops, so we must
            # NOT mutate them in place.
            node.inputs[1] = gs.Constant(
                name=f"{node.name}_starts_norm",
                values=s_vals.astype(starts.values.dtype),
            )
            node.inputs[2] = gs.Constant(
                name=f"{node.name}_ends_norm",
                values=e_vals.astype(ends.values.dtype),
            )
            rewritten += 1
    if rewritten:
        new_model = gs.export_onnx(graph)
        new_model.ir_version = model.ir_version
        return new_model, rewritten
    return model, 0


def propagate_static_shapes(model: onnx.ModelProto, max_iters: int = 8) -> onnx.ModelProto:
    """Iteratively fold Shape ops and re-run data-propagating shape
    inference until no more changes occur. Required for downstream
    Where-lowering / broadcast inference to see static dims instead of
    symbolic ``unk__N`` placeholders.
    """
    for it in range(max_iters):
        model, n_folded = fold_shape_ops(model)
        model, n_slice = resolve_negative_slices(model)
        logger.info(
            "(shape-prop) iter %d: folded %d node(s), normalized %d slice op(s)",
            it, n_folded, n_slice,
        )
        del model.graph.value_info[:]
        try:
            model = shape_inference.infer_shapes(
                model, check_type=False, strict_mode=False, data_prop=True
            )
        except Exception as e:
            logger.warning(
                "(shape-prop) shape inference reported issues at iter %d: %s", it, e,
            )
        if n_folded == 0 and n_slice == 0:
            logger.info("(shape-prop) converged after %d iter(s)", it)
            break
    return model


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

def drop_empty_name_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ``graph.value_info`` entries with an empty name.

    Rare leftover when an RNN optional output still uses ``""``; fails ONNX
    validation: ``Field 'name' of 'value_info' is required to be non-empty``.
    """
    graph = model.graph
    kept = [vi for vi in graph.value_info if vi.name]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return model


def finalize_torq_ready_onnx(
    model: onnx.ModelProto,
    *,
    max_ir_version: int = 11,
    symbolic_shape_infer: bool = True,
) -> onnx.ModelProto:
    """Post-process ONNX for Torq import and layer extraction.

    - Optionally runs ONNX Runtime symbolic shape inference so ``unk__`` dims
      become static where ORT can derive them.
    - Drops ``graph.value_info`` entries whose names duplicate ``graph.output``
      (avoids rank mismatches in torch-onnx import for isolated subgraphs).
    - Caps ``ir_version`` for broader onnxruntime / tooling compatibility.
    - Refreshes standard ONNX shape inference when possible.

    Mutates ``model`` unless a shape-inference step returns a replacement
    model. Requires ``onnxruntime`` with ``tools.symbolic_shape_infer`` for the
    first step; if unavailable or it fails, subsequent steps still run.
    """
    work = model
    if symbolic_shape_infer:
        try:
            from onnxruntime.tools.symbolic_shape_infer import (
                SymbolicShapeInference,
            )

            work = SymbolicShapeInference.infer_shapes(
                work,
                auto_merge=True,
                guess_output_rank=True,
                verbose=0,
            )
        except Exception as exc:
            logger.debug("Symbolic shape inference skipped: %s", exc)

    work.ir_version = min(int(work.ir_version), max_ir_version)

    work = drop_empty_name_value_info(work)

    graph = work.graph
    out_names = {o.name for o in graph.output}
    kept = [vi for vi in graph.value_info if vi.name and vi.name not in out_names]
    del graph.value_info[:]
    graph.value_info.extend(kept)

    try:
        work = shape_inference.infer_shapes(work)
    except Exception as exc:
        logger.debug("shape_inference.infer_shapes after finalize skipped: %s", exc)

    try:
        onnx.checker.check_model(work, full_check=False)
    except Exception as exc:
        logger.warning("ONNX checker warning after finalize_torq_ready_onnx: %s", exc)

    return work


def upgrade_model(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    if (curr_opset := get_model_opset(model)) >= target_opset:
        logger.info("Model already at opset %d >= %d, skipping upgrade", curr_opset, target_opset)
        return model
    upgraded = onnx.version_converter.convert_version(model, target_opset)
    logger.info("Upgraded model opset to %d", target_opset)
    return upgraded


if __name__ == "__main__":
    pass
