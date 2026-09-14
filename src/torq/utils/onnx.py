# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import logging
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference


logger = logging.getLogger(__name__)


__all__ = [
    # CLI helpers
    "add_onnx_args",
    "validate_onnx_source_dir",

    # model inspection
    "get_model_opset",
    "get_model_ops_count",
    "check_dynamic_shapes",
    "print_onnx_model_inputs_outputs_info",

    # Transformations
    "drop_empty_name_value_info",
    "save_onnx_split_weights",
    "finalize_torq_ready_onnx",
]


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def add_onnx_args(
    parser: argparse.ArgumentParser,
    *,
    model_dtypes: list[str] | None = None,
    dynamic_quantize: bool = True,
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
        help="Directory containing source ONNX models (skips the source download)",
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
    group.add_argument(
        "--no-onnx-cleanup",
        action="store_true",
        default=False,
        help="Skip the torq.model_export.cleanup pipeline (collapse unrolled "
             "Concats, fold constants, fold Conv+BatchNorm) that runs on each "
             "exported component before dtype conversion",
    )
    group.add_argument(
        "--split-weights",
        action="store_true",
        default=False,
        help=(
            "Write the exported models (and --dump-after-edit files) with tensor "
            "data above 1024 bytes in an external <model>.onnx.data file (ONNX "
            "convention) so the .onnx stays lightweight and opens fast in a model "
            "viewer. Only tensor data above 1024 bytes is externalized; smaller "
            "constants stay inline so onnxruntime can still resolve shape-op "
            "inputs (Squeeze/Reshape axes, Slice/Pad parameters) at load time."
        ),
    )
    if allow_no_opt:
        group.add_argument(
            "--no-optimize",
            action="store_true",
            default=False,
            help="Do no optimize exported ONNX models via onnxruntime",
        )
    if dynamic_quantize:
        group.add_argument(
            "--dynamic-quantize",
            action="store_true",
            default=False,
            help="Dynamically quantize the model to 8-bit integer"
        )
        group.add_argument(
            "--dynamic-quantize-uint8-weights",
            action="store_true",
            default=False,
            help="Quantize weights with unsigned values"
        )
        group.add_argument(
            "--dynamic-quantize-per-tensor",
            action="store_true",
            default=False,
            help="Quantize weights with per-tensor scale and zero point"
        )
        group.add_argument(
            "--dynamic-quantization-skip-model",
            type=str,
            nargs="+",
            metavar="COMPONENT",
            default=None,
            help="Skip dynamic quantization for the given model component(s) "
            "(e.g. quantize the transformer but not the lm_head)",
        )
        group.add_argument(
            "--dynamic-quantize-analyze-nodes",
            action="store_true",
            default=False,
            help="Also run a per-node quantization-sensitivity report (one quantize+inference "
            "pass per candidate node; slow on large models). A fast whole-model summary "
            "always runs regardless of this flag.",
        )
    if convert_dtypes:
        group.add_argument(
            "--convert-dtypes",
            action="store_true",
            default=False,
            help="Convert the exported model to the dtypes Torq supports (float -> bf16, int64 -> int32)"
        )
        group.add_argument(
            "--preserve-io-dtypes",
            action="store_true",
            default=False,
            help="Preserve model input/output dtypes by adding runtime casts"
        )


def validate_onnx_source_dir(
    onnx_source_dir: str | os.PathLike | None,
    required_files: tuple[str, ...] = (),
) -> Path | None:
    if onnx_source_dir is None:
        return None

    source_dir = Path(onnx_source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"ONNX source directory does not exist: '{source_dir}'")
    for filename in ("config.json", *required_files):
        if not (source_dir / filename).is_file():
            raise FileNotFoundError(
                f"Expected {filename} in ONNX source directory: '{source_dir}'"
            )
    return source_dir


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


def save_onnx_split_weights(model: onnx.ModelProto, path: str | os.PathLike) -> None:
    """Save ``model`` to ``path`` with large tensor data in ``<path>.data``.

    Follows the ONNX external-data convention (``model.onnx`` + ``model.onnx.data``).
    Only tensor data above 1024 bytes (onnx's ``size_threshold``) is externalized; 
    smaller constants (e.g. Squeeze/Reshape axes, Slice/Pad parameters) stay inline.
    """
    path = Path(path)
    data_path = path.parent / f"{path.name}.data"
    # onnx.save appends to an existing external-data file (each tensor lands at
    # a fresh offset beyond the old content), so a rewritten dump would
    # accumulate stale data across saves; start from a clean file.
    if data_path.exists():
        data_path.unlink()
    onnx.save(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=1024,
    )


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


if __name__ == "__main__":
    pass
