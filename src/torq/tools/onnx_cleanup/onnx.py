# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Model-agnostic ONNX cleanup: undo common exporter artifacts.

Composes registered graph edits with ORT-backed constant folding into one
pipeline for models that don't go through a full ``torq.models`` exporter
(which already runs its own graph-edit blocks):

1. ``CollapseUnrolledConcat`` — rewrite per-element unrolled stack/unbind
   Concats back into their source tensor (or a single Slice of it).
2. Constant folding — evaluate all-constant subgraphs (positional-embedding
   builders, constant weight prep, ...) into initializers via
   ``onnx_graphsurgeon``'s ORT-backed ``fold_constants``.
3. ``FoldConvBatchNorm`` — fold exported eval-mode BatchNorm
   (``Conv -> Mul -> Add`` with per-channel constants) into the conv.

Run on fp32 graphs, before ``torq.tools.convert_dtype``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Collection

import onnx
import onnx_graphsurgeon as gs

from ...graph_edit.edits import CollapseUnrolledConcat, FoldConvBatchNorm
from ...graph_edit.onnx import OnnxGraphEditor

logger = logging.getLogger(__name__)

PASSES = ("collapse-concat", "fold-constants", "fold-conv-bn")

GRAPH_NAME = "onnx_cleanup"


def cleanup_onnx_model(
    model: onnx.ModelProto,
    *,
    min_fanin: int = 32,
    skip: Collection[str] = (),
    fold_size_threshold: int | None = None,
) -> onnx.ModelProto:
    """Return a cleaned copy of ``model`` (the input proto is not modified)."""
    unknown = set(skip) - set(PASSES)
    if unknown:
        raise ValueError(f"Unknown cleanup pass(es): {sorted(unknown)}")

    # The collapse edit only rewrites what static shapes prove safe, so give
    # it as much shape information as possible up front.
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as exc:
        logger.warning("shape inference failed, continuing without: %s", exc)

    editor = OnnxGraphEditor(gs.import_onnx(model), GRAPH_NAME)
    with editor:
        if "collapse-concat" not in skip:
            editor.apply_edit(
                CollapseUnrolledConcat(editor.graph, GRAPH_NAME, min_fanin)
            )
        if "fold-constants" not in skip:
            editor.graph.fold_constants(size_threshold=fold_size_threshold)
        if "fold-conv-bn" not in skip:
            editor.apply_edit(FoldConvBatchNorm(editor.graph, GRAPH_NAME))
        cleaned = editor.to_onnx()

    logger.info(
        "cleanup: %d -> %d nodes, %d -> %d initializers",
        len(model.graph.node), len(cleaned.graph.node),
        len(model.graph.initializer), len(cleaned.graph.initializer),
    )
    return cleaned


def _static_fp32_input_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    initializer_names = {init.name for init in model.graph.initializer}
    shapes: dict[str, list[int]] = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        ttype = inp.type.tensor_type
        if ttype.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(
                f"--verify needs fp32 graph inputs; '{inp.name}' is not FLOAT"
            )
        dims = [d.dim_value if d.HasField("dim_value") else -1
                for d in ttype.shape.dim]
        if -1 in dims:
            raise ValueError(
                f"--verify needs static input shapes; '{inp.name}' has {dims}"
            )
        shapes[inp.name] = dims
    return shapes


def add_onnx_cleanup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX model")
    parser.add_argument(
        "--min-fanin", type=int, default=32,
        help="Only collapse Concat nodes with at least this many inputs (default: 32)",
    )
    parser.add_argument(
        "--skip", action="append", choices=PASSES, default=None, metavar="PASS",
        help=f"Skip one of the cleanup passes {PASSES}. Repeatable.",
    )
    parser.add_argument(
        "--fold-size-threshold", type=int, default=None, metavar="BYTES",
        help="Don't fold constants larger than this many bytes (default: no limit)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check cleaned vs original outputs on random inputs "
             "(onnxruntime; needs static fp32 graph inputs)",
    )


def onnx_cleanup_from_args(args: argparse.Namespace) -> None:
    model = onnx.load(args.input)
    cleaned = cleanup_onnx_model(
        model,
        min_fanin=args.min_fanin,
        skip=tuple(args.skip or ()),
        fold_size_threshold=args.fold_size_threshold,
    )
    if args.verify:
        from ...utils.onnx_verify import verify_equivalence

        verify_equivalence(model, cleaned, _static_fp32_input_shapes(model))
        logger.info("verify: cleaned model matches the original")
    onnx.save(cleaned, args.output)
    logger.info("saved %s", args.output)
