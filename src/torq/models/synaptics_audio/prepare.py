# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Run the preparation flow for a Synaptics audio recipe.

Flow:

1. Resolve the source FP32 ONNX path (auto-fetch from HF Hub via
   :func:`fetch.fetch_source` if no explicit ``src`` is given).
2. Auto-discover its static input shapes from ``graph.input`` (with optional
   per-recipe overrides).
3. Apply :data:`passes.PIPELINE` to fold and simplify the graph (still FP32);
   the passes' internal ``shape_inference`` resolves the output shapes from
   the now-fixed input shapes.
4. Verify the simplified FP32 model is numerically equivalent to the source
   on a single batch of random inputs (:func:`verify.verify_equivalence`).
5. Convert the simplified FP32 graph to BF16 and write ``dst``. If ``dst`` is
   a directory, the output file keeps the source ONNX stem and appends
   ``_torq_bf16.onnx``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import onnx

from torq.tools.convert_dtype.onnx import convert_model

from .fetch import fetch_sources
from .passes import PIPELINE
from .passes.base import PassContext
from .recipes import Recipe
from .verify import verify_equivalence

logger = logging.getLogger("synaptics-audio.prepare")

TORQ_BF16_SUFFIX = "_torq_bf16"


def _resolve_input_shapes(
    model: onnx.ModelProto,
    overrides: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    """Auto-discover each ``graph.input``'s static shape; overrides win."""
    resolved: dict[str, tuple[int, ...]] = {}
    unresolved: list[tuple[str, list]] = []
    for vi in model.graph.input:
        if vi.name in overrides:
            resolved[vi.name] = tuple(int(d) for d in overrides[vi.name])
            continue
        dims: list[int | None] = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        if any(x is None for x in dims):
            unresolved.append((vi.name, [d if d is not None else "?" for d in dims]))
        else:
            resolved[vi.name] = tuple(d for d in dims if d is not None)
    if unresolved:
        details = ", ".join(f"{n}={s}" for n, s in unresolved)
        raise ValueError(
            f"source ONNX has dynamic input dims that cannot be auto-resolved: "
            f"{details}. Provide Recipe.input_shape_overrides for the affected input(s)."
        )
    return resolved


def _resolve_output_path(src_path: Path, dst: Path) -> Path:
    """Return explicit ``dst`` or ``dst/<source_stem>_torq_bf16.onnx``."""
    dst = Path(dst)
    if _is_output_directory(dst):
        return dst / f"{src_path.stem}{TORQ_BF16_SUFFIX}.onnx"
    return dst


def _is_output_directory(dst: Path) -> bool:
    return (dst.exists() and dst.is_dir()) or dst.suffix == ""


def _prepare_one(recipe: Recipe, src_path: Path, dst: Path) -> Path:
    dst = _resolve_output_path(src_path, Path(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading FP32 source: %s", src_path)
    fp32 = onnx.load(str(src_path))

    input_shapes = _resolve_input_shapes(fp32, recipe.input_shape_overrides)
    logger.info("resolved input shapes: %s", input_shapes)

    ctx = PassContext(input_shapes=input_shapes)

    simplified = fp32
    for pass_ in PIPELINE:
        before = len(simplified.graph.node)
        simplified = pass_(simplified, ctx)
        after = len(simplified.graph.node)
        logger.info("pass %-32s nodes %4d -> %4d", pass_.name, before, after)

    logger.info("verifying FP32 equivalence on %d input(s)", len(input_shapes))
    verify_equivalence(fp32, simplified, input_shapes)

    with tempfile.TemporaryDirectory(prefix="synaptics_audio_") as tmp:
        tmp_fp32 = Path(tmp) / f"{src_path.stem}_simplified_fp32.onnx"
        onnx.save(simplified, str(tmp_fp32))
        logger.info("converting to BF16 -> %s", dst)
        convert_model(
            str(tmp_fp32),
            str(dst),
            convert_dtype="bf16",
            convert_io=True,
            skip_torq_onnx_finalize=True,
            preserve_unused_node_outputs=True,
        )

    return dst


def prepare(
    recipe: Recipe,
    dst: Path,
    *,
    src: Path | str | None = None,
) -> Path | list[Path]:
    """Run the FP32 -> simplified FP32 -> BF16 flow for ``recipe``.

    If ``src`` is ``None``, every source ONNX declared by the recipe is
    auto-fetched from the HuggingFace Hub. Recipes with multiple sources must
    be prepared into a directory so each output can keep its source stem.
    """
    dst_path = Path(dst)
    if src is not None:
        src_paths = [Path(src)]
    else:
        source_filenames = recipe.source_filenames()
        if len(source_filenames) > 1 and not _is_output_directory(dst_path):
            raise ValueError(
                f"recipe {recipe.key!r} declares {len(source_filenames)} source filenames; "
                "use a directory destination so each output can keep its source name"
            )
        src_paths = fetch_sources(recipe)

    if not src_paths:
        raise ValueError(
            f"recipe {recipe.key!r} has no source_filename; cannot auto-fetch from HF. "
            f"Either set source_filename in the recipe or provide an explicit src path."
        )

    outputs = [_prepare_one(recipe, src_path, dst_path) for src_path in src_paths]
    return outputs[0] if len(outputs) == 1 else outputs
