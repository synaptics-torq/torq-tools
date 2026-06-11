# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

""":class:`Recipe` -- per-target preparation declaration.

A recipe declares conversion-target identity (``key``, ``repo_id``) plus the
source ONNX file(s) inside that HuggingFace repo (``source_filename``). Input
shapes are auto-discovered from each source ONNX's ``graph.input`` by
:func:`prepare.prepare`; output shapes are derived by ``shape_inference`` as
the simplification pipeline runs.

The optional ``input_shape_overrides`` field is a safety hatch for the rare
case where the source ONNX has dynamic input ``dim_param``s that cannot be
auto-resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Recipe:
    """Declarative description of one Synaptics audio conversion target.

    Attributes:
        key: short, lowercase, underscore-separated identifier (CLI-facing).
        repo_id: HuggingFace repo identifier (e.g. ``"Synaptics/Voice-Filter"``).
        source_filename: path or paths of source FP32 ONNX files inside the
            HF repo. When ``None``, ``prepare`` requires an explicit ``src``
            argument. Multiple source filenames are all converted when the CLI
            destination is a directory.
        input_shape_overrides: optional ``{input_name: static_shape}``. Only
            needed when the source ONNX's ``graph.input`` has dynamic dim_params
            that auto-discovery cannot resolve. Default: empty.
    """

    key: str
    repo_id: str
    source_filename: str | tuple[str, ...] | None = None
    input_shape_overrides: Mapping[str, Sequence[int]] = field(default_factory=dict)

    def source_filenames(self) -> tuple[str, ...]:
        """Return all HF source filenames declared by this recipe."""
        if self.source_filename is None:
            return ()
        if isinstance(self.source_filename, str):
            return (self.source_filename,)
        return self.source_filename
