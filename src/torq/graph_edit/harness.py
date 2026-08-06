# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Declarative graph-edit harness.

Lets callers describe the graph edits to apply as an ordered list of named
:class:`EditSpec` entries instead of hard-coding imperative editor calls.  The
same specs power the ``--apply-graph-edit`` / ``--apply-graph-edits-from-file``
CLI flags (and their ``--exclude`` counterparts) as well as
``--view-graph-edits``.

Exporter-defined *default* specs may reference runtime graph objects (e.g. a
``cur_len`` tensor built mid-export) via :func:`ctx` placeholders that are
resolved against an export-time context mapping.  User-supplied specs from
flags/files carry *literal* arguments only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .onnx import OnnxGraphEdit

__all__ = [
    "ContextRef",
    "ctx",
    "EditSpec",
    "GraphEditHarness",
    "edit_registry",
    "resolve_args",
    "parse_edit_flag",
    "load_edit_specs_file",
    "load_exclude_names_file",
    "add_graph_edit_harness_args",
    "render_graph_edit_plan",
]


@dataclass(frozen=True)
class ContextRef:
    """Placeholder for a runtime value supplied via the export context."""

    name: str

    def __repr__(self) -> str:  # nicer rendering in --view-graph-edits
        return f"${self.name}"


def ctx(name: str) -> ContextRef:
    return ContextRef(name)


@dataclass(frozen=True)
class EditSpec:
    """A named graph edit plus its positional constructor arguments."""

    name: str
    args: tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    def render(self) -> str:
        if not self.args:
            return self.name
        rendered = ", ".join(repr(a) for a in self.args)
        return f"{self.name}:[{rendered}]"


# -----------------------------------------------------------------------------
# Registry: edit class name -> class
# -----------------------------------------------------------------------------

_REGISTRY: dict[str, type[OnnxGraphEdit]] | None = None


def edit_registry() -> dict[str, type[OnnxGraphEdit]]:
    """Return a mapping of registered edit name -> :class:`OnnxGraphEdit` class.

    Built lazily by scanning the ``torq.graph_edit.edits`` package for concrete
    :class:`OnnxGraphEdit` subclasses.
    """
    global _REGISTRY
    if _REGISTRY is None:
        from . import edits as _edits_pkg

        registry: dict[str, type[OnnxGraphEdit]] = {}
        for obj in vars(_edits_pkg).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, OnnxGraphEdit)
                and obj is not OnnxGraphEdit
            ):
                registry[obj.__name__] = obj
        _REGISTRY = registry
    return _REGISTRY


def resolve_args(args: tuple[Any, ...], context: dict[str, Any]) -> list[Any]:
    """Resolve any :class:`ContextRef` placeholders against ``context``."""
    resolved: list[Any] = []
    for arg in args:
        if isinstance(arg, ContextRef):
            if arg.name not in context:
                raise KeyError(
                    f"Graph edit context is missing required value '{arg.name}'"
                )
            resolved.append(context[arg.name])
        else:
            resolved.append(arg)
    return resolved


# -----------------------------------------------------------------------------
# Parsing (CLI flags + files)
# -----------------------------------------------------------------------------

def _coerce_args(parsed: Any) -> tuple[Any, ...]:
    if parsed is None:
        return ()
    if isinstance(parsed, list):
        return tuple(parsed)
    return (parsed,)


def parse_edit_flag(value: str) -> EditSpec:
    """Parse a ``--apply-graph-edit`` value of the form ``Name`` or ``Name:<yaml>``.

    The argument portion after the first ``:`` is parsed as a YAML expression,
    so ``EliminateExpand:[Add, Mul]`` yields args ``('Add', 'Mul')`` and
    ``WidenStridedDepthwiseConv:[64, 4]`` yields ``(64, 4)``.
    """
    import yaml

    name, sep, rest = value.partition(":")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(
            f"Invalid graph edit '{value}': empty edit name"
        )
    if not sep or not rest.strip():
        return EditSpec(name, ())
    try:
        parsed = yaml.safe_load(rest)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid graph edit args for '{name}': {exc}"
        ) from exc
    return EditSpec(name, _coerce_args(parsed))


def _spec_from_item(item: Any) -> EditSpec:
    if isinstance(item, str):
        return EditSpec(item.strip(), ())
    if isinstance(item, dict):
        if "name" not in item:
            raise ValueError(f"Graph edit entry is missing 'name': {item!r}")
        return EditSpec(str(item["name"]).strip(), _coerce_args(item.get("args")))
    raise ValueError(f"Unsupported graph edit entry: {item!r}")


def _load_edit_items(path: str | Path) -> list[Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return []
    if isinstance(data, dict):
        for key in ("edits", "excludes", "graph_edits"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                f"Graph edit file '{path}' must contain a list or an 'edits' key"
            )
    if not isinstance(data, list):
        raise ValueError(f"Graph edit file '{path}' must contain a YAML list")
    return data


def load_edit_specs_file(path: str | Path) -> list[EditSpec]:
    return [_spec_from_item(item) for item in _load_edit_items(path)]


def load_exclude_names_file(path: str | Path) -> list[str]:
    return [_spec_from_item(item).name for item in _load_edit_items(path)]


# -----------------------------------------------------------------------------
# Harness
# -----------------------------------------------------------------------------

@dataclass
class GraphEditHarness:
    """Holds the parsed user overrides and merges them with exporter defaults.

    Precedence (high -> low) is flags > file > defaults.  An overriding edit
    keeps the original position of the default it replaces; brand-new edits are
    appended.  Excluded edits are removed last.
    """

    apply_flag: list[EditSpec] = field(default_factory=list)
    apply_file: list[EditSpec] = field(default_factory=list)
    exclude: set[str] = field(default_factory=set)
    view: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GraphEditHarness":
        apply_flag = list(getattr(args, "apply_graph_edit", None) or [])
        apply_file: list[EditSpec] = []
        edits_file = getattr(args, "apply_graph_edits_from_file", None)
        if edits_file:
            apply_file = load_edit_specs_file(edits_file)

        exclude: set[str] = set(getattr(args, "exclude_graph_edit", None) or [])
        exclude_file = getattr(args, "exclude_graph_edits_from_file", None)
        if exclude_file:
            exclude.update(load_exclude_names_file(exclude_file))

        return cls(
            apply_flag=apply_flag,
            apply_file=apply_file,
            exclude=exclude,
            view=bool(getattr(args, "view_graph_edits", False)),
        )

    def _validate_names(self) -> None:
        registry = edit_registry()
        unknown = {
            spec.name
            for spec in (*self.apply_flag, *self.apply_file)
            if spec.name not in registry
        } | {name for name in self.exclude if name not in registry}
        if unknown:
            raise ValueError(
                "Unknown graph edit(s): "
                + ", ".join(sorted(unknown))
                + f". Available: {', '.join(sorted(registry))}"
            )

    def finalize(self, defaults: list[EditSpec]) -> list[EditSpec]:
        """Merge ``defaults`` with the user overrides into the final ordered list."""
        self._validate_names()

        default_names = {spec.name for spec in defaults}
        overrides: dict[str, tuple[Any, ...]] = {}
        extras: dict[str, tuple[Any, ...]] = {}
        extras_order: list[str] = []

        # file first, then flags so flags win on ties (flags > file).
        for spec in (*self.apply_file, *self.apply_flag):
            if spec.name in default_names:
                # Only override args when the user actually supplied some, so a
                # bare name never wipes a default's (possibly runtime) args.
                if spec.args:
                    overrides[spec.name] = spec.args
            else:
                if spec.name not in extras:
                    extras_order.append(spec.name)
                extras[spec.name] = spec.args

        result: list[EditSpec] = []
        for spec in defaults:
            args = overrides.get(spec.name, spec.args)
            result.append(EditSpec(spec.name, args))
        for name in extras_order:
            result.append(EditSpec(name, extras[name]))

        return [spec for spec in result if spec.name not in self.exclude]


# -----------------------------------------------------------------------------
# CLI integration
# -----------------------------------------------------------------------------

def add_graph_edit_harness_args(parser: argparse.ArgumentParser) -> None:
    """Add the graph-edit harness flags to an export subparser."""
    group = parser.add_argument_group("Graph edit harness args")
    group.add_argument(
        "--apply-graph-edit",
        type=parse_edit_flag,
        action="append",
        metavar="NAME[:ARGS]",
        default=None,
        help=(
            "Apply a graph edit by name, optionally with YAML args, e.g. "
            "'EliminateExpand:[Add, Mul]'. Repeatable. Takes priority over "
            "the exporter's default edits."
        ),
    )
    group.add_argument(
        "--apply-graph-edits-from-file",
        type=str,
        metavar="FILE",
        default=None,
        help="Apply graph edits listed in a YAML file (list of names/{name,args}).",
    )
    group.add_argument(
        "--exclude-graph-edit",
        type=str,
        action="append",
        metavar="NAME",
        default=None,
        help="Exclude a graph edit by name. Repeatable.",
    )
    group.add_argument(
        "--exclude-graph-edits-from-file",
        type=str,
        metavar="FILE",
        default=None,
        help="Exclude graph edits named in a YAML file.",
    )
    group.add_argument(
        "--view-graph-edits",
        action="store_true",
        default=False,
        help="Print the finalized graph edits for this export command and exit.",
    )


def render_graph_edit_plan(blocks: "dict[str, list[EditSpec]]") -> str:
    """Render finalized edit blocks (name -> ordered specs) as printable text."""
    lines: list[str] = ["Graph edits to be applied:"]
    if not blocks:
        lines.append("  (none)")
        return "\n".join(lines)
    for block_name, specs in blocks.items():
        lines.append(f"\n[{block_name}]")
        if not specs:
            lines.append("  (none)")
            continue
        for i, spec in enumerate(specs, 1):
            lines.append(f"  {i:>2}. {spec.render()}")
    return "\n".join(lines)
