# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://www.apache.org/licenses/LICENSE-2.0 WITH LLVM-exception
"""Recommend batch-prefill sizes from isolated matmul tiling probes.

Keep the largest batch size per weight-strip plateau and report the choices
in ``prefill_sizes.json``. Fused producers can change the full model's tiling.
See the batch prefill user manual for the CLI and compiler workflow.
"""

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from shutil import which

from ..utils.compile import _resolve_compiler

logger = logging.getLogger("torq.model_export.prefill_sizes")

# Default sweep cap; explicit sizes are passed to the exporter.
DEFAULT_MAX_BATCH_SIZE = 128

# User flags stripped from probe compiles: they do not apply to a bare
# linalg matmul probe, or this module sets them itself.
PROBE_FLAG_DENYLIST = (
    "--iree-input-type",
    "--iree-hal-target-backends",
    "--torq-hw",
    "--torq-target-host-triple",
    "--torq-enable-batch-prefill",
    "--torq-prefill-",  # cap and explicit sizes
    "--torq-convert-io-dtype",
    "--torq-debug-info",
    "--torq-enable-profiling",
    "--torq-dump-profiling",
    "--mlir-print",
    "--dump-compilation-phases-to",
)

PROBE_TIMEOUT = 120  # seconds per probe compile
MAX_PROBE_JOBS = max(1, min(16, os.cpu_count() or 16))  # the probes are torq-compile subprocesses

# Bumped when the probe template or the parse changes: stale work-dir caches
# must not serve results produced by different probe behavior.
PROBE_CACHE_VERSION = 2

# The probe weight is a non-splat constant (splats can change codegen);
# the values are irrelevant to tiling, so one deterministic array per shape.
_PROBE_TEMPLATE = """func.func @main(%x: tensor<{m}x{k}xbf16>) -> tensor<{m}x{n}xbf16> {{
  %w = arith.constant dense<"0x{hex}"> : tensor<{k}x{n}xbf16>
  %z = arith.constant 0.0 : bf16
  %e = tensor.empty() : tensor<{m}x{n}xbf16>
  %f = linalg.fill ins(%z : bf16) outs(%e : tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16>
  %r = linalg.matmul ins(%x, %w : tensor<{m}x{k}xbf16>, tensor<{k}x{n}xbf16>) outs(%f : tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16>
  return %r : tensor<{m}x{n}xbf16>
}}
"""


class ProbeError(RuntimeError):
    """A probe compile or parse failed; the caller must choose explicit sizes."""


@dataclass(frozen=True)
class ProbeResult:
    """What one probe compile showed for a weight shape at batch size m."""

    m: int
    width: Optional[int]  # N-tile step; None = weight not tiled along N
    m_tiled: bool
    m_outer: bool  # M loop outside the N loop: the weight strip is reloaded
    # per M pass (re-streamed)

    def signature(self):
        return (self.width, self.m_tiled, self.m_outer)

    @property
    def restreams_weight(self) -> bool:
        return self.m_tiled and self.m_outer


def snap(width: int, decode_width: int) -> int:
    """Mirror the bf16 family grid and aligned divisor choice in TileAndFuse."""
    vector = 32
    grid = 0
    unit = vector
    while unit <= decode_width:
        grid = max(grid, unit, 3 * unit if 3 * unit <= decode_width else 0)
        unit *= 2
    step = grid or decode_width
    alignment = vector if step % vector == 0 and width >= vector else 1
    cap = min(width, step) // alignment
    best = 1
    for d in range(1, math.isqrt(step // alignment) + 1):
        if (step // alignment) % d == 0:
            for divisor in (d, step // alignment // d):
                if divisor <= cap:
                    best = max(best, divisor)
    snapped = best * alignment
    return width if snapped == 1 or snapped * 2 < width else snapped


def filter_probe_flags(compiler_options: List[str]) -> List[str]:
    """The user's compiler options minus the probe denylist."""
    kept = []
    for opt in compiler_options:
        name = opt.split("=", 1)[0]
        if not any(name.startswith(prefix) for prefix in PROBE_FLAG_DENYLIST):
            kept.append(opt)
    return kept


# -- weight-shape extraction -------------------------------------------------

def _squeeze_leading_ones(dims: List[int]) -> Optional[Tuple[int, int]]:
    """A weight of rank 2 (or rank > 2 with leading 1s) as (K, N)."""
    while len(dims) > 2 and dims[0] == 1:
        dims = dims[1:]
    if len(dims) != 2 or dims[0] <= 0 or dims[1] <= 0:
        return None
    return dims[0], dims[1]


def extract_weight_shapes(model_path) -> List[Tuple[int, int]]:
    """The distinct [K, N] matmul weight shapes of the model, in order."""
    path = Path(model_path)
    if path.suffix == ".onnx":
        return _weight_shapes_onnx(path)
    if path.suffix == ".mlir":
        return _weight_shapes_mlir(path)
    raise ProbeError(f"no matmul weight shapes from {path.name} (need .onnx or .mlir)")


def _weight_shapes_onnx(path: Path) -> List[Tuple[int, int]]:
    return _weight_shapes_from_onnx(onnx_load(path))


def onnx_load(path: Path):
    import onnx

    return onnx.load(str(path), load_external_data=False)  # Only metadata is needed.


def _weight_shapes_from_onnx(model) -> List[Tuple[int, int]]:
    inits = {i.name: list(i.dims) for i in model.graph.initializer}
    shapes: List[Tuple[int, int]] = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm") or len(node.input) < 2:
            continue
        dims = inits.get(node.input[1])
        if dims is None:
            continue  # weight not an initializer
        shape = _squeeze_leading_ones(dims)
        if shape is None:
            continue
        if node.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in node.attribute):
            shape = (shape[1], shape[0])
        if shape not in shapes:
            shapes.append(shape)
    return shapes


_CONST_DEF_RE = re.compile(
    r"^(?P<indent>\s*)%(?P<name>\w+) = (?:torch\.operator \"onnx\.Constant\"\(\) \{[^}]*: |"
    r"arith\.constant dense[^:]*: |util\.global\.load @\w+ : )"
    r"tensor<(?P<dims>[\dx]+)x(?P<elt>[a-z0-9]+)>"
)
_MATMUL_RE = re.compile(r"torch\.operator \"onnx\.MatMul\"\((?P<a>%\w+),\s*(?P<b>%\w+)\)")
_GEMM_RE = re.compile(
    r"torch\.operator \"onnx\.Gemm\"\((?P<a>%\w+),\s*(?P<b>%\w+),[^)]*\)(?:\s*\{(?P<attrs>[^}]*)\})?"
)


def _weight_shapes_mlir(path: Path) -> List[Tuple[int, int]]:
    """[K, N] of onnx.MatMul/onnx.Gemm weights in an imported ONNX MLIR.

    A weight counts only when its operand is defined by a constant-like op
    (onnx.Constant, arith.constant dense, or util.global.load)."""
    const_dims: Dict[str, Tuple[int, int]] = {}
    shapes: List[Tuple[int, int]] = []
    with path.open() as fh:
        for line in fh:
            if line.lstrip().startswith("{-#"):
                break  # Shapes are in the body, before the resource payloads.
            if re.match(r"\s*func\.func\b", line):
                const_dims.clear()  # SSA names are local to each function.
            cm = _CONST_DEF_RE.match(line)
            if cm:
                dims = [int(d) for d in cm.group("dims").split("x")]
                shape = _squeeze_leading_ones(dims)
                if shape:
                    const_dims["%" + cm.group("name")] = shape
                continue
            alias = re.match(r"\s*(%\w+) = torch_c\.from_builtin_tensor (%\w+)\b", line)
            if alias and alias[2] in const_dims:
                const_dims[alias[1]] = const_dims[alias[2]]
                continue
            mm = _MATMUL_RE.search(line)
            if mm:
                shape = const_dims.get(mm.group("b"))
                if shape and shape not in shapes:
                    shapes.append(shape)
                continue
            gm = _GEMM_RE.search(line)
            if gm:
                shape = const_dims.get(gm.group("b"))
                if shape:
                    if gm.group("attrs") and re.search(r"transB = 1(?!\d)", gm.group("attrs")):
                        shape = (shape[1], shape[0])
                    if shape not in shapes:
                        shapes.append(shape)
    return shapes


# -- probe compile -----------------------------------------------------------

def _weight_hex(k: int, n: int) -> str:
    w = np.random.RandomState(0).randn(k, n).astype(np.float32) * 0.05
    return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes().hex().upper()


_IDX_CONST_RE = re.compile(r"%(?P<name>\w+) = arith\.constant (?P<value>\d+) : index")
_FOR_RE = re.compile(
    r"^(?P<indent>\s*)(?:%\w+ = )?scf\.for %(?P<iv>\w+) = %(?P<lb>\w+) to %(?P<ub>\w+) step %(?P<step>\w+)"
)
_WEIGHT_DEF_RE = re.compile(
    r"%(?P<name>\w+) = arith\.constant dense[^:]*: tensor<(?P<k>\d+)x(?P<n>\d+)xbf16>"
)


def _region_body(lines: List[str], start: int, indent: int) -> str:
    """The text of the region opened at lines[start], up to the first line
    at or above its indentation."""
    body = []
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip(" "))) <= indent:
            break
        body.append(line)
    return "\n".join(body)


def _parse_tile_and_fuse(stderr: str, m: int, k: int, n: int) -> ProbeResult:
    """Read tiling from the final dump of the probe's single matmul function.

    Expect an arith.constant weight; unsupported IR raises a probe error.
    """
    sections = [
        s for s in stderr.split("// -----// IR Dump After ")
        if re.match(r"\S+(?: \(|: )torq-tile-and-fuse\b", s)
    ]
    if not sections:
        raise ProbeError("no torq-tile-and-fuse dump in the probe output")
    text = sections[-1]
    consts = {c.group("name"): int(c.group("value")) for c in _IDX_CONST_RE.finditer(text)}
    weight = None
    for wd in _WEIGHT_DEF_RE.finditer(text):
        if int(wd.group("k")) == k and int(wd.group("n")) == n:
            weight = wd.group("name")
    if weight is None:
        raise ProbeError(f"no tensor<{k}x{n}xbf16> weight constant in the probe output")

    lines = text.splitlines()
    loops = []
    for i, line in enumerate(lines):
        fm = _FOR_RE.match(line)
        if not fm:
            continue
        lb = consts.get(fm.group("lb"))
        ub = consts.get(fm.group("ub"))
        step = consts.get(fm.group("step"))
        if lb != 0 or ub is None or step is None:
            continue  # only constant-bound loops from zero
        indent = len(fm.group("indent"))
        loops.append(
            {"iv": fm.group("iv"), "ub": ub, "step": step, "indent": indent,
             "body": _region_body(lines, i, indent)}
        )

    weight_loop = None
    w_extract = re.compile(
        r"tensor\.extract_slice " + re.escape("%" + weight) + r"\[0, " + re.escape("%") + r"(?P<iv>\w+)\]"
    )
    for loop in loops:
        for em in w_extract.finditer(loop["body"]):
            if em.group("iv") == loop["iv"]:  # the loop tiles the weight's N dim
                if weight_loop is None or loop["indent"] > weight_loop["indent"]:
                    weight_loop = loop  # the innermost N-tile loop
    m_loop = None
    for loop in loops:
        if loop is weight_loop or loop["ub"] != m:
            continue
        if re.search(r"tensor\.extract_slice %\w+\[" + re.escape("%" + loop["iv"]) + r", 0\]", loop["body"]):
            m_loop = loop
    if weight_loop is None:
        return ProbeResult(m, None, m_loop is not None, False)
    return ProbeResult(
        m,
        weight_loop["step"],
        m_loop is not None,
        m_loop is not None and m_loop["indent"] < weight_loop["indent"],
    )


def run_probe(tool: str, probe_mlir: Path, chip: str, flags: List[str]) -> str:
    """Compile the probe and return the stderr carrying the IR dump."""
    cmd = [
        tool, str(probe_mlir), "-o", os.devnull,
        "--iree-input-type=none",
        "--iree-hal-target-backends=torq",
        f"--torq-hw={chip}",
        "--torq-target-host-triple=native",
        "--torq-disable-slicing",
        "--mlir-print-ir-after=torq-tile-and-fuse",
        "--mlir-elide-elementsattrs-if-larger=16",
        *flags,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"probe timed out after {PROBE_TIMEOUT}s: {' '.join(cmd)}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise ProbeError(f"probe compile failed (code {proc.returncode}): {' | '.join(tail)}")
    return proc.stderr.decode("utf-8", "replace")


# -- plateau scan ------------------------------------------------------------

@dataclass
class _ShapeScan:
    k: int
    n: int
    decode_width: Optional[int]
    plateaus: List[Dict] = field(default_factory=list)  # b_start, b_end, width, snapped, m_tiled, m_outer
    points: Dict[int, ProbeResult] = field(default_factory=dict)

    @property
    def constrained(self) -> bool:
        return self.decode_width is not None


def _plateau(b_start: int, b_end: int, sig) -> Dict:
    width, m_tiled, m_outer = sig
    return {
        "b_start": b_start,
        "b_end": b_end,
        "width": width,
        "snapped": None,  # filled in by recommend() when split constants are on
        "m_tiled": m_tiled,
        "m_outer": m_outer,
    }


def _scan_shape(shape, cap: int, probe: Callable[[int], ProbeResult]) -> _ShapeScan:
    """Record every supported batch size, including narrow dips and returns."""
    k, n = shape
    decode = probe(1)
    scan = _ShapeScan(k, n, decode.width)
    scan.points[1] = decode
    if not scan.constrained:
        return scan
    for b in range(4, cap + 1):
        result = scan.points[b] = probe(b)
        if scan.plateaus and result.signature() == (
            scan.plateaus[-1]["width"], scan.plateaus[-1]["m_tiled"], scan.plateaus[-1]["m_outer"]
        ):
            scan.plateaus[-1]["b_end"] = b
        else:
            scan.plateaus.append(_plateau(b, b, result.signature()))
    return scan


# -- reporting ---------------------------------------------------------------

@dataclass
class ProbeReport:
    chosen: List[int]
    shapes: List[_ShapeScan]
    cap: int
    context: Optional[int]
    candidates: List[Dict] = field(default_factory=list)  # b, kept, reason
    warnings: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    def sizes_flag(self) -> Optional[str]:
        if not self.chosen:
            return None
        return "--native-batch-prefill " + ",".join(str(b) for b in self.chosen)

    def log_lines(self) -> List[str]:
        lines = [
            f"prefill probe: chosen sizes {self.chosen} (cap {self.cap}, probe {self.elapsed:.0f}s)"
        ]
        for scan in self.shapes:
            if not scan.constrained:
                lines.append(f"prefill probe: [{scan.k},{scan.n}] untiled at decode; unconstrained")
                continue
            parts = [
                f"B {p['b_start']}-{p['b_end']}: width {p['width']}"
                + (f" (snapped {p['snapped']})" if p.get("snapped") is not None else "")
                + (f", M tiled{' (M-outer)' if p['m_outer'] else ''}" if p["m_tiled"] else "")
                for p in scan.plateaus
            ]
            lines.append(
                f"prefill probe: [{scan.k},{scan.n}] decode {scan.decode_width}; " + "; ".join(parts)
            )
        return lines

    def to_json(self, model: str, chip: str, tool: str, flags: List[str]) -> dict:
        return {
            "model": str(model),
            "chip": chip,
            "tool": str(tool),
            "flags": flags,
            "cap": self.cap,
            "context": self.context,
            "elapsed_s": round(self.elapsed, 1),
            "shapes": [
                {
                    "k": s.k,
                    "n": s.n,
                    "decode_width": s.decode_width,
                    "constrained": s.constrained,
                    "plateaus": s.plateaus,
                }
                for s in self.shapes
            ],
            "candidates": self.candidates,
            "chosen": self.chosen,
            "warnings": self.warnings,
        }


# -- cache -------------------------------------------------------------------

_CACHE_NAME = "prefill_probe_cache.json"
_JSON_NAME = "prefill_sizes.json"


class _ProbeCache:
    """Work-dir probe results keyed by (shape, m, chip, filtered flags, tool).

    The key uses the tool's mtime rather than ``torq-compile --version``
    (which prints a constant "unknown" in dev trees): a rebuilt compiler
    must invalidate the cache. PROBE_CACHE_VERSION invalidates it when the
    probe template or parser changes.
    """

    def __init__(self, work_dir: Path, chip: str, flags: List[str], tool: str):
        self.path = Path(work_dir) / _CACHE_NAME
        tool_path = Path(tool).resolve()
        stat = tool_path.stat()
        self.key_extra = (PROBE_CACHE_VERSION, chip, tuple(flags), str(tool_path), stat.st_mtime_ns)
        self.entries: Dict[str, Dict] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                entries = data.get("entries") if isinstance(data, dict) else None
                self.entries = entries if isinstance(entries, dict) else {}
            except (OSError, json.JSONDecodeError):
                self.entries = {}

    def _key(self, k: int, n: int, m: int) -> str:
        raw = f"{k}x{n}|m={m}|{json.dumps(self.key_extra, default=str)}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def get(self, k: int, n: int, m: int) -> Optional[ProbeResult]:
        e = self.entries.get(self._key(k, n, m))
        if e is None:
            return None
        return ProbeResult(e["m"], e["width"], e["m_tiled"], e["m_outer"])

    def put(self, k: int, n: int, result: ProbeResult) -> None:
        self.entries[self._key(k, n, result.m)] = {
            "m": result.m,
            "width": result.width,
            "m_tiled": result.m_tiled,
            "m_outer": result.m_outer,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"entries": self.entries}, indent=1))


# -- recommendation ------------------------------------------------------------

def _context_length_from_onnx(model) -> Optional[int]:
    """Best-effort context length from [1, heads, context, head_dim] KV
    inputs of the supported decoders. None when not apparent; the
    exporter's explicit context check remains the safety net."""
    contexts = []
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[0].dim_value == 1 and dims[2].dim_value > 0:
            contexts.append(dims[2].dim_value)
    return min(contexts, default=None)


def recommend(
    model_path,
    chip: str,
    compiler_options: List[str],
    work_dir,
    cap: Optional[int] = None,
    context_length: Optional[int] = None,
    tool: Optional[str] = None,
) -> Optional[ProbeReport]:
    """Probe the tiler and recommend sizes.

    Returns a report with a non-empty ``chosen`` list, or None when the
    probe warns and the caller should keep the exporter's explicit sizes
    (no shapes, nothing constrained, every candidate dropped, or any
    probe failure). Writes ``prefill_sizes.json`` to ``work_dir`` whenever
    a sweep ran.
    """
    started = time.perf_counter()

    model_path = Path(model_path)
    work_dir = Path(work_dir)
    selected = _resolve_compiler(tool)
    tool = which(selected)
    if tool is None:
        raise ProbeError(f"compiler executable not found: {selected}")
    flags = filter_probe_flags(compiler_options)
    warnings: List[str] = []

    if cap is None:
        cap = DEFAULT_MAX_BATCH_SIZE
    if type(cap) is not int or not 4 <= cap <= 1 << 20:
        raise ProbeError("max batch size must be an integer in [4, 1048576]")
    if context_length is not None and (type(context_length) is not int or context_length < 4):
        raise ProbeError("context length must be an integer of at least four")

    onnx_model = None
    if model_path.suffix == ".onnx":
        onnx_model = onnx_load(model_path)  # one load for context + shapes
        if context_length is None:
            context_length = _context_length_from_onnx(onnx_model)

    shapes = (
        _weight_shapes_from_onnx(onnx_model) if onnx_model is not None
        else extract_weight_shapes(model_path)
    )
    if not shapes:
        warnings.append(f"no matmul weight shapes found in {model_path.name}; choose explicit sizes")
        _finish(None, warnings, started, cap, context_length, work_dir, model_path, chip, tool, flags, [], None)
        return None

    if context_length is not None:
        cap = min(cap, context_length)

    probe_dir = work_dir / "prefill_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    cache = _ProbeCache(work_dir, chip, flags, tool)
    hexes: Dict[Tuple[int, int], str] = {}
    hexes_lock = threading.Lock()

    def make_probe(k: int, n: int, m: int) -> ProbeResult:
        cached = cache.get(k, n, m)
        if cached is not None:
            return cached
        with hexes_lock:  # hexes is written from pool threads
            if (k, n) not in hexes:
                hexes[(k, n)] = _weight_hex(k, n)
            hex = hexes[(k, n)]
        mlir = probe_dir / f"probe_{k}x{n}_m{m}.mlir"
        mlir.write_text(_PROBE_TEMPLATE.format(m=m, k=k, n=n, hex=hex))
        try:
            stderr = run_probe(tool, mlir, chip, flags)
            result = _parse_tile_and_fuse(stderr, m, k, n)
        finally:
            mlir.unlink(missing_ok=True)  # the cache holds the result
        cache.put(k, n, result)
        return result

    scans: List[_ShapeScan] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROBE_JOBS) as pool:
            decode = list(pool.map(lambda shape: make_probe(*shape, 1), shapes))
            jobs = [(k, n, b) for (k, n), result in zip(shapes, decode)
                    if result.width is not None for b in range(4, cap + 1)]
            list(pool.map(lambda job: make_probe(*job), jobs))
        for shape in shapes:
            scans.append(_scan_shape(shape, cap, lambda b, shape=shape: cache.get(*shape, b)))

        constrained = [s for s in scans if s.constrained]

        split_requested = any(
            o.partition("=")[0] == "--torq-enable-split-constants-optimization"
            and o.partition("=")[2] not in ("false", "0") for o in flags
        )
        for scan in constrained:
            for plateau in scan.plateaus:
                if plateau["width"] is not None:
                    plateau["snapped"] = (
                        snap(plateau["width"], scan.decode_width) if split_requested else plateau["width"]
                    )

        # Candidates: the largest B of each joint plateau = every constrained
        # shape's plateau end, plus the limit (cap bounded by the context when
        # known). Use the sweep results to keep or drop each candidate.
        limit = cap if context_length is None else min(cap, context_length)
        candidate_set = {4, limit} if limit >= 4 else {limit}
        for scan in constrained:
            candidate_set.update(p["b_end"] for p in scan.plateaus if p["width"] is not None and p["b_end"] <= limit)
        # Four is the minimum supported by every native decoder exporter.
        candidates = sorted(b for b in candidate_set if b >= 4)

        sig_at = {(scan.k, scan.n, b): scan.points[b]
                  for scan in constrained for b in candidates}
    except (ProbeError, OSError):
        cache.save()  # keep the successful probes for the next run
        raise

    if not constrained:
        warnings.append(
            "no weight is tiled along N at decode; the sizes are unconstrained, choose explicit sizes"
        )
        _finish(None, warnings, started, cap, context_length, work_dir, model_path, chip, tool, flags, scans, None, cache)
        return None

    chosen: List[int] = []
    report = ProbeReport(chosen, scans, cap, context_length, warnings=warnings)
    for b in candidates:
        reason = "kept"
        for scan in constrained:
            r = sig_at[(scan.k, scan.n, b)]
            if r.width is None:
                reason = f"[{scan.k},{scan.n}] no tile fit (skipped by the tiler)"
                break
            if r.restreams_weight:
                reason = f"[{scan.k},{scan.n}] M-tiled with M-outer loop order (weight re-streamed)"
                break
        report.candidates.append({"b": b, "kept": reason == "kept", "reason": reason})
        if reason == "kept":
            chosen.append(b)

    # Domination: with split constants the effective tile count is the SNAPPED
    # width; a smaller B whose per-shape (snapped width, M tiling, loop order)
    # equals a larger kept B does strictly fewer tokens per weight pass and is
    # dropped.
    def effective_sig(b):
        return tuple(
            (
                snap(sig_at[(scan.k, scan.n, b)].width, scan.decode_width)
                if split_requested and sig_at[(scan.k, scan.n, b)].width is not None
                else sig_at[(scan.k, scan.n, b)].width,
                sig_at[(scan.k, scan.n, b)].m_tiled,
                sig_at[(scan.k, scan.n, b)].m_outer,
            )
            for scan in constrained
        )

    sigs = {b: effective_sig(b) for b in chosen}
    for c in report.candidates:
        b = c["b"]
        if not c["kept"] or b == 4:
            continue
        dominator = next(
            (b2 for b2 in chosen if b2 > b and sigs[b2] == sigs[b]), None
        )
        if dominator is not None:
            c["kept"] = False
            c["reason"] = (
                f"identical effective tile counts as {dominator} after the "
                "split-constants snap (fewer tokens per weight pass)"
            )
            chosen.remove(b)
    if not chosen:
        warnings.append(
            "no probe candidates (cap too small)" if not candidates
            else "every probed candidate was dropped; choose explicit sizes"
        )
        _finish(None, warnings, started, cap, context_length, work_dir, model_path, chip, tool, flags, scans, report, cache)
        return None
    return _finish(chosen, warnings, started, cap, context_length, work_dir, model_path, chip, tool, flags, scans, report, cache)


def _finish(
    chosen: Optional[List[int]],
    warnings: List[str],
    started: float,
    cap: int,
    context: Optional[int],
    work_dir: Path,
    model: Path,
    chip: str,
    tool: str,
    flags: List[str],
    scans: List[_ShapeScan],
    report: Optional[ProbeReport] = None,
    cache: Optional[_ProbeCache] = None,
) -> Optional[ProbeReport]:
    """Log the outcome, persist the JSON, and (re)build the report."""
    if report is None:
        report = ProbeReport(chosen or [], scans, cap, context, warnings=list(warnings),
                             elapsed=time.perf_counter() - started)
    else:
        report.chosen = chosen or []
        report.warnings = list(warnings)
        report.elapsed = time.perf_counter() - started
    for w in warnings:
        logger.warning("prefill probe: %s", w)
    for line in report.log_lines():
        logger.info(line)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / _JSON_NAME).write_text(
            json.dumps(report.to_json(model, chip, tool, flags), indent=1, default=str)
        )
        if cache is not None:
            cache.save()
    except OSError as exc:
        logger.warning("prefill probe: could not write %s: %s", _JSON_NAME, exc)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    # Split the user's torq-compile flags off the probe's own arguments; a
    # trailing bare "--" would otherwise confuse argparse.
    if "--" in argv:
        i = argv.index("--")
        argv, user_flags = argv[:i], argv[i + 1:]
    else:
        user_flags = []
    parser = argparse.ArgumentParser(
        prog="python -m torq.model_export.prefill_sizes",
        description="Probe the matmul tiler and print the recommended "
        "--native-batch-prefill sizes for the model exporter.",
    )
    parser.add_argument("model", help="the .onnx or imported .mlir source")
    parser.add_argument("--chip", default="SL2610")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help=f"probe range cap (default {DEFAULT_MAX_BATCH_SIZE})")
    parser.add_argument("--context-length", type=int, default=None,
                        help="drop sizes above the model context (default: guess from the ONNX)")
    parser.add_argument("--work-dir", default=None,
                        help="where prefill_sizes.json and the probe cache go "
                        "(default: <cwd>/<model-stem>-prefill-probes)")
    parser.add_argument("--tool", default=None, help="path to torq-compile")
    args = parser.parse_args(argv)
    model = Path(args.model)
    if not model.exists():
        print(f"no such model file: {model}", file=sys.stderr)
        return 2
    work_dir = Path(args.work_dir or f"{Path(args.model).stem}-prefill-probes")
    try:
        report = recommend(
            model, args.chip, list(user_flags), work_dir,
            cap=args.max_batch_size, context_length=args.context_length, tool=args.tool,
        )
    except Exception as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        report = None
    if report is None or not report.chosen:
        print("no recommendation; supply explicit sizes after full-model validation", file=sys.stderr)
        return 1
    print(report.sizes_flag())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
