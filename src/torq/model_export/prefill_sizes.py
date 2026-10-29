# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://www.apache.org/licenses/LICENSE-2.0 WITH LLVM-exception
"""Recommend batch-prefill sizes from representative projection tiling probes.

See ``doc/prefill_sizes.md`` for the probe flow, selection algorithm,
quantization handling, cache, observed families and limitations.
"""

import concurrent.futures
import hashlib
import io
import json
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from shutil import which

from ..utils.compile import _resolve_compiler

logger = logging.getLogger("torq.model_export.prefill_sizes")

# Default sweep cap; explicit sizes are passed to the exporter.
DEFAULT_MAX_BATCH_SIZE = 128

# User flags stripped from isolated projection compiles, or set here.
PROBE_FLAG_DENYLIST = (
    "--iree-input-type",
    "--iree-hal-target-backends",
    "--torq-hw",
    "--torq-target-host-triple",
    "--torq-enable-multi-graph",
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
PROBE_CACHE_VERSION = 3


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
    snap_grid: bool = True  # only direct BF16 weights use the BF16 family grid

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


# -- projection extraction ---------------------------------------------------

@dataclass(frozen=True)
class _Projection:
    k: int
    n: int
    template: bytes

    @cached_property
    def key(self):
        return hashlib.sha256(self.template).hexdigest()

    def write(self, path: Path, m: int):
        from iree.compiler import ir

        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(self.template)
            func = next(iter(module.body.operations))
            block = func.regions[0].blocks[0]
            ret = list(block.operations)[-1]
            # Only the activation and projection result carry the token extent.
            inp = _batch_type(block.arguments[0].type, m)
            out = _batch_type(ret.operands[0].type, m)
            block.arguments[0].set_type(inp)
            ret.operands[0].set_type(out)
            func.attributes['function_type'] = ir.TypeAttr.get(ir.FunctionType.get([inp], [out]))
            module.operation.verify()
            with path.open('wb') as stream:
                module.operation.write_bytecode(stream)


def _batch_type(typ, m):
    from iree.compiler import ir

    match = re.fullmatch(r'!torch.vtensor<\[([^]]+)\],([^>]+)>', str(typ))
    if not match:
        raise ProbeError(f'unsupported projection tensor type: {typ}')
    dims = match[1].split(',')
    if len(dims) < 2 or any(d.strip() not in ('1', '?') for d in dims[:-1]):
        raise ProbeError(f'projection must have one token and singleton batch dimensions: {typ}')
    dims = ['1'] * (len(dims) - 2) + [str(m), dims[-1]]
    return ir.Type.parse(f'!torch.vtensor<[{",".join(dims)}],{match[2]}>')


def onnx_load(path: Path):
    import onnx

    return onnx.load(str(path), load_external_data=False)


def extract_weight_shapes(model_path) -> List[Tuple[int, int]]:
    """Distinct logical projection shapes, including constant dequantization."""
    return list(dict.fromkeys((p.k, p.n) for p in extract_projections(model_path)))


def extract_projections(model_path, model=None, exclude_lm_head=False) -> List[_Projection]:
    """Keep one actual constant-backed projection per graph/type structure."""
    from iree.compiler import ir

    path = Path(model_path)
    with ir.Context(), ir.Location.unknown():
        if path.suffix in ('.mlir', '.mlirbc'):
            return _projections_mlir(ir.Module.parse(path.read_bytes()))
        if path.suffix != '.onnx':
            raise ProbeError(f'need .onnx or imported .mlir/.mlirbc: {path.name}')
        return _projections_onnx(model if model is not None else onnx_load(path), path.parent, exclude_lm_head)


def _projections_onnx(model, data_dir, exclude_lm_head=False):
    import numpy as np
    import onnx
    from iree.compiler import ir
    from iree.compiler.extras import onnx_importer
    from .prefill import _tensor_context_cache

    # Infer metadata before loading one projection's weights at a time.
    model = onnx.shape_inference.infer_shapes(model)
    extractor = onnx.utils.Extractor(model)
    inits = {t.name for t in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    constant = dict.fromkeys(inits, True)

    def is_constant(name):
        if not name:
            return True  # optional ONNX operand
        if name not in constant:
            node = producers.get(name)
            constant[name] = node is not None and (
                node.op_type == 'Constant' or bool(node.input) and all(is_constant(i) for i in node.input)
            )
        return constant[name]

    projections = []
    seen = set()
    for node in model.graph.node:
        if exclude_lm_head and 'logits' in node.output:
            continue  # Native families compile the language model head separately.
        if node.op_type not in ('MatMul', 'Gemm') or not is_constant(node.input[1]):
            continue  # attention matmuls do not have constant weights
        if not all(is_constant(i) for i in node.input[2:]):
            raise ProbeError(f'projection has a runtime bias: {node.name}')
        if node.op_type == 'Gemm' and any(a.name == 'transA' and a.i for a in node.attribute):
            raise ProbeError('transposed Gemm activations are unsupported; choose explicit sizes')
        sub = extractor.extract_model([node.input[0]], [node.output[0]])
        sub.graph.name = 'main'
        onnx.load_external_data_for_model(sub, str(data_dir))
        # Normalize names and large nonsplat payloads for structural deduplication.
        # The chosen representative and its cache identity retain the real values.
        signature = onnx.ModelProto()
        signature.CopyFrom(sub)
        names = {}
        def rename(name):
            return names.setdefault(name, f'v{len(names)}') if name else ''
        for tensor, original in zip(signature.graph.initializer, sub.graph.initializer):
            tensor.name = rename(tensor.name)
            if math.prod(tensor.dims) > 16:
                values = onnx.numpy_helper.to_array(original)
                payload = values.reshape(-1)[:1].tobytes() if np.all(values == values.flat[0]) else b'nonsplat'
                tensor.ClearField('raw_data')
                tensor.ClearField('external_data')
                for field_name in ('float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data'):
                    tensor.ClearField(field_name)
                tensor.ClearField('data_location')
                tensor.raw_data = payload
        for op in signature.graph.node:
            op.name = ''
            op.input[:] = [rename(i) for i in op.input]
            op.output[:] = [rename(o) for o in op.output]
        for value in (*signature.graph.input, *signature.graph.output, *signature.graph.value_info):
            value.name = rename(value.name)
        key = signature.graph.SerializeToString()
        if key in seen:
            continue
        seen.add(key)
        info = onnx_importer.ModelInfo(sub)
        module = info.create_module()
        cache = _tensor_context_cache()(module.context)
        importer = onnx_importer.NodeImporter.define_function(info.main_graph, module.operation, context_cache=cache)
        importer.import_all()
        with module.context, ir.Location.unknown():
            projections.extend(_projections_mlir(module))
    return projections


def _projections_mlir(module):
    from iree.compiler import ir
    from iree.compiler.dialects import func

    projections = []
    seen = set()
    for source in module.body.operations:
        if source.operation.name != 'func.func' or not source.regions[0].blocks:
            continue
        for op in source.regions[0].blocks[0].operations:
            if op.operation.name != 'torch.operator' or str(op.attributes['name']) not in ('"onnx.MatMul"', '"onnx.Gemm"'):
                continue
            chain = []
            visited = set()
            def visit(value):
                if value in visited:
                    return True
                if not isinstance(value, ir.OpResult):
                    return False
                producer = value.owner
                if producer.regions or producer.name == 'util.global.load':
                    return False
                if not all(visit(i) for i in producer.operands):
                    return False
                visited.update(producer.results)
                chain.append(producer)
                return True
            if not visit(op.operands[1]):
                continue
            if not all(visit(v) for v in list(op.operands)[2:]):
                raise ProbeError('projection has a runtime bias; choose explicit sizes')
            if 'torch.onnx.transA' in op.attributes and ir.IntegerAttr(op.attributes['torch.onnx.transA']).value:
                raise ProbeError('transposed Gemm activations are unsupported; choose explicit sizes')
            weight = re.fullmatch(r'!torch.vtensor<\[([^]]+)\],([^>]+)>', str(op.operands[1].type))
            if not weight:
                raise ProbeError(f'unsupported projection weight type: {op.operands[1].type}')
            dims = [int(d) for d in weight[1].split(',')]
            # LFM's depthwise-convolution rewrite is [C,1,K] @ [C,K,1].
            # C indexes independent kernels, not tokens sharing a projection.
            if (str(op.attributes['name']) == '"onnx.MatMul"' and len(dims) == 3
                    and dims[0] > 1 and dims[1] > 0 and dims[2] == 1
                    and re.fullmatch(rf'!torch.vtensor<\[{dims[0]},1,{dims[1]}\],[^>]+>',
                                     str(op.operands[0].type))):
                logger.info('Skipping channel-wise convolution MatMul with weight shape %s', dims)
                continue
            while len(dims) > 2 and dims[0] == 1:
                dims.pop(0)
            if len(dims) != 2 or min(dims) <= 0:
                raise ProbeError(f'unsupported projection weight shape: {dims}')
            k, n = dims
            if 'torch.onnx.transB' in op.attributes and ir.IntegerAttr(op.attributes['torch.onnx.transB']).value:
                k, n = n, k
            # Ignore SSA/resource names when deduplicating imported MLIR too.
            numbers = {op.operands[0]: 0}
            structure = []
            for producer in [*chain, op.operation]:
                for v in producer.results:
                    numbers[v] = len(numbers)
                structure.append((producer.name, tuple(numbers[v] for v in producer.operands),
                                  tuple(str(v.type) for v in producer.results),
                                  tuple((a, re.sub(r'dense_resource<[^>]+>', 'dense_resource<weight>', str(producer.attributes[a])))
                                        for a in producer.attributes)))
            structure = tuple(structure)
            if structure in seen:
                continue
            seen.add(structure)
            result = ir.Module.create()
            attrs = {a: source.attributes[a] for a in source.attributes if a.startswith('torch.onnx_meta.')}
            with ir.InsertionPoint(result.body):
                function = func.FuncOp('main', ([op.operands[0].type], [op.results[0].type]))
                for name, attr in attrs.items():
                    function.attributes[name] = attr
            block = function.add_entry_block()
            values = {op.operands[0]: block.arguments[0]}
            with ir.InsertionPoint(block):
                for producer in [*chain, op.operation]:
                    clone = producer.clone()
                    for index, operand in enumerate(producer.operands):
                        clone.operands[index] = values[operand]
                    values.update(zip(producer.results, clone.results))
                func.ReturnOp([values[op.results[0]]])
            result.operation.verify()
            stream = io.BytesIO()
            result.operation.write_bytecode(stream)
            projections.append(_Projection(k, n, stream.getvalue()))
    return projections


# -- probe compile -----------------------------------------------------------

_IDX_CONST_RE = re.compile(r"%(?P<name>\w+) = arith\.constant (?P<value>\d+) : index")
_FOR_RE = re.compile(
    r"^(?P<indent>\s*)(?:%\w+ = )?scf\.for %(?P<iv>\w+) = %(?P<lb>\w+) to %(?P<ub>\w+) step %(?P<step>\w+)"
)
_WEIGHT_DEF_RE = re.compile(
    r"%(?P<name>\w+) = arith\.constant dense[^:]*: tensor<(?P<dims>(?:\d+x)+)(?P<elt>[a-z0-9]+)>"
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

    Follow constant weight strips, including blocked quantization tensors.
    Unsupported IR raises a probe error.
    """
    sections = [
        s for s in stderr.split("// -----// IR Dump After ")
        if re.match(r"\S+: torq-tile-and-fuse\b", s)
    ]
    if not sections:
        raise ProbeError("no torq-tile-and-fuse dump in the probe output")
    weight = None
    for text in reversed(sections):
        for wd in _WEIGHT_DEF_RE.finditer(text):
            dims = [int(d) for d in wd['dims'].rstrip('x').split('x')]
            if dims[-1] == n and math.prod(dims[:-1]) == k:
                weight = wd['name']
                rank = len(dims)
                snap_grid = wd['elt'] == 'bf16'
                break
        if weight is not None:
            break
    if weight is None:
        raise ProbeError(f'no constant-backed [{k},{n}] projection in the probe output')
    consts = {c.group("name"): int(c.group("value")) for c in _IDX_CONST_RE.finditer(text)}

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

    weight_loop = m_loop = None
    extracts = re.compile(r'tensor\.extract_slice %(\w+)\[([^]]+)\]')
    for loop in loops:
        for name, offsets in extracts.findall(loop['body']):
            offsets = [o.strip() for o in offsets.split(',')]
            iv = '%' + loop['iv']
            if (name == weight and len(offsets) == rank and offsets[-1] == iv
                    and all(o == '0' for o in offsets[:-1])):
                if weight_loop is None or loop['indent'] > weight_loop['indent']:
                    weight_loop = loop
            elif (loop['ub'] == m and len(offsets) >= 2 and offsets[-2:] == [iv, '0']
                  and all(o == '0' for o in offsets[:-2])):
                m_loop = loop
    if weight_loop is None:
        return ProbeResult(m, None, m_loop is not None, False, snap_grid)
    return ProbeResult(
        m, weight_loop['step'], m_loop is not None,
        m_loop is not None and m_loop['indent'] < weight_loop['indent'], snap_grid,
    )


def run_probe(tool: str, probe_mlir: Path, chip: str, flags: List[str]) -> str:
    """Compile the probe and return the stderr carrying the IR dump."""
    cmd = [
        tool, str(probe_mlir), "-o", os.devnull,
        "--iree-input-type=onnx-torq",
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
    snap_grid: bool = True
    projection: str = ""

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
    scan = _ShapeScan(k, n, decode.width, snap_grid=decode.snap_grid)
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
        return "--native-batch-prefill --batch-prefill-sizes " + ",".join(str(b) for b in self.chosen)

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
                    "projection": s.projection,
                    "snap_grid": s.snap_grid,
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
    """Work-dir results keyed by (projection content, m, chip, flags, tool).

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

    def _key(self, projection: str, m: int) -> str:
        raw = f"{projection}|m={m}|{json.dumps(self.key_extra, default=str)}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def get(self, projection: str, m: int) -> Optional[ProbeResult]:
        e = self.entries.get(self._key(projection, m))
        if e is None:
            return None
        return ProbeResult(e["m"], e["width"], e["m_tiled"], e["m_outer"], e["snap_grid"])

    def put(self, projection: str, result: ProbeResult) -> None:
        self.entries[self._key(projection, result.m)] = {
            "m": result.m,
            "width": result.width,
            "m_tiled": result.m_tiled,
            "m_outer": result.m_outer,
            "snap_grid": result.snap_grid,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"entries": self.entries}, indent=1))


# -- recommendation ------------------------------------------------------------

def _context_length_from_onnx(model) -> Optional[int]:
    """Infer capacity from legacy full-cache or time-major read-only KV I/O."""
    outputs = {out.name: out for out in model.graph.output}
    contexts = []
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if not inp.name.startswith("past_key_values.") or len(dims) != 4 or dims[0].dim_value != 1:
            continue
        out = outputs.get(inp.name.replace("past_key_values.", "present."))
        out_dims = out.type.tensor_type.shape.dim if out is not None else []
        read_only = len(out_dims) == 4 and dims[1].dim_value != out_dims[1].dim_value
        capacity = dims[1 if read_only else 2].dim_value
        if capacity > 0:
            contexts.append(capacity)
    return min(contexts, default=None)


def recommend(
    model_path,
    chip: str,
    compiler_options: List[str],
    work_dir,
    cap: Optional[int] = None,
    context_length: Optional[int] = None,
    tool: Optional[str] = None,
    exclude_lm_head: bool = False,
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

    projections = extract_projections(model_path, onnx_model, exclude_lm_head)
    if not projections:
        warnings.append(f"no matmul weight shapes found in {model_path.name}; choose explicit sizes")
        _finish(None, warnings, started, cap, context_length, work_dir, model_path, chip, tool, flags, [], None)
        return None

    if context_length is not None:
        cap = min(cap, context_length)

    probe_dir = work_dir / "prefill_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    cache = _ProbeCache(work_dir, chip, flags, tool)

    def make_probe(projection: _Projection, m: int) -> ProbeResult:
        cached = cache.get(projection.key, m)
        if cached is not None:
            return cached
        mlir = probe_dir / f"probe_{projection.key}_m{m}.mlirbc"
        try:
            projection.write(mlir, m)
            stderr = run_probe(tool, mlir, chip, flags)
            result = _parse_tile_and_fuse(stderr, m, projection.k, projection.n)
        finally:
            mlir.unlink(missing_ok=True)
        cache.put(projection.key, result)
        return result

    scans: List[_ShapeScan] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROBE_JOBS) as pool:
            decode = list(pool.map(lambda p: make_probe(p, 1), projections))
            jobs = [(p, b) for p, result in zip(projections, decode)
                    if result.width is not None for b in range(4, cap + 1)]
            list(pool.map(lambda job: make_probe(*job), jobs))
        for p in projections:
            scan = _scan_shape((p.k, p.n), cap, lambda b, p=p: cache.get(p.key, b))
            scan.projection = p.key
            scans.append(scan)

        constrained = [s for s in scans if s.constrained]

        split_requested = any(
            o.partition("=")[0] == "--torq-enable-split-constants-optimization"
            and o.partition("=")[2] not in ("false", "0") for o in flags
        )
        for scan in constrained:
            for plateau in scan.plateaus:
                if plateau["width"] is not None:
                    plateau["snapped"] = (
                        snap(plateau["width"], scan.decode_width) if split_requested and scan.snap_grid else plateau["width"]
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

        sig_at = {(i, b): scan.points[b]
                  for i, scan in enumerate(constrained) for b in candidates}
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
        for i, scan in enumerate(constrained):
            r = sig_at[(i, b)]
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
                snap(sig_at[(i, b)].width, scan.decode_width)
                if split_requested and scan.snap_grid and sig_at[(i, b)].width is not None
                else sig_at[(i, b)].width,
                sig_at[(i, b)].m_tiled,
                sig_at[(i, b)].m_outer,
            )
            for i, scan in enumerate(constrained)
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
        "--native-batch-prefill --batch-prefill-sizes options for the model exporter.",
    )
    parser.add_argument("model", help="the .onnx or imported .mlir/.mlirbc source")
    parser.add_argument("--chip", help="HW chip identifier, passed to --torq-hw in torq-compile", 
                        default="SL2610", choices=["SL2610", "SR200"])
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
