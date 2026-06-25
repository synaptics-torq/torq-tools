#!/usr/bin/env python3
"""Replace `_upsample_nearest1d*` Gather nodes with a structural upsample.

Safe version of replace_upsample_nearest1d.py — only replaces Gather nodes
whose indices are STATIC initializers that match a perfect repeat pattern.
Skips Gather nodes with dynamically-computed indices (e.g. the x20 upsample
whose indices depend on audio length via Floor/Div/Min).

The original script incorrectly replaced all matching Gather nodes, including
the x20 upsample with data-dependent indices, producing wrong outputs
(max_abs=1.52 on x20_context).

For each matching Gather (axis=2, data rank 3, output rank 3, scale >= 2, tail
in {0, 1}, STATIC indices matching repeat pattern), rewrite as:

    data [N, C, L_in]
      -> Reshape [N, C, L_in, 1]
      -> Expand  [N, C, L_in, scale]
      -> Reshape [N, C, L_in * scale]
      -> (if tail==1) Concat(axis=2, [_, Slice(data, axis=2, start=L_in-1)])
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CORR_THRESHOLD = 0.9999
SAMPLES_JSON = "tmp_conv_verify/tsuki_original_samples.json"
DEFAULT_PART_B_DIR = "tests/testdata/onnx_models/tsuki_static_new_fp32_split_stft_final_s50_4s"


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def const_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def static_shapes(graph):
    out = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                ok = False
                break
        if ok:
            out[vi.name] = dims
    for init in graph.initializer:
        out.setdefault(init.name, list(init.dims))
    return out


def elem_types(graph):
    out = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        out.setdefault(init.name, init.data_type)
    return out


def find_targets(graph, shapes):
    init_names = {i.name for i in graph.initializer}
    init_data = {}
    for i in graph.initializer:
        init_data[i.name] = i

    targets = []
    for node in graph.node:
        if node.op_type != "Gather":
            continue
        if "upsample_nearest1d" not in (node.name or "").lower():
            continue
        if get_attr_int(node, "axis", 0) != 2:
            continue
        d = shapes.get(node.input[0])
        o = shapes.get(node.output[0])
        if not d or not o or len(d) != 3 or len(o) != 3:
            continue
        if d[0] != o[0] or d[1] != o[1] or d[0] != 1:
            continue
        L_in, L_out = d[2], o[2]
        if L_in <= 0 or L_out <= L_in:
            continue
        scale = L_out // L_in
        tail = L_out - scale * L_in
        if scale < 2 or tail not in (0, 1):
            continue

        idx_name = node.input[1]
        if idx_name not in init_names:
            print(f"  SKIP {node.name}: indices '{idx_name}' are dynamically computed")
            continue

        idx_array = numpy_helper.to_array(init_data[idx_name]).flatten()
        expected = np.repeat(np.arange(L_in, dtype=idx_array.dtype), scale)
        if tail:
            expected = np.concatenate([expected, [L_in - 1]])
        if not np.array_equal(idx_array, expected):
            print(f"  SKIP {node.name}: static indices don't match repeat pattern")
            continue

        targets.append({
            "node": node,
            "name": node.name,
            "data": node.input[0],
            "out": node.output[0],
            "N": d[0], "C": d[1], "L_in": L_in, "L_out": L_out,
            "scale": scale, "tail": tail,
        })
    return targets


def make_replacement(target, elem_type, used_names):
    """Emit (nodes, initializers, value_infos) for one Gather rewrite."""
    data = target["data"]
    out = target["out"]
    N, C, L_in = target["N"], target["C"], target["L_in"]
    scale, tail = target["scale"], target["tail"]
    L_main = L_in * scale
    prefix = out

    def uniq(base):
        name = base
        n = 0
        while name in used_names:
            n += 1
            name = f"{base}__{n}"
        used_names.add(name)
        return name

    inits = []
    nodes = []
    vis = []

    # Reshape data → [N, C, L_in, 1]
    shape_4d = uniq(f"{prefix}__shape_4d")
    inits.append(const_i64(shape_4d, [N, C, L_in, 1]))
    data_4d = uniq(f"{prefix}__data_4d")
    nodes.append(helper.make_node("Reshape", [data, shape_4d], [data_4d],
                                  name=uniq(f"{prefix}__reshape_in")))
    vis.append(helper.make_tensor_value_info(data_4d, elem_type, [N, C, L_in, 1]))

    # Expand → [N, C, L_in, scale]
    expand_to = uniq(f"{prefix}__expand_to")
    inits.append(const_i64(expand_to, [N, C, L_in, scale]))
    expanded = uniq(f"{prefix}__expanded")
    nodes.append(helper.make_node("Expand", [data_4d, expand_to], [expanded],
                                  name=uniq(f"{prefix}__expand")))
    vis.append(helper.make_tensor_value_info(expanded, elem_type, [N, C, L_in, scale]))

    # Reshape → [N, C, L_main]
    shape_flat = uniq(f"{prefix}__shape_flat")
    inits.append(const_i64(shape_flat, [N, C, L_main]))

    if tail == 0:
        # Write directly to the original output name; no Slice/Concat needed.
        nodes.append(helper.make_node("Reshape", [expanded, shape_flat], [out],
                                      name=uniq(f"{prefix}__reshape_out")))
    else:
        flat = uniq(f"{prefix}__flat")
        nodes.append(helper.make_node("Reshape", [expanded, shape_flat], [flat],
                                      name=uniq(f"{prefix}__reshape_flat")))
        vis.append(helper.make_tensor_value_info(flat, elem_type, [N, C, L_main]))

        # Slice last element of `data` along axis 2: [N, C, L_in-1:L_in]
        starts = uniq(f"{prefix}__tail_starts")
        ends   = uniq(f"{prefix}__tail_ends")
        axes   = uniq(f"{prefix}__tail_axes")
        steps  = uniq(f"{prefix}__tail_steps")
        inits.extend([
            const_i64(starts, [L_in - 1]),
            const_i64(ends,   [L_in]),
            const_i64(axes,   [2]),
            const_i64(steps,  [1]),
        ])
        tail_slice = uniq(f"{prefix}__tail_slice")
        nodes.append(helper.make_node("Slice",
                                       [data, starts, ends, axes, steps],
                                       [tail_slice],
                                       name=uniq(f"{prefix}__tail_slice_op")))
        vis.append(helper.make_tensor_value_info(tail_slice, elem_type, [N, C, 1]))

        nodes.append(helper.make_node("Concat", [flat, tail_slice], [out],
                                      name=uniq(f"{prefix}__concat_tail"),
                                      axis=2))

    return nodes, inits, vis


def prune_unused(graph):
    """Reachability pruning from graph outputs (keeps graph inputs too)."""
    producer = {o: n for n in graph.node for o in n.output if o}
    keep_tensors = {o.name for o in graph.output} | {i.name for i in graph.input}
    keep_nodes = set()
    work = list(keep_tensors)
    while work:
        t = work.pop()
        p = producer.get(t)
        if p is None:
            continue
        nid = id(p)
        if nid in keep_nodes:
            continue
        keep_nodes.add(nid)
        for inp in p.input:
            if inp and inp not in keep_tensors:
                keep_tensors.add(inp)
                work.append(inp)
        for outp in p.output:
            if outp and outp not in keep_tensors:
                keep_tensors.add(outp)

    kn = [n for n in graph.node if id(n) in keep_nodes]
    dropped_n = len(graph.node) - len(kn)
    del graph.node[:]
    graph.node.extend(kn)

    ki = [init for init in graph.initializer if init.name in keep_tensors]
    dropped_i = len(graph.initializer) - len(ki)
    del graph.initializer[:]
    graph.initializer.extend(ki)

    kv = [vi for vi in graph.value_info if vi.name in keep_tensors]
    dropped_v = len(graph.value_info) - len(kv)
    del graph.value_info[:]
    graph.value_info.extend(kv)

    return dropped_n, dropped_i, dropped_v


def rewrite_model(model):
    graph = model.graph
    shapes = static_shapes(graph)
    types = elem_types(graph)

    targets = find_targets(graph, shapes)
    if not targets:
        return [], 0, 0, 0

    by_id = {id(t["node"]): t for t in targets}
    used = set(shapes.keys()) | {n.name for n in graph.node if n.name}
    # Reserve initializer / value_info names too
    used.update(init.name for init in graph.initializer)

    new_nodes = []
    new_inits = []
    new_vis = []

    for node in graph.node:
        t = by_id.get(id(node))
        if t is None:
            new_nodes.append(node)
            continue
        et = types.get(t["out"], TensorProto.BFLOAT16)
        nodes, inits, vis = make_replacement(t, et, used)
        new_nodes.extend(nodes)
        new_inits.extend(inits)
        new_vis.extend(vis)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    graph.value_info.extend(new_vis)

    dropped = prune_unused(graph)
    return targets, *dropped


# ---------------------------------------------------------------------------
# --verify-audio harness
# ---------------------------------------------------------------------------

def _load_wav_float(path):
    with wave.open(str(path)) as w:
        sw = w.getsampwidth()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=dtype)
    return raw.astype(np.float64) / float(2 ** (8 * sw - 1))


def _run_inference(part_a, part_b, stft_kernels, text, output_wav):
    cmd = [
        sys.executable, "inference_fixed.py", text,
        "--part-a", str(part_a),
        "--part-b", str(part_b),
        "--stft-kernels", str(stft_kernels),
        "--output", str(output_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def verify_audio(input_path, output_path, *, part_b, stft_kernels,
                 samples_json, save_dir=None):
    samples = json.loads(Path(samples_json).read_text())
    rows = []
    work_root = Path(save_dir) if save_dir else None
    if work_root:
        work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for s in samples:
            sid = int(s.get("id"))
            text = s.get("A") or s.get("text")
            if not text:
                continue
            try:
                before_wav = td / f"id{sid:02d}_before.wav"
                after_wav = td / f"id{sid:02d}_after.wav"
                _run_inference(input_path, part_b, stft_kernels, text, before_wav)
                _run_inference(output_path, part_b, stft_kernels, text, after_wav)
            except subprocess.CalledProcessError as exc:
                rows.append({"id": sid, "text": text, "status": f"inference failed ({exc.returncode})"})
                continue

            a = _load_wav_float(before_wav)
            b = _load_wav_float(after_wav)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            d = np.abs(a - b)
            a0, b0 = a - a.mean(), b - b.mean()
            corr_num = (a0 * b0).sum()
            corr_den = float(np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()))
            corr = float(corr_num / corr_den) if corr_den > 0 else float("nan")
            rows.append({
                "id": sid, "text": text, "frames": n,
                "mean_abs": float(d.mean()),
                "max_abs": float(d.max()),
                "corr": corr,
                "status": "ok",
            })
            if work_root:
                import shutil
                shutil.copy(after_wav, work_root / f"id{sid:02d}_after.wav")
                shutil.copy(before_wav, work_root / f"id{sid:02d}_before.wav")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-audio", action="store_true",
                        help=f"Run end-to-end audio A/B against the un-rewritten input. "
                             f"Pass criterion: corr ≥ {CORR_THRESHOLD} on every sample.")
    parser.add_argument("--part-b", default=f"{DEFAULT_PART_B_DIR}/part_b_post_stft_4s.onnx")
    parser.add_argument("--stft-kernels", default=f"{DEFAULT_PART_B_DIR}/stft_kernels.npz")
    parser.add_argument("--samples-json", default=SAMPLES_JSON)
    parser.add_argument("--save-audio-dir", type=Path, default=None,
                        help="If set, copy per-sample before/after WAVs here.")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    targets, dropped_n, dropped_i, dropped_v = rewrite_model(model)

    print(f"Upsample rewrites: {len(targets)}")
    for t in targets:
        print(f"  {t['name']:40s}  L_in={t['L_in']:5d} -> L_out={t['L_out']:5d}  "
              f"scale={t['scale']} tail={t['tail']}")
    print(f"Pruned by reachability: nodes={dropped_n} initializers={dropped_i} value_info={dropped_v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))

    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker: OK")
    if args.shape_infer:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        print("Strict shape inference: OK")

    print(f"Wrote: {args.output}")

    if args.verify_audio:
        print()
        print(f"--- --verify-audio (corr threshold: {CORR_THRESHOLD}) ---")
        rows = verify_audio(args.input, args.output,
                            part_b=args.part_b, stft_kernels=args.stft_kernels,
                            samples_json=args.samples_json,
                            save_dir=args.save_audio_dir)
        all_ok = True
        print(f'{"id":>3s} {"frames":>7s} {"mean_abs":>11s} {"max_abs":>11s} {"corr":>10s}  status')
        for r in rows:
            if r.get("status") != "ok":
                print(f'{r["id"]:>3d} {"-":>7s} {"-":>11s} {"-":>11s} {"-":>10s}  {r["status"]}')
                all_ok = False
                continue
            corr = r["corr"]
            ok = corr >= CORR_THRESHOLD
            if not ok:
                all_ok = False
            mark = "OK " if ok else "BAD"
            print(f'{r["id"]:>3d} {r["frames"]:>7d} {r["mean_abs"]:>11.6f} '
                  f'{r["max_abs"]:>11.6f} {corr:>10.6f}  {mark}')
        if not all_ok:
            print("VERIFICATION FAILED")
            return 2
        print("VERIFICATION PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
