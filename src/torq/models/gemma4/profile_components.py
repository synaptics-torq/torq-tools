# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Profile Gemma4 ONNX performance component-by-component.

The full checkpoint (~9.3GB bf16 / ~19GB fp32) does not fit in memory on this
machine, so instead of exporting/running the merged decoder graph end-to-end
this breaks the architecture into the pieces `Gemma4TextModel.forward` chains
together -- the two embedding tables, one representative decoder layer per
(sliding/full-attention x shared/non-shared-KV) combination, the per-layer
input projection, the final norm, and the LM head -- exports each to its own
tiny ONNX graph with *randomly initialized* weights of the real shape/dtype,
and profiles it individually with onnxruntime.

Random weights are a deliberate simplification, not a shortcut around the
memory limit: ONNX Runtime's per-op latency and a tensor's memory footprint
both depend only on shape/dtype, never on the actual weight values, so this
gives the same performance numbers `torq-export-model gemma4` would produce
against the real checkpoint -- it just can't validate numerical correctness
(that's what `export.py`'s `validate_onnx()` is for, against real weights).

One exception: `embed_tokens_per_layer` (the largest single tensor at ~2.35B
params / ~4.7GB in bf16) is profiled with a *downsized* vocab surrogate,
because materializing its real 262144-row table would itself risk OOMing
this machine. This is safe for the *latency* number specifically because a
Gather op's cost scales with the row width being copied, not the table's
total row count -- but it means the surrogate's own weight-size printout is
meaningless; the reported "real weight" size for that component is computed
analytically from config instead of measured. Every other component is
profiled at true, full scale.

Each component is exported/run in its own subprocess so peak-RSS
measurements reflect that component in isolation, not a running total
across the whole profiling session.

Usage (inside a venv with this dir's requirements.txt installed):
    python -m torq.models.gemma4.profile_components --past-len 128
"""

import argparse
import gc
import multiprocessing
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Final

DEFAULT_HF_REPO: Final[str] = "principled-intelligence/gemma-4-E2B-it-text-only"
DEFAULT_PAST_LEN: Final[int] = 128
DEFAULT_SURROGATE_VOCAB: Final[int] = 4096
DEFAULT_N_WARMUP: Final[int] = 3
DEFAULT_N_ITERS: Final[int] = 20

# (component name, decoder-layer index, is_kv_shared) for the four
# attention-layer archetypes -- see export.py's Gemma4TextKVWrapper docstring
# for why only these combinations exist.
_LAYER_PROBES: Final[tuple[tuple[str, int, bool], ...]] = (
    ("layer_sliding_nonshared", 0, False),
    ("layer_full_nonshared", 4, False),
    ("layer_sliding_shared", 15, True),
    ("layer_full_shared", 19, True),
)

_DTYPE_BYTES: Final[dict[str, int]] = {"fp32": 4, "bf16": 2}


def add_gemma4_profile_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hf-repo", type=str, default=DEFAULT_HF_REPO,
        help="HuggingFace repo to pull just config.json from (no weights downloaded) (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dtype", type=str, choices=["fp32", "bf16"], default="fp32",
        help="dtype for the ONNX graphs (default: %(default)s; bf16 has no onnxruntime CPU MatMul kernel)",
    )
    parser.add_argument(
        "--past-len", type=int, default=DEFAULT_PAST_LEN,
        help="Simulated KV-cache length (decode step context) to profile at (default: %(default)s)",
    )
    parser.add_argument(
        "--surrogate-vocab", type=int, default=DEFAULT_SURROGATE_VOCAB,
        help="Downsized vocab used only for embed_tokens_per_layer (default: %(default)s) -- see module docstring",
    )
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    parser.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op thread count")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Where per-component .onnx files are written (default: a tmp dir under --models-dir)",
    )
    parser.add_argument("--models-dir", type=str, default="models", metavar="DIR")
    parser.add_argument(
        "--components", type=str, nargs="+", default=None, metavar="NAME",
        help="Profile only these components (default: all). Names printed at the end of a full run.",
    )


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class _RssSampler:
    """Background thread sampling this process's own RSS."""

    def __init__(self, interval: float = 0.005):
        self._interval = interval
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return max(self._samples) if self._samples else _rss_bytes()

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(_rss_bytes())
            self._stop.wait(self._interval)


def _build_component(name: str, config, dtype, past_len: int, surrogate_vocab: int):
    """Returns (torch_module, example_args, input_names, output_names, real_weight_bytes)."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextRotaryEmbedding,
        Gemma4TextScaledWordEmbedding,
    )

    from .export import _layer_head_dim, _layer_kv_heads

    dtype_bytes = torch.finfo(dtype).bits // 8
    hidden_states = torch.randn(1, 1, config.hidden_size, dtype=dtype)

    if name == "embed_tokens":
        module = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
        ).to(dtype)
        input_ids = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.long)
        real_bytes = config.vocab_size * config.hidden_size * dtype_bytes
        return module, (input_ids,), ["input_ids"], ["inputs_embeds"], real_bytes

    if name == "embed_tokens_per_layer":
        row_width = config.num_hidden_layers * config.hidden_size_per_layer_input
        module = Gemma4TextScaledWordEmbedding(
            surrogate_vocab, row_width, config.pad_token_id,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        ).to(dtype)
        input_ids = torch.randint(0, surrogate_vocab, (1, 1), dtype=torch.long)
        real_bytes = config.vocab_size_per_layer_input * row_width * dtype_bytes
        return module, (input_ids,), ["input_ids"], ["per_layer_inputs_raw"], real_bytes

    if name == "per_layer_projection":
        class PerLayerProjectionProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                row_width = config.num_hidden_layers * config.hidden_size_per_layer_input
                self.proj = torch.nn.Linear(config.hidden_size, row_width, bias=False)
                self.scale = config.hidden_size**-0.5
                self.norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
                self.n_layers = config.num_hidden_layers
                self.per_layer_size = config.hidden_size_per_layer_input

            def forward(self, inputs_embeds):
                projection = self.proj(inputs_embeds) * self.scale
                projection = projection.reshape(*inputs_embeds.shape[:-1], self.n_layers, self.per_layer_size)
                return self.norm(projection)

        module = PerLayerProjectionProbe().to(dtype)
        real_bytes = config.hidden_size * config.num_hidden_layers * config.hidden_size_per_layer_input * dtype_bytes
        return module, (hidden_states,), ["inputs_embeds"], ["per_layer_projection"], real_bytes

    if name == "final_norm":
        module = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
        real_bytes = config.hidden_size * dtype_bytes
        return module, (hidden_states,), ["hidden_states"], ["normed"], real_bytes

    if name == "lm_head":
        class LMHeadProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.softcap = config.final_logit_softcapping

            def forward(self, hidden_states):
                logits = self.lm_head(hidden_states)
                if self.softcap is not None:
                    logits = torch.tanh(logits / self.softcap) * self.softcap
                return logits

        module = LMHeadProbe().to(dtype)
        real_bytes = config.hidden_size * config.vocab_size * dtype_bytes
        return module, (hidden_states,), ["hidden_states"], ["logits"], real_bytes

    for probe_name, layer_idx, is_shared in _LAYER_PROBES:
        if name != probe_name:
            continue
        layer_type = config.layer_types[layer_idx]
        head_dim = _layer_head_dim(config, layer_type)
        n_kv_heads = _layer_kv_heads(config, layer_type)
        per_layer_input = torch.randn(1, 1, config.hidden_size_per_layer_input, dtype=dtype)
        position_ids = torch.tensor([[past_len]], dtype=torch.long)
        k_in = torch.randn(1, n_kv_heads, past_len, head_dim, dtype=dtype)
        v_in = torch.randn(1, n_kv_heads, past_len, head_dim, dtype=dtype)

        # Single-token decode-step attention needs no explicit mask: a query
        # at position `past_len` may attend to everything at/before it (all
        # of `[0, past_len]`), and DynamicSlidingWindowLayer's own eviction
        # already keeps sliding-window caches within the window -- so
        # `attention_mask=None` is not a shortcut, it is what the real
        # merged-decoder graph reduces to at this shape. See export.py.
        if is_shared:
            kv_shared_layer_index = _kv_shared_layer_index(config, layer_idx)

            class SharedLayerProbe(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.decoder_layer = Gemma4TextDecoderLayer(config, layer_idx)
                    self.rotary = Gemma4TextRotaryEmbedding(config)

                def forward(self, hidden_states, per_layer_input, position_ids, k_shared, v_shared):
                    cache = DynamicCache(config=config)
                    cache.shared_layers = {kv_shared_layer_index: (k_shared, v_shared)}
                    position_embeddings = self.rotary(hidden_states, position_ids, layer_type)
                    return self.decoder_layer(
                        hidden_states, per_layer_input,
                        position_embeddings=position_embeddings,
                        attention_mask=None, position_ids=position_ids,
                        past_key_values=cache,
                    )

            module = SharedLayerProbe().to(dtype)
            real_bytes = sum(p.numel() for p in module.decoder_layer.parameters()) * dtype_bytes
            return (
                module, (hidden_states, per_layer_input, position_ids, k_in, v_in),
                ["hidden_states", "per_layer_input", "position_ids", "k_shared", "v_shared"],
                ["hidden_states_out"], real_bytes,
            )

        class NonSharedLayerProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_layer = Gemma4TextDecoderLayer(config, layer_idx)
                self.rotary = Gemma4TextRotaryEmbedding(config)

            def forward(self, hidden_states, per_layer_input, position_ids, k_in, v_in):
                cache = DynamicCache(config=config)
                if k_in.shape[-2] > 0:
                    cache.update(k_in, v_in, layer_idx)
                position_embeddings = self.rotary(hidden_states, position_ids, layer_type)
                out = self.decoder_layer(
                    hidden_states, per_layer_input,
                    position_embeddings=position_embeddings,
                    attention_mask=None, position_ids=position_ids,
                    past_key_values=cache,
                )
                layer = cache.layers[layer_idx]
                return out, layer.keys, layer.values

        module = NonSharedLayerProbe().to(dtype)
        real_bytes = sum(p.numel() for p in module.decoder_layer.parameters()) * dtype_bytes
        return (
            module, (hidden_states, per_layer_input, position_ids, k_in, v_in),
            ["hidden_states", "per_layer_input", "position_ids", "past_key", "past_value"],
            ["hidden_states_out", "present_key", "present_value"], real_bytes,
        )

    raise ValueError(f"Unknown component '{name}'")


def _kv_shared_layer_index(config, layer_idx: int) -> int:
    """Mirrors `Gemma4TextAttention.__init__`'s shared-layer lookup."""
    n_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    first_shared = config.num_hidden_layers - n_shared
    prev_layers = list(config.layer_types[:first_shared])
    return len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])


def _profile_one_component(name: str, args_dict: dict, conn):
    """Runs in a dedicated subprocess: build -> export -> profile -> report back."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args_dict["hf_repo"])
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args_dict["dtype"]]

    module, example_args, input_names, output_names, real_weight_bytes = _build_component(
        name, config, dtype, args_dict["past_len"], args_dict["surrogate_vocab"],
    )
    module = module.eval()
    weight_params = sum(p.numel() for p in module.parameters())

    out_path = Path(args_dict["out_dir"]) / f"{name}.onnx"
    torch.onnx.export(
        module, example_args, str(out_path), dynamo=True,
        input_names=input_names, output_names=output_names,
    )
    onnx_file_bytes = out_path.stat().st_size

    del module
    gc.collect()

    so = ort.SessionOptions()
    if args_dict["threads"]:
        so.intra_op_num_threads = args_dict["threads"]
    if args_dict["dtype"] == "bf16":
        # No CPU bf16 MatMul kernel in onnxruntime -- report sizing only, skip timing.
        conn.send({
            "name": name, "skipped": "bf16 has no onnxruntime CPU MatMul kernel",
            "weight_params": weight_params, "real_weight_bytes": real_weight_bytes,
            "onnx_file_bytes": onnx_file_bytes,
        })
        return

    session = ort.InferenceSession(str(out_path), sess_options=so, providers=["CPUExecutionProvider"])
    feeds = {inp.name: a.to(torch.float32).numpy() if a.dtype == torch.bfloat16 else a.numpy()
             for inp, a in zip(session.get_inputs(), example_args)}

    for _ in range(args_dict["n_warmup"]):
        session.run(None, feeds)

    sampler = _RssSampler()
    sampler.start()
    latencies = []
    for _ in range(args_dict["n_iters"]):
        t0 = time.perf_counter()
        session.run(None, feeds)
        latencies.append(time.perf_counter() - t0)
    peak_rss = sampler.stop()

    lat_ms = sorted(t * 1e3 for t in latencies)
    conn.send({
        "name": name,
        "weight_params": weight_params,
        "real_weight_bytes": real_weight_bytes,
        "onnx_file_bytes": onnx_file_bytes,
        "peak_rss_bytes": peak_rss,
        "mean_ms": statistics.fmean(lat_ms),
        "median_ms": statistics.median(lat_ms),
        "p90_ms": lat_ms[max(0, min(len(lat_ms) - 1, int(round(0.9 * (len(lat_ms) - 1)))))],
    })


def _run_component(name: str, args_dict: dict, timeout_s: float = 600) -> dict:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_profile_one_component, args=(name, args_dict, child_conn))
    proc.start()

    result = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if parent_conn.poll(1.0):
            result = parent_conn.recv()
            break
        if not proc.is_alive():
            # Child exited (crashed or otherwise) without sending a result.
            break
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
    if result is None:
        return {"name": name, "error": f"subprocess produced no result (exitcode={proc.exitcode})"}
    return result


def _fmt_mb(b: int) -> str:
    return f"{b / (1024 * 1024):.1f} MB"


def profile_gemma4_components(
    hf_repo: str = DEFAULT_HF_REPO,
    dtype: str = "fp32",
    past_len: int = DEFAULT_PAST_LEN,
    surrogate_vocab: int = DEFAULT_SURROGATE_VOCAB,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_iters: int = DEFAULT_N_ITERS,
    threads: int | None = None,
    out_dir: str | os.PathLike | None = None,
    models_dir: str | os.PathLike = "models",
    components: list[str] | None = None,
) -> list[dict]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_repo)
    n_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    n_cache_layers = config.num_hidden_layers - n_shared

    all_names = ["embed_tokens", "embed_tokens_per_layer", "per_layer_projection"]
    all_names += [probe_name for probe_name, _, _ in _LAYER_PROBES]
    all_names += ["final_norm", "lm_head"]
    names = components or all_names

    out_dir = Path(out_dir) if out_dir else Path(models_dir) / hf_repo / "profile" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "hf_repo": hf_repo, "dtype": dtype, "past_len": past_len,
        "surrogate_vocab": surrogate_vocab, "n_warmup": n_warmup, "n_iters": n_iters,
        "threads": threads, "out_dir": str(out_dir),
    }

    results = []
    for name in names:
        print(f"-- profiling '{name}' (isolated subprocess) --")
        result = _run_component(name, args_dict)
        results.append(result)
        if "error" in result:
            print(f"   ERROR: {result['error']}")
        elif "skipped" in result:
            print(f"   skipped timing ({result['skipped']}); real weight {_fmt_mb(result['real_weight_bytes'])}")
        else:
            print(
                f"   mean={result['mean_ms']:.3f}ms  p90={result['p90_ms']:.3f}ms  "
                f"peak_rss={_fmt_mb(result['peak_rss_bytes'])}  "
                f"real_weight={_fmt_mb(result['real_weight_bytes'])}"
            )

    _print_summary(results, config, n_cache_layers, past_len)
    return results


def _print_summary(results: list[dict], config, n_cache_layers: int, past_len: int):
    by_name = {r["name"]: r for r in results if "error" not in r}

    print("\n=== Component summary ===")
    header = f"{'component':<26}{'params':>14}{'real weight':>14}{'peak RSS':>12}{'mean latency':>14}{'p90 latency':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<26}{'ERROR: ' + r['error']}")
            continue
        lat = f"{r['mean_ms']:.3f} ms" if "mean_ms" in r else "n/a"
        p90 = f"{r['p90_ms']:.3f} ms" if "p90_ms" in r else "n/a"
        rss = _fmt_mb(r["peak_rss_bytes"]) if "peak_rss_bytes" in r else "n/a"
        print(
            f"{r['name']:<26}{r['weight_params']:>14,}{_fmt_mb(r['real_weight_bytes']):>14}"
            f"{rss:>12}{lat:>14}{p90:>13}"
        )

    # Whole-model extrapolation: sum real weight bytes across the actual
    # layer-type population (not just the 4 profiled samples), and sum
    # per-decode-step latency the same way. Layers not covered by a probe
    # (e.g. if --components narrowed the run) are skipped with a note.
    n_full_attn = sum(1 for lt in config.layer_types if lt == "full_attention")
    n_sliding_attn = config.num_hidden_layers - n_full_attn
    n_full_nonshared = sum(
        1 for i, lt in enumerate(config.layer_types[:n_cache_layers]) if lt == "full_attention"
    )
    n_sliding_nonshared = n_cache_layers - n_full_nonshared
    n_full_shared = n_full_attn - n_full_nonshared
    n_sliding_shared = n_sliding_attn - n_sliding_nonshared

    layer_counts = {
        "layer_sliding_nonshared": n_sliding_nonshared,
        "layer_full_nonshared": n_full_nonshared,
        "layer_sliding_shared": n_sliding_shared,
        "layer_full_shared": n_full_shared,
    }

    total_weight_bytes = 0
    total_latency_ms = 0.0
    missing = []
    for comp in ("embed_tokens", "embed_tokens_per_layer", "per_layer_projection", "final_norm", "lm_head"):
        r = by_name.get(comp)
        if r is None:
            missing.append(comp)
            continue
        total_weight_bytes += r["real_weight_bytes"]
        total_latency_ms += r.get("mean_ms", 0.0)
    for comp, count in layer_counts.items():
        r = by_name.get(comp)
        if r is None:
            missing.append(comp)
            continue
        total_weight_bytes += r["real_weight_bytes"] * count
        total_latency_ms += r.get("mean_ms", 0.0) * count

    print(f"\n=== Whole-model extrapolation (from components, at past_len={past_len}) ===")
    print(f"  Estimated total weight size: {_fmt_mb(total_weight_bytes)}")
    if total_latency_ms:
        print(f"  Estimated per-token decode latency: {total_latency_ms:.2f} ms  (~{1000.0/total_latency_ms:.2f} tok/s)")
    else:
        print("  Estimated per-token decode latency: n/a (no timed components)")
    if missing:
        print(f"  (extrapolation excludes missing components: {missing})")
    print(
        "  Note: lm_head is tied to embed_tokens in the real checkpoint "
        "(tie_word_embeddings=true) -- this sum double-counts that weight; "
        "subtract one embed_tokens-sized block for an on-disk size estimate."
    )


def main():
    parser = argparse.ArgumentParser(description="Profile Gemma4 ONNX performance component-by-component")
    add_gemma4_profile_args(parser)
    args = parser.parse_args()
    profile_gemma4_components(
        hf_repo=args.hf_repo,
        dtype=args.dtype,
        past_len=args.past_len,
        surrogate_vocab=args.surrogate_vocab,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        threads=args.threads,
        out_dir=args.out_dir,
        models_dir=args.models_dir,
        components=args.components,
    )


if __name__ == "__main__":
    main()
