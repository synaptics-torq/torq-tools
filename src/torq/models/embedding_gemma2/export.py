# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Export EmbeddingGemma-2 (text + image + video) to static Torq graphs.

Source: the fp32 ``onnx-community/embeddinggemma-2-ONNX`` graphs. Produces

    export/onnx/{fp32,bf16}/static/vision_<H>x<W>.onnx   pixel_patches -> image_embeds
    export/onnx/{fp32,bf16}/static/text_body_s<S>.onnx   inputs_embeds + attention_bias -> last_hidden_state
    export/iree/bf16/static/*.vmfb                       (unless --no-compile)
    export/assets/                                       token_embeddings.npy (bf16), tokenizer, manifest

Token-embedding gather, media scatter, mean pooling and L2 normalisation run on the host
(see :mod:`.preprocess`).
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import onnx

from . import preprocess as pp
from ._static import AUDIO_FRAMES, build_static_audio, build_static_text, build_static_vision, op_histogram

logger = logging.getLogger("embedding_gemma2.export")

HF_REPO = "onnx-community/embeddinggemma-2-ONNX"
SOURCE_FILES = (
    "onnx/model.onnx", "onnx/model.onnx_data",
    "onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data",
    "onnx/audio_encoder.onnx", "onnx/audio_encoder.onnx_data",
    "tokenizer.json", "tokenizer_config.json", "config.json",
    "config_sentence_transformers.json", "processor_config.json", "preprocessor_config.json",
)
ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json", "config_sentence_transformers.json")

TORQ_FLAGS: tuple[str, ...] = (
    "--torq-hw=SL2610",
    "--torq-disable-slicing",
    "--torq-enable-transpose-optimization",
    "--torq-convert-dtypes",
    "--torq-enable-annotate-tied-operands",
    "--torq-convert-io-dtype",
    "--torq-enable-split-constants-optimization",
)
# NSS program budget per graph. The budget is reserved in every loaded network's XRAM
# map, so oversized budgets make the vision + text networks fail to load together
# ("failed to acquire hardware"). Measured needs: vision 384x384 31.1 MB, audio 1120
# frames 10.4 MB, text S=128 3.5 MB, text S=512 12.4 MB.
def nss_programs_size(name: str) -> int:
    if name.startswith("vision_"):
        return 32 << 20
    if name.startswith("audio_"):
        return 16 << 20
    seq_len = int(name.rsplit("_s", 1)[1])
    return (8 << 20) if seq_len <= 128 else (16 << 20)


def _parse_grid(s: str) -> tuple[int, int]:
    h, w = (int(x) for x in s.lower().split("x"))
    if h % 48 or w % 48:
        raise argparse.ArgumentTypeError(f"{s}: image sides must be multiples of 48 px")
    return h, w


def add_embedding_gemma2_export_args(parser: argparse.ArgumentParser):
    from ...utils.compile import add_torq_args
    from ...utils.logging import add_logging_args

    parser.add_argument("--models-dir", default="models", metavar="DIR",
                        help="Base directory for source and export models (default: %(default)s)")
    parser.add_argument("--onnx-source-dir", default=None, metavar="DIR",
                        help=f"Directory holding the {HF_REPO} files (skips the download)")
    parser.add_argument("--seq-lens", default="128,512",
                        help="Static text sequence lengths, comma separated, each <= 513 (default: %(default)s)")
    parser.add_argument("--image-sizes", default="384x384", type=lambda s: [_parse_grid(x) for x in s.split(",")],
                        help="Static vision input sizes HxW in px, comma separated (default: %(default)s)")
    parser.add_argument("--no-audio", action="store_true", help="Skip the audio encoder")
    parser.add_argument("--audio-frames", default=f"{AUDIO_FRAMES},{AUDIO_FRAMES // 2},{AUDIO_FRAMES // 4},{AUDIO_FRAMES // 8}",
                        help="Static audio windows in 10 ms mel frames, comma separated; each is a separate "
                             "audio encoder (default: %(default)s = 11.2/5.6/2.8/1.4 s)")
    parser.add_argument("--weights", choices=["fp32", "q4"], default="fp32",
                        help="Source weights: fp32 graphs (bf16 on Torq) or the onnx-community *_q4 graphs "
                             "(int4 weights, bf16 activations; exported under export-q4) (default: %(default)s)")
    parser.add_argument("--no-compile", action="store_true", help="Stop after the bf16 ONNX graphs")
    parser.add_argument("--skip-validate", action="store_true", help="Skip ORT validation against the source")
    add_torq_args(parser)
    add_logging_args(parser)


def _download_source(dst: Path, weights: str = "fp32") -> Path:
    from huggingface_hub import hf_hub_download

    files = list(SOURCE_FILES)
    if weights != "fp32":
        files += [f.replace(".onnx", f"_{weights}.onnx") for f in SOURCE_FILES if f.startswith("onnx/")]
    for f in files:
        if not (dst / f).exists():
            logger.info("downloading %s/%s", HF_REPO, f)
            hf_hub_download(HF_REPO, f, local_dir=dst)
    return dst


class _SourceGraphs:
    """Source graph per component for the static builders. For q4 weights, the builders
    run on an exact float rewrite of the q4 graph (cached under onnx/q4_dequantized/) and
    :meth:`requantize` restores the int4 weights in the built graph."""

    def __init__(self, source: Path, weights: str):
        self.source, self.weights = source, weights
        self.registry: dict[str, dict] = {}

    def path(self, name: str) -> Path:
        if self.weights == "fp32":
            return self.source / "onnx" / f"{name}.onnx"
        from .q4 import dequantize_q4_graph

        cached = self.source / "onnx" / f"{self.weights}_dequantized" / f"{name}.onnx"
        if name not in self.registry or not cached.exists():
            model, self.registry[name] = dequantize_q4_graph(onnx.load(str(_source_graph(self.source, name, self.weights))))
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                onnx.save(model, str(cached), save_as_external_data=True, location=f"{name}.onnx_data")
        return cached

    def requantize(self, name: str, model: onnx.ModelProto) -> onnx.ModelProto:
        if self.weights == "fp32":
            return model
        from .q4 import requantize_matmuls

        model, hit, _ = requantize_matmuls(model, self.registry[name])
        if hit == 0:
            raise RuntimeError(f"{name}: no q4 weight found in the static graph")
        return model


def _save(model: onnx.ModelProto, path: Path):
    # The vmfb entrypoint is named after the graph; torq.runtime invokes "main".
    model.graph.name = "main"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = path.with_suffix(".onnx_data")
    if data.exists():
        data.unlink()
    onnx.save(model, str(path), save_as_external_data=True, location=data.name)


def _to_bf16(src: Path, dst: Path, weights: str = "fp32"):
    """The bf16 graph torq-compile builds. For q4, also writes the MatMulNBits graph that
    the ONNX Runtime backend runs, next to the q4 static graph (``q4_ort``)."""
    from torq.lab.model_tools.dtype_conversion.onnx import convert_model

    dst.parent.mkdir(parents=True, exist_ok=True)
    if weights != "q4":
        convert_model(str(src), str(dst), "bf16", convert_io=True)
        return
    from .q4 import split_int4_matmul_rows, to_matmulnbits

    ort_model, _ = to_matmulnbits(onnx.load(str(src)))
    _save(ort_model, src.parent.parent.parent / "q4_ort" / "static" / src.name)
    model = onnx.load(str(src))
    if split_int4_matmul_rows(model):
        split_src = dst.with_name(dst.stem + "_q4split.onnx")
        _save(model, split_src)
        convert_model(str(split_src), str(dst), "bf16", convert_io=True)
        split_src.unlink()
        split_src.with_suffix(".onnx_data").unlink()
    else:
        convert_model(str(src), str(dst), "bf16", convert_io=True)


def _static_session(path: Path):
    """ORT session for a static graph. Graph optimizations are off: ORT fuses
    DequantizeLinear(INT4) + MatMul into its own int4 kernel, which computes at lower
    precision than the q4 source graphs (MatMulNBits without accuracy_level is fp32)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _source_graph(source: Path, name: str, weights: str) -> Path:
    return source / "onnx" / (f"{name}_{weights}.onnx" if weights != "fp32" else f"{name}.onnx")


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def validate_vision(source: Path, static: Path, size: tuple[int, int], seed: int = 0, weights: str = "fp32") -> float:
    import onnxruntime as ort
    from PIL import Image

    ref = ort.InferenceSession(str(_source_graph(source, "vision_encoder", weights)), providers=["CPUExecutionProvider"])
    st = _static_session(static)
    img = Image.fromarray(np.random.default_rng(seed).integers(0, 255, (size[0] // 4, size[1] // 4, 3), np.uint8))
    v = pp.patchify(img, size)
    r = ref.run(None, {"pixel_values": v.patches[None], "pixel_position_ids": v.positions[None]})[0]
    o = st.run(None, {"pixel_patches": v.patches[None]})[0][0]
    return float(_cos(r, o).min())


def audio_frontend_tables(source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mel filter bank [257, 128] and analysis window [320] of the Gemma4 audio front end."""
    cfg = json.loads((source / "preprocessor_config.json").read_text())
    n_fft = pp.AUDIO_FFT_LENGTH
    n = np.arange(pp.AUDIO_FRAME_LENGTH, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * n / pp.AUDIO_FRAME_LENGTH)).astype(np.float32)  # periodic Hann

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    n_mels = int(cfg.get("feature_size", 128))
    fmin, fmax = float(cfg.get("min_frequency", 0.0)), float(cfg.get("max_frequency", 8000.0))
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fft_freqs = np.linspace(0, pp.AUDIO_SAMPLE_RATE // 2, n_fft // 2 + 1)
    # triangular filters, no normalisation (transformers.audio_utils.mel_filter_bank, norm=None, htk)
    fdiff = np.diff(mel_pts)
    slopes = mel_pts[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    filters = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)
    return filters, window


def validate_audio(source: Path, static: Path, num_frames: int, weights: str = "fp32") -> float:
    import onnxruntime as ort

    ref = ort.InferenceSession(str(_source_graph(source, "audio_encoder", weights)), providers=["CPUExecutionProvider"])
    st = _static_session(static)
    filters, window = audio_frontend_tables(source)
    rng = np.random.default_rng(0)
    t = np.arange(int(3.3 * pp.AUDIO_SAMPLE_RATE)) / pp.AUDIO_SAMPLE_RATE
    clip = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) + 0.05 * rng.standard_normal(len(t)))
    # a clip that fits the window (~90% of it), so the check covers padding as well
    clip = clip[: int(0.9 * num_frames * pp.AUDIO_HOP_LENGTH)]
    mel, mask = pp.audio_features(clip.astype(np.float32), filters, window)
    n = pp.audio_num_tokens(mask)
    r = ref.run(None, {"input_features": mel[None], "input_features_mask": mask[None]})[0]
    padded, _ = pp.audio_features(clip.astype(np.float32), filters, window, num_frames=num_frames)
    o = st.run(None, {"input_features": padded[None]})[0][0][:n]
    return float(_cos(r, o).min())


def validate_text(source: Path, static: Path, seq_len: int, lut: np.ndarray, weights: str = "fp32") -> float:
    from .reference import OrtEmbeddingGemma2

    ref = OrtEmbeddingGemma2(source, vision=False, variant="" if weights == "fp32" else weights)
    st = _static_session(static)
    worst = 1.0
    for text, prompt in (("cats sleeping on a couch", "query"),
                         ("Mars is often referred to as the Red Planet. " * 4, "document")):
        ids = pp.text_ids(ref.tokenizer, text, prompt)[:seq_len]
        emb_ref = ref.run_text_model(ids)[1]
        embeds, mask = pp.build_inputs_embeds(ids, lut, seq_len)
        bias = pp.attention_bias(mask)
        h = st.run(None, {"inputs_embeds": embeds, "attention_bias": bias})[0]
        worst = min(worst, float(pp.pool_and_normalize(h, mask) @ emb_ref))
    return worst


def xram_span_mb(vmfb: Path) -> float | None:
    """Largest per-dispatch XRAM span of a vmfb (what the runtime maps into the NPU IOMMU),
    from `torq-run-module --dump_dispatches`; None if the tool is unavailable."""
    import subprocess
    import tempfile

    tool = shutil.which("torq-run-module")
    build = Path(__import__("os").environ.get("IREE_BUILD_DIR", "")) / "runtime" / "tools" / "torq-run-module"
    if tool is None and build.exists():
        tool = str(build)
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "dump.json"
        subprocess.run([tool, str(vmfb), f"--dump_dispatches={dump}", f"--module={vmfb}"],
                       check=True, capture_output=True)
        data = json.loads(dump.read_text())
    spans = [max(s["address"] + s["size"] for s in d["segments"]) - min(s["address"] for s in d["segments"])
             for d in data["dispatches"] if d["segments"]]
    return round(max(spans) / 1e6, 1) if spans else 0.0


def export_embedding_gemma2(models_dir: str | Path = "models", onnx_source_dir: str | Path | None = None,
                            seq_lens=(128, 512), image_sizes=((384, 384),), compile_vmfb: bool = True,
                            validate: bool = True, compiler_args: list[str] | None = None,
                            audio: bool = True, audio_frames=(AUDIO_FRAMES,),
                            use_binary: bool = False, compiler_path: str | Path | None = None,
                            weights: str = "fp32") -> Path:
    """``weights``: "fp32" builds bf16 graphs from the fp32 source; "q4" builds from the
    onnx-community ``*_q4`` graphs and keeps their 4-bit block-quantized weights as
    DequantizeLinear(INT4) -> MatMul (bf16 activations), exported under ``export-q4``."""
    if weights not in ("fp32", "q4"):
        raise ValueError(f"unknown weights {weights!r}")
    root = Path(models_dir) / "embeddinggemma-2"
    source = Path(onnx_source_dir) if onnx_source_dir else _download_source(root / "source", weights)
    out = root / ("export" if weights == "fp32" else f"export-{weights}")
    fp32_dir = out / "onnx" / ("fp32" if weights == "fp32" else weights) / "static"
    bf16_dir = out / "onnx" / "bf16" / "static"
    iree_dir = out / "iree" / "bf16" / "static"
    assets = out / "assets"
    sources = _SourceGraphs(source, weights)
    assets.mkdir(parents=True, exist_ok=True)
    graphs: list[tuple[str, Path]] = []

    for (h, w) in image_sizes:
        name = f"vision_{h}x{w}"
        model = build_static_vision(sources.path("vision_encoder"), (h // pp.PATCH_SIZE, w // pp.PATCH_SIZE))
        model = sources.requantize("vision_encoder", model)
        logger.info("%s ops: %s", name, op_histogram(model))
        _save(model, fp32_dir / f"{name}.onnx")
        if validate:
            c = validate_vision(source, fp32_dir / f"{name}.onnx", (h, w), weights=weights)
            logger.info("%s: min token cos vs source %.6f", name, c)
            if c < 0.9999:
                raise RuntimeError(f"{name} does not match the source graph (cos {c})")
        _to_bf16(fp32_dir / f"{name}.onnx", bf16_dir / f"{name}.onnx", weights)
        graphs.append(("vision", bf16_dir / f"{name}.onnx"))

    for frames in (audio_frames if audio else ()):
        name = f"audio_{frames}"
        model = build_static_audio(sources.path("audio_encoder"), frames)
        model = sources.requantize("audio_encoder", model)
        logger.info("%s ops: %s", name, op_histogram(model))
        _save(model, fp32_dir / f"{name}.onnx")
        if validate:
            c = validate_audio(source, fp32_dir / f"{name}.onnx", frames, weights=weights)
            logger.info("%s: min token cos vs source %.6f", name, c)
            if c < 0.9999:
                raise RuntimeError(f"{name} does not match the source graph (cos {c})")
        _to_bf16(fp32_dir / f"{name}.onnx", bf16_dir / f"{name}.onnx", weights)
        graphs.append(("audio", bf16_dir / f"{name}.onnx"))
    if audio:
        filters, window = audio_frontend_tables(source)
        np.save(assets / "mel_filters.npy", filters)
        np.save(assets / "mel_window.npy", window)

    lut = None
    for s in seq_lens:
        name = f"text_body_s{s}"
        model, lut = build_static_text(sources.path("model"), s)
        model = sources.requantize("model", model)
        logger.info("%s ops: %s", name, op_histogram(model))
        _save(model, fp32_dir / f"{name}.onnx")
        if validate:
            c = validate_text(source, fp32_dir / f"{name}.onnx", s, lut, weights=weights)
            logger.info("%s: min sentence cos vs source %.7f", name, c)
            if c < 0.9999:
                raise RuntimeError(f"{name} does not match the source graph (cos {c})")
        _to_bf16(fp32_dir / f"{name}.onnx", bf16_dir / f"{name}.onnx", weights)
        graphs.append(("text", bf16_dir / f"{name}.onnx"))

    if lut is not None:
        import ml_dtypes

        np.save(assets / "token_embeddings.npy", lut.astype(ml_dtypes.bfloat16))
    for f in ASSET_FILES:
        shutil.copy2(source / f, assets / f)
    manifest = {
        "model": "google/embeddinggemma-2",
        "source": HF_REPO,
        "weights": weights if weights == "fp32" else f"{weights} (int4 block-32 weights, bf16 activations)",
        "seq_lens": list(seq_lens),
        "image_sizes": [list(x) for x in image_sizes],
        "patch_size": pp.PATCH_SIZE,
        "pooling_kernel": pp.POOL_KERNEL,
        "embedding_dim": 768,
        "mrl_dims": list(pp.MRL_DIMS),
        "text_inputs": ["inputs_embeds", "attention_bias"],
        "text_output": "last_hidden_state",
        "vision_input": "pixel_patches",
        "vision_output": "image_embeds",
        "mask_bias": pp.MASK_BIAS,
        "token_ids": {"bos": pp.BOS_ID, "eos": pp.EOS_ID, "pad": pp.PAD_ID, "boi": pp.BOI_ID,
                      "eoi": pp.EOI_ID, "image": pp.IMAGE_TOKEN_ID, "video": pp.VIDEO_TOKEN_ID,
                      "boa": pp.BOA_ID, "eoa": pp.EOA_ID, "audio": pp.AUDIO_TOKEN_ID},
        "video": {"fps": 1, "max_frames": 7},
        "audio": {"num_frames": max(audio_frames), "windows": sorted(audio_frames),
                  "max_tokens": max(audio_frames) // 4, "sample_rate": pp.AUDIO_SAMPLE_RATE} if audio else None,
        "prompts": pp.PROMPTS,
    }
    (assets / "embeddinggemma2_manifest.json").write_text(json.dumps(manifest, indent=2))

    if compile_vmfb:
        from ...utils.compile import export_torq

        for _, path in graphs:
            args = list(TORQ_FLAGS) + [f"--torq-max-nss-programs-size={nss_programs_size(path.stem)}"]
            args += list(compiler_args or [])
            logger.info("compiling %s", path.name)
            export_torq(path, iree_dir, compiler_args=args, use_binary=use_binary, compiler_path=compiler_path)
        spans = {}
        for _, path in graphs:
            span = xram_span_mb(iree_dir / f"{path.stem}.vmfb")
            if span is not None:
                spans[path.stem] = span
        if spans:
            # the runner keeps the resident networks within the board's NPU mapping budget
            manifest["xram_span_mb"] = spans
            (assets / "embeddinggemma2_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def export_embedding_gemma2_from_args(args: argparse.Namespace):
    from ...utils.logging import configure_logging

    configure_logging(args.logging)
    export_embedding_gemma2(
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        seq_lens=tuple(int(s) for s in args.seq_lens.split(",")),
        image_sizes=tuple(args.image_sizes),
        compile_vmfb=not args.no_compile,
        validate=not args.skip_validate,
        compiler_args=args.compile_flags,
        use_binary=args.use_binary,
        compiler_path=args.compiler_path,
        audio=not args.no_audio,
        audio_frames=tuple(int(f) for f in args.audio_frames.split(",")),
        weights=args.weights,
    )
