"""
Export UsefulSensors/moonshine-streaming-tiny (safetensors) to ONNX + INT8.

Produces in ./moonshine_streaming_tiny/:
  preprocessor.onnx                 — audio → CNN embedder features (fp32)
  encoder_model.onnx                — embedder features → encoder hidden states (fp32)
  decoder_model.onnx                — first decode step, no KV cache (fp32)
  decoder_with_past_model.onnx      — subsequent decode steps (fp32)
  decoder_token_embeddings.npy      — vocab embedding matrix for external logit computation
  preprocessor_int8.onnx            — INT8 dynamic-quantized preprocessor
  encoder_model_int8.onnx           — INT8 dynamic-quantized encoder
  decoder_model_int8.onnx           — INT8 dynamic-quantized decoder
  decoder_with_past_model_int8.onnx — INT8 dynamic-quantized decoder_with_past

Usage:
    pip install "transformers>=5.2.0" "huggingface_hub>=0.23" torch onnx onnxruntime
    python export_moonshine_streaming_tiny.py

The decoders output last_hidden_state (not logits). Logits are computed
externally: logits = last_hidden_state @ decoder_token_embeddings.T

INT8 strategy: dynamic quantization (weight-only).
  - No calibration dataset needed.
  - Weights stored as int8; activations remain float32 at runtime.
  - ~4× smaller model files; typically 1.5-2× faster on CPU.

For full static INT8 (weights + activations), enable STATIC_QUANT=True,
provide calibration audio files in CALIBRATION_AUDIO_DIR, and:
    pip install soundfile
"""

import shutil
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoProcessor,
    MoonshineStreamingForConditionalGeneration,
)
from types import SimpleNamespace
from transformers.cache_utils import EncoderDecoderCache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

MODEL_ID = "UsefulSensors/moonshine-streaming-tiny"
# Weights cached locally
MODEL_LOCAL_DIR = Path("moonshine_streaming_tiny/weights")
# Export output mirrors non-streaming layout:
#   models/UsefulSensors/<model>/export/onnx/float/{static,dynamic}/
OUTPUT_DIR = Path("models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic")
OPSET = 18
DEVICE = "cpu"

# ── Quantization settings ────────────────────────────────────────────────────
# Dynamic (weight-only) INT8: no calibration needed, always safe.
DYNAMIC_QUANT = False

# Static INT8: requires calibration audio. Set to True and point
# CALIBRATION_AUDIO_DIR at a directory of 16-kHz WAV files.
STATIC_QUANT = False
CALIBRATION_AUDIO_DIR = Path("calibration_audio")   # 10-100 files recommended

# Tiny model dims (from config.json)
NUM_DECODER_LAYERS = 6
DECODER_HEADS = 8
HEAD_DIM = 40        # 320 hidden / 8 heads
ENC_HIDDEN = 320     # raw encoder output dim (decoder adapter projects this to 320 internally)


# ── Download helpers ─────────────────────────────────────────────────────────

def download_model(model_id: str, local_dir: Path) -> Path:
    """
    Download all model files from HuggingFace Hub into `local_dir`
    (not ~/.cache/huggingface). Returns the local directory path.
    """
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} → {local_dir.resolve()} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        # Don't use the shared HF cache; files go directly into local_dir
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"  Download complete.")
    return local_dir


# ── Wrapper modules ──────────────────────────────────────────────────────────

class FullEncoderWrapper(torch.nn.Module):
    """Wraps full encoder (embedder + transformer). Used for validation reference."""

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        encoder_outputs = self.encoder(
            input_values, attention_mask=attention_mask, return_dict=True,
        )
        return encoder_outputs.last_hidden_state  # (B, T_enc, hidden)


class PreprocessorWrapper(torch.nn.Module):
    """CNN embedder: raw audio + attention_mask → feature sequence + frame-level mask.

    Wraps model.model.encoder.embedder which contains:
      CMVN → Asinh compression → Linear+SiLU → CausalConv1d(stride=2) → CausalConv1d(stride=2)
    Output shapes: (B, seq_len, hidden_size) and (B, seq_len) where seq_len ≈ audio_len / 320.
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.embedder = model.model.encoder.embedder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor):
        hidden_states, padding_mask = self.embedder(input_values, padding_mask=attention_mask)
        return hidden_states, padding_mask  # (B, seq_len, hidden), (B, seq_len)


class TransformerEncoderWrapper(torch.nn.Module):
    """Transformer encoder layers + final norm with sliding-window attention.

    Input: embedder features (B, seq_len, hidden_size) + frame-level mask (B, seq_len)
    Output: encoder hidden states (B, seq_len, hidden_size)

    Replicates the per-layer sliding-window mask computation from
    MoonshineStreamingEncoder.forward().
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.layers = model.model.encoder.layers
        self.final_norm = model.model.encoder.final_norm
        self.config = model.model.encoder.config

    def forward(self, input_features: torch.FloatTensor, attention_mask: torch.Tensor) -> torch.FloatTensor:
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )

        hidden_states = input_features

        # Build per-layer sliding-window masks (same as MoonshineStreamingEncoder.forward)
        for layer_idx, encoder_layer in enumerate(self.layers):
            layer_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                and_mask_function=sliding_window_mask_function(
                    self.config.sliding_windows[layer_idx]
                ),
            )
            layer_out = encoder_layer(hidden_states, attention_mask=layer_mask)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self.final_norm(hidden_states)
        return hidden_states  # (B, seq_len, hidden_size)


class DecoderWrapper(torch.nn.Module):
    """
    First decode step — no past KV cache.
    Returns last_hidden_state + flat past KV tensors (self + cross).
    Logits are computed externally via decoder_token_embeddings.npy.
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.base_model = model.model  # MoonshineStreamingModel (no proj_out/lm_head)
        self.n_layers = NUM_DECODER_LAYERS

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,       # (B, T_dec)
        encoder_hidden_states: torch.FloatTensor,  # (B, T_enc, hidden)
    ):
        # Run the base model (encoder+decoder) without lm_head
        out = self.base_model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=None,
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden_state = out.last_hidden_state  # (B, T_dec, hidden)
        pkv: EncoderDecoderCache = out.past_key_values

        # Flatten into individual tensors for ONNX
        self_cache = pkv.self_attention_cache
        cross_cache = pkv.cross_attention_cache

        flat = [hidden_state]
        for layer in self_cache.layers:
            flat.append(layer.keys)    # (B, heads, T_dec, head_dim)
            flat.append(layer.values)
        for layer in cross_cache.layers:
            flat.append(layer.keys)    # (B, heads, T_enc, head_dim)
            flat.append(layer.values)

        return tuple(flat)


def _layer_from_kv(k, v):
    """Create a DynamicLayer pre-filled with key/value tensors."""
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


class DecoderWithPastWrapper(torch.nn.Module):
    """
    Subsequent decode steps — accepts and returns flat KV tensors.
    Returns last_hidden_state (not logits). Logits computed externally.
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.base_model = model.model  # MoonshineStreamingModel (no proj_out/lm_head)
        self.n_layers = NUM_DECODER_LAYERS

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,        # (B, 1)
        encoder_hidden_states: torch.FloatTensor,   # (B, T_enc, hidden)
        # past self-attention KV — n_layers × key + n_layers × value (interleaved)
        *flat_past,
    ):
        n = self.n_layers
        # Reconstruct EncoderDecoderCache (DynamicCache v5: layers[i].keys/.values)
        self_cache = DynamicCache()
        cross_cache = DynamicCache()

        for i in range(n):
            self_cache.layers.append(_layer_from_kv(flat_past[2 * i], flat_past[2 * i + 1]))
        for i in range(n):
            cross_cache.layers.append(_layer_from_kv(flat_past[2 * n + 2 * i], flat_past[2 * n + 2 * i + 1]))

        pkv = EncoderDecoderCache(self_cache, cross_cache)

        out = self.base_model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=None,
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        hidden_state = out.last_hidden_state  # (B, 1, hidden)
        new_pkv: EncoderDecoderCache = out.past_key_values

        new_self = new_pkv.self_attention_cache
        new_cross = new_pkv.cross_attention_cache

        flat_out = [hidden_state]
        for layer in new_self.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)
        for layer in new_cross.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)

        return tuple(flat_out)


# ── Dynamic axes helpers ─────────────────────────────────────────────────────

def _kv_self_axes(n_layers, prefix):
    """Dynamic axes for n_layers × (key, value) self-attention tensors."""
    axes = {}
    for i in range(n_layers):
        axes[f"{prefix}_self_key_{i}"]   = {0: "batch", 2: "past_seq"}
        axes[f"{prefix}_self_value_{i}"] = {0: "batch", 2: "past_seq"}
    return axes


def _kv_cross_axes(n_layers, prefix):
    axes = {}
    for i in range(n_layers):
        axes[f"{prefix}_cross_key_{i}"]   = {0: "batch", 2: "enc_seq"}
        axes[f"{prefix}_cross_value_{i}"] = {0: "batch", 2: "enc_seq"}
    return axes


def _kv_output_names(n_layers, self_prefix="present", cross_prefix="present"):
    names = []
    for i in range(n_layers):
        names.append(f"{self_prefix}_self_key_{i}")
        names.append(f"{self_prefix}_self_value_{i}")
    for i in range(n_layers):
        names.append(f"{cross_prefix}_cross_key_{i}")
        names.append(f"{cross_prefix}_cross_value_{i}")
    return names


def _kv_input_names(n_layers):
    names = []
    for i in range(n_layers):
        names.append(f"past_self_key_{i}")
        names.append(f"past_self_value_{i}")
    for i in range(n_layers):
        names.append(f"past_cross_key_{i}")
        names.append(f"past_cross_value_{i}")
    return names


# ── DynamicLayer patch ───────────────────────────────────────────────────────
# DynamicLayer.lazy_initialization creates torch.tensor([]) (1D) then update()
# does cat([1D_empty, 4D_key_states]) which confuses both TorchScript and dynamo.
# Fix: on first call (not initialized), skip the cat and assign directly.

def _patched_dynamic_update(self, key_states, value_states, cache_kwargs=None):
    if not self.is_initialized:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True
        return self.keys, self.values
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values

DynamicLayer.update = _patched_dynamic_update


# ── Export functions ─────────────────────────────────────────────────────────
# We use the dynamo ONNX exporter (dynamo=True) for all models. Unlike the
# TorchScript tracer, dynamo properly handles dynamic shapes without baking
# Python .shape[i] values as constants.

from torch.onnx._internal.torchscript_exporter import registration, symbolic_helper

# asinh symbolic only needed for TorchScript encoder export path
def _asinh_symbolic(g, input):
    return g.op("Asinh", input)
torch.onnx.register_custom_op_symbolic("aten::asinh", _asinh_symbolic, opset_version=18)


def _consolidate_onnx(output_path: Path):
    """Embed external tensor data into the .onnx file and remove .data files.

    The dynamo exporter sometimes creates external .onnx.data files for weights.
    This merges everything into a single self-contained .onnx protobuf.
    """
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    data_path = Path(str(output_path) + ".data")
    if not data_path.exists():
        return  # Already single-file, nothing to do

    model = onnx.load(str(output_path), load_external_data=True)
    convert_model_from_external_data(model)
    onnx.save(model, str(output_path))
    data_path.unlink()


def export_preprocessor(model, output_path: Path):
    """Export CNN embedder: raw audio → feature sequence + frame-level mask."""
    wrapper = PreprocessorWrapper(model).eval()
    dummy_audio = torch.randn(1, 32000)   # 2 seconds; must be multiple of 80
    dummy_mask = torch.ones(1, 32000, dtype=torch.long)

    batch = torch.export.Dim("batch", min=1)
    audio_len = torch.export.Dim("audio_length", min=80, max=960000)  # up to 60s

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_values", "attention_mask"],
            output_names=["input_features", "padding_mask"],
            dynamic_shapes={
                "input_values": {0: batch, 1: audio_len},
                "attention_mask": {0: batch, 1: audio_len},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  preprocessor → {output_path}")


def export_encoder(model, output_path: Path):
    """Export transformer encoder layers: embedder features + mask → hidden states."""
    wrapper = TransformerEncoderWrapper(model).eval()
    # ~2s of audio after CNN (100 post-CNN frames)
    dummy_features = torch.randn(1, 100, ENC_HIDDEN)
    dummy_mask = torch.ones(1, 100, dtype=torch.bool)

    batch = torch.export.Dim("batch", min=1)
    seq_len = torch.export.Dim("seq_length", min=1, max=3000)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_features", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_shapes={
                "input_features": {0: batch, 1: seq_len},
                "attention_mask": {0: batch, 1: seq_len},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  encoder → {output_path}")


def export_decoder(model, output_path: Path):
    wrapper = DecoderWrapper(model).eval()
    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, ENC_HIDDEN)

    # Dynamo exporter with dynamic shapes
    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden),
            str(output_path),
            dynamo=True,
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["last_hidden_state"] + _kv_output_names(NUM_DECODER_LAYERS),
            dynamic_shapes={
                "decoder_input_ids": {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc_seq},
            },
        )
    _consolidate_onnx(output_path)
    print(f"  decoder → {output_path}")


def export_decoder_with_past(model, output_path: Path):
    wrapper = DecoderWithPastWrapper(model).eval()

    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, ENC_HIDDEN)

    B, H, HEAD = 1, DECODER_HEADS, HEAD_DIM
    dummy_self_past = [(torch.randn(B, H, 5, HEAD), torch.randn(B, H, 5, HEAD))
                       for _ in range(NUM_DECODER_LAYERS)]
    dummy_cross_past = [(torch.randn(B, H, 50, HEAD), torch.randn(B, H, 50, HEAD))
                        for _ in range(NUM_DECODER_LAYERS)]
    flat_past = []
    for k, v in dummy_self_past:
        flat_past += [k, v]
    for k, v in dummy_cross_past:
        flat_past += [k, v]

    # Dynamic dims for dynamo
    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)
    past_seq = torch.export.Dim("past_seq", min=1)

    # Build dynamic_shapes: *flat_past maps to a single "flat_past" key with a list
    n = NUM_DECODER_LAYERS
    flat_past_shapes = []
    for i in range(2 * n):  # self-attention pairs
        flat_past_shapes.append({0: batch, 2: past_seq})
    for i in range(2 * n):  # cross-attention pairs
        flat_past_shapes.append({0: batch, 2: enc_seq})

    dyn_shapes = {
        "decoder_input_ids": {0: batch},
        "encoder_hidden_states": {0: batch, 1: enc_seq},
        "flat_past": tuple(flat_past_shapes),
    }

    input_names = ["decoder_input_ids", "encoder_hidden_states"] + _kv_input_names(NUM_DECODER_LAYERS)
    output_names = ["last_hidden_state"] + _kv_output_names(NUM_DECODER_LAYERS)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, *flat_past),
            str(output_path),
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dyn_shapes,
        )
    _consolidate_onnx(output_path)
    print(f"  decoder_with_past → {output_path}")


def save_token_embeddings(model, output_path: Path):
    """Extract decoder token embeddings (tied to proj_out) and save as .npy.

    Logits are computed externally: logits = last_hidden_state @ embeddings.T
    """
    embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
    np.save(str(output_path), embeddings)
    print(f"  decoder_token_embeddings → {output_path}  "
          f"(shape {embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB)")


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_dynamic_int8(src: Path, dst: Path):
    """Weight-only dynamic INT8 quantization — no calibration needed."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        # Quantize all MatMul nodes (the bulk of transformer compute)
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,      # per-tensor is faster; set True for slightly better accuracy
        reduce_range=False,
        extra_options={"WeightSymmetric": True},
    )
    print(f"  quantized (dynamic int8) → {dst}")


def quantize_static_int8(src: Path, dst: Path, processor, audio_dir: Path):
    """Full static INT8 — both weights and activations quantized."""
    import soundfile as sf
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        CalibrationDataReader,
        QuantFormat,
    )
    # Build calibration data from WAV files
    class AudioCalibReader(CalibrationDataReader):
        def __init__(self):
            wav_paths = sorted(audio_dir.glob("*.wav"))[:50]
            if not wav_paths:
                raise FileNotFoundError(f"No .wav files found in {audio_dir}")
            self._data = []
            for p in wav_paths:
                audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if sr != 16000:
                    raise ValueError(f"{p}: expected 16000 Hz, got {sr}")
                inputs = processor(audio, sampling_rate=16000, return_tensors="np")
                self._data.append({"input_values": inputs.input_values})
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=AudioCalibReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  quantized (static int8) → {dst}")


# ── Validation ───────────────────────────────────────────────────────────────

def validate(model, output_dir: Path):
    """Numerical check: split ONNX pipeline (preprocessor → encoder) vs PyTorch full encoder."""
    import onnxruntime as ort

    # Test at multiple audio lengths to exercise dynamic shapes
    test_lengths = [16000, 48000, 80000]  # 1s, 3s, 5s
    print()

    for audio_len in test_lengths:
        dummy_audio = np.random.randn(1, audio_len).astype(np.float32)
        dummy_mask = np.ones((1, audio_len), dtype=np.int64)

        # PyTorch reference (full encoder)
        with torch.no_grad():
            full_wrapper = FullEncoderWrapper(model).eval()
            pt_out = full_wrapper(
                torch.from_numpy(dummy_audio),
                torch.from_numpy(dummy_mask),
            ).numpy()

        # ONNX two-step pipeline: preprocessor → encoder
        preproc_sess = ort.InferenceSession(str(output_dir / "preprocessor.onnx"))
        enc_sess = ort.InferenceSession(str(output_dir / "encoder.onnx"))

        preproc_outs = preproc_sess.run(None, {
            "input_values": dummy_audio,
            "attention_mask": dummy_mask,
        })
        features = preproc_outs[0]
        padding_mask = preproc_outs[1]

        ort_out = enc_sess.run(None, {
            "input_features": features,
            "attention_mask": padding_mask,
        })[0]

        max_diff = np.abs(pt_out - ort_out).max()
        duration_s = audio_len / 16000
        print(f"  Split pipeline validation ({duration_s:.0f}s audio) — "
              f"shape {ort_out.shape}, max diff: {max_diff:.6f}")
        assert max_diff < 1e-4, f"Validation failed! max_diff={max_diff}"

    print("  ALL PASSED")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download safetensors + configs to local dir (skips if already present)
    if not (MODEL_LOCAL_DIR / "model.safetensors").exists():
        download_model(MODEL_ID, MODEL_LOCAL_DIR)
    else:
        print(f"Using cached weights in {MODEL_LOCAL_DIR.resolve()}")

    print(f"\nLoading model from {MODEL_LOCAL_DIR} ...")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(MODEL_LOCAL_DIR),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",  # avoids SDPA enable_gqa=True export bug
    ).eval().to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_LOCAL_DIR),
        local_files_only=True,
    )

    model.config.use_cache = True

    print("\nExporting to ONNX ...")
    export_preprocessor(model, OUTPUT_DIR / "preprocessor.onnx")
    export_encoder(model, OUTPUT_DIR / "encoder.onnx")
    export_decoder(model, OUTPUT_DIR / "decoder.onnx")
    export_decoder_with_past(model, OUTPUT_DIR / "decoder_with_past.onnx")
    save_token_embeddings(model, OUTPUT_DIR / "decoder_token_embeddings.npy")

    # Copy tokenizer for inference
    tok_src = MODEL_LOCAL_DIR / "tokenizer.json"
    tok_dst = OUTPUT_DIR / "tokenizer.json"
    shutil.copy2(tok_src, tok_dst)
    print(f"  tokenizer → {tok_dst}")

    print("\nValidating split encoder pipeline ...")
    validate(model, OUTPUT_DIR)

    fp32_models = [
        OUTPUT_DIR / "preprocessor.onnx",
        OUTPUT_DIR / "encoder.onnx",
        OUTPUT_DIR / "decoder.onnx",
        OUTPUT_DIR / "decoder_with_past.onnx",
    ]

    if DYNAMIC_QUANT:
        print("\nQuantizing (dynamic INT8) ...")
        for src in fp32_models:
            dst = src.with_name(src.stem + "_int8.onnx")
            quantize_dynamic_int8(src, dst)

    if STATIC_QUANT:
        print("\nQuantizing (static INT8) — encoder only ...")
        enc_src = OUTPUT_DIR / "encoder.onnx"
        enc_dst = OUTPUT_DIR / "encoder_static_int8.onnx"
        quantize_static_int8(enc_src, enc_dst, processor, CALIBRATION_AUDIO_DIR)

    print(f"\nDone. All files in: {OUTPUT_DIR.resolve()}/")
    for f in sorted(OUTPUT_DIR.glob("*.onnx")) + sorted(OUTPUT_DIR.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        tag = "INT8" if "int8" in f.name else ("numpy" if f.suffix == ".npy" else "fp32")
        print(f"  {f.name:50s}  {size_mb:7.1f} MB  [{tag}]")


if __name__ == "__main__":
    main()
