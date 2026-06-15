# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path
from typing import Literal, Final
from types import SimpleNamespace

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import torch
import ml_dtypes
from transformers import AutoConfig
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

from ...utils.logging import (
    configure_logging,
)

from . import (
    ONNX_DTYPES,
    OPTIMUM_DTYPES,
    STATIC_MODEL_COMPONENTS,
    add_moonshine_streaming_export_args,
)

from ._graph import MoonshineStreaming5SplitOnnxGraphEditor
from ._inference import MoonshineStreaming5Split
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig


# Helper to construct cache layer from KV tensors
def _layer_from_kv(k, v):
    from transformers.cache_utils import DynamicLayer
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


# Patch Asinh compression to use basic ops to avoid ONNX export issues with legacy exporter
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import MoonshineStreamingAsinhCompression
def _patched_asinh_forward(self, x):
    val = torch.exp(self.log_k) * x
    return torch.log(val + torch.sqrt(val**2 + 1.0))
MoonshineStreamingAsinhCompression.forward = _patched_asinh_forward


# ── Wrapper modules ──────────────────────────────────────────────────────────

class StatefulPreprocessorWrapper(torch.nn.Module):
    """Stateful preprocessor: audio chunk + buffers -> features + updated buffers."""

    def __init__(self, model):
        super().__init__()
        embedder = model.model.encoder.embedder
        self.cmvn = embedder.cmvn
        self.comp = embedder.comp
        self.linear = embedder.linear
        self.conv1 = embedder.conv1
        self.conv2 = embedder.conv2
        self.frame_len = embedder.frame_len

    def forward(self, audio_chunk, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count):
        # 1. Concatenate past active sample buffer and new chunk
        active_past = sample_buffer[:, :sample_len[0]]
        samples = torch.cat([active_past, audio_chunk], dim=1)

        total_samples = samples.shape[1]
        n_frames = total_samples // self.frame_len
        processed_len = n_frames * self.frame_len

        frames_samples = samples[:, :processed_len]

        # 2. Remaining samples are saved in sample_buffer_out
        remainder = samples[:, processed_len:]
        sample_len_out = torch.tensor([total_samples - processed_len], device=samples.device, dtype=torch.int64)

        # Pad remainder to 79 elements using the concat + slice trick
        zeros = torch.zeros(1, 79, device=samples.device, dtype=samples.dtype)
        sample_buffer_out = torch.cat([remainder, zeros], dim=1)[:, :79]

        # 3. Reshape and forward through linear embeds
        x = frames_samples.reshape(1, -1, self.frame_len)
        x = self.cmvn(x)
        x = self.comp(x)
        x = torch.nn.functional.silu(self.linear(x))

        # 4. Causal convolutions
        # Transpose to channel-first for Conv1D: [B, C, T]
        x = x.transpose(1, 2)

        # Conv1
        x1_padded = torch.cat([conv1_buffer, x], dim=2)
        x1_conv = torch.nn.functional.conv1d(
            x1_padded,
            self.conv1.weight,
            self.conv1.bias,
            stride=self.conv1.stride,
            dilation=self.conv1.dilation
        )
        conv1_buffer_out = x1_padded[:, :, -4:]
        x1_silu = torch.nn.functional.silu(x1_conv)

        # Conv2
        x2_padded = torch.cat([conv2_buffer, x1_silu], dim=2)
        x2_conv = torch.nn.functional.conv1d(
            x2_padded,
            self.conv2.weight,
            self.conv2.bias,
            stride=self.conv2.stride,
            dilation=self.conv2.dilation
        )
        conv2_buffer_out = x2_padded[:, :, -4:]

        # Features output: shape [B, T_out, hidden]
        features = x2_conv.transpose(1, 2)
        frame_count_out = frame_count + n_frames

        return features, sample_buffer_out, sample_len_out, conv1_buffer_out, conv2_buffer_out, frame_count_out


class StatefulEncoderWrapper(torch.nn.Module):
    """
    Stateful streaming encoder with per-layer left-context hidden-state buffers.

    Processes new stable frames alongside a fixed right-context window so each
    stable frame is encoded exactly once.  The right-context window is always
    total_lookahead = sum(right_ctx_per_layer) frames wide (16 for tiny).

    Inputs
    ------
    stable_features  [1, T_stable, hidden]        dynamic T_stable
    right_ctx        [1, total_lookahead, hidden]  static — lookahead frames
    buf_i            [1, left_ctx_i, hidden]       static — per-layer cache

    Outputs
    -------
    encoded_stable   [1, T_stable, hidden]
    buf_i_out        [1, left_ctx_i, hidden]       updated caches
    """

    def __init__(self, model):
        super().__init__()
        enc = model.model.encoder
        self.layers = enc.layers
        self.final_norm = enc.final_norm
        self.config = enc.config
        self._left_ctx = [int(w[0]) for w in self.config.sliding_windows]
        self._right_ctx = [int(w[1]) for w in self.config.sliding_windows]
        self._total_lookahead = sum(self._right_ctx)

    def forward(
        self,
        stable_features: torch.Tensor,
        right_ctx: torch.Tensor,
        buf_0: torch.Tensor,
        buf_1: torch.Tensor,
        buf_2: torch.Tensor,
        buf_3: torch.Tensor,
        buf_4: torch.Tensor,
        buf_5: torch.Tensor,
    ) -> tuple:
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )
        bufs = [buf_0, buf_1, buf_2, buf_3, buf_4, buf_5]
        bufs_out = []
        # rc is a Python int constant — enables static end-anchored slices in ONNX
        rc = self._total_lookahead

        stable_in = stable_features
        right_ctx_h = right_ctx

        for layer_idx, (layer, buf) in enumerate(zip(self.layers, bufs)):
            lc = self._left_ctx[layer_idx]        # Python int constant
            layer_rc = self._right_ctx[layer_idx] # Python int constant

            window = torch.cat([buf, stable_in, right_ctx_h], dim=1)

            attn_mask = torch.ones(
                window.shape[0], window.shape[1], dtype=torch.bool, device=window.device
            )
            layer_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=window,
                attention_mask=attn_mask,
                and_mask_function=sliding_window_mask_function((lc, layer_rc)),
            )

            out = layer(window, attention_mask=layer_mask)
            out = out[0] if isinstance(out, tuple) else out

            # Buffer: last lc frames of [buf | stable_in] — static end-anchored slice
            bufs_out.append(torch.cat([buf, stable_in], dim=1)[:, -lc:, :])

            # Peel off left buf and right ctx with static offsets to avoid dynamic indices
            out_trimmed = out[:, lc:, :]           # removes left_buf portion [1, T+rc, hidden]
            stable_in = out_trimmed[:, :-rc, :]    # [1, T_stable, hidden]
            right_ctx_h = out_trimmed[:, -rc:, :]  # [1, rc, hidden]

        return (self.final_norm(stable_in), *bufs_out)


class EncoderWrapper(torch.nn.Module):
    """Pure Transformer encoder layer forwarding features."""

    def __init__(self, model):
        super().__init__()
        self.layers = model.model.encoder.layers
        self.final_norm = model.model.encoder.final_norm
        self.config = model.model.encoder.config

    def forward(self, input_features: torch.FloatTensor) -> torch.FloatTensor:
        from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
            create_bidirectional_mask,
            sliding_window_mask_function,
        )

        hidden_states = input_features
        # Causal encoder attention mask with sliding window
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)

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
        return hidden_states


class AdapterWrapper(torch.nn.Module):
    """Decoupled projection / positional embedder."""

    def __init__(self, decoder):
        super().__init__()
        self.pos_emb = decoder.pos_emb
        self.proj = decoder.proj

    def forward(self, encoded, pos_offset):
        seq_len = encoded.shape[1]
        arange = torch.arange(seq_len, device=encoded.device)
        indices = pos_offset.to(torch.int64) + arange

        position_embeddings = self.pos_emb(indices)
        if position_embeddings.ndim == 2:
            position_embeddings = position_embeddings.unsqueeze(0)

        memory = encoded + position_embeddings
        memory = self.proj(memory)
        return memory


class CrossKVGeneratorWrapper(torch.nn.Module):
    """Generates stacked cross key-values from adapter memory."""

    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers
        self.depth = len(decoder.layers)
        self.num_heads = decoder.config.num_key_value_heads
        self.head_dim = getattr(decoder.config, "head_dim", decoder.config.hidden_size // decoder.config.num_attention_heads)

    def forward(self, memory):
        bsz, seq_len = memory.shape[:-1]
        k_list = []
        v_list = []

        for layer in self.layers:
            attn = layer.encoder_attn
            k_proj = attn.k_proj(memory).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_proj = attn.v_proj(memory).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_list.append(k_proj)
            v_list.append(v_proj)

        k_cross = torch.stack(k_list, dim=0)
        v_cross = torch.stack(v_list, dim=0)
        return k_cross, v_cross


class ZeroEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, indices):
        return torch.zeros(indices.shape[0], self.embedding_dim, device=indices.device)


class DecoderKVWrapper(torch.nn.Module):
    """Decoder executing with token input and stacked self/cross caches."""

    def __init__(self, model):
        super().__init__()
        self.base_model = model.model
        self.proj_out = model.proj_out
        self.n_layers = len(self.base_model.decoder.layers)
        self.decoder = self.base_model.decoder

        # Replace pos_emb and proj with proper nn.Modules during initialization
        self.decoder.pos_emb = ZeroEmbedding(self.decoder.pos_emb.embedding_dim)
        self.decoder.proj = torch.nn.Identity()

    def forward(self, token, k_self, v_self, out_k_cross, out_v_cross):
        # Construct past_key_values cache
        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        for i in range(self.n_layers):
            self_cache.layers.append(_layer_from_kv(k_self[i], v_self[i]))
            cross_cache.layers.append(_layer_from_kv(out_k_cross[i], out_v_cross[i]))

        pkv = EncoderDecoderCache(self_cache, cross_cache)

        # Inform layers that cross-cache is initialized so they bypass projection
        for i in range(self.n_layers):
            pkv.is_updated[i] = True

        # Create dummy encoder_hidden_states of matching shape to trigger cross-attention
        enc_seq_len = out_k_cross.shape[3]
        dummy_encoder_hidden = torch.zeros(
            1, enc_seq_len, self.decoder.config.hidden_size,
            dtype=k_self.dtype, device=k_self.device
        )

        # Call decoder
        dec_out = self.decoder(
            input_ids=token,
            past_key_values=pkv,
            use_cache=True,
            encoder_hidden_states=dummy_encoder_hidden,
        )

        logits = self.proj_out(dec_out.last_hidden_state)

        # Extract updated caches
        updated_k_self = torch.stack([layer.keys for layer in pkv.self_attention_cache.layers], dim=0)
        updated_v_self = torch.stack([layer.values for layer in pkv.self_attention_cache.layers], dim=0)

        return logits, updated_k_self, updated_v_self, out_k_cross, out_v_cross


# ── Model Exporter ───────────────────────────────────────────────────────────

class MoonshineStreaming5SplitExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["tiny", "small"] = "tiny",
        model_dtype: str = "float",
        static_models: bool = True,
        *,
        extract_embeddings: bool = False,
        hf_repo: str | None = None,
        max_audio_s: int = 5,
        max_tok_per_s: int = 6,
        models_dir: str | os.PathLike = "models",
        onnx_source_dir: str | os.PathLike | None = None,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        skip_export: list[str] | None = None,
        **edit_args
    ):
        self._model_size = model_size
        self._extract_embeddings = extract_embeddings
        self._onnx_source_dir = onnx_source_dir
        self._hf_repo = hf_repo or f"UsefulSensors/moonshine-streaming-{self._model_size}"
        self._config = AutoConfig.from_pretrained(self._hf_repo)
        self._num_samples = max_audio_s * 16_000
        self._max_tokens = max_audio_s * max_tok_per_s
        self._hidden_size = int(self._config.hidden_size)
        self._vocab_size = int(self._config.vocab_size)

        # Standard conv layers of moonshine preprocessor do two strided causal convs of stride 2
        # giving a total input reduction of stride 4.
        self._enc_seq_len = self._num_samples // 320

        self._n_layers = getattr(self._config, "decoder_num_hidden_layers", getattr(self._config, "num_hidden_layers", 6))
        self._broadcast_ops = edit_args.get("broadcast_ops", None)

        dec_heads = getattr(self._config, "num_attention_heads", 8)
        self._num_kv_heads = getattr(self._config, "num_key_value_heads", 8)
        self._head_dim = self._hidden_size // dec_heads

        opt_configs = {
            comp: ORTOptimizerConfig(
                num_heads=dec_heads,
                hidden_size=self._config.hidden_size
            ) for comp in STATIC_MODEL_COMPONENTS
        }

        super().__init__(
            model_dtype,
            static_models,
            self._config,
            Path(models_dir) / self._hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=convert_dtypes,
            opt_configs=opt_configs,
            skip_export=skip_export,
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir = self._models_dir / "source" / "onnx" / "merged" / self._model_size / self._model_dtype
        if self._static_models:
            onnx_dir = onnx_dir / f"static_i{self._num_samples // 16000}"
        export_dir = (
            self._models_dir
            / "export"
            / "onnx"
            / self._model_dtype
            / ("static" if self._static_models else "dynamic")
        )
        convert_dir = (
            self._models_dir
            / "export"
            / "onnx"
            / "converted"
            / ("static" if self._static_models else "dynamic")
        )
        iree_dir = (
            self._models_dir
            / "export"
            / "iree"
            / ("converted" if self._convert_dtypes else self._model_dtype)
            / ("static" if self._static_models else "dynamic")
        )
        return onnx_dir, export_dir, convert_dir, iree_dir

    def _generate_source_onnx(self):
        from huggingface_hub import snapshot_download
        from transformers import MoonshineStreamingForConditionalGeneration

        local_dir = self._models_dir / "weights" / self._model_size
        if not (local_dir / "model.safetensors").exists():
            self._logger.info("Downloading %s to %s ...", self._hf_repo, str(local_dir))
            snapshot_download(
                repo_id=self._hf_repo,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )

        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            str(local_dir),
            torch_dtype=torch.float32,
            local_files_only=True,
            attn_implementation="eager",
        ).eval()

        self._onnx_dir.mkdir(parents=True, exist_ok=True)

        # Save token embeddings
        embeddings = model.model.decoder.embed_tokens.weight.detach().cpu().numpy()
        np.save(self._onnx_dir / "decoder_token_embeddings.npy", embeddings)

        # Save tokenizer
        shutil.copy2(local_dir / "tokenizer.json", self._onnx_dir / "tokenizer.json")
        shutil.copy2(local_dir / "config.json", self._onnx_dir / "config.json")

        self._logger.info("Exporting Stateful Preprocessor (frontend) to ONNX...")
        preproc = StatefulPreprocessorWrapper(model).eval()

        # Set convolution buffers sizes based on size configuration
        enc_hidden = self._hidden_size
        c1 = enc_hidden * 2
        c2 = enc_hidden

        # Chunk len defaults to 80 (5ms) for dummy input
        dummy_audio = torch.randn(1, 80)
        dummy_sample_buf = torch.zeros(1, 79)
        dummy_sample_len = torch.zeros(1, dtype=torch.int64)
        dummy_conv1_buf = torch.zeros(1, enc_hidden, 4)
        dummy_conv2_buf = torch.zeros(1, c1, 4)
        dummy_frame_cnt = torch.zeros(1, dtype=torch.int64)

        if self._static_models:
            torch.onnx.utils.export(
                preproc,
                (dummy_audio, dummy_sample_buf, dummy_sample_len, dummy_conv1_buf, dummy_conv2_buf, dummy_frame_cnt),
                str(self._onnx_dir / "frontend.onnx"),
                opset_version=17,
                input_names=["audio_chunk", "sample_buffer", "sample_len", "conv1_buffer", "conv2_buffer", "frame_count"],
                output_names=["features", "sample_buffer_out", "sample_len_out", "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"],
            )
        else:
            torch.onnx.utils.export(
                preproc,
                (dummy_audio, dummy_sample_buf, dummy_sample_len, dummy_conv1_buf, dummy_conv2_buf, dummy_frame_cnt),
                str(self._onnx_dir / "frontend.onnx"),
                opset_version=17,
                input_names=["audio_chunk", "sample_buffer", "sample_len", "conv1_buffer", "conv2_buffer", "frame_count"],
                output_names=["features", "sample_buffer_out", "sample_len_out", "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"],
                dynamic_axes={
                    "audio_chunk": {0: "batch", 1: "chunk_len"},
                    "features": {0: "batch", 1: "feat_len"},
                },
            )

        if self._static_models:
            self._logger.info("Exporting pure Transformer Encoder to ONNX...")
            encoder = EncoderWrapper(model).eval()
            dummy_features = torch.randn(1, 5, enc_hidden)
            torch.onnx.export(
                encoder,
                (dummy_features,),
                str(self._onnx_dir / "encoder.onnx"),
                dynamo=True,
                input_names=["features"],
                output_names=["encoded"],
            )
        else:
            self._logger.info("Exporting Stateful Streaming Encoder as encoder.onnx...")
            streaming_enc = StatefulEncoderWrapper(model).eval()
            n_enc_layers = len(streaming_enc.layers)
            total_la = streaming_enc._total_lookahead
            left_ctxs = streaming_enc._left_ctx

            dummy_stable = torch.randn(1, 10, enc_hidden)
            dummy_right_ctx = torch.zeros(1, total_la, enc_hidden)
            dummy_bufs = [torch.zeros(1, lc, enc_hidden) for lc in left_ctxs]

            buf_in_names = [f"buf_{i}" for i in range(n_enc_layers)]
            buf_out_names = [f"buf_{i}_out" for i in range(n_enc_layers)]
            t_stable = torch.export.Dim("t_stable", min=1)
            # dynamic_shapes as a list — positional match to (stable_features, right_ctx, buf_0..N)
            # Only stable_features has a dynamic dim; all buffers and right_ctx are static.
            dynamic_shapes_list = [{1: t_stable}] + [None] * (1 + n_enc_layers)

            torch.onnx.export(
                streaming_enc,
                (dummy_stable, dummy_right_ctx, *dummy_bufs),
                str(self._onnx_dir / "encoder.onnx"),
                dynamo=True,
                input_names=["stable_features", "right_ctx"] + buf_in_names,
                output_names=["encoded_stable"] + buf_out_names,
                dynamic_shapes=dynamic_shapes_list,
            )

        self._logger.info("Exporting Adapter wrapper to ONNX...")
        adapter = AdapterWrapper(model.model.decoder).eval()
        dummy_encoded = torch.randn(1, 5, enc_hidden)
        dummy_pos_offset = torch.zeros(1, dtype=torch.int64)

        if self._static_models:
            torch.onnx.export(
                adapter,
                (dummy_encoded, dummy_pos_offset),
                str(self._onnx_dir / "adapter.onnx"),
                dynamo=True,
                input_names=["encoded", "pos_offset"],
                output_names=["memory"],
            )
        else:
            batch = torch.export.Dim("batch", min=1)
            seq_len = torch.export.Dim("seq_length", min=1, max=3000)
            torch.onnx.export(
                adapter,
                (dummy_encoded, dummy_pos_offset),
                str(self._onnx_dir / "adapter.onnx"),
                dynamo=True,
                input_names=["encoded", "pos_offset"],
                output_names=["memory"],
                dynamic_shapes={
                    "encoded": {0: batch, 1: seq_len},
                    "pos_offset": None,
                },
            )

        self._logger.info("Exporting Cross KV Generator wrapper to ONNX...")
        cross_kv = CrossKVGeneratorWrapper(model.model.decoder).eval()
        dummy_memory = torch.randn(1, 5, self._config.hidden_size)

        if self._static_models:
            torch.onnx.export(
                cross_kv,
                (dummy_memory,),
                str(self._onnx_dir / "cross_kv.onnx"),
                dynamo=True,
                input_names=["memory"],
                output_names=["k_cross", "v_cross"],
            )
        else:
            batch = torch.export.Dim("batch", min=1)
            seq_len = torch.export.Dim("seq_length", min=1, max=3000)
            torch.onnx.export(
                cross_kv,
                (dummy_memory,),
                str(self._onnx_dir / "cross_kv.onnx"),
                dynamo=True,
                input_names=["memory"],
                output_names=["k_cross", "v_cross"],
                dynamic_shapes={
                    "memory": {0: batch, 1: seq_len},
                },
            )

        self._logger.info("Exporting Decoder KV wrapper to ONNX...")
        decoder_kv = DecoderKVWrapper(model).eval()
        dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
        dummy_k_self = torch.randn(self._n_layers, 1, self._num_kv_heads, 1, self._head_dim)
        dummy_v_self = torch.randn(self._n_layers, 1, self._num_kv_heads, 1, self._head_dim)
        dummy_k_cross = torch.randn(self._n_layers, 1, self._num_kv_heads, 5, self._head_dim)
        dummy_v_cross = torch.randn(self._n_layers, 1, self._num_kv_heads, 5, self._head_dim)

        batch = torch.export.Dim("batch", min=1)
        dec_seq = torch.export.Dim("dec_seq", min=1)
        past_seq = torch.export.Dim("past_seq", min=1)
        enc_seq = torch.export.Dim("enc_seq", min=1)

        if self._static_models:
            torch.onnx.export(
                decoder_kv,
                (dummy_dec_ids, dummy_k_self, dummy_v_self, dummy_k_cross, dummy_v_cross),
                str(self._onnx_dir / "decoder_kv.onnx"),
                dynamo=True,
                input_names=["token", "k_self", "v_self", "out_k_cross", "out_v_cross"],
                output_names=["logits", "out_k_self", "out_v_self", "out_k_cross_out", "out_v_cross_out"],
            )
        else:
            torch.onnx.export(
                decoder_kv,
                (dummy_dec_ids, dummy_k_self, dummy_v_self, dummy_k_cross, dummy_v_cross),
                str(self._onnx_dir / "decoder_kv.onnx"),
                dynamo=True,
                input_names=["token", "k_self", "v_self", "out_k_cross", "out_v_cross"],
                output_names=["logits", "out_k_self", "out_v_self", "out_k_cross_out", "out_v_cross_out"],
                dynamic_shapes={
                    "token": {0: batch, 1: dec_seq},
                    "k_self": {1: batch, 3: past_seq},
                    "v_self": {1: batch, 3: past_seq},
                    "out_k_cross": {1: batch, 3: enc_seq},
                    "out_v_cross": {1: batch, 3: enc_seq},
                },
            )

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        source_files = {
            "frontend": self._onnx_dir / "frontend.onnx",
            "encoder": self._onnx_dir / "encoder.onnx",
            "adapter": self._onnx_dir / "adapter.onnx",
            "cross_kv": self._onnx_dir / "cross_kv.onnx",
            "decoder_kv": self._onnx_dir / "decoder_kv.onnx",
        }

        # Check if source ONNX files exist, if not generate them!
        any_missing = any(not path.exists() for path in source_files.values())
        if any_missing:
            self._logger.info("Source ONNX models not found. Downloading PyTorch model and exporting...")
            self._generate_source_onnx()

        return {
            comp: onnx.load(path)
            for comp, path in source_files.items()
        }

    def _make_frontend_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model, "frontend", self._onnx_export_dtype)
        # Fix input shape of audio chunk to 80 (5ms)
        editor.fix_frontend_io(chunk_len=80)
        editor.decompose_layer_normalization()

        # Extract non-zero pads to explicit Pad nodes in frontend
        for node in list(editor._graph.nodes):
            if node.op == "Conv":
                weight = node.inputs[1]
                if ("kernel_shape" not in node.attrs or not node.attrs["kernel_shape"]) and weight.shape is not None:
                    if len(weight.shape) == 3:
                        node.attrs["kernel_shape"] = [weight.shape[2]]

        editor._graph.cleanup().toposort()
        editor.decompose_strided_conv1d()
        editor.replace_pad_with_concat()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_encoder_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model, "encoder", self._onnx_export_dtype)
        # Sequence length is self._enc_seq_len (number of frames)
        editor.fix_encoder_io(self._enc_seq_len)
        editor.decompose_layer_normalization()
        editor.decompose_gelu()
        editor.decompose_boolean_and()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_adapter_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model, "adapter", self._onnx_export_dtype)
        editor.fix_adapter_io(self._enc_seq_len)
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_cross_kv_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model, "cross_kv", self._onnx_export_dtype)
        editor.fix_cross_kv_io(self._enc_seq_len)
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_decoder_kv_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model, "decoder_kv", self._onnx_export_dtype)
        # Decode step length is 1 (autoregressive step-by-step)
        editor.fix_decoder_kv_io(dec_seq_len=1, enc_seq_len=self._enc_seq_len)
        editor.decompose_layer_normalization()
        editor.decompose_gelu()
        editor.decompose_boolean_and()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def make_static(self):
        self._logger.info("Verifying and finalizing static dimensions...")
        self._components["frontend"] = self._make_frontend_static(self._components["frontend"])
        self._components["encoder"] = self._make_encoder_static(self._components["encoder"])
        self._components["adapter"] = self._make_adapter_static(self._components["adapter"])
        self._components["cross_kv"] = self._make_cross_kv_static(self._components["cross_kv"])
        self._components["decoder_kv"] = self._make_decoder_kv_static(self._components["decoder_kv"])

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        # Move embeddings/tokenizer
        emb_src = self._onnx_dir / "decoder_token_embeddings.npy"
        emb_dst = Path(model_path).parent / "decoder_token_embeddings.npy"
        if emb_src.exists():
            shutil.copy2(emb_src, emb_dst)

        if "encoder" in component:
            editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.remove_identity_gather_nd()
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        if "frontend" in component:
            editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.decompose_reduce_sum()
            editor.decompose_asinh()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        if "decoder_kv" in component:
            editor = MoonshineStreaming5SplitOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        for filename in ("tokenizer.json", "config.json"):
            src = self._onnx_dir / filename
            dst = Path(model_path).parent / filename
            if src.exists():
                shutil.copy2(src, dst)

    def export_onnx(self, validate: bool = True):
        super().export_onnx(validate=False)
        for filename in (
            "decoder_token_embeddings.npy", "tokenizer.json", "config.json",
        ):
            src = self._onnx_dir / filename
            dst = self._export_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 5):
        self._logger.info("Validating exported 5-split ONNX models...")
        runner = MoonshineStreaming5Split.from_onnx(
            frontend_model=self._export_dir / "frontend.onnx",
            encoder_model=self._export_dir / "encoder.onnx",
            adapter_model=self._export_dir / "adapter.onnx",
            cross_kv_model=self._export_dir / "cross_kv.onnx",
            decoder_model=self._export_dir / "decoder_kv.onnx",
            model_size=self._model_size,
        )

        wav_path = Path(__file__).parent.parent / "moonshine_streaming" / "OSR_us_000_0010_8k.wav"
        if wav_path.exists():
            self._logger.info("Loading test audio file '%s' for validation...", wav_path.name)
            import soundfile as sf
            from scipy.signal import resample_poly
            from tokenizers import Tokenizer

            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != 16000:
                data = resample_poly(data, up=16000, down=sr).astype(np.float32)

            speech = data.astype(np.float32)[np.newaxis, :]
            tokens = runner.run(speech)

            tokenizer_path = self._export_dir / "tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                transcribed = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
                self._logger.info("Validation transcription: '%s'", transcribed)
            else:
                self._logger.info("Successfully ran validation, tokens: %s", str(tokens))
        else:
            self._logger.warning("Test audio file '%s' not found, running validation with dummy audio", wav_path)
            # 5 seconds of audio
            dummy_audio = np.random.randn(1, 80000).astype(np.float32)
            tokens = runner.run(dummy_audio)
            self._logger.info("Successfully transcribed dummy audio, tokens: %s", str(tokens))


def export_moonshine_streaming_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = MoonshineStreaming5SplitExporter(
        args.model_size,
        args.dtype,
        not args.dynamic_models,
        extract_embeddings=args.extract_embeddings,
        hf_repo=args.hf_repo,
        max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec,
        models_dir=args.models_dir,
        onnx_source_dir=args.onnx_source_dir,
        show_model_info=args.show_model_info,
        convert_dtypes=args.convert_dtypes,
        skip_export=args.skip_export,
        broadcast_ops=args.broadcast_ops
    )
    exporter.export_onnx(validate=not args.skip_validation)
    if args.convert_dtypes:
        exporter.convert_models(preserve_io=args.preserve_io_dtypes)
    if args.skip_torq is None or "all" not in args.skip_torq:
        exporter.export_torq(
            torq_compile_args=args.compile_flags or [],
            use_binary=args.use_binary,
            skip=args.skip_torq or [],
            local_compile=args.local_compile,
            compiler_path=args.compiler_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Export 5-Split Moonshine Streaming to Torq")
    add_moonshine_streaming_export_args(parser)
    export_moonshine_streaming_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
