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
from transformers import AutoConfig, AutoProcessor
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

from ...utils.logging import (
    configure_logging,
)

from . import (
    ONNX_DTYPES,
    OPTIMUM_DTYPES,
    STATIC_MODEL_COMPONENTS,
    STATIC_MODEL_COMPONENTS_UNFOLDED,
    add_moonshine_streaming_export_args,
)

from ._graph import MoonshineStreamingOnnxGraphEditor
from ._inference import MoonshineStreamingDynamic, MoonshineStreamingStatic
from ...model_export.onnx import OnnxModelExporterBase, ORTOptimizerConfig


# ── Wrapper modules ──────────────────────────────────────────────────────────

class PreprocessorWrapper(torch.nn.Module):
    """CNN embedder: raw audio + attention_mask → feature sequence + frame-level mask."""

    def __init__(self, model):
        super().__init__()
        self.embedder = model.model.encoder.embedder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor):
        hidden_states, padding_mask = self.embedder(input_values, padding_mask=attention_mask)
        return hidden_states, padding_mask  # (B, seq_len, hidden), (B, seq_len)


class TransformerEncoderWrapper(torch.nn.Module):
    """Transformer encoder layers + final norm with sliding-window attention."""

    def __init__(self, model):
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
    """First decode step — no past KV cache."""

    def __init__(self, model, n_layers):
        super().__init__()
        self.base_model = model.model
        self.n_layers = n_layers

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,       # (B, T_dec)
        encoder_hidden_states: torch.FloatTensor,  # (B, T_enc, hidden)
    ):
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
        pkv = out.past_key_values

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
    from transformers.cache_utils import DynamicLayer
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


# Patch DynamicLayer to avoid empty tensor dynamic cat issue with Dynamo ONNX export
from transformers.cache_utils import DynamicLayer
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


class DecoderWithPastWrapper(torch.nn.Module):
    """Subsequent decode steps — accepts and returns flat KV tensors."""

    def __init__(self, model, n_layers):
        super().__init__()
        self.base_model = model.model
        self.n_layers = n_layers

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,        # (B, 1)
        encoder_hidden_states: torch.FloatTensor,   # (B, T_enc, hidden)
        # past self-attention KV + past cross-attention KV
        *flat_past,
    ):
        n = self.n_layers
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
        new_pkv = out.past_key_values

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


def _kv_output_names(n_layers):
    names = []
    for i in range(n_layers):
        names.append(f"present_self_key_{i}")
        names.append(f"present_self_value_{i}")
    for i in range(n_layers):
        names.append(f"present_cross_key_{i}")
        names.append(f"present_cross_value_{i}")
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


# ── Model Exporter ───────────────────────────────────────────────────────────

class MoonshineStreamingModelExporter(OnnxModelExporterBase):

    def __init__(
        self,
        model_size: Literal["tiny", "small"] = "tiny",
        model_dtype: str = "float",
        static_models: bool = True,
        *,
        extract_embeddings: bool = False,
        fold_encoder_cache: bool = True,
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
        self._fold_encoder_cache = fold_encoder_cache
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

        enc_heads = getattr(
            self._config,
            "encoder_num_attention_heads",
            getattr(
                getattr(self._config, "encoder_config", None),
                "num_attention_heads",
                8
            )
        )
        dec_heads = getattr(
            self._config,
            "decoder_num_attention_heads",
            getattr(self._config, "num_attention_heads", 8)
        )

        components_list = STATIC_MODEL_COMPONENTS if fold_encoder_cache else STATIC_MODEL_COMPONENTS_UNFOLDED
        opt_configs = {
            comp: ORTOptimizerConfig(
                num_heads=enc_heads if comp == "encoder" else dec_heads,
                hidden_size=self._config.hidden_size
            ) for comp in components_list
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

        self._logger.info("Exporting Preprocessor wrapper to ONNX...")
        preproc = PreprocessorWrapper(model).eval()
        dummy_audio = torch.randn(1, self._num_samples)
        dummy_mask = torch.ones(1, self._num_samples, dtype=torch.long)

        if self._static_models:
            torch.onnx.export(
                preproc,
                (dummy_audio, dummy_mask),
                str(self._onnx_dir / "preprocessor.onnx"),
                dynamo=True,
                input_names=["input_values", "attention_mask"],
                output_names=["input_features", "padding_mask"],
            )
        else:
            batch = torch.export.Dim("batch", min=1)
            audio_len = torch.export.Dim("audio_length", min=80, max=960000)
            torch.onnx.export(
                preproc,
                (dummy_audio, dummy_mask),
                str(self._onnx_dir / "preprocessor.onnx"),
                dynamo=True,
                input_names=["input_values", "attention_mask"],
                output_names=["input_features", "padding_mask"],
                dynamic_shapes={
                    "input_values": {0: batch, 1: audio_len},
                    "attention_mask": {0: batch, 1: audio_len},
                },
            )

        self._logger.info("Exporting Transformer Encoder wrapper to ONNX...")
        encoder = TransformerEncoderWrapper(model).eval()
        dummy_features = torch.randn(1, self._enc_seq_len, self._hidden_size)
        dummy_feat_mask = torch.ones(1, self._enc_seq_len, dtype=torch.bool)

        if self._static_models:
            torch.onnx.export(
                encoder,
                (dummy_features, dummy_feat_mask),
                str(self._onnx_dir / "encoder.onnx"),
                dynamo=True,
                input_names=["input_features", "attention_mask"],
                output_names=["last_hidden_state"],
            )
        else:
            batch = torch.export.Dim("batch", min=1)
            seq_len = torch.export.Dim("seq_length", min=1, max=3000)
            torch.onnx.export(
                encoder,
                (dummy_features, dummy_feat_mask),
                str(self._onnx_dir / "encoder.onnx"),
                dynamo=True,
                input_names=["input_features", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_shapes={
                    "input_features": {0: batch, 1: seq_len},
                    "attention_mask": {0: batch, 1: seq_len},
                },
            )

        self._logger.info("Exporting Decoder wrapper to ONNX...")
        decoder = DecoderWrapper(model, self._n_layers).eval()
        dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
        dummy_enc_hidden = torch.randn(1, self._enc_seq_len, self._hidden_size)

        if self._static_models:
            torch.onnx.export(
                decoder,
                (dummy_dec_ids, dummy_enc_hidden),
                str(self._onnx_dir / "decoder.onnx"),
                dynamo=True,
                input_names=["decoder_input_ids", "encoder_hidden_states"],
                output_names=["last_hidden_state"] + _kv_output_names(self._n_layers),
            )
        else:
            batch = torch.export.Dim("batch", min=1)
            enc_seq = torch.export.Dim("enc_seq", min=1)
            torch.onnx.export(
                decoder,
                (dummy_dec_ids, dummy_enc_hidden),
                str(self._onnx_dir / "decoder.onnx"),
                dynamo=True,
                input_names=["decoder_input_ids", "encoder_hidden_states"],
                output_names=["last_hidden_state"] + _kv_output_names(self._n_layers),
                dynamic_shapes={
                    "decoder_input_ids": {0: batch},
                    "encoder_hidden_states": {0: batch, 1: enc_seq},
                },
            )

        self._logger.info("Exporting Decoder with past wrapper to ONNX...")
        decoder_past = DecoderWithPastWrapper(model, self._n_layers).eval()

        B, H = 1, getattr(self._config, "decoder_num_attention_heads", getattr(self._config, "num_attention_heads", 8))
        HEAD = self._hidden_size // H

        past_len = 5
        dummy_self_past = [(torch.randn(B, H, past_len, HEAD), torch.randn(B, H, past_len, HEAD))
                           for _ in range(self._n_layers)]
        dummy_cross_past = [(torch.randn(B, H, self._enc_seq_len if self._static_models else 50, HEAD), 
                             torch.randn(B, H, self._enc_seq_len if self._static_models else 50, HEAD))
                            for _ in range(self._n_layers)]

        flat_past = []
        for k, v in dummy_self_past:
            flat_past += [k, v]
        for k, v in dummy_cross_past:
            flat_past += [k, v]

        batch = torch.export.Dim("batch", min=1)
        enc_seq = torch.export.Dim("enc_seq", min=1)
        past_seq = torch.export.Dim("past_seq", min=1)

        flat_past_shapes = []
        for i in range(2 * self._n_layers):
            flat_past_shapes.append({0: batch, 2: past_seq})
        for i in range(2 * self._n_layers):
            flat_past_shapes.append({0: batch, 2: enc_seq})

        dyn_shapes = {
            "decoder_input_ids": {0: batch},
            "encoder_hidden_states": {0: batch, 1: enc_seq},
            "flat_past": tuple(flat_past_shapes),
        }

        torch.onnx.export(
            decoder_past,
            (dummy_dec_ids, dummy_enc_hidden, *flat_past),
            str(self._onnx_dir / "decoder_with_past.onnx"),
            dynamo=True,
            input_names=["decoder_input_ids", "encoder_hidden_states"] + _kv_input_names(self._n_layers),
            output_names=["last_hidden_state"] + _kv_output_names(self._n_layers),
            dynamic_shapes=dyn_shapes,
        )

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        source_files = {
            "preprocessor": self._onnx_dir / "preprocessor.onnx",
            "encoder": self._onnx_dir / "encoder.onnx",
            "decoder": self._onnx_dir / "decoder.onnx",
            "decoder_with_past": self._onnx_dir / "decoder_with_past.onnx",
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

    def _make_preprocessor_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model, "preprocessor", self._onnx_export_dtype)
        editor.fix_preprocessor_io(self._num_samples)
        editor.decompose_layer_normalization()

        # Pre-process Conv nodes for DecomposeStridedConv1D compatibility:
        # 1. Pre-populate kernel_shape if missing from weights shape
        # 2. Extract non-zero pads to explicit Pad nodes
        import numpy as np
        import onnx_graphsurgeon as gs
        
        for node in list(editor._graph.nodes):
            if node.op == "Conv":
                weight = node.inputs[1]
                if ("kernel_shape" not in node.attrs or not node.attrs["kernel_shape"]) and weight.shape is not None:
                    if len(weight.shape) == 3:
                        node.attrs["kernel_shape"] = [weight.shape[2]]

        for node in list(editor._graph.nodes):
            if node.op == "Conv":
                pads = node.attrs.get("pads", [0, 0])
                if any(p != 0 for p in pads):
                    conv_in = node.inputs[0]
                    rank = len(conv_in.shape) if conv_in.shape is not None else 3
                    
                    pad_width = [0] * rank
                    pad_width[-1] = pads[0]
                    pad_width_after = [0] * rank
                    pad_width_after[-1] = pads[1]
                    
                    pads_array = np.array(pad_width + pad_width_after, dtype=np.int64)
                    pads_const = gs.Constant(
                        name=node.name + "_explicit_pads_const",
                        values=pads_array
                    )
                    
                    padded_shape = list(conv_in.shape) if conv_in.shape is not None else [1, 1, 1]
                    if conv_in.shape is not None:
                        padded_shape[-1] = conv_in.shape[-1] + pads[0] + pads[1]
                        
                    padded_in = gs.Variable(
                        name=node.name + "_padded_input",
                        dtype=conv_in.dtype,
                        shape=padded_shape
                    )
                    
                    pad_node = gs.Node(
                        op="Pad",
                        name=node.name + "_explicit_pad",
                        inputs=[conv_in, pads_const],
                        outputs=[padded_in],
                        attrs={"mode": "constant"}
                    )
                    editor._graph.nodes.append(pad_node)
                    node.inputs[0] = padded_in
                    node.attrs["pads"] = [0, 0]
                    
        editor._graph.cleanup().toposort()

        editor.decompose_strided_conv1d()
        editor.replace_pad_with_concat()
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_encoder_model_static(self, model: onnx.ModelProto) -> onnx.ModelProto:
        editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model, "encoder", self._onnx_export_dtype)
        editor.fix_encoder_io(self._enc_seq_len)
        editor.decompose_layer_normalization()
        editor.decompose_gelu()
        # Replace And(i1,i1)->i1 with Cast(int8)+Mul+Cast(bool) to avoid hardware
        # DMA assertion failures on bit-packed boolean tensors, then make the
        # resulting Mul broadcasting explicit.
        editor.decompose_boolean_and()
        # editor.broadcast_op_inputs(["Mul"])
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    def _make_decoder_model_static(self, model: onnx.ModelProto, with_past: bool) -> onnx.ModelProto:
        graph: gs.Graph = gs.import_onnx(model)
        comp = "decoder" + ("_with_past" if with_past else "")
        dec_heads = getattr(self._config, "decoder_num_attention_heads", getattr(self._config, "num_attention_heads", 8))
        pad_len = (self._config.hidden_size // dec_heads) % 8

        editor = MoonshineStreamingOnnxGraphEditor(graph, comp, self._onnx_export_dtype)
        editor.fix_decoder_io(self._enc_seq_len, self._max_tokens, with_past)
        editor.decompose_layer_normalization()
        editor.decompose_gelu()

        # Remove redundant Cast ops
        editor.remove_redundant_casts()
        # Remove isNaN ops
        editor.remove_isNaN()
        # Move model output if it's fed by a Concat node which has a Pad consumer
        if not with_past:
            editor.move_output_from_concat(pad_len=pad_len)

        if with_past:
            cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
            graph.inputs.append(cur_len_2d)
            cur_len = graph.layer(
                name="current_len_to_1d",
                op="Squeeze",
                inputs=[cur_len_2d, [0]],
                outputs=[gs.Variable(cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1])],
            )[0]

            (
                editor
                # Replace dynamic KV cache
                .replace_dynamic_kv_cache(cur_len, self._max_tokens)
                # Add causal attention score mask
                .mask_future_attn_scores(cur_len, self._max_tokens)
                # Replace dynamic sequence length getter with `cur_len`
                .add_curr_len_input(cur_len)
                # Replace dynamic index computation `Range(start, start + 1, 1) -> index`
                .convert_to_static_index()
            )

        # Replace And(i1,i1)->i1 with Cast(int8)+Mul+Cast(bool) to avoid hardware
        # DMA assertion failures on bit-packed boolean tensors, then make the
        # resulting Mul broadcasting explicit.
        editor.decompose_boolean_and()
        # editor.broadcast_op_inputs(["Mul"])
        new_model = editor.to_onnx(override_ir=model.ir_version)
        return self.check_model(new_model)

    @staticmethod
    def extract_encoder_cache(static_decoder: onnx.ModelProto) -> onnx.ModelProto:
        """Extract the cross-attention KV cache generation subgraph from the static decoder.

        The cross-attn KV caches (present_cross_key/value_*) depend only on encoder_hidden_states,
        so they can be computed independently in a separate model.

        Args:
            static_decoder: The full static decoder model

        Returns:
            An ONNX model that takes encoder_hidden_states and produces all cross-attn KV caches
        """
        graph = gs.import_onnx(static_decoder)

        encoder_cache_output_names = {out.name for out in graph.outputs if "cross" in out.name}
        encoder_input = next(
            inp for inp in graph.inputs
            if "encoder" in inp.name
        )

        # Keep only the encoder cache outputs and encoder_hidden_states input
        graph.outputs = [out for out in graph.outputs if out.name in encoder_cache_output_names]
        graph.inputs = [encoder_input]

        # Fold constants (Shape ops produce known values in a static graph)
        # This severs dependencies on input_ids through shape-computation paths
        graph.fold_constants(size_threshold=1024 * 1024)  # 1 MiB limit to avoid bloat
        graph.name = "main"
        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()

        gen_encoder_cache = gs.export_onnx(graph)
        gen_encoder_cache.ir_version = static_decoder.ir_version
        return gen_encoder_cache

    @staticmethod
    def fold_encoder_cache(
        encoder_model: onnx.ModelProto,
        gen_encoder_cache: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Merge gen_encoder_cache into encoder, so encoder outputs cross-attn KV caches directly.

        Connects encoder's output (last_hidden_state or encoder_hidden_states) to gen_encoder_cache's input,
        making the intermediate state an internal tensor.

        Args:
            encoder_model: The static encoder model
            gen_encoder_cache: The extracted cross-attn KV cache generation model

        Returns:
            A merged ONNX model: input_values/input_features → cross-attn KV caches
        """
        enc_graph = gs.import_onnx(encoder_model)
        cache_graph = gs.import_onnx(gen_encoder_cache)

        # Rename all nodes and tensors in cache_graph to avoid name collisions
        # inputs[0] is the rewired input (encoder_hidden_states), which we don't want to rename
        rewired_input_name = cache_graph.inputs[0].name
        
        # We also don't want to rename outputs since they are the final model outputs
        output_names = {out.name for out in cache_graph.outputs}

        for node in cache_graph.nodes:
            if node.name:
                node.name = f"cache_fold/{node.name}"
            for inp in node.inputs:
                if isinstance(inp, (gs.Variable, gs.Constant)):
                    if inp.name != rewired_input_name and inp.name not in output_names:
                        inp.name = f"cache_fold/{inp.name}"
            for out in node.outputs:
                if isinstance(out, (gs.Variable, gs.Constant)):
                    if out.name not in output_names:
                        out.name = f"cache_fold/{out.name}"

        # The encoder has one output (last_hidden_state)
        # gen_encoder_cache has one input (encoder_hidden_states)
        # Wire them together by replacing cache_graph's input with encoder's output tensor
        enc_output_tensor = enc_graph.outputs[0]
        cache_input_tensor = cache_graph.inputs[0]

        # Rewire all consumers in cache_graph that reference the cache input
        for node in cache_graph.nodes:
            for i, inp in enumerate(node.inputs):
                if inp is cache_input_tensor:
                    node.inputs[i] = enc_output_tensor

        # Merge: add all cache_graph nodes and initializers into enc_graph
        enc_graph.nodes.extend(cache_graph.nodes)
        # Set outputs to be the cache outputs
        enc_graph.outputs = cache_graph.outputs

        enc_graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()

        merged = gs.export_onnx(enc_graph)
        merged.ir_version = encoder_model.ir_version
        return merged

    def make_static(self):
        self._logger.info("Verifying and finalizing static dimensions...")
        self._components["preprocessor"] = self._make_preprocessor_model_static(self._components["preprocessor"])

        decoder = self._components.pop("decoder")
        decoder_with_past = self._components.pop("decoder_with_past")

        self._logger.info("(encoder) Making encoder static...")
        self._components["encoder"] = self._make_encoder_model_static(self._components["encoder"])

        self._logger.info("(decoder) Making full decoder static for encoder cache extraction...")
        static_full_decoder = self._make_decoder_model_static(decoder, False)

        self._logger.info("(gen_encoder_cache) Extracting cross-attn KV cache subgraph...")
        gen_encoder_cache = self.extract_encoder_cache(static_full_decoder)

        if self._fold_encoder_cache:
            # Fold gen_encoder_cache into encoder
            self._logger.info("(encoder) Folding gen_encoder_cache into encoder...")
            self._components["encoder"] = self.fold_encoder_cache(
                self._components["encoder"], gen_encoder_cache
            )
        else:
            self._components["gen_encoder_cache"] = gen_encoder_cache

        # Use decoder_with_past as the unified decoder
        self._logger.info("(decoder) Making unified decoder static...")
        self._components["decoder"] = self._make_decoder_model_static(
            decoder_with_past, True
        )
        expected = set(STATIC_MODEL_COMPONENTS if self._fold_encoder_cache else STATIC_MODEL_COMPONENTS_UNFOLDED)
        assert set(self._components) >= expected

    def _dedup_decoder_embeddings_npy(self, emb_dir: str | os.PathLike):
        emb_dir = Path(emb_dir)
        if (d_emb_p := emb_dir / f"decoder_token_embeddings.npy").exists() \
            and (dp_emb_p := emb_dir / f"decoder_with_past_token_embeddings.npy").exists():
            d_emb = np.load(d_emb_p)
            dp_emb = np.load(dp_emb_p)
            if np.array_equal(d_emb, dp_emb):
                dp_emb_p.unlink()

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        # Move token embeddings and tokenizer over to the export folder
        emb_src = self._onnx_dir / "decoder_token_embeddings.npy"
        emb_dst = Path(model_path).parent / "decoder_token_embeddings.npy"
        if emb_src.exists():
            shutil.copy2(emb_src, emb_dst)

        if "encoder" in component:
            editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.remove_identity_gather_nd()
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        if "preprocessor" in component:
            editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.decompose_reduce_sum()
            editor.decompose_asinh()
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)

        if "decoder" in component:
            editor = MoonshineStreamingOnnxGraphEditor.from_onnx(model_path, component, self._onnx_export_dtype)
            editor.eliminate_transposes()
            editor.collapse_reshape_chains()
            if self._extract_embeddings:
                # Extract token embeddings LUT
                embeddings_npy = Path(model_path).parent / f"{component}_token_embeddings.npy"
                embeddings_inp = "token_embedding"
                editor.extract_token_embeddings(
                    self._hidden_size,
                    self._vocab_size,
                    embeddings_npy,
                    inp_name=embeddings_inp
                )
                editor.reorder_graph_input(embeddings_inp, 0)
            new_model = editor.to_onnx(override_ir=onnx.load(model_path).ir_version)
            onnx.save(new_model, model_path)
            if self._extract_embeddings:
                self._dedup_decoder_embeddings_npy(Path(model_path).parent)

        tok_src = self._onnx_dir / "tokenizer.json"
        tok_dst = Path(model_path).parent / "tokenizer.json"
        if tok_src.exists():
            shutil.copy2(tok_src, tok_dst)

        cfg_src = self._onnx_dir / "config.json"
        cfg_dst = Path(model_path).parent / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, cfg_dst)

    def export_onnx(self, validate: bool = True):
        super().export_onnx(validate=False)
        for filename in ("decoder_token_embeddings.npy", "tokenizer.json", "config.json"):
            src = self._onnx_dir / filename
            dst = self._export_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 5):
        self._logger.info("Validating exported ONNX models...")
        if self._static_models:
            gen_encoder_cache_model = None if self._fold_encoder_cache else self._export_dir / "gen_encoder_cache.onnx"
            runner = MoonshineStreamingStatic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                gen_encoder_cache_model=gen_encoder_cache_model,
                decoder_model=self._export_dir / "decoder.onnx",
                model_size=self._model_size,
                preprocessor_model=self._export_dir / "preprocessor.onnx",
            )
        else:
            runner = MoonshineStreamingDynamic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                decoder_model=self._export_dir / "decoder.onnx",
                decoder_with_past_model=self._export_dir / "decoder_with_past.onnx",
                model_size=self._model_size,
                preprocessor_model=self._export_dir / "preprocessor.onnx",
                max_inp_len=self._num_samples,
            )

        wav_path = Path(__file__).parent / "OSR_us_000_0010_8k.wav"
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
            self._logger.warning("Test audio file '%s' not found, running with dummy audio...", wav_path)
            dummy_audio = np.random.randn(runner.max_inp_len or 32000).astype(np.float32)
            tokens = runner.run(dummy_audio)
            self._logger.info("Successfully transcribed dummy audio, tokens: %s", str(tokens))

    def convert_models(
        self, 
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
        skip: list[str] | None = None,
    ):
        skip = skip or []
        external_data = None if any(m in self._skip_export for m in ("decoder", "decoder_with_past")) else \
        [(self._export_paths["decoder"].parent / "decoder_token_embeddings.npy", np.dtype(ml_dtypes.bfloat16))]
        super().convert_models(
            convert_dir=convert_dir,
            preserve_io=preserve_io,
            skip=skip,
            external_data=external_data,
        )


def export_moonshine_streaming_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = MoonshineStreamingModelExporter(
        args.model_size,
        args.dtype,
        not args.dynamic_models,
        extract_embeddings=args.extract_embeddings,
        fold_encoder_cache=not args.no_fold_encoder_cache,
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
    parser = argparse.ArgumentParser(description="Export Moonshine Streaming to Torq")
    add_moonshine_streaming_export_args(parser)
    export_moonshine_streaming_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
