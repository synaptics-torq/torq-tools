# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import os
import shutil
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from . import add_gemma4_export_args
from ...model_export.onnx import OnnxModelExporterBase
from ...utils.logging import configure_logging

_TORCH_DTYPES: Final[dict[str, torch.dtype]] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

# Files copied alongside the exported ONNX for tokenization / chat templating.
_TOKENIZER_ASSETS: Final[tuple[str, ...]] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class Gemma4TextKVWrapper(torch.nn.Module):
    """Causal-LM decoder with a flat, per-persistent-layer KV-cache I/O.

    Gemma4 only keeps a real KV cache for the first
    ``num_hidden_layers - num_kv_shared_layers`` layers. Every later layer of
    the same type (`is_kv_shared_layer` in transformers'
    ``modeling_gemma4.Gemma4TextAttention.forward``) reuses one of those
    entries via ``past_key_values.shared_layers[...]``, a dict populated
    fresh on every forward call -- it never persists its own cache, so it
    needs no ONNX I/O. This wrapper therefore only exposes
    ``num_hidden_layers - num_kv_shared_layers`` KV pairs, not one per layer.

    Sliding-attention and full-attention layers additionally use different
    head dims (``config.head_dim`` vs ``config.global_head_dim``), which is
    why the KV pairs are exposed individually rather than as one stacked
    tensor.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        config = model.config
        n_shared = getattr(config, "num_kv_shared_layers", 0) or 0
        self.n_cache_layers = config.num_hidden_layers - n_shared
        self.layer_types = list(config.layer_types[: self.n_cache_layers])

    def forward(self, input_ids, attention_mask, position_ids, *kv_and_extras):
        n = self.n_cache_layers
        k_in = kv_and_extras[0::2]
        v_in = kv_and_extras[1::2]

        # Rebuild the persistent cache from the flat inputs, then let the
        # model append this step's KV via its own `past_key_values.update()`
        # calls (see Gemma4TextAttention.forward) -- the shared-layer entries
        # are recomputed transiently inside this same forward call.
        past_key_values = DynamicCache(config=self.model.config)
        for i in range(n):
            if k_in[i].shape[-2] > 0:
                past_key_values.update(k_in[i], v_in[i], i)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        outputs = (out.logits,)
        for i in range(n):
            layer = out.past_key_values.layers[i]
            outputs = outputs + (layer.keys, layer.values)
        return outputs


def _layer_head_dim(config, layer_type: str) -> int:
    is_sliding = layer_type == "sliding_attention"
    if not is_sliding and config.global_head_dim:
        return int(config.global_head_dim)
    return int(config.head_dim)


def _layer_kv_heads(config, layer_type: str) -> int:
    is_sliding = layer_type == "sliding_attention"
    use_alt = bool(config.attention_k_eq_v) and not is_sliding
    if use_alt and config.num_global_key_value_heads:
        return int(config.num_global_key_value_heads)
    return int(config.num_key_value_heads)


class Gemma4ModelExporter(OnnxModelExporterBase):

    def __init__(
        self,
        *,
        hf_repo: str = "principled-intelligence/gemma-4-E2B-it-text-only",
        model_dtype: str = "fp32",
        max_seq_len: int = 4096,
        models_dir: str | os.PathLike = "models",
        show_model_info: bool = False,
    ):
        self._hf_repo = hf_repo
        self._max_seq_len = max_seq_len
        try:
            config = AutoConfig.from_pretrained(hf_repo, local_files_only=True)
        except OSError:
            config = AutoConfig.from_pretrained(hf_repo)

        super().__init__(
            model_dtype,
            False,  # static_models: only the dynamic (autoregressive, per-step KV) graph is supported so far
            config,
            Path(models_dir) / hf_repo,
            show_model_info=show_model_info,
            convert_dtypes=False,
            opt_configs={},
        )

    def _setup_dirs(self) -> list[Path]:
        onnx_dir = self._models_dir / "source" / self._model_dtype
        export_dir = self._models_dir / "export" / "onnx" / self._model_dtype / "dynamic"
        convert_dir = self._models_dir / "export" / "onnx" / "converted" / "dynamic"
        torq_dir = self._models_dir / "export" / "torq" / self._model_dtype / "dynamic"
        return onnx_dir, export_dir, convert_dir, torq_dir

    @property
    def _weights_dir(self) -> Path:
        return self._models_dir / "source" / "weights"

    def _download_source(self):
        from huggingface_hub import snapshot_download

        if (self._weights_dir / "model.safetensors").exists():
            return
        self._logger.info("Downloading %s to %s ...", self._hf_repo, str(self._weights_dir))
        snapshot_download(
            repo_id=self._hf_repo,
            local_dir=str(self._weights_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "*.gguf"],
        )

    def _generate_source_onnx(self):
        self._download_source()

        torch_dtype = _TORCH_DTYPES[self._model_dtype]
        self._logger.info(
            "Loading %s from '%s' (dtype=%s) ...", self._hf_repo, str(self._weights_dir), torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(self._weights_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval()

        self._onnx_dir.mkdir(parents=True, exist_ok=True)
        for fname in _TOKENIZER_ASSETS:
            src = self._weights_dir / fname
            if src.exists():
                shutil.copy2(src, self._onnx_dir / fname)

        wrapper = Gemma4TextKVWrapper(model).eval()
        n = wrapper.n_cache_layers
        layer_types = wrapper.layer_types
        self._logger.info(
            "%d/%d layers persist a KV cache entry (the rest reuse one within the same forward call)",
            n, model.config.num_hidden_layers,
        )

        dummy_input_ids = torch.ones(1, 1, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 1, dtype=torch.long)
        dummy_position_ids = torch.zeros(1, 1, dtype=torch.long)
        kv_dummies = []
        kv_in_names, kv_out_names = [], []
        for i, layer_type in enumerate(layer_types):
            head_dim = _layer_head_dim(model.config, layer_type)
            n_heads = _layer_kv_heads(model.config, layer_type)
            kv_dummies.append(torch.zeros(1, n_heads, 0, head_dim, dtype=torch_dtype))
            kv_dummies.append(torch.zeros(1, n_heads, 0, head_dim, dtype=torch_dtype))
            kv_in_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
            kv_out_names += [f"present.{i}.key", f"present.{i}.value"]

        # Dynamic axes: the KV "past length" grows every decode step.
        # Sliding-window layers are capped at `sliding_window` tokens by
        # DynamicSlidingWindowLayer's own eviction; full-attention layers
        # grow up to `--max-seq-len`. `attention_mask` always tracks the
        # *absolute* position count (a full-attention layer's length + 1 for
        # the new token), independent of any individual sliding layer's
        # capped size -- the sliding-window mask itself is derived from
        # `position_ids`, not from the mask's raw length.
        #
        # This wrapper + dynamic_shapes spec was validated end-to-end (matching
        # `model.generate()` token-for-token, including onnxruntime execution
        # past the sliding-window eviction point) against a small randomly
        # initialized Gemma4ForCausalLM built from the *real* E2B config shape
        # (35 layers / 20 shared / mixed head dims) -- see the transformers
        # 5.5.0 `modeling_gemma4.py` source for the KV-sharing logic this
        # mirrors. It has not been run against the actual 9.3GB checkpoint
        # (no machine with enough RAM was available while writing this), so
        # rerun `validate_onnx()` once you have the real weights loaded.
        full_len = torch.export.Dim("full_kv_len", min=0, max=self._max_seq_len)
        sliding_len = torch.export.Dim(
            "sliding_kv_len", min=0, max=int(model.config.sliding_window)
        )
        attn_len = full_len + 1

        # `Gemma4TextKVWrapper.forward` collects the KV tensors via `*kv_and_extras`,
        # so `torch.export` sees them as a single nested tuple argument, not flattened
        # positional args -- the dynamic_shapes spec must mirror that nesting.
        kv_shape_specs = ()
        for layer_type in layer_types:
            dim = sliding_len if layer_type == "sliding_attention" else full_len
            kv_shape_specs += ({2: dim}, {2: dim})
        dynamic_shapes = (None, {1: attn_len}, None, kv_shape_specs)

        onnx_path = self._onnx_dir / "model.onnx"
        self._logger.info("Exporting Gemma4TextKVWrapper to ONNX @ '%s' ...", str(onnx_path))
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids, *kv_dummies),
            str(onnx_path),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            input_names=["input_ids", "attention_mask", "position_ids", *kv_in_names],
            output_names=["logits", *kv_out_names],
        )

    def _load_onnx(self) -> dict[str, onnx.ModelProto]:
        model_path = self._onnx_dir / "model.onnx"
        if not model_path.exists():
            self._generate_source_onnx()
        return {"model": onnx.load(model_path)}

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = True) -> onnx.ModelProto:
        # Full data propagation assumes mostly-concrete shapes; this graph's
        # KV-cache axes are intentionally dynamic, so skip it by default.
        return super().check_model(model, skip_data_prop=skip_data_prop)

    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str):
        raise NotImplementedError(
            "Static (fixed-KV-cache) export is not implemented yet for gemma4 -- "
            "only the dynamic, autoregressive per-step ONNX is currently supported."
        )

    def make_static(self):
        raise NotImplementedError(
            "Static (fixed-KV-cache) export is not implemented yet for gemma4 -- "
            "only the dynamic, autoregressive per-step ONNX is currently supported."
        )

    def export_onnx(self, validate: bool = True):
        super().export_onnx(validate=False)
        for fname in _TOKENIZER_ASSETS:
            src = self._onnx_dir / fname
            if src.exists():
                shutil.copy2(src, self._export_dir / fname)
        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 3):
        """Greedy-decode a couple of short prompts through the exported ONNX
        graph (via onnxruntime) and compare against `model.generate()` on the
        same PyTorch model used for export.
        """
        import onnxruntime as ort

        prompts = [
            "The capital of France is",
            "1, 2, 3, 4,",
            "Hello, my name is",
        ]

        self._logger.info("Loading PyTorch model for reference generation...")
        torch_dtype = _TORCH_DTYPES[self._model_dtype]
        model = AutoModelForCausalLM.from_pretrained(
            str(self._weights_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(self._weights_dir), local_files_only=True)

        session = ort.InferenceSession(str(self._export_paths["model"]))
        n_cache_layers = model.config.num_hidden_layers - (
            getattr(model.config, "num_kv_shared_layers", 0) or 0
        )
        layer_types = list(model.config.layer_types[:n_cache_layers])
        np_dtype = np.float32 if self._model_dtype == "fp32" else None
        if np_dtype is None:
            raise ValueError(
                f"ONNX-Runtime validation requires --dtype fp32 (got '{self._model_dtype}'); "
                "onnxruntime has no CPU bf16 MatMul kernel."
            )

        n_gen_tokens = 8
        n_mismatches = 0
        for i in range(min(n_iters, len(prompts))):
            prompt = prompts[i]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            with torch.no_grad():
                ref_out = model.generate(
                    input_ids, max_new_tokens=n_gen_tokens, do_sample=False, use_cache=True
                )
            ref_tokens = ref_out[0, input_ids.shape[1]:].tolist()

            kv_cache = {}
            for j, layer_type in enumerate(layer_types):
                head_dim = _layer_head_dim(model.config, layer_type)
                n_heads = _layer_kv_heads(model.config, layer_type)
                kv_cache[f"past_key_values.{j}.key"] = np.zeros((1, n_heads, 0, head_dim), dtype=np_dtype)
                kv_cache[f"past_key_values.{j}.value"] = np.zeros((1, n_heads, 0, head_dim), dtype=np_dtype)

            gen_tokens = []
            prompt_tokens = input_ids[0].tolist()
            curr_len = 0
            next_token = None
            for token in prompt_tokens + [None] * n_gen_tokens:
                if token is None:
                    token = next_token
                    gen_tokens.append(token)
                inputs = {
                    "input_ids": np.array([[token]], dtype=np.int64),
                    "attention_mask": np.ones((1, curr_len + 1), dtype=np.int64),
                    "position_ids": np.array([[curr_len]], dtype=np.int64),
                    **kv_cache,
                }
                logits, *cache_out = session.run(None, inputs)
                for j in range(len(layer_types)):
                    kv_cache[f"past_key_values.{j}.key"] = cache_out[2 * j]
                    kv_cache[f"past_key_values.{j}.value"] = cache_out[2 * j + 1]
                next_token = int(logits[0, -1].argmax())
                curr_len += 1

            match = gen_tokens == ref_tokens
            n_mismatches += 0 if match else 1
            self._logger.info(
                "(ONNX-validation) [%r] onnx=%s ref=%s match=%s",
                prompt, gen_tokens, ref_tokens, match,
            )

        if n_mismatches:
            self._logger.warning(
                "(ONNX-validation) %d/%d prompt(s) mismatched -- check the KV-cache "
                "wrapper logic before relying on this export",
                n_mismatches, min(n_iters, len(prompts)),
            )
        else:
            self._logger.info("(ONNX-validation) All prompts matched the PyTorch reference")


def export_gemma4_from_args(args: argparse.Namespace):
    configure_logging(args.logging)
    exporter = Gemma4ModelExporter(
        hf_repo=args.hf_repo,
        model_dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        models_dir=args.models_dir,
    )
    exporter.export_onnx(validate=not args.skip_validation)


def main():
    parser = argparse.ArgumentParser(description="Export Gemma4 to ONNX (dynamic, no Torq compile yet)")
    add_gemma4_export_args(parser)
    export_gemma4_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
