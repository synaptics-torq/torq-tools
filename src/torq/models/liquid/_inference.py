# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None
try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

from ...inference.runners import (
    InferenceRunner,
    ORTInferenceRunner,
    VMFBInferenceRunner,
)
from ...inference.transformers import (
    DecoderOnlyConfig,
    DynamicDecoderOnlyRunner,
    StaticDecoderOnlyRunner,
    find_single_data_file,
)

DEFAULT_SYS_PROMPT: Final[str] = "You are a helpful AI assistant. Provide concise answers."


def _download_asset(repo_id: str, asset_name: str) -> str:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub python API not available in environment"
        )
    return hf_hub_download(repo_id, asset_name)


def _load_tokenizer(tokenizer_path: str | os.PathLike):
    if Tokenizer is None:
        raise RuntimeError(
            "tokenizers python API not available in environment"
        )
    return Tokenizer.from_file(str(tokenizer_path))


@dataclass(frozen=True)
class ModelConfig(DecoderOnlyConfig):
    conv_dim: int = 0
    conv_L_cache: int = 3
    layer_types: tuple[str, ...] = ()

    @classmethod
    def from_json_config(cls, json_file: str | os.PathLike, instruct_model: bool = False) -> "ModelConfig":
        with open(json_file) as f:
            config = json.load(f)
        try:
            head_dim = config.get("head_dim") or (
                config["hidden_size"] // config["num_attention_heads"]
            )
            return cls(
                n_layers=config["num_hidden_layers"],
                n_kv_heads=config["num_key_value_heads"],
                head_dim=head_dim,
                bos_token_id=config["bos_token_id"],
                eos_token_id=config["eos_token_id"],
                pad_token_id=config.get("pad_token_id"),
                instruct_model=instruct_model,
                conv_dim=config.get("conv_dim", config["hidden_size"]),
                conv_L_cache=config.get("conv_L_cache", 3),
                layer_types=tuple(config["layer_types"]),
            )
        except KeyError as e:
            raise ValueError(f"Model config missing required metadata: {e}")


class LiquidBase:
    """LFM-specific hooks for the shared decoder-only inference runners."""

    def _set_liquid_config(self, config: ModelConfig) -> None:
        self._conv_dim = config.conv_dim
        self._conv_L_cache = config.conv_L_cache
        self._layer_types = config.layer_types

    def _init_model_metadata(self):
        sess = getattr(self._model, "_sess", None)
        self._input_names: set[str] | None = (
            {i.name for i in sess.get_inputs()} if sess is not None else None
        )
        self._nl_token_id = self._tokenizer.encode("\n").ids[-1]

    def _declares_input(self, name: str, default: bool = False) -> bool:
        """Whether the loaded model declares a graph input called `name`.

        Falls back to `default` for runners that cannot report their
        signature (e.g. VMFB, which is fed positionally).
        """
        if self._input_names is None:
            return default
        return name in self._input_names

    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(input).ids
        # LFM2.5 ChatML format: <|im_start|>role\n{content}<|im_end|>\n
        if role == "assistant":
            ids = self._tokenizer.encode("<|im_start|>assistant\n").ids
        else:
            ids = self._tokenizer.encode(
                f"<|im_start|>{role}\n{input}<|im_end|>\n"
            ).ids
        # Strip any auto-prepended bos — caller prepends it once.
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _prepare_run_input_tokens(self, input: str) -> list[int]:
        if self._instruct_model:
            # ChatML: <|startoftext|> + <|im_start|>user\n…<|im_end|>\n + <|im_start|>assistant\n
            inp_tokens = [self._bos_token_id]
            inp_tokens += self._tokenize_input(input, "user")
            inp_tokens += self._tokenize_input("", "assistant")
            return inp_tokens
        return self._tokenize_input(input, "user")

    def _warmup_tokens(self) -> list[int]:
        # ChatML: <|startoftext|> + <|im_start|>system\n…<|im_end|>\n
        return [self._bos_token_id] + self._tokenize_input(self._sys_prompt, "system")

    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool:
        if next_token == self._eos_token_id:
            return True
        if not self._instruct_model and len(gen_tokens) > 2:
            return all(t == self._nl_token_id for t in gen_tokens[-2:])
        return False


def _kv_input_name(layer: int, kind: str, combined: bool = False) -> str:
    """Return KV cache input tensor name for an attention layer."""
    if combined:
        return f"past_key_values.{layer}.key_value"
    return f"past_key_values.{layer}.{kind}"


def _conv_input_name(layer: int) -> str:
    return f"past_conv.{layer}"


class LiquidDynamic(LiquidBase, DynamicDecoderOnlyRunner):

    DEFAULT_REPO_ID: Final[str] = "LiquidAI/LFM2.5-350M"

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int | None = None,
        max_gen_tokens: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ):
        if config_path is None:
            config_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "config.json")
        if tokenizer_path is None:
            tokenizer_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "tokenizer.json")
        config = ModelConfig.from_json_config(config_path, instruct_model)
        self._set_liquid_config(config)
        DynamicDecoderOnlyRunner.__init__(
            self,
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            include_position_ids=False,
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidDynamic":
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_inp_len: int | None = None,
        max_gen_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
    ) -> "LiquidDynamic":
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for i, lt in enumerate(self._layer_types):
            if lt == "conv":
                cache[_conv_input_name(i)] = np.zeros(
                    (1, self._conv_dim, self._conv_L_cache), dtype=np.float32
                )
            else:
                cache[_kv_input_name(i, "key")] = np.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
                )
                cache[_kv_input_name(i, "value")] = np.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
                )
        return cache

    def _llm_step(
        self, token: int, curr_seq_len: int
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([[token]], dtype=np.int64)
        attn_mask = np.ones([1, curr_seq_len + 1], dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        # Source exports disagree on the third input: the upstream LiquidAI
        # ONNX takes a `num_logits_to_keep` scalar, the Synaptics mirror takes
        # `position_ids`.  Feed whichever this model declares, keeping
        # `num_logits_to_keep` when the runner cannot report its signature.
        if self._declares_input("position_ids"):
            inputs["position_ids"] = np.array([[curr_seq_len]], dtype=np.int64)
        if self._declares_input("num_logits_to_keep", default=True):
            inputs["num_logits_to_keep"] = np.array(1, dtype=np.int64)
        inputs.update(self._kv_cache)
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

class LiquidStatic(LiquidBase, StaticDecoderOnlyRunner):

    DEFAULT_REPO_ID: Final[str] = "LiquidAI/LFM2.5-350M"

    def __init__(
        self,
        model: InferenceRunner,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
        prefill_model: InferenceRunner | None = None,
        prefill_size: int | None = None,
        lm_head: InferenceRunner | None = None,
    ):
        if config_path is None:
            config_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "config.json")
        if tokenizer_path is None:
            tokenizer_path = _download_asset(repo_id or self.DEFAULT_REPO_ID, "tokenizer.json")
        if (prefill_model is None) != (prefill_size is None):
            raise ValueError("prefill_model and prefill_size must be provided together")
        if prefill_size is not None and prefill_size < 1:
            raise ValueError(f"prefill_size must be positive, got {prefill_size}")
        config = ModelConfig.from_json_config(config_path, instruct_model)
        self._set_liquid_config(config)
        # When `--extract-embeddings` was used at export time the model's
        # input is `token_embedding` rather than `input_ids`; load the LUT
        # from a sibling `token_embeddings.npy` if present.
        token_embeddings = self._find_token_embeddings(model.model_path)
        # The model's KV-cache + attention_mask shape is FIXED at the
        # `max_gen_tokens` chosen at export time.  Generation can stop
        # earlier (smaller `_max_gen_tokens`) but the tensor shapes must
        # always be sized at the compiled length.
        self._kv_cache_len = max_gen_tokens
        StaticDecoderOnlyRunner.__init__(
            self,
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            _load_tokenizer(tokenizer_path),
            DEFAULT_SYS_PROMPT if instruct_model else None,
            combined_kv_io=combined_kv_io,
            token_embeddings=token_embeddings,
            lm_head=lm_head,
            prefill_model=prefill_model,
            prefill_size=prefill_size,
        )
        if self._lm_head is not None:
            self._logger.info("Loaded split LM head '%s'", str(self._lm_head.model_path))
        if self._prefill_model is not None:
            self._logger.info(
                "Loaded %d-token prefill model '%s'",
                self._prefill_size,
                str(self._prefill_model.model_path),
            )

    @staticmethod
    def _find_token_embeddings(
        model_path: str | os.PathLike,
        emb_pattern: str = "token_embeddings.npy",
    ) -> np.ndarray | None:
        paths = list(Path(model_path).parent.glob(emb_pattern))
        if not paths:
            return None
        if len(paths) > 1:
            raise RuntimeError(f"Found multiple embedding files: {paths}")
        return np.load(paths[0])

    @staticmethod
    def _find_prefill_model(
        model_path: str | os.PathLike,
        prefill_pattern: str,
    ) -> Path | None:
        """Locate the batched prefill model exported alongside the transformer."""
        return find_single_data_file(model_path, prefill_pattern, "prefill model")

    @staticmethod
    def _find_lm_head(
        model_path: str | os.PathLike,
        lm_head_pattern: str,
    ) -> Path | None:
        """Locate the split LM head exported alongside the decode model."""
        return find_single_data_file(model_path, lm_head_pattern, "split LM head")

    @staticmethod
    def _infer_prefill_size(prefill_model: InferenceRunner) -> int:
        for input_name in ("token_embedding", "input_ids"):
            shape = prefill_model.input_shapes.get(input_name)
            if shape is not None and len(shape) >= 2 and isinstance(shape[1], int):
                return shape[1]
        raise ValueError(
            f"Could not determine fixed prefill size from '{prefill_model.model_path}'"
        )

    @classmethod
    def from_onnx(
        cls,
        model_path: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
        prefill_model_path: str | os.PathLike | None = None,
        prefill_size: int | None = None,
        lm_head_path: str | os.PathLike | None = None,
    ) -> "LiquidStatic":
        prefill_model_path = prefill_model_path or cls._find_prefill_model(
            model_path, f"{Path(model_path).stem}_prefill.onnx"
        )
        prefill_model = (
            ORTInferenceRunner(prefill_model_path, n_threads=n_threads)
            if prefill_model_path else None
        )
        if prefill_model is not None and prefill_size is None:
            prefill_size = cls._infer_prefill_size(prefill_model)
        lm_head_path = lm_head_path or cls._find_lm_head(model_path, "lm_head.onnx")
        return cls(
            ORTInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            prefill_model=prefill_model,
            prefill_size=prefill_size,
            lm_head=(
                ORTInferenceRunner(lm_head_path, n_threads=n_threads)
                if lm_head_path else None
            ),
        )

    @classmethod
    def from_vmfb(
        cls,
        model_path: str | os.PathLike,
        max_gen_tokens: int,
        max_inp_len: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        repo_id: str | None = None,
        combined_kv_io: bool = True,
        config_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
        prefill_model_path: str | os.PathLike | None = None,
        prefill_size: int | None = None,
        lm_head_path: str | os.PathLike | None = None,
    ) -> "LiquidStatic":
        prefill_model_path = prefill_model_path or cls._find_prefill_model(
            model_path, f"{Path(model_path).stem}_prefill.vmfb"
        )
        lm_head_path = lm_head_path or cls._find_lm_head(model_path, "lm_head.vmfb")
        return cls(
            VMFBInferenceRunner(model_path, n_threads=n_threads),
            max_prompt_tokens=max_inp_len,
            max_gen_tokens=max_gen_tokens,
            instruct_model=instruct_model,
            repo_id=repo_id,
            combined_kv_io=combined_kv_io,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            prefill_model=(
                VMFBInferenceRunner(prefill_model_path, n_threads=n_threads)
                if prefill_model_path else None
            ),
            prefill_size=prefill_size,
            lm_head=(
                VMFBInferenceRunner(lm_head_path, n_threads=n_threads)
                if lm_head_path else None
            ),
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for i, lt in enumerate(self._layer_types):
            if lt == "conv":
                cache[_conv_input_name(i)] = np.zeros(
                    (1, self._conv_dim, self._conv_L_cache), dtype=np.float32
                )
            else:
                if self._combined_kv_io:
                    cache[f"past_key_values.{i}.key_value"] = np.zeros(
                        (1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
                else:
                    cache[_kv_input_name(i, "key")] = np.zeros(
                        (1, self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
                    cache[_kv_input_name(i, "value")] = np.zeros(
                        (1, self._n_kv_heads, self._max_gen_tokens, self._head_dim), dtype=np.float32
                    )
        return cache

    def _llm_tokens_step(
        self,
        model: InferenceRunner,
        tokens: list[int],
        curr_seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        token_ids = np.asarray(tokens, dtype=np.int64)
        if isinstance(self._token_embeddings, np.ndarray):
            inputs = {
                "token_embedding": np.expand_dims(self._token_embeddings[token_ids], axis=0),
            }
        else:
            inputs = {
                "input_ids": np.expand_dims(token_ids, axis=0),
            }
        inputs["position_ids"] = np.array([[curr_seq_len]], dtype=np.int64)
        inputs.update(self._kv_cache)
        # If the static model still exposes attention_mask, supply a full
        # mask sized at the *compiled* KV-cache length (not the runtime
        # generation cap, which may be smaller).
        if self._declares_input("attention_mask"):
            inputs["attention_mask"] = np.ones(
                [1, self._kv_cache_len], dtype=np.int64
            )
        logits, *cache = model.infer(inputs)
        if self._lm_head is not None:
            # With a split LM head the decode model's and prefill model's
            # first output is the hidden state, not logits; run it through
            # the standalone head.
            logits = self._lm_head.infer({"last_hidden_states": logits})[0]
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

if __name__ == "__main__":
    pass
