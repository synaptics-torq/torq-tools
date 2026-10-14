# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Synaptics Incorporated.

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns as _perf_counter_ns

import numpy as np

from .runners import InferenceRunner
from . import ptd as ptd_core


@dataclass(frozen=True)
class DecoderOnlyConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bos_token_id: int
    eos_token_id: int | list[int]
    pad_token_id: int | None = None
    instruct_model: bool = False


class DecoderOnlyRunner(ABC):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        tokenizer,
        sys_prompt: str | None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._max_prompt_tokens = max_prompt_tokens
        self._max_gen_tokens = max_gen_tokens
        self._tokenizer = tokenizer
        self._sys_prompt = sys_prompt
        self._max_total_tokens = self._calc_max_total_tokens(
            self._max_prompt_tokens, self._max_gen_tokens
        )
        self._max_user_tokens: int | None = None
        self._temperature = temperature
        self._top_p = top_p

        self._n_layers: int = config.n_layers
        self._n_kv_heads: int = config.n_kv_heads
        self._head_dim: int = config.head_dim
        self._instruct_model: bool = config.instruct_model
        self._bos_token_id: int = config.bos_token_id
        # HF configs may list several EOS ids (gemma-3-1b: [1, 106]); the
        # first is the model's primary EOS.
        eos_token_id = config.eos_token_id
        self._eos_token_id: int = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        self._bos_token: str = self._tokenizer.decode(
            [self._bos_token_id], skip_special_tokens=False
        )
        self._eos_token: str = self._tokenizer.decode(
            [self._eos_token_id], skip_special_tokens=False
        )
        self._init_model_metadata()
        self._logger.info("Loaded model '%s'", str(self._model.model_path))

        self._n_tokens_gen: int = 0
        self._infer_times: deque[float] = deque(maxlen=100)
        self._kv_cache = self._init_cache()
        self._warmup_len = self.warmup() if self._instruct_model else 0
        self._reset_cache_state = deepcopy(self._kv_cache)

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else 0.0

    @property
    def max_inp_len(self) -> int | None:
        return self._max_user_tokens if self._max_user_tokens is not None else self._max_prompt_tokens

    def _init_model_metadata(self):
        pass

    @abstractmethod
    def _init_cache(self) -> dict[str, np.ndarray]: ...

    def _update_cache(self, new_values: list[np.ndarray]):
        for k, v in zip(self._kv_cache.keys(), new_values):
            self._kv_cache[k] = v

    @abstractmethod
    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]: ...

    @abstractmethod
    def _stop_decoding(self, next_token: int, gen_tokens: list[int]) -> bool: ...

    @abstractmethod
    def _run(self, input: list[int], max_gen_tokens: int | None = None) -> list[int]: ...

    @abstractmethod
    def _tokenize_input(self, input: str, role: str | None = None) -> list[int]: ...

    @staticmethod
    def _calc_max_total_tokens(max_prompt_tokens: int | None, max_gen_tokens: int | None) -> int | None:
        if isinstance(max_prompt_tokens, (int, float)) and isinstance(max_gen_tokens, (int, float)):
            return int(max_prompt_tokens + max_gen_tokens)
        return None

    def _reset_cache(self):
        self._kv_cache.update(self._reset_cache_state)

    def _format_input_tokens(self, input: list[int]) -> list[int]:
        """Cap the prompt at ``max_inp_len``.

        This is a cap, not a pad target: decoder-only exports bake the causal
        mask in (static) or feed it as all-ones (dynamic, one token per step),
        so right-padding with ``pad_token_id`` would be attended over as real
        content and degenerate the output (E-11).
        """
        max_len = self.max_inp_len
        if isinstance(max_len, int) and len(input) > max_len:
            self._logger.warning("Truncating input from %d to %d", len(input), max_len)
            input = input[: max_len]

        return input

    def sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        x = logits.astype(np.float64)
        temperature = temperature or self._temperature
        top_p = top_p or self._top_p
        if temperature <= 0:
            return int(x.argmax())

        x = x / temperature
        x = x - x.max()
        probs = np.exp(x)
        probs = probs / probs.sum()
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cdf = np.cumsum(sorted_probs)
        cut = np.searchsorted(cdf, top_p) + 1
        keep = order[:cut]
        kept_probs = probs[keep]
        kept_probs = kept_probs / kept_probs.sum()

        return int(np.random.choice(keep, p=kept_probs))

    def _prefill_prompt(self, prompt_tokens: list[int], start_seq_len: int = 0) -> tuple[int, int]:
        num_tokens_gen = start_seq_len
        next_token: int | None = None
        for token in prompt_tokens:
            next_token, cache = self._llm_step(token, num_tokens_gen)
            self._update_cache(cache)
            num_tokens_gen += 1
        return next_token, num_tokens_gen

    def _prepare_run_input_tokens(self, input: str) -> list[int]:
        return self._tokenize_input(input, "user")

    def _warmup_tokens(self) -> list[int]:
        return self._tokenize_input(self._sys_prompt, "system")

    def run(self, input: str, max_gen_tokens: int | None = None) -> str:
        return self._tokenizer.decode(self.run_tokens(input, max_gen_tokens))

    def run_tokens(self, input: str, max_gen_tokens: int | None = None) -> list[int]:
        """Generate token IDs without decoding away special tokens or token boundaries."""
        self._reset_cache()
        inp_tokens = self._prepare_run_input_tokens(input)
        st = time.perf_counter_ns()
        out_tokens = self._run(inp_tokens, max_gen_tokens)
        et = time.perf_counter_ns()
        self._infer_times.append(et - st)
        return out_tokens

    def warmup(self) -> int:
        if not self._instruct_model:
            self._logger.warning("Not an instruct model, skipping system prompt warm-up")
            return 0
        sys_tokens = self._warmup_tokens()
        if isinstance(self._max_prompt_tokens, int):
            if len(sys_tokens) > self._max_prompt_tokens:
                self._logger.warning(
                    "Truncating system prompt from %d to %d",
                    len(sys_tokens),
                    self.max_inp_len,
                )
                sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
            if self._max_user_tokens < 1:
                self._logger.warning("No tokens left for user prompt")
        warmup_len = len(sys_tokens)
        self._prefill_prompt(sys_tokens, start_seq_len=0)
        if self._max_user_tokens is not None:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt, %d tokens remaining for user input",
                warmup_len,
                self._max_user_tokens,
            )
        else:
            self._logger.debug(
                "Warm-up complete: %d tokens consumed by system prompt",
                warmup_len,
            )
        return warmup_len


class DynamicDecoderOnlyRunner(DecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int | None,
        max_gen_tokens: int | None,
        tokenizer,
        sys_prompt: str | None,
        *,
        include_position_ids: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._include_position_ids = include_position_ids
        super().__init__(
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            tokenizer,
            sys_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                (1, self._n_kv_heads, 0, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([[token]], dtype=np.int64)
        attn_mask = np.ones([1, curr_seq_len + 1], dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        if self._include_position_ids:
            inputs["position_ids"] = np.array([[curr_seq_len]], dtype=np.int64)
        inputs.update(self._kv_cache)
        logits, *cache = self._model.infer(inputs)
        next_token = self.sample_next_token(logits[0, -1])
        return next_token, cache

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None,
    ) -> list[int]:
        self._max_gen_tokens = max_gen_tokens or self._max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = [next_token]
        while not self._stop_decoding(next_token, gen_tokens):
            if isinstance(self._max_gen_tokens, int) and len(gen_tokens) >= self._max_gen_tokens:
                self._logger.warning("Max generation tokens reached, stopping early")
                break
            next_token, cache = self._llm_step(next_token, curr_seq_len)
            self._update_cache(cache)
            gen_tokens.append(next_token)
            curr_seq_len += 1
        self._n_tokens_gen = len(gen_tokens)
        return gen_tokens


class StaticDecoderOnlyRunner(DecoderOnlyRunner):
    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        tokenizer,
        sys_prompt: str | None,
        *,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
        token_id_lut: np.ndarray | None = None,
        lm_head: InferenceRunner | None = None,
        prefill_model: InferenceRunner | None = None,
        prefill_size: int | None = None,
        hidden_states_name: str = "last_hidden_states",
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self._combined_kv_io = combined_kv_io
        self._token_embeddings = token_embeddings
        self._token_id_lut = token_id_lut
        self._lm_head = lm_head
        if (prefill_model is None) != (prefill_size is None):
            raise ValueError("prefill_model and prefill_size must be provided together")
        if prefill_size is not None and prefill_size < 1:
            raise ValueError(f"prefill_size must be positive, got {prefill_size}")
        self._prefill_model = prefill_model
        self._prefill_size = prefill_size
        self._hidden_states_name = hidden_states_name
        super().__init__(
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            tokenizer,
            sys_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    def _init_cache(self) -> dict[str, np.ndarray]:
        if self._combined_kv_io:
            return {
                f"past_key_values.{i}.key_value": np.zeros(
                    [1, 2 * self._n_kv_heads, self._max_gen_tokens, self._head_dim],
                    dtype=np.float32,
                )
                for i in range(self._n_layers)
            }
        return {
            f"past_key_values.{i}.{typ}": np.zeros(
                [1, self._n_kv_heads, self._max_gen_tokens, self._head_dim],
                dtype=np.float32,
            )
            for i in range(self._n_layers)
            for typ in ("key", "value")
        }

    def _llm_tokens_step(
        self,
        model: InferenceRunner,
        tokens: list[int],
        curr_seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        token_ids = np.asarray(tokens, dtype=np.int64)
        if isinstance(self._token_embeddings, np.ndarray):
            inputs = {
                "token_embedding": np.expand_dims(self._token_embeddings[token_ids], axis=0)
            }
        else:
            inputs = {
                "input_ids": np.expand_dims(token_ids, axis=0)
            }
        pos_ids = np.array([[curr_seq_len]], dtype=np.int64)
        inputs.update({
            "position_ids": pos_ids,
            **self._kv_cache,
        })
        logits, *cache = model.infer(inputs)
        if self._lm_head is not None:
            # With a split LM head the first output is the hidden state, not
            # logits; run it through the standalone head to get the logits.
            logits = self._lm_head.infer({self._hidden_states_name: logits})[0]
        next_token = self.sample_next_token(logits[0, -1])
        if self._token_id_lut is not None:
            if next_token >= len(self._token_id_lut):
                raise RuntimeError(
                    f"Sampled compact token index {next_token} outside token ID LUT "
                    f"with {len(self._token_id_lut)} entries"
                )
            next_token = int(self._token_id_lut[next_token])
        return next_token, cache

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
        return self._llm_tokens_step(self._model, [token], curr_seq_len)

    def _prefill_prompt(self, prompt_tokens: list[int], start_seq_len: int = 0) -> tuple[int, int]:
        if self._prefill_model is None or self._prefill_size is None:
            return super()._prefill_prompt(prompt_tokens, start_seq_len)

        curr_seq_len = start_seq_len
        next_token: int | None = None
        chunk_start = 0
        while chunk_start + self._prefill_size <= len(prompt_tokens):
            chunk = prompt_tokens[chunk_start:chunk_start + self._prefill_size]
            next_token, cache = self._llm_tokens_step(
                self._prefill_model,
                chunk,
                curr_seq_len,
            )
            self._update_cache(cache)
            chunk_start += self._prefill_size
            curr_seq_len += self._prefill_size

        if chunk_start < len(prompt_tokens):
            next_token, curr_seq_len = super()._prefill_prompt(
                prompt_tokens[chunk_start:],
                start_seq_len=curr_seq_len,
            )
        return next_token, curr_seq_len

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None,
    ) -> list[int]:
        if isinstance(max_gen_tokens, int) and 0 <= max_gen_tokens < self._max_gen_tokens:
            self._max_gen_tokens = max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = [next_token]
        while not self._stop_decoding(next_token, gen_tokens):
            if curr_seq_len >= self._max_gen_tokens:
                self._logger.warning("Max generation tokens reached, stopping early")
                break
            next_token, cache = self._llm_step(next_token, curr_seq_len)
            self._update_cache(cache)
            gen_tokens.append(next_token)
            curr_seq_len += 1
        self._n_tokens_gen = len(gen_tokens)
        return gen_tokens


class StaticPTDRunner(StaticDecoderOnlyRunner):
    """Static decode runner with Progressive Tree Drafting (PTD).

    Uses one fixed-row body and optional block head for every phase. Prefill
    packs causal prompt chains; decoding packs draft/verification trees.
    The host promotes valid scratch KVs into the committed cache. Near cache
    capacity, fewer real rows are packed and remaining rows are masked padding.

    PTD greedy output is token-identical to ordinary greedy decoding of the
    same model (anchor row = ordinary step; each accepted continuation is the
    ordinary greedy prediction for its prefix; the stop condition is scanned
    per emitted token).

    Progressive Tree Drafting (PTD): Gao et al. 2026,
    https://arxiv.org/abs/2607.10661 (see torq.inference.ptd for attribution
    and the domain-specific adaptations).
    """

    def __init__(
        self,
        model: InferenceRunner,
        config: DecoderOnlyConfig,
        max_prompt_tokens: int,
        max_gen_tokens: int,
        tokenizer,
        sys_prompt: str | None,
        *,
        ptd_rows: int,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
        token_id_lut: np.ndarray | None = None,
        lm_head: InferenceRunner | None = None,
        ptd_width: int = 4,
        ptd_depth: int = 6,
        pool_max_nodes: int = 16,
        pool_max_entries: int = 64,
        min_draft_rows: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if ptd_rows < 2:
            raise ValueError(f"ptd_rows must be >= 2, got {ptd_rows}")
        if list(model.input_shapes.get("tree_vis", ())) != [1, 1, ptd_rows, ptd_rows - 1]:
            raise ValueError("PTD requires a body with matching fixed-row tree_vis input")
        if temperature != 0.0 or top_p != 1.0:
            raise ValueError("PTD supports greedy decoding only (temperature=0, top_p=1)")
        self._ptd_rows = ptd_rows
        self._ptd_width = ptd_width
        self._ptd_depth = ptd_depth
        self._ptd_pool_max_nodes = pool_max_nodes
        self._ptd_pool_max_entries = pool_max_entries
        self._ptd_min_draft_rows = max(0, min_draft_rows)
        self._ptd_pool = None
        self._ptd_tree = None
        self._ptd_pad_token = config.pad_token_id if config.pad_token_id is not None else 0
        self._ptd_out_names: list[str] = list(model.output_names)
        self._ptd_first_out = next(
            (n for n in self._ptd_out_names if n in ("logits", "last_hidden_states")), None
        )
        if self._ptd_first_out is None:
            raise ValueError(f"PTD model has no logits/last_hidden_states output: {self._ptd_out_names[:4]}...")
        if self._ptd_first_out == "last_hidden_states" and lm_head is None:
            raise ValueError("PTD hidden-state output requires a split LM head")
        self._ptd_log: list[dict] = []
        self._ptd_rounds: int = 0
        super().__init__(
            model,
            config,
            max_prompt_tokens,
            max_gen_tokens,
            tokenizer,
            sys_prompt,
            combined_kv_io=combined_kv_io,
            token_embeddings=token_embeddings,
            token_id_lut=token_id_lut,
            lm_head=lm_head,
        )
        self._logger.info(
            "Loaded PTD body '%s' (rows=%d, width=%d, depth=%d)",
            str(model.model_path), ptd_rows, ptd_width, ptd_depth,
        )

    # -- PTD round machinery -------------------------------------------------

    def _ptd_token_feed(self, tokens: np.ndarray) -> dict[str, np.ndarray]:
        if isinstance(self._token_embeddings, np.ndarray):
            return {"token_embedding": self._token_embeddings[tokens][None, ...]}
        return {"input_ids": tokens.astype(np.int64)[None, ...]}

    def _ptd_forward(self, packing) -> dict[str, np.ndarray]:
        feeds = self._ptd_token_feed(packing.tokens)
        feeds.update({
            "position_ids": packing.positions[None, ...],
            "tree_vis": packing.vis[None, None, ...],
            **self._kv_cache,
        })
        return dict(zip(self._ptd_out_names, self._model.infer(feeds)))

    def _ptd_logits(self, out_map: dict) -> np.ndarray:
        first = out_map[self._ptd_first_out]
        if self._lm_head is not None:
            first = self._lm_head.infer({self._hidden_states_name: first})[0]
        return first.reshape(self._ptd_rows, -1)

    def _llm_step(self, token: int, curr_seq_len: int) -> tuple[int, list[np.ndarray]]:
        packing = ptd_core.pack_round(
            curr_seq_len, token, [], [], self._ptd_rows, self._ptd_pad_token,
        )
        out_map = self._ptd_forward(packing)
        next_token = int(self._ptd_row_tokens(self._ptd_logits(out_map))[0])
        return next_token, self._ptd_promote(out_map, packing, [], curr_seq_len)

    def _prefill_prompt(self, prompt_tokens: list[int], start_seq_len: int = 0) -> tuple[int, int]:
        if start_seq_len + len(prompt_tokens) > self._max_gen_tokens:
            raise ValueError("PTD prompt exceeds the remaining cache capacity")
        curr_seq_len = start_seq_len
        next_token = None
        for offset in range(0, len(prompt_tokens), self._ptd_rows):
            chunk = prompt_tokens[offset:offset + self._ptd_rows]
            nodes = []
            for depth, token in enumerate(chunk[1:], start=1):
                nodes.append(ptd_core.PTDNode(token, depth, parent=nodes[-1] if nodes else None))
            packing = ptd_core.pack_round(
                curr_seq_len, chunk[0], nodes, [], self._ptd_rows, self._ptd_pad_token,
            )
            out_map = self._ptd_forward(packing)
            self._update_cache(self._ptd_promote(out_map, packing, nodes, curr_seq_len))
            curr_seq_len += len(chunk)
            # Keep the split head off the prefill path until the final batch.
            if offset + len(chunk) == len(prompt_tokens):
                next_token = int(self._ptd_row_tokens(self._ptd_logits(out_map))[len(chunk) - 1])
        return next_token, curr_seq_len

    def _ptd_row_tokens(self, logits: np.ndarray) -> np.ndarray:
        """argmax per row, remapped through the token ID LUT when trimmed."""
        row_tokens = np.argmax(logits, axis=-1).astype(np.int64)
        if self._token_id_lut is not None:
            if np.any(row_tokens >= len(self._token_id_lut)):
                raise RuntimeError("PTD row token outside token ID LUT")
            row_tokens = self._token_id_lut[row_tokens]
        return row_tokens

    def _ptd_round(self, anchor_token: int, L: int, gen_tokens: list[int]) -> tuple[list[int], bool]:
        """One PTD round. Returns (emitted tokens, stopped). Updates the
        committed cache, draft tree, pool, and metrics."""
        round_start = _perf_counter_ns()
        rows = self._ptd_rows
        free = min(rows, self._max_gen_tokens - L) - 1

        # 1. Retrieve candidates (key = this round's anchor = last emitted
        # token) and allocate the row budget: candidates first, keeping at
        # least one draft row while the draft tree is non-empty.
        candidate_tree = self._ptd_pool.retrieve(anchor_token) if self._ptd_pool else None
        cand_nodes = candidate_tree.bfs_nodes() if candidate_tree is not None else []
        draft_nodes = self._ptd_tree.bfs_nodes()
        if draft_nodes and self._ptd_min_draft_rows > 0:
            n_cand = min(len(cand_nodes), free - self._ptd_min_draft_rows)
        else:
            n_cand = min(len(cand_nodes), free)
        n_draft = min(len(draft_nodes), free - max(n_cand, 0))
        cand_nodes = cand_nodes[: max(n_cand, 0)]
        draft_nodes = draft_nodes[:n_draft]

        # 2. Pack the round.
        packing = ptd_core.pack_round(
            L, anchor_token, cand_nodes, draft_nodes, rows, self._ptd_pad_token
        )

        # 3. Body (+ block head).
        t0 = _perf_counter_ns()
        out_map = self._ptd_forward(packing)
        first = self._ptd_logits(out_map)
        body_ms = (_perf_counter_ns() - t0) / 1e6

        # 4. Greedy verification. The stop adapter matches ordinary greedy:
        # the base _stop_decoding expects the candidate included as the last
        # element of the generated-token list, while the PTD scan hands over
        # the tokens committed *before* the candidate.
        row_tokens = self._ptd_row_tokens(first.reshape(rows, -1))
        max_accepted = self._max_gen_tokens - L - 1
        result = ptd_core.greedy_verify(
            row_tokens, cand_nodes, packing.candidate_slots,
            gen_tokens,
            lambda token, committed: self._stop_decoding(token, list(committed) + [token]),
            max_accepted,
        )

        # 5. Commit: present cache (anchor at L) + promote accepted path KVs.
        new_cache = self._ptd_promote(out_map, packing, result.accepted, L)
        self._update_cache(new_cache)

        # 6. Evolve the draft tree and pool (bounded, request-local):
        # packed draft nodes grow their argmax children, the tree steps and
        # re-anchors on the round's first emitted token, and the pool absorbs
        # the packed nodes' subtrees keyed by token value.
        for node in packing.draft_slots:
            self._ptd_tree.add_child(node, int(row_tokens[packing.draft_slots[node]]))
        self._ptd_tree.step_and_prune()
        self._ptd_tree.reroot(result.emitted[0])
        for node in packing.draft_slots:
            self._ptd_pool.merge_node(node)

        self._ptd_rounds += 1
        self._ptd_log.append({
            "round": self._ptd_rounds,
            "L": L,
            "rows": rows,
            "n_candidates": packing.n_candidates,
            "n_drafts": packing.n_drafts,
            "emitted": len(result.emitted),
            "accepted": len(result.accepted),
            "hit": result.hit,
            "pool_entries": len(self._ptd_pool) if self._ptd_pool else 0,
            "body_ms": body_ms,
            "round_ms": (_perf_counter_ns() - round_start) / 1e6,
        })
        return result.emitted, result.stopped

    def _ptd_promote(self, out_map: dict, packing, accepted: list, L: int) -> list[np.ndarray]:
        """Present caches (anchor committed at L by the model) + accepted-path
        scratch KVs promoted to rows L+1.., in place on the present outputs.

        Returns the updated cache values in ``self._kv_cache`` key order (the
        base ``_update_cache`` zips them by position).
        """
        cache = self._kv_cache
        if self._combined_kv_io:
            for l in range(self._n_layers):
                committed = out_map[f"present.{l}.key_value"]
                scratch = out_map.get(f"present.{l}.key_value_scratch")
                if scratch is None:  # Older exports kept scratch K/V separate.
                    scratch = np.concatenate(
                        [out_map[f"present.{l}.key_scratch"], out_map[f"present.{l}.value_scratch"]],
                        axis=1,
                    )
                for j, node in enumerate(accepted):
                    s = packing.candidate_slots[node] - 1  # scratch row index
                    committed[0, :, L + 1 + j, :] = scratch[0, :, s, :]
            return [out_map[name.replace("past_key_values", "present")] for name in cache]
        for l in range(self._n_layers):
            for role in ("key", "value"):
                scratch = out_map[f"present.{l}.{role}_scratch"]
                committed = out_map[f"present.{l}.{role}"]
                for j, node in enumerate(accepted):
                    s = packing.candidate_slots[node] - 1
                    committed[0, :, L + 1 + j, :] = scratch[0, :, s, :]
        return [out_map[name.replace("past_key_values", "present")] for name in cache]

    def _run(
        self,
        inp_tokens: list[int],
        max_gen_tokens: int | None = None,
    ) -> list[int]:
        if isinstance(max_gen_tokens, int) and 0 <= max_gen_tokens < self._max_gen_tokens:
            self._max_gen_tokens = max_gen_tokens
        inp_tokens = self._format_input_tokens(inp_tokens)
        next_token, curr_seq_len = self._prefill_prompt(inp_tokens, start_seq_len=self._warmup_len)
        gen_tokens = [next_token]

        # Request-local PTD state (fresh draft tree; the pool warms up).
        self._ptd_tree = ptd_core.PTDTree(width=self._ptd_width, max_depth=self._ptd_depth)
        self._ptd_pool = ptd_core.PTDPool(
            max_entry_nodes=self._ptd_pool_max_nodes,
            max_entries=self._ptd_pool_max_entries,
            width=self._ptd_width,
        )
        self._ptd_log = []
        self._ptd_rounds = 0

        M = self._max_gen_tokens
        if self._stop_decoding(next_token, gen_tokens):
            self._n_tokens_gen = len(gen_tokens)
            return gen_tokens
        while True:
            # Same loop gate as the base runner: the cache (size M) is full
            # once curr_seq_len reaches M.
            if curr_seq_len >= M:
                self._logger.warning("Max generation tokens reached, stopping early")
                break
            emitted, stopped = self._ptd_round(next_token, curr_seq_len, gen_tokens)
            gen_tokens.extend(emitted)
            curr_seq_len += len(emitted)  # anchor row + accepted input nodes
            if stopped:
                break
            next_token = emitted[-1]
        self._n_tokens_gen = len(gen_tokens)
        return gen_tokens

    @property
    def ptd_round_log(self) -> list[dict]:
        return self._ptd_log


@dataclass(frozen=True)
class EncoderDecoderConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    start_token_id: int
    end_token_id: int
    encoder_pad_id: int


class EncoderDecoderRunner(ABC):
    def __init__(
        self,
        config: EncoderDecoderConfig,
        max_inp_len: int | None,
        combined_kv_io: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_inp_len = max_inp_len
        self._combined_kv_io = combined_kv_io
        self._n_layers: int = config.n_layers
        self._n_kv_heads: int = config.n_kv_heads
        if self._combined_kv_io:
            self._n_kv_heads *= 2
        self._head_dim: int = config.head_dim
        self._start_token_id: int = config.start_token_id
        self._end_token_id: int = config.end_token_id
        self._encoder_pad_id: int = config.encoder_pad_id
        self._n_tokens_gen: int = 0
        self._infer_times: deque[int] = deque(maxlen=100)
        self._kv_cache = self._init_cache()
        self._all_cache_names: list[str] = list(self._kv_cache)
        self._dec_cache_names: list[str] = [
            k for k in self._all_cache_names if "encoder" not in k
        ]

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else 0.0

    @property
    def max_inp_len(self) -> int | None:
        return self._max_inp_len

    def _init_cache(self) -> dict[str, np.ndarray]:
        if self._combined_kv_io:
            return {
                f"past_key_values.{i}.{a}.key_value": np.zeros(
                    (1, self._n_kv_heads, 1, self._head_dim),
                    dtype=np.float32,
                )
                for i in range(self._n_layers)
                for a in ("decoder", "encoder")
            }
        return {
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self._n_kv_heads, 1, self._head_dim), dtype=np.float32
            )
            for i in range(self._n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }

    def _size_input(self, input: np.ndarray) -> np.ndarray:
        input = input.flatten()
        if len(input) > self._max_inp_len:
            self._logger.warning("Truncating input from %d to %d", len(input), self.max_inp_len)
            input = input[: self._max_inp_len]
        elif len(input) < self._max_inp_len:
            self._logger.info("Padding input from %d to %d", len(input), self.max_inp_len)
            input = np.pad(
                input,
                (0, self._max_inp_len - len(input)),
                constant_values=self._encoder_pad_id,
            )
        return input.reshape((1, self._max_inp_len))

    @abstractmethod
    def run(self, input: np.ndarray, max_tokens: int | None = None) -> np.ndarray: ...


class DynamicEncoderDecoderRunner(EncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        config: EncoderDecoderConfig,
        max_inp_len: int | None = None,
        *,
        decoder_log_name: str = "decoder",
    ):
        super().__init__(config, max_inp_len, combined_kv_io=False)
        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._decoder = decoder
        self._logger.info("Loaded %s '%s'", decoder_log_name, str(self._decoder.model_path))

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        for k, v in zip(self._kv_cache.keys(), new_values):
            if update_all or "decoder" in k:
                self._kv_cache[k] = v

    def _run_decoder(
        self,
        input_tokens: list[int],
        encoder_out: np.ndarray,
        *,
        seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        input_ids = np.array([input_tokens], dtype=np.int64)
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_out,
            **self._kv_cache,
            "use_cache_branch": np.array([seq_len > 0], dtype=np.bool_),
        }
        logits, *cache = self._decoder.infer(decoder_inputs)
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def run(
        self,
        input: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        if max_tokens is None:
            max_tokens = int((input.shape[-1] / 16000) * 6)
        if isinstance(self.max_inp_len, int):
            input = self._size_input(input)

        st = time.time()
        next_token = self._start_token_id
        tokens = [next_token]
        encoder_out = self._encoder.infer({"input_values": input})[0].astype(np.float32)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder([next_token], encoder_out, seq_len=i)
            self._update_cache(cache, update_all=i < 1)

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


class StaticEncoderDecoderRunner(EncoderDecoderRunner):
    def __init__(
        self,
        encoder: InferenceRunner,
        gen_encoder_cache: InferenceRunner | None,
        decoder: InferenceRunner,
        config: EncoderDecoderConfig,
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None,
        *,
        combined_kv_io: bool = True,
        token_embeddings: np.ndarray | None = None,
    ):
        super().__init__(config, max_inp_len, combined_kv_io)
        self._encoder = encoder
        self._logger.info("Loaded encoder '%s'", str(self._encoder.model_path))
        self._gen_encoder_cache = gen_encoder_cache
        if self._gen_encoder_cache is not None:
            self._logger.info("Loaded gen_encoder_cache '%s'", str(self._gen_encoder_cache.model_path))
        else:
            self._logger.info("Encoder cache folded into encoder")
        self._decoder = decoder
        self._logger.info("Loaded decoder '%s'", str(self._decoder.model_path))
        self._preprocessor = preprocessor
        if self._preprocessor is not None:
            self._logger.info("Loaded preprocessor '%s'", str(self._preprocessor.model_path))
        self._max_dec_len = max_dec_len
        self._dec_cache_shapes: dict[str, tuple[int, ...]] = {
            cache_name: (1, self._n_kv_heads, self._max_dec_len, self._head_dim)
            for cache_name in self._dec_cache_names
        }
        for cache_name, shape in self._dec_cache_shapes.items():
            self._kv_cache[cache_name] = np.zeros(shape, dtype=np.float32)
        self._token_embeddings = token_embeddings

    def _pad_cache_tensor(
        self,
        cache_name: str,
        cache_values: np.ndarray,
    ) -> np.ndarray:
        if not (req_shape := self._dec_cache_shapes.get(cache_name)):
            return cache_values
        if cache_values.shape == req_shape:
            return cache_values
        if cache_values.ndim != len(req_shape):
            raise ValueError(
                f"Invalid cache tensor dims: got {cache_values.ndim}, expected {len(req_shape)}"
            )
        pad_width = []
        for cache_dim, req_dim in zip(cache_values.shape, req_shape):
            if cache_dim > req_dim:
                raise ValueError(
                    f"Unexpected dim for cache tensor: {cache_values.shape}, expected: {req_shape}"
                )
            before = 0
            after = req_dim - cache_dim
            pad_width.append((before, after))

        cache_padded = np.pad(
            cache_values, pad_width, mode="constant", constant_values=0
        )
        return cache_padded

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self._all_cache_names if update_all else self._dec_cache_names
        if (curr_len := len(cache_tensors)) != (new_len := len(new_values)):
            raise RuntimeError(
                f"Cache tensors mismatch: expected {curr_len} new values, got {new_len}"
            )
        for k, v in zip(cache_tensors, new_values):
            self._kv_cache[k] = self._pad_cache_tensor(k, v)

    def _get_decoder_token_input(
        self,
        input_tokens: list[int],
    ) -> dict[str, np.ndarray]:
        if isinstance(self._token_embeddings, np.ndarray):
            last_tok = input_tokens[-1]
            return {"token_embedding": np.expand_dims(self._token_embeddings[last_tok], axis=(0, 1))}
        return {"input_ids": np.array([input_tokens], dtype=np.int64)}

    def _run_decoder(
        self,
        input_tokens: list[int],
        *,
        seq_len: int,
    ) -> tuple[int, list[np.ndarray]]:
        decoder_inputs = self._get_decoder_token_input(input_tokens)
        decoder_inputs.update({
            **self._kv_cache,
            "current_len": np.array([[seq_len]], dtype=np.int64),
        })
        logits, *cache = self._decoder.infer(decoder_inputs)
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def run(
        self,
        input: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        self._n_tokens_gen = 0
        max_tokens = (
            max_tokens
            if isinstance(max_tokens, int) and max_tokens < self._max_dec_len
            else self._max_dec_len
        )

        st = time.time()
        next_token = self._start_token_id
        tokens = [next_token]
        input = self._size_input(input)
        if self._preprocessor is not None:
            input = self._preprocessor.infer({"input_values": input})[0]
        encoder_input_name = "input_features" if self._preprocessor is not None else "input_values"
        encoder_out = self._encoder.infer({encoder_input_name: input})

        if self._gen_encoder_cache is not None:
            encoder_hidden_states = encoder_out[0].astype(np.float32)
            encoder_cache_outputs = self._gen_encoder_cache.infer(
                {"encoder_hidden_states": encoder_hidden_states}
            )
        else:
            encoder_cache_outputs = encoder_out

        enc_cache_names = [k for k in self._all_cache_names if "encoder" in k]
        for k, v in zip(enc_cache_names, encoder_cache_outputs):
            self._kv_cache[k] = v

        next_token, cache = self._run_decoder(tokens, seq_len=0)
        self._update_cache(cache)
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder(
                [next_token], seq_len=i + 1
            )
            self._update_cache(cache)

            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


def find_single_data_file(
    model_path: str | os.PathLike,
    pattern: str,
    description: str,
) -> Path | None:
    paths = list(Path(model_path).parent.glob(pattern))
    if not paths:
        return None

    paths = list({p.resolve(): p for p in paths}.values())
    if len(paths) > 1:
        raise RuntimeError(
            f"Expected a single {description} file, found {len(paths)}: {paths}"
        )
    return paths[0]
