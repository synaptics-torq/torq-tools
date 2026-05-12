# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Per-layer quantization sensitivity analysis.

Uses teacher-forced autoregressive evaluation: for each MatMul weight layer,
the weight is quantised → dequantised at a given bit-width, the model is
run step-by-step feeding baseline tokens, and output logit divergence is
measured against the fp32 baseline.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .config import SensitivityResult, SensitivityResults
from .quantize import dequantize_weight, quantize_weight

logger = logging.getLogger(__name__)

# Gemma-3 defaults
_BOS_ID = 2
_EOS_ID = 1
_END_TURN_ID = 106
_HEAD_DIM = 256
_MAX_SEQ = 256
_SYS_PROMPT = "You are a helpful AI assistant named Gemma. Answer in 1-2 sentences."

_DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "The speed of light is approximately",
]


class LayerSensitivityAnalyzer:
    """Analyse per-layer quantization sensitivity of an ONNX model.

    Uses teacher-forced autoregressive evaluation: process prompt tokens
    one-by-one with KV cache, generate N tokens, and compare logits against
    the unmodified baseline using KL divergence.

    Parameters
    ----------
    model_path : path to the fp32/bf16 ONNX model
    embeddings_path : path to token_embeddings.npy
    tokenizer_path : path to tokenizer.json (Gemma-3 / SentencePiece)
    calibration_prompts : list of text prompts; ``None`` → use defaults
    num_tokens : number of generation tokens to evaluate per prompt
    num_layers : number of transformer layers (for KV cache init)
    skip_layers : layer-name substrings to skip (e.g. ``["lm_head"]``)
    """

    def __init__(
        self,
        model_path: str | Path,
        embeddings_path: str | Path,
        tokenizer_path: str | Path | None = None,
        token_lut_path: str | Path | None = None,
        calibration_prompts: list[str] | None = None,
        num_tokens: int = 5,
        num_layers: int = 18,
        skip_layers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.embeddings_path = Path(embeddings_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.token_lut_path = Path(token_lut_path) if token_lut_path else None
        self.prompts = calibration_prompts or _DEFAULT_PROMPTS
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.skip_layers = skip_layers or []
        self._embeddings: np.ndarray | None = None
        self._token_lut: np.ndarray | None = None  # reduced_idx → full_vocab_id
        self._reverse_lut: dict[int, int] | None = None  # full_vocab_id → reduced_idx
        self._tokenizer = None

    # --- public API ----------------------------------------------------------

    def analyze(
        self,
        bits_options: list[int] | None = None,
        output_path: str | Path | None = None,
        config_output_path: str | Path | None = None,
        bf16_threshold: float = 0.1,
        int8_threshold: float = 0.01,
    ) -> SensitivityResults:
        """Run sensitivity analysis.

        Parameters
        ----------
        bits_options : list of bit-widths to test (default ``[4, 8]``)
        output_path : save detailed results JSON here
        config_output_path : save quantization config JSON here
        bf16_threshold : KL threshold above which layers stay bf16
        int8_threshold : KL threshold above which layers use int8

        Returns
        -------
        SensitivityResults with per-layer metrics
        """
        bits_options = bits_options or [4, 8]
        import onnxruntime as ort

        # Load embeddings
        emb = np.load(str(self.embeddings_path))
        if emb.dtype != np.float32:
            u16 = emb.view(np.uint16)
            fp32 = np.zeros(u16.shape, dtype=np.float32)
            fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
            emb = fp32
        self._embeddings = emb

        # Load token LUT for reduced-vocab models
        if self.token_lut_path and self.token_lut_path.exists():
            self._token_lut = np.load(str(self.token_lut_path)).astype(np.int64)
            self._reverse_lut = {
                int(full_id): idx for idx, full_id in enumerate(self._token_lut)
            }
            logger.info(
                "Loaded token LUT: %d reduced vocab entries", len(self._token_lut)
            )

        # Tokenize prompts
        token_sequences = self._tokenize_prompts()
        logger.info("Tokenized %d prompts", len(token_sequences))

        # Load and convert model to fp32 for CPU inference
        logger.info("Loading model %s", self.model_path)
        model_proto = onnx.load(str(self.model_path), load_external_data=True)
        fp32_model = self._convert_to_fp32(model_proto)

        # Create baseline session
        logger.info("Creating baseline session...")
        base_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx.save(fp32_model, base_tmp.name)
        base_tmp.close()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 4
        base_sess = ort.InferenceSession(
            base_tmp.name, opts, providers=["CPUExecutionProvider"]
        )

        # Collect baseline logits (teacher-forced)
        logger.info("Collecting baseline logits...")
        baseline_all = []
        for i, ids in enumerate(token_sequences):
            t0 = time.time()
            bl = self._collect_logits(base_sess, ids, self.num_tokens)
            elapsed = (time.time() - t0) * 1000
            tokens_text = self._decode_tokens([t for t, _ in bl])
            logger.info(
                "  Prompt %d: %.0fms, %d tokens: \"%s\"",
                i + 1, elapsed, len(bl), tokens_text[:60],
            )
            baseline_all.append(bl)
        del base_sess

        # Get baseline reference tokens for teacher forcing
        ref_tokens_all = [[t for t, _ in bl] for bl in baseline_all]

        # Find MatMul weight layers
        init_map = {i.name: i for i in fp32_model.graph.initializer}
        fp32_init_idx = {
            i.name: idx for idx, i in enumerate(fp32_model.graph.initializer)
        }
        layers = []
        for node in fp32_model.graph.node:
            if node.op_type != "MatMul":
                continue
            if any(s in node.name for s in self.skip_layers):
                continue
            for inp in node.input:
                if inp in init_map:
                    init = init_map[inp]
                    if len(init.dims) == 2:
                        layers.append(
                            (node.name, inp, fp32_init_idx[inp])
                        )
                        break

        logger.info("Found %d MatMul layers to test", len(layers))

        # Per-layer sensitivity testing
        results = SensitivityResults()
        for layer_idx, (node_name, weight_name, init_idx) in enumerate(layers):
            original_init = fp32_model.graph.initializer[init_idx]
            original_data = original_init.raw_data
            fp32_w = numpy_helper.to_array(original_init).astype(np.float32)

            layer_kl = {}
            layer_cos = {}
            layer_top1 = {}

            for bits in bits_options:
                logger.info(
                    "[%d/%d] Testing %s at %d-bit",
                    layer_idx + 1, len(layers), node_name, bits,
                )

                # Quantize → dequantize this layer
                qw = quantize_weight(fp32_w, bits)
                deq_w = dequantize_weight(qw)

                # Replace weight in model
                new_init = numpy_helper.from_array(deq_w, name=weight_name)
                fp32_model.graph.initializer[init_idx].CopyFrom(new_init)

                # Save modified model and create session
                onnx.save(fp32_model, base_tmp.name)
                mod_sess = ort.InferenceSession(
                    base_tmp.name, opts, providers=["CPUExecutionProvider"]
                )

                # Collect logits with teacher forcing (using baseline tokens)
                kl_divs = []
                cos_sims = []
                top1_matches = []
                for prompt_idx, ids in enumerate(token_sequences):
                    ref_tokens = ref_tokens_all[prompt_idx]
                    mod_results = self._collect_logits(
                        mod_sess, ids, self.num_tokens, ref_tokens=ref_tokens,
                    )

                    # Compare per-step logits
                    bl_results = baseline_all[prompt_idx]
                    for step_idx in range(min(len(bl_results), len(mod_results))):
                        bl_logits = bl_results[step_idx][1]
                        mod_logits = mod_results[step_idx][1]
                        kl_divs.append(_kl_divergence(bl_logits, mod_logits))
                        cos_sims.append(_cosine_similarity(bl_logits, mod_logits))
                        top1_matches.append(
                            1.0 if bl_logits.argmax() == mod_logits.argmax() else 0.0
                        )

                del mod_sess

                mean_kl = float(np.mean(kl_divs)) if kl_divs else float("inf")
                mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
                mean_top1 = float(np.mean(top1_matches)) if top1_matches else 0.0

                layer_kl[bits] = mean_kl
                layer_cos[bits] = mean_cos
                layer_top1[bits] = mean_top1

                sev = _classify(mean_kl)
                logger.info(
                    "  %d-bit: KL=%.6f cos=%.6f top1=%.2f [%s]",
                    bits, mean_kl, mean_cos, mean_top1, sev,
                )

                # Restore original weight
                original_init_restored = onnx.TensorProto()
                original_init_restored.name = weight_name
                original_init_restored.dims[:] = list(fp32_w.shape)
                original_init_restored.data_type = TensorProto.FLOAT
                original_init_restored.raw_data = original_data
                fp32_model.graph.initializer[init_idx].CopyFrom(
                    original_init_restored
                )

            # Classify by worst-case (int4) KL
            worst_kl = max(layer_kl.values()) if layer_kl else 0.0
            result = SensitivityResult(
                layer_name=node_name,
                kl_divergence=layer_kl,
                cosine_similarity=layer_cos,
                top1_match=layer_top1,
                classification=_classify(worst_kl),
            )
            results.layers.append(result)

            # Summary line
            parts = [f"{b}bit={layer_kl[b]:.6f}" for b in sorted(layer_kl)]
            logger.info(
                "  → %s: %s [%s]", node_name, ", ".join(parts), result.classification,
            )

        # Cleanup
        os.unlink(base_tmp.name)

        # Summary
        classifications = {}
        for r in results.layers:
            classifications[r.classification] = (
                classifications.get(r.classification, 0) + 1
            )
        logger.info("Sensitivity summary: %s", classifications)

        # Save results
        if output_path:
            results.save(output_path)
        if config_output_path:
            config = results.to_config(
                bf16_threshold=bf16_threshold,
                int8_threshold=int8_threshold,
            )
            config.save(config_output_path)
            logger.info("Saved quantization config to %s", config_output_path)

        return results

    # --- internals -----------------------------------------------------------

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.tokenizer_path and self.tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
                return self._tokenizer
            except ImportError:
                logger.warning("tokenizers not installed")
        return None

    def _tokenize_prompts(self) -> list[list[int]]:
        """Tokenise calibration prompts using Gemma chat template."""
        tok = self._load_tokenizer()
        if tok is None:
            raise RuntimeError(
                "Tokenizer is required for sensitivity analysis. "
                "Provide --tokenizer path/to/tokenizer.json"
            )

        def encode_turn(content, role):
            """Encode a single turn in Gemma chat format."""
            start_turn = tok.decode([105], skip_special_tokens=False)
            end_turn = tok.decode([106], skip_special_tokens=False)
            if role == "model":
                text = start_turn + "model\n"
            else:
                text = start_turn + role + "\n" + content + end_turn + "\n"
            ids = tok.encode(text).ids
            if ids and ids[0] == _BOS_ID:
                ids = ids[1:]
            return ids

        all_ids = []
        for prompt in self.prompts:
            ids = (
                [_BOS_ID]
                + encode_turn(_SYS_PROMPT, "system")
                + encode_turn(prompt, "user")
                + encode_turn("", "model")
            )
            all_ids.append(ids)
        return all_ids

    def _decode_tokens(self, token_ids: list[int]) -> str:
        tok = self._load_tokenizer()
        if tok:
            return tok.decode(token_ids)
        return str(token_ids)

    def _reduced_to_full(self, reduced_idx: int) -> int:
        """Map reduced-vocab index to full-vocab token ID."""
        if self._token_lut is not None:
            return int(self._token_lut[reduced_idx])
        return reduced_idx

    def _full_to_reduced(self, full_id: int) -> int | None:
        """Map full-vocab token ID to reduced-vocab index (None if not in vocab)."""
        if self._reverse_lut is not None:
            return self._reverse_lut.get(full_id)
        return full_id

    def _collect_logits(
        self,
        sess,
        prompt_ids: list[int],
        n_gen: int,
        ref_tokens: list[int] | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Run autoregressive inference, returning (full_token_id, logits) per step.

        Parameters
        ----------
        sess : ORT session
        prompt_ids : tokenized prompt (full-vocab token IDs)
        n_gen : number of tokens to generate after prompt
        ref_tokens : if provided, feed these full-vocab token IDs instead of
                     model predictions (teacher forcing)
        """
        emb = self._embeddings
        kv = {
            f"past_key_values.{i}.key_value": np.zeros(
                (1, 2, _MAX_SEQ, _HEAD_DIM), dtype=np.float32
            )
            for i in range(self.num_layers)
        }
        out_names = [o.name for o in sess.get_outputs()]

        pos = 0
        logits = None

        # Process prompt tokens (full-vocab IDs → embedding lookup)
        for tok in prompt_ids:
            logits = self._step(sess, emb, kv, tok, pos, out_names)
            pos += 1

        # Generate tokens
        results = []
        # logits are over reduced vocab; argmax gives reduced index
        reduced_idx = int(logits.argmax())
        full_tok = self._reduced_to_full(reduced_idx)
        results.append((full_tok, logits.copy()))

        for i in range(n_gen - 1):
            # Teacher forcing: ref_tokens are full-vocab IDs
            feed_tok = ref_tokens[i] if ref_tokens and i < len(ref_tokens) else full_tok
            if feed_tok in (_EOS_ID, _END_TURN_ID):
                break
            logits = self._step(sess, emb, kv, feed_tok, pos, out_names)
            pos += 1
            reduced_idx = int(logits.argmax())
            full_tok = self._reduced_to_full(reduced_idx)
            results.append((full_tok, logits.copy()))

        return results

    @staticmethod
    def _step(sess, emb, kv, token_id, pos, out_names):
        """Run a single autoregressive step."""
        e = emb[token_id].astype(np.float32).reshape(1, 1, -1)
        feeds = {
            "token_embedding": e,
            "position_ids": np.array([[pos]], dtype=np.int64),
        }
        feeds.update(kv)
        outs = sess.run(None, feeds)
        # Update KV cache
        for i, name in enumerate(out_names[1:], 1):
            kv[name.replace("present.", "past_key_values.")] = outs[i]
        return outs[0][0, -1]  # logits for last position

    @staticmethod
    def _convert_to_fp32(model_proto: onnx.ModelProto) -> onnx.ModelProto:
        """Convert a bf16 model to fp32 for CPU inference with ORT."""
        new_inits = []
        for init in model_proto.graph.initializer:
            if init.data_type == TensorProto.BFLOAT16:
                dims = list(init.dims)
                n = 1
                for d in dims:
                    n *= d
                raw = init.raw_data
                if len(raw) == n * 2:
                    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
                    fp32 = np.zeros(dims, dtype=np.float32)
                    fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
                else:
                    fp32 = np.frombuffer(raw, dtype=np.float32).reshape(dims)
                new_inits.append(numpy_helper.from_array(fp32, name=init.name))
            else:
                new_inits.append(init)

        new_inputs = []
        for inp in model_proto.graph.input:
            t = inp.type.tensor_type
            if t.elem_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
                shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.shape.dim
                ]
                new_inputs.append(
                    helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, shape)
                )
            else:
                new_inputs.append(inp)

        new_outputs = []
        for out in model_proto.graph.output:
            t = out.type.tensor_type
            if t.elem_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
                shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.shape.dim
                ]
                new_outputs.append(
                    helper.make_tensor_value_info(out.name, TensorProto.FLOAT, shape)
                )
            else:
                new_outputs.append(out)

        new_nodes = []
        for node in model_proto.graph.node:
            if node.op_type == "Cast":
                new_attrs = []
                for attr in node.attribute:
                    if attr.name == "to" and attr.i in (
                        TensorProto.BFLOAT16, TensorProto.FLOAT16
                    ):
                        new_attr = onnx.AttributeProto()
                        new_attr.name = "to"
                        new_attr.type = onnx.AttributeProto.INT
                        new_attr.i = TensorProto.FLOAT
                        new_attrs.append(new_attr)
                    else:
                        new_attrs.append(attr)
                nn = helper.make_node(
                    node.op_type, list(node.input), list(node.output), name=node.name
                )
                nn.attribute.extend(new_attrs)
                new_nodes.append(nn)
            else:
                new_nodes.append(node)

        fp32_graph = helper.make_graph(
            new_nodes, model_proto.graph.name, new_inputs, new_outputs, new_inits
        )
        fp32_model = helper.make_model(fp32_graph)
        fp32_model.ir_version = model_proto.ir_version
        del fp32_model.opset_import[:]
        for op in model_proto.opset_import:
            fp32_model.opset_import.append(op)

        return fp32_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL divergence D(P || Q) from logits."""
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    p -= p.max()
    q -= q.max()
    p_exp = np.exp(p)
    q_exp = np.exp(q)
    p_sum = p_exp.sum()
    q_sum = q_exp.sum()
    log_p = p - np.log(p_sum)
    log_q = q - np.log(q_sum)
    p_prob = p_exp / p_sum
    kl = float(np.sum(p_prob * (log_p - log_q)))
    return max(0.0, kl)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _classify(kl_divergence: float) -> str:
    if kl_divergence > 1.0:
        return "CRITICAL"
    if kl_divergence > 0.1:
        return "HIGH"
    if kl_divergence > 0.01:
        return "MEDIUM"
    return "LOW"
