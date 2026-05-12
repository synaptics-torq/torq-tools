# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Per-layer quantization sensitivity analysis.

Measures the impact of quantizing each MatMul weight layer independently
by comparing output logits against the fp32 baseline using KL divergence
and other metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from .config import QuantizationConfig, SensitivityResult, SensitivityResults
from .quantize import dequantize_weight, quantize_weight

logger = logging.getLogger(__name__)

# Default calibration prompts (tokenised IDs for Gemma-3)
_DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "The speed of light is approximately",
]


class LayerSensitivityAnalyzer:
    """Analyse per-layer quantization sensitivity of an ONNX model.

    Runs teacher-forced inference: for each MatMul layer the weight is
    quantised → dequantised at a given bit-width, the forward pass is run
    with the same input tokens, and output logit divergence is measured
    against the unmodified fp32 baseline.

    Parameters
    ----------
    model_path : path to the fp32 ONNX model
    tokenizer_path : path to tokenizer.json (Gemma-3 / SentencePiece)
    calibration_prompts : list of text prompts; ``None`` → use defaults
    num_tokens : number of output tokens to evaluate per prompt
    skip_layers : layer-name substrings to skip (e.g. ``["lm_head"]``)
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        calibration_prompts: list[str] | None = None,
        num_tokens: int = 5,
        skip_layers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.prompts = calibration_prompts or _DEFAULT_PROMPTS
        self.num_tokens = num_tokens
        self.skip_layers = skip_layers or []

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

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime is required for sensitivity analysis. "
                "Install with: pip install onnxruntime"
            )

        logger.info("Loading model %s", self.model_path)
        model = onnx.load(str(self.model_path), load_external_data=True)

        # Collect MatMul weight layers
        init_map = {i.name: i for i in model.graph.initializer}
        layers = []
        for node in model.graph.node:
            if node.op_type != "MatMul":
                continue
            if any(s in node.name for s in self.skip_layers):
                continue
            for inp in node.input:
                if inp in init_map:
                    init = init_map[inp]
                    if len(init.dims) == 2 and init.data_type == TensorProto.FLOAT:
                        layers.append((node.name, inp, init))
                        break

        logger.info("Found %d MatMul layers to analyse", len(layers))

        # Tokenise calibration prompts
        token_sequences = self._tokenize_prompts()

        # Run baseline inference (unmodified model)
        logger.info("Running baseline inference...")
        baseline_logits = self._run_inference(model, token_sequences, ort)

        # Per-layer analysis
        results = SensitivityResults()
        for layer_idx, (node_name, weight_name, init) in enumerate(layers):
            fp32_w = numpy_helper.to_array(init).astype(np.float32)

            best_result = None
            for bits in bits_options:
                logger.info(
                    "[%d/%d] Testing %s at %d-bit",
                    layer_idx + 1,
                    len(layers),
                    node_name,
                    bits,
                )
                # Quantize → dequantize this single layer
                qw = quantize_weight(fp32_w, bits)
                deq_w = dequantize_weight(qw)

                # Replace weight in model temporarily
                original_data = init.raw_data
                init.CopyFrom(numpy_helper.from_array(deq_w, name=weight_name))

                # Run inference with modified weight
                modified_logits = self._run_inference(model, token_sequences, ort)

                # Restore original weight
                init.raw_data = original_data

                # Compute metrics
                metrics = self._compute_metrics(baseline_logits, modified_logits)
                result = SensitivityResult(
                    layer_name=node_name,
                    kl_divergence=metrics["kl_divergence"],
                    cosine_similarity=metrics["cosine_similarity"],
                    top1_match=metrics["top1_match"],
                    top5_match=metrics["top5_match"],
                    mse=metrics["mse"],
                    bits_tested=bits,
                    classification=self._classify(metrics["kl_divergence"]),
                )

                if best_result is None or result.kl_divergence < best_result.kl_divergence:
                    best_result = result

            if best_result is not None:
                results.layers.append(best_result)

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

    def _tokenize_prompts(self) -> list[list[int]]:
        """Tokenise calibration prompts.

        Falls back to simple character-level encoding if no tokenizer is
        available.
        """
        if self.tokenizer_path and self.tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(str(self.tokenizer_path))
                return [tok.encode(p).ids for p in self.prompts]
            except ImportError:
                logger.warning(
                    "tokenizers not installed; using fallback encoding"
                )

        # Fallback: encode as UTF-8 byte values (crude but works for testing)
        return [[2] + list(p.encode("utf-8")[:64]) for p in self.prompts]

    def _run_inference(
        self,
        model: onnx.ModelProto,
        token_sequences: list[list[int]],
        ort,
    ) -> list[np.ndarray]:
        """Run inference and collect output logits for each prompt.

        Returns a list of logit arrays, one per prompt.
        """
        # Serialize model to bytes for ORT session
        model_bytes = model.SerializeToString()
        sess = ort.InferenceSession(
            model_bytes,
            providers=["CPUExecutionProvider"],
        )

        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        logits_output = output_names[0]  # First output is typically logits

        all_logits = []
        for tokens in token_sequences:
            prompt_logits = []
            # Build feeds — we only need the first few tokens
            eval_tokens = tokens[: self.num_tokens + 1]

            # Create dummy inputs matching model signature
            feeds = self._build_feeds(sess, eval_tokens)
            if feeds is None:
                continue

            try:
                outputs = sess.run([logits_output], feeds)
                prompt_logits.append(outputs[0].flatten())
            except Exception as e:
                logger.warning("Inference failed for prompt: %s", e)
                continue

            if prompt_logits:
                all_logits.append(np.concatenate(prompt_logits))

        return all_logits

    def _build_feeds(self, sess, tokens: list[int]) -> dict | None:
        """Build input feeds for ORT session from token list."""
        feeds = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dtype_str = inp.type.replace("tensor(", "").replace(")", "")
            dtype_map = {
                "float": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "float16": np.float16,
            }
            dtype = dtype_map.get(dtype_str, np.float32)

            if "input_ids" in inp.name or "token" in inp.name.lower():
                val = np.array(tokens[:shape[-1]], dtype=dtype).reshape(shape)
            elif "attention_mask" in inp.name or "mask" in inp.name.lower():
                val = np.ones(shape, dtype=dtype)
            elif "position" in inp.name:
                val = np.zeros(shape, dtype=dtype)
            else:
                # KV cache or other — fill with zeros
                val = np.zeros(shape, dtype=dtype)
            feeds[inp.name] = val
        return feeds

    @staticmethod
    def _compute_metrics(
        baseline_logits: list[np.ndarray],
        modified_logits: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute divergence metrics between baseline and modified logits."""
        if not baseline_logits or not modified_logits:
            return {
                "kl_divergence": float("inf"),
                "cosine_similarity": 0.0,
                "top1_match": 0.0,
                "top5_match": 0.0,
                "mse": float("inf"),
            }

        kl_divs = []
        cos_sims = []
        top1_matches = []
        top5_matches = []
        mses = []

        for bl, ml in zip(baseline_logits, modified_logits):
            min_len = min(len(bl), len(ml))
            bl, ml = bl[:min_len], ml[:min_len]

            # KL divergence (log-softmax)
            bl_log_p = bl - np.logaddexp.reduce(bl)
            ml_log_q = ml - np.logaddexp.reduce(ml)
            bl_p = np.exp(bl_log_p)
            kl = np.sum(bl_p * (bl_log_p - ml_log_q))
            kl_divs.append(max(0.0, float(kl)))

            # Cosine similarity
            norm_b = np.linalg.norm(bl)
            norm_m = np.linalg.norm(ml)
            if norm_b > 0 and norm_m > 0:
                cos_sims.append(float(np.dot(bl, ml) / (norm_b * norm_m)))
            else:
                cos_sims.append(0.0)

            # Top-1 match
            top1_matches.append(1.0 if np.argmax(bl) == np.argmax(ml) else 0.0)

            # Top-5 match
            top5_b = set(np.argsort(bl)[-5:])
            top5_m = set(np.argsort(ml)[-5:])
            top5_matches.append(len(top5_b & top5_m) / 5.0)

            # MSE
            mses.append(float(np.mean((bl - ml) ** 2)))

        return {
            "kl_divergence": float(np.mean(kl_divs)) if kl_divs else float("inf"),
            "cosine_similarity": float(np.mean(cos_sims)) if cos_sims else 0.0,
            "top1_match": float(np.mean(top1_matches)) if top1_matches else 0.0,
            "top5_match": float(np.mean(top5_matches)) if top5_matches else 0.0,
            "mse": float(np.mean(mses)) if mses else float("inf"),
        }

    @staticmethod
    def _classify(kl_divergence: float) -> str:
        if kl_divergence > 1.0:
            return "CRITICAL"
        if kl_divergence > 0.1:
            return "HIGH"
        if kl_divergence > 0.01:
            return "MEDIUM"
        return "LOW"
