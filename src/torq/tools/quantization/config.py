# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Quantization configuration: per-layer quantization assignments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LayerQuantConfig:
    """Quantization settings for a single layer."""

    bits: int = 8  # 4, 8, or 16 (16 = bf16, no quantization)
    block_size: int = 32

    def __post_init__(self):
        if self.bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8, or 16, got {self.bits}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")


@dataclass
class QuantizationConfig:
    """Per-layer quantization configuration.

    ``layers`` maps layer names (e.g. ``/model/layers.0/attn/q_proj/MatMul``)
    to their quantization settings.  Layers not listed use ``default``.
    """

    default: LayerQuantConfig = field(default_factory=lambda: LayerQuantConfig(bits=8))
    layers: dict[str, LayerQuantConfig] = field(default_factory=dict)

    # --- factories -----------------------------------------------------------

    @classmethod
    def uniform(cls, bits: int, block_size: int = 32) -> QuantizationConfig:
        """All layers use the same quantization."""
        return cls(default=LayerQuantConfig(bits=bits, block_size=block_size))

    # --- per-layer access ----------------------------------------------------

    def get(self, layer_name: str) -> LayerQuantConfig:
        """Return the config for *layer_name*, falling back to ``default``."""
        return self.layers.get(layer_name, self.default)

    # --- serialisation -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "default": asdict(self.default),
            "layers": {k: asdict(v) for k, v in self.layers.items()},
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved quantization config to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> QuantizationConfig:
        """Load config from JSON."""
        data = json.loads(Path(path).read_text())
        default = LayerQuantConfig(**data.get("default", {}))
        layers = {k: LayerQuantConfig(**v) for k, v in data.get("layers", {}).items()}
        return cls(default=default, layers=layers)


@dataclass
class SensitivityResult:
    """Per-layer sensitivity metrics from analysis."""

    layer_name: str
    kl_divergence: float = 0.0
    cosine_similarity: float = 1.0
    top1_match: float = 1.0
    top5_match: float = 1.0
    mse: float = 0.0
    bits_tested: int = 4
    classification: str = "LOW"  # LOW / MEDIUM / HIGH / CRITICAL


@dataclass
class SensitivityResults:
    """Collection of per-layer sensitivity results."""

    layers: list[SensitivityResult] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.layers], indent=2))
        logger.info("Saved sensitivity results (%d layers) to %s", len(self.layers), path)

    @classmethod
    def load(cls, path: str | Path) -> SensitivityResults:
        data = json.loads(Path(path).read_text())
        return cls(layers=[SensitivityResult(**d) for d in data])

    def to_config(
        self,
        bf16_threshold: float = 0.1,
        int8_threshold: float = 0.01,
        block_size: int = 32,
    ) -> QuantizationConfig:
        """Convert sensitivity results to a quantization config.

        Layers with KL divergence above *bf16_threshold* are kept in bf16,
        those above *int8_threshold* use int8, and the rest use int4.
        """
        layers: dict[str, LayerQuantConfig] = {}
        for r in self.layers:
            if r.kl_divergence > bf16_threshold:
                layers[r.layer_name] = LayerQuantConfig(bits=16, block_size=block_size)
            elif r.kl_divergence > int8_threshold:
                layers[r.layer_name] = LayerQuantConfig(bits=8, block_size=block_size)
            else:
                layers[r.layer_name] = LayerQuantConfig(bits=4, block_size=block_size)
        return QuantizationConfig(
            default=LayerQuantConfig(bits=4, block_size=block_size),
            layers=layers,
        )
