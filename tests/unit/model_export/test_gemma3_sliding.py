# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from types import SimpleNamespace

import numpy as np
import pytest

from torq.models.gemma3.export import Gemma3ModelExporter
from torq.model_export.validation import _validate_long_context


@pytest.mark.parametrize("capacity", [4, 16])
@pytest.mark.parametrize("legacy", [False, True])
def test_gemma_sliding_configuration(capacity, legacy):
    exporter = object.__new__(Gemma3ModelExporter)
    config = dict(sliding_window=4, num_hidden_layers=3, sliding_window_pattern=3)
    if not legacy:
        config["layer_types"] = ["sliding_attention", "sliding_attention", "full_attention"]
    exporter._broadcast_ops = None
    exporter._extract_embeddings = False
    exporter._trim_vocab = False
    exporter._split_lm_head = False
    exporter._batch_prefill = None
    exporter._config = SimpleNamespace(**config)
    exporter._max_gen_tokens = capacity
    expected = (4, frozenset({0, 1})) if capacity > 4 else (None, frozenset())
    assert exporter._sliding_attention() == expected
    spec = exporter.graph_edit_blocks()["model.static (KV cache)"][1]
    assert spec.args[-2:] == expected


def test_long_context_uses_reference_tokens_and_checks_boundary():
    class Runner:
        def __init__(self, fail=False):
            self.seen = []
            self.fail = fail

        def _tokenize_input(self, text):
            return [1, 2, 3]

        def _init_cache(self):
            self.seen.clear()
            return {}

        def sample_next_token(self, logits):
            return int(logits.argmax())

        def _llm_step(self, token, position):
            self.seen.append(token)
            logits = np.array([1., 2., 3.])
            if self.fail and position == 4:
                logits[0] += 0.1  # Same top token, different attention output.
            return self.sample_next_token(logits), []

        def _update_cache(self, cache, position):
            pass

    logger = SimpleNamespace(info=lambda *args: None)
    runner, reference = Runner(), Runner()
    _validate_long_context(logger, runner, reference, 4, 16)
    assert runner.seen == reference.seen
    assert len(runner.seen) == 16
    with pytest.raises(ValueError, match="position 4"):
        _validate_long_context(logger, Runner(fail=True), reference, 4, 16)
