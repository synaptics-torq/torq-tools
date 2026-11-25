# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from types import SimpleNamespace

import numpy as np
import pytest

from torq.model_export.validation import _validate_long_context


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
