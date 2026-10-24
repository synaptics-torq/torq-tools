# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torq.model_export.validation import validate_decoder_only_onnx
from torq.models.gemma3._inference import Gemma3Dynamic, Gemma3Static


def test_tool_prompt_requires_checkpoint_template(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    runner = Gemma3Dynamic.__new__(Gemma3Dynamic)
    with pytest.raises(ValueError, match="source checkpoint's chat template"):
        runner._load_tool_template(str(tmp_path / "tokenizer.json"), [])


@pytest.mark.parametrize("runner_cls", [Gemma3Dynamic, Gemma3Static])
def test_tool_prompt_uses_local_template_and_one_bos(tmp_path, runner_cls):
    backend = Tokenizer(WordLevel({
        "<unk>": 0, "<bos>": 1, "developer": 2, "temperature": 3, "London": 4, "model": 5,
    }, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>", unk_token="<unk>",
        chat_template="{{ bos_token }}{{ messages[0]['role'] }} {{ messages[0]['content'] }} "
                      "{{ tools[0]['function']['name'] }} {{ messages[1]['content'] }} "
                      "{% if add_generation_prompt %}model{% endif %}",
    )
    tokenizer.save_pretrained(tmp_path)
    runner = runner_cls.__new__(runner_cls)
    tools = [{"type": "function", "function": {"name": "temperature"}}]
    runner._load_tool_template(str(tmp_path / "tokenizer.json"), tools)
    runner._bos_token_id = tokenizer.bos_token_id
    runner._max_user_tokens = 100
    runner._max_prompt_tokens = 100
    expected = tokenizer.apply_chat_template([
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": "London"},
    ], tools=tools, add_generation_prompt=True)
    assert runner._warmup_tokens() + runner._prepare_run_input_tokens("London") == expected
    assert {1, 2, 3, 4, 5} <= set(expected)
    runner._max_user_tokens = 0
    with pytest.raises(ValueError, match="tool prompt exceeds"):
        runner._prepare_run_input_tokens("London")


@pytest.mark.parametrize("static_models", [False, True])
def test_validation_passes_tools_and_complete_prompt_limit(static_models):
    runner = Mock(last_infer_time=1, avg_infer_time=1)
    runner.run.return_value = "call:temperature{}"
    static_cls = SimpleNamespace(from_onnx=Mock(return_value=runner))
    dynamic_cls = SimpleNamespace(from_onnx=Mock(return_value=runner))
    tools = [{"type": "function", "function": {"name": "temperature"}}]
    validate_decoder_only_onnx(
        logging.getLogger(__name__), static_cls, dynamic_cls, "model.onnx", "source.onnx",
        static_models=static_models, max_gen_tokens=256, instruct_model=True,
        prefill_size=64, max_inp_len=224, prompts=("London",), n_iters=1,
        runner_kwargs={"tools": tools},
    )
    for call in static_cls.from_onnx.call_args_list + dynamic_cls.from_onnx.call_args_list:
        assert call.kwargs["tools"] == tools
        assert call.kwargs["max_inp_len"] == 224
    assert runner.run.call_count == 2
    runner.run.assert_called_with("London", skip_special_tokens=False)
