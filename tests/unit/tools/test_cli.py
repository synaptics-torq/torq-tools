# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import pytest

from torq.utils.cli import parse_remainder_args_to_dict, promote_string_val


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("-7", -7),
        ("3.5", 3.5),
        ("1", 1),
        ("0", 0),
        ("true", True),
        ("True", True),
        ("FALSE", False),
        ("quant", "quant"),
        ("", ""),
    ],
)
def test_promote_string_val(raw, expected):
    result = promote_string_val(raw)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("args", [None, []])
def test_parse_remainder_args_to_dict_empty(args):
    assert parse_remainder_args_to_dict(args, "extra") == {}


def test_parse_remainder_args_to_dict_pairs():
    parsed = parse_remainder_args_to_dict(
        ["op_types_to_quantize", "MatMul", "reduce_range", "False"], "extra"
    )
    assert parsed == {"op_types_to_quantize": "MatMul", "reduce_range": False}


def test_parse_remainder_args_to_dict_odd_count_raises():
    with pytest.raises(ValueError, match="Trailing or missing value"):
        parse_remainder_args_to_dict(["per_channel"], "extra")
