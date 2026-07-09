# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from .arithmetic import (
    DequantizeProjectionsMatMul,
    FoldScalarMatMul,
    RemoveIsNaN,
    RemoveRedundantCasts,
    ReplaceConstantDivWithMul,
    ReplaceInt64FloatCast,
)
from .artifacts import ExtractConstantLUT, SplitLMHead, TrimLMHeadVocab
from .conv import DecomposeStridedConv1D, WidenStridedDepthwiseConv
from .mixins import CommonGraphEditsMixin
from .padding import AbsorbPadding, ReplacePadWithConcat, RewriteNegativePads
from .rnn import DecomposeBidirectionalRnn
from .shape import (
    BroadcastOpInputs,
    CollapseReshapeChain,
    ConstantBroadcastPolicy,
    EliminateExpand,
    EliminateRank0Gather,
    EliminateSingletonGatherUnsqueeze,
    EliminateTranspose,
)
from .transformer import (
    AddCurrLenInput,
    CollapseGQABroadcast,
    CombineKVCacheMixin,
    ConvertToStaticIndex,
    MaskFutureAttentionScores,
    ReplaceDynamicKVCache,
    RetargetCrossAttnKeyLayout,
)
from .custom_ops import (
    ReplaceGroupQueryAttention,
    ReplaceSimplifiedLayerNorm,
    ReplaceSkipSimplifiedLayerNorm,
)

__all__ = [
    "ReplaceDynamicKVCache",
    "MaskFutureAttentionScores",
    "AddCurrLenInput",
    "ConvertToStaticIndex",
    "DequantizeProjectionsMatMul",
    "RemoveIsNaN",
    "RemoveRedundantCasts",
    "FoldScalarMatMul",
    "ReplaceConstantDivWithMul",
    "ReplaceInt64FloatCast",
    "ConstantBroadcastPolicy",
    "BroadcastOpInputs",
    "ExtractConstantLUT",
    "EliminateExpand",
    "EliminateTranspose",
    "RetargetCrossAttnKeyLayout",
    "CollapseReshapeChain",
    "CollapseGQABroadcast",
    "TrimLMHeadVocab",
    "SplitLMHead",
    "EliminateRank0Gather",
    "EliminateSingletonGatherUnsqueeze",
    "RewriteNegativePads",
    "AbsorbPadding",
    "ReplacePadWithConcat",
    "WidenStridedDepthwiseConv",
    "DecomposeStridedConv1D",
    "DecomposeBidirectionalRnn",
    "CombineKVCacheMixin",
    "CommonGraphEditsMixin",
    "ReplaceSimplifiedLayerNorm",
    "ReplaceSkipSimplifiedLayerNorm",
    "ReplaceGroupQueryAttention",
]
