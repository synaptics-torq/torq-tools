# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

from .arithmetic import (
    DecomposeLayerNormalization,
    DequantizeProjectionsMatMul,
    FoldScalarMatMul,
    RemoveIsNaN,
    RemoveRedundantCasts,
    ReplaceConstantDivWithMul,
    ReplaceInt64FloatCast,
)
from .artifacts import ComputeDequantizedLUT, ExtractConstantLUT, MaterializeTiedLMHead, SplitLMHead, TakeLastToken, TrimLMHeadVocab
from .conv import DecomposeStridedConv1D, FoldConvBatchNorm, WidenStridedDepthwiseConv
from .mixins import CommonGraphEditsMixin, DecoderGraphEditor
from .padding import AbsorbPadding, ReplacePadWithConcat, RewriteNegativePads
from .rnn import DecomposeBidirectionalRnn
from .shape import (
    BroadcastOpInputs,
    CollapseReshapeChain,
    CollapseUnrolledConcat,
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
    GroupQueryHeads,
    MaskFutureAttentionScores,
    PrecomputeRotaryTables,
    ReadOnlyKVCache,
    ReplaceDynamicKVCache,
    ReplaceLegacyRotaryCache,
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
    "GroupQueryHeads",
    "ReadOnlyKVCache",
    "PrecomputeRotaryTables",
    "ReplaceLegacyRotaryCache",
    "DequantizeProjectionsMatMul",
    "DecomposeLayerNormalization",
    "RemoveIsNaN",
    "RemoveRedundantCasts",
    "FoldScalarMatMul",
    "ReplaceConstantDivWithMul",
    "ReplaceInt64FloatCast",
    "ConstantBroadcastPolicy",
    "BroadcastOpInputs",
    "ExtractConstantLUT",
    "ComputeDequantizedLUT",
    "EliminateExpand",
    "EliminateTranspose",
    "RetargetCrossAttnKeyLayout",
    "CollapseReshapeChain",
    "CollapseUnrolledConcat",
    "CollapseGQABroadcast",
    "MaterializeTiedLMHead",
    "TrimLMHeadVocab",
    "SplitLMHead",
    "TakeLastToken",
    "EliminateRank0Gather",
    "EliminateSingletonGatherUnsqueeze",
    "RewriteNegativePads",
    "AbsorbPadding",
    "ReplacePadWithConcat",
    "WidenStridedDepthwiseConv",
    "DecomposeStridedConv1D",
    "FoldConvBatchNorm",
    "DecomposeBidirectionalRnn",
    "CombineKVCacheMixin",
    "CommonGraphEditsMixin",
    "DecoderGraphEditor",
    "ReplaceSimplifiedLayerNorm",
    "ReplaceSkipSimplifiedLayerNorm",
    "ReplaceGroupQueryAttention",
]
