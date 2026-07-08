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


class CommonGraphEditsMixin:
    """
    Mixin providing convenience methods for common graph edits.
    
    Must be used with OnnxGraphEditor (defines self._graph, self._graph_name, 
    self._export_dtype, self.apply_edit).
    """

    def replace_dynamic_kv_cache(self, cur_len, max_tokens):
        self.apply_edit(ReplaceDynamicKVCache(self._graph, self._graph_name, cur_len, max_tokens))
        return self

    def mask_future_attn_scores(self, cur_len, max_tokens):
        self.apply_edit(MaskFutureAttentionScores(self._graph, self._graph_name, cur_len, max_tokens, self._export_dtype))
        return self

    def add_curr_len_input(self, cur_len):
        self.apply_edit(AddCurrLenInput(self._graph, self._graph_name, cur_len))
        return self

    def convert_to_static_index(self):
        self.apply_edit(ConvertToStaticIndex(self._graph, self._graph_name))
        return self

    def replace_simplified_layer_norm(self):
        self.apply_edit(ReplaceSimplifiedLayerNorm(self._graph, self._graph_name))
        return self

    def replace_skip_simplified_layer_norm(self):
        self.apply_edit(ReplaceSkipSimplifiedLayerNorm(self._graph, self._graph_name))
        return self

    def replace_group_query_attention(self, num_heads, kv_num_heads, head_dim):
        self.apply_edit(ReplaceGroupQueryAttention(self._graph, self._graph_name, num_heads, kv_num_heads, head_dim))
        return self

    def dequantize_projections_matmul(self, hidden_size, vocab_size):
        self.apply_edit(DequantizeProjectionsMatMul(self._graph, self._graph_name, hidden_size, vocab_size, self._export_dtype))
        return self

    def remove_isNaN(self):
        self.apply_edit(RemoveIsNaN(self._graph, self._graph_name))
        return self

    def remove_redundant_casts(
        self
    ):
        self.apply_edit(RemoveRedundantCasts(self._graph, self._graph_name))
        return self

    def fold_scalar_matmul(self):
        self.apply_edit(FoldScalarMatMul(self._graph, self._graph_name))
        return self
    
    def replace_constant_div_with_mul(self):
        self.apply_edit(ReplaceConstantDivWithMul(self._graph, self._graph_name, self._export_dtype))
        return self

    def replace_int64_float_cast(self, max_int: int):
        self.apply_edit(ReplaceInt64FloatCast(self._graph, self._graph_name, max_int))
        return self

    def broadcast_op_inputs(self, ops, output_idx=0, inputs_idx=None, constants_policy=ConstantBroadcastPolicy.SKIP):
        self.apply_edit(BroadcastOpInputs(self._graph, self._graph_name, ops, output_idx, inputs_idx, constants_policy))
        return self

    def extract_token_embeddings(self, hidden_size, vocab_size, save_to, inp_name="token_embedding"):
        self.apply_edit(ExtractConstantLUT(self._graph, self._graph_name, (vocab_size, hidden_size), save_to, inp_name))
        return self

    def eliminate_expands(self, ops: list[str]):
        self.apply_edit(EliminateExpand(self._graph, self._graph_name, ops))
        return self

    def eliminate_transposes(self):
        self.apply_edit(EliminateTranspose(self._graph, self._graph_name))
        return self

    def collapse_reshape_chains(self):
        self.apply_edit(CollapseReshapeChain(self._graph, self._graph_name))
        return self

    def retarget_cross_attn_key_layout(self):
        self.apply_edit(RetargetCrossAttnKeyLayout(self._graph, self._graph_name))
        return self

    def collapse_gqa_broadcast(self):
        self.apply_edit(CollapseGQABroadcast(self._graph, self._graph_name))
        return self

    def trim_lm_head_vocab(self, kept_token_ids, save_lut=None, output_name="logits", include_argmax=False):
        self.apply_edit(TrimLMHeadVocab(self._graph, self._graph_name, kept_token_ids, output_name, save_lut, include_argmax))
        return self

    def split_lm_head(self, save_to, output_name="logits", hidden_states_name="last_hidden_states"):
        self.apply_edit(SplitLMHead(self._graph, self._graph_name, save_to, output_name, hidden_states_name))
        return self

    def eliminate_rank0_gather(self):
        self.apply_edit(EliminateRank0Gather(self._graph, self._graph_name))
        return self

    def eliminate_singleton_gather_unsqueeze(self):
        self.apply_edit(EliminateSingletonGatherUnsqueeze(self._graph, self._graph_name))
        return self

    def rewrite_negative_pads(self):
        self.apply_edit(RewriteNegativePads(self._graph, self._graph_name))
        return self

    def absorb_padding(self):
        self.apply_edit(AbsorbPadding(self._graph, self._graph_name))
        return self

    def replace_pad_with_concat(self):
        self.apply_edit(ReplacePadWithConcat(self._graph, self._graph_name))
        return self

    def widen_strided_depthwise_conv(self, bus_width_bytes: int = 72, sg_groups_max: int = 4):
        self.apply_edit(
            WidenStridedDepthwiseConv(
                self._graph, self._graph_name, bus_width_bytes, sg_groups_max
            )
        )
        return self

    def decompose_strided_conv1d(self):
        self.apply_edit(DecomposeStridedConv1D(self._graph, self._graph_name))
        return self

    def decompose_bidirectional_rnn(self, max_chunk_len: int | None = None):
        self.apply_edit(
            DecomposeBidirectionalRnn(self._graph, self._graph_name, max_chunk_len)
        )
        return self
