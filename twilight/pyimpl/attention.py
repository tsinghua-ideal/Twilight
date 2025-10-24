# This is a Python implementation of Several Approximate Attention Algorithms, hack
# the Llama

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.

# Reference:
#   - https://github.com/mit-han-lab/Quest/blob/main/evaluation/quest_attention.py


import json
import math
from typing import Optional, Tuple, Dict

import torch
from torch import nn

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention

# Prefill compression policies
from .snap_kv import snap_kv_compression, prepare_inputs_for_generation_llama_snapkv

# Index selector policies
from .streaming_llm import streaming_selector
from .quest import quest_selector
from .top_k import oracle_topk_selector
from .tidal_decode import tidal_decode_selector
from .sparq import sparq_selector
from .double_sparse import double_sparse_selector, init_model_channel_config

# Weight estimator policies
from .top_p import top_p_unnormalized
from .elementwise_threshold import elementwise_threshold
from .quantize import min_max_per_token_quant_kv, max_per_token_quant_kv

# States
from .state import (
    CompressorType,
    IndexSelectorType,
    WeightEstimatorType,
    WeightPrunerType,
    LocalState,
    HistoryBudgetInfo,
    HistoryAccumulatedScoreInfo,
)


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    torch.cuda.empty_cache()

    bsz, q_len, _ = hidden_states.size()

    # Shape conversion:
    #   Q: [bs, num_heads, q_len, head_dim]
    #   K: [bs, num_kv_heads, q_len, head_dim]
    #   V: [bs, num_kv_heads, q_len, head_dim]
    bs, q_len, _ = hidden_states.size()

    # print(q_len)

    query_states = (
        self.q_proj(hidden_states)
        .view(bs, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bs, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bs, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):  # [SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Apply rotary embedding
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Concat current KV with past key value cache
    # New cache format
    # if isinstance(past_key_value, DynamicCache):
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos}
        if q_len == 1:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            # KV cache compression in prefill stage
            cache_key, cache_value = key_states, value_states
            if self.state.compressor == CompressorType.SNAP_KV:
                cache_key, cache_value = snap_kv_compression(
                    query_states,
                    key_states,
                    value_states,
                    **self.state.compressor_args,
                )
            past_key_value.update(cache_key, cache_value, self.layer_idx, cache_kwargs)
    # Legacy cache format
    # else:
    #     if past_key_value is not None:
    #         # reuse k, v, self_attention
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     past_key_value = (key_states, value_states) if use_cache else None

    # Prefill stage
    if q_len > 1:
        # Flash Attention's layout: [bs, seqlen, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=False,  # flash-attn >= 2.1
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # Currently we assert Attention Mask to be None
    assert attention_mask is None

    # Computate full attention weights, which is OK (will not OOM) for decode
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    # torch.set_printoptions(profile="full")
    # print(f"Layer {self.layer_id}, Real Weights: ", attn_weights)

    start_layer = 2 if self.state.skip_first_two_layers else 0

    if self.layer_id >= start_layer:
        # Index Selector
        if self.state.selector == IndexSelectorType.NONE:
            mask = torch.ones_like(attn_weights, dtype=torch.bool)
        elif self.state.selector == IndexSelectorType.STREAMING:
            mask = streaming_selector(
                attn_weights,
                self.state.selector_args["token_budget"],
                self.state.selector_args["num_sinks"],
            )
        elif self.state.selector == IndexSelectorType.QUEST:
            mask = quest_selector(
                self,
                query_states,
                key_states,
                **self.state.selector_args,
            )
        elif self.state.selector == IndexSelectorType.ORACLE_TOPK:
            mask = oracle_topk_selector(
                attn_weights,
                self.state.selector_args["token_budget"],
            )
        elif self.state.selector == IndexSelectorType.TIDAL_DECODE:
            mask = tidal_decode_selector(
                self.state,
                attn_weights,
                self.state.selector_args["token_budget"],
                self.layer_id,
                self.state.selector_args["reselection_layers"],
            )
        elif self.state.selector == IndexSelectorType.SPARQ:
            mask = sparq_selector(
                query_states,
                key_states,
                **self.state.selector_args,
            )
        elif self.state.selector == IndexSelectorType.DS:
            selector_args = self.state.selector_args.copy()
            selector_args.pop("config_path")
            selector_args.pop("selected_channel")
            mask = double_sparse_selector(
                query_states,
                key_states,
                self.sorted_channel,
                **selector_args,
            )

        # Weight Estimator
        if self.state.weight_estimator == WeightEstimatorType.NONE:
            estimated_weights = attn_weights
        elif self.state.weight_estimator == WeightEstimatorType.MIN_MAX_QUANT:
            quantized_key = min_max_per_token_quant_kv(
                key_states,
                self.state.weight_estimator_args["quant_bit"],
            )
            estimated_weights = torch.matmul(
                query_states, quantized_key.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
        elif self.state.weight_estimator == WeightEstimatorType.MAX_QUANT:
            quantized_key = max_per_token_quant_kv(
                key_states,
                self.state.weight_estimator_args["quant_bit"],
                self.state.weight_estimator_args["smooth_k"],
            )
            estimated_weights = torch.matmul(
                query_states, quantized_key.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)

        # Prune the weights
        # The input is un-normalized
        if self.state.weight_pruner == WeightPrunerType.THRESHOLD:
            pruned_mask = elementwise_threshold(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )
        elif self.state.weight_pruner == WeightPrunerType.TOP_P:
            pruned_mask = top_p_unnormalized(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )

        if self.state.weight_pruner != WeightPrunerType.NONE:
            mask = mask & pruned_mask

        # Record budget
        if self.state.budget_info is not None:
            self.state.budget_info.update(self.layer_id, mask)

            # if self.layer_id == 31:
            #     # print(self.state.budget_info.get_avg_budget_cur_query())
            #     self.state.budget_info.print_budget_info_single_query_with_B0() / 0

        # Record the score
        if self.state.score_info is not None:
            self.state.score_info.update(self.layer_id, attn_weights, mask)

        if self.state.use_estimated_weights_in_attn:
            attn_weights = estimated_weights

        # Mask unselected tokens
        attn_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    # Evaluate cos similarity

    # attn_weights_real = torch.matmul(
    #     query_states, key_states.transpose(2, 3)
    # ) / math.sqrt(self.head_dim)
    # attn_weights_real = nn.functional.softmax(
    #     attn_weights_real, dim=-1, dtype=torch.float32
    # ).to(query_states.dtype)
    # attn_output_real = torch.matmul(attn_weights_real, value_states)

    # print(
    #     "avg cos sim: ",
    #     nn.CosineSimilarity(dim=-1, eps=1e-6)(attn_output, attn_output_real)
    #     .flatten()
    #     .mean(),
    # )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global local_state
local_state = LocalState()


"""
Enable Sparse Attention through a monkey patching way.
"""


def enable_sparse_attention(
    model,
    sparse_config: Dict,
    enable_budget_info: bool = False,
    enable_score_info: bool = False,
) -> (Optional[HistoryBudgetInfo], Optional[HistoryAccumulatedScoreInfo]):
    """
    Enable Sparse Attention via config dict.

    Config template:
    {
        "selector": {
            "type": "type_name",
            ... // Other args
        }
        "weight_estimator": {
            "type": "type_name",
            ... // Other args
        }
        "weight_pruner": {
            "type": "type_name",
            ... // Other args
        }
        "use_estimated_weights_in_attn": False,
        "skip_first_two_layers": True,
    }
    """

    def _split_type_args(config: Dict) -> Tuple:
        config1 = config.copy()
        type_name = config1["type"]
        del config1["type"]
        return type_name, config1

    compressor_type, compressor_args = _split_type_args(sparse_config["compressor"])
    selector_type, selector_args = _split_type_args(sparse_config["selector"])
    weight_estimator_type, weight_estimator_args = _split_type_args(
        sparse_config["weight_estimator"]
    )
    weight_pruner_type, weight_pruner_args = _split_type_args(
        sparse_config["weight_pruner"]
    )

    budget_info = HistoryBudgetInfo() if enable_budget_info else None
    score_info = HistoryAccumulatedScoreInfo() if enable_score_info else None

    use_estimated_weights_in_attn = sparse_config.get(
        "use_estimated_weights_in_attn", False
    )
    skip_first_two_layers = sparse_config.get("skip_first_two_layers", True)

    _enable_sparse_attention(
        model,
        compressor=CompressorType.from_str(compressor_type),
        compressor_args=compressor_args,
        selector=IndexSelectorType.from_str(selector_type),
        selector_args=selector_args,
        weight_estimator=WeightEstimatorType.from_str(weight_estimator_type),
        weight_estimator_args=weight_estimator_args,
        weight_pruner=WeightPrunerType.from_str(weight_pruner_type),
        weight_pruner_args=weight_pruner_args,
        budget_info=budget_info,
        score_info=score_info,
        use_estimated_weights_in_attn=use_estimated_weights_in_attn,
        skip_first_two_layers=skip_first_two_layers,
    )

    return budget_info, score_info


def _enable_sparse_attention(
    model,
    # KV cache compression in prefill stage
    compressor: CompressorType = CompressorType.NONE,
    compressor_args: Dict = {},
    # Sparse attention index selector
    selector: IndexSelectorType = IndexSelectorType.NONE,
    selector_args: Dict = {},
    # Weight estimator and pruner
    weight_estimator: WeightEstimatorType = WeightEstimatorType.NONE,
    weight_estimator_args: Dict = {},
    weight_pruner: WeightPrunerType = WeightPrunerType.NONE,
    weight_pruner_args: Dict = {},
    # Global Args
    budget_info: Optional[HistoryBudgetInfo] = None,
    score_info: Optional[HistoryAccumulatedScoreInfo] = None,
    # Whether to use estimated weights as real weights
    use_estimated_weights_in_attn: bool = False,
    # To ensure a fair comparsion, we skip the first two layers
    skip_first_two_layers: bool = True,
) -> None:
    """Enable Sparse Attention.

    Can't be called twice since the local_state isn't reset.
    """

    if not local_state.arg_set:
        local_state.set_args(
            compressor=compressor,
            compressor_args=compressor_args,
            selector=selector,
            selector_args=selector_args,
            weight_estimator=weight_estimator,
            weight_estimator_args=weight_estimator_args,
            weight_pruner=weight_pruner,
            weight_pruner_args=weight_pruner_args,
            budget_info=budget_info,
            score_info=score_info,
            use_estimated_weights_in_attn=use_estimated_weights_in_attn,
            skip_first_two_layers=skip_first_two_layers,
        )

        if selector == IndexSelectorType.DS:
            config_path = selector_args["config_path"]

            with open(config_path, "r") as f:
                channel_config = json.load(f)

            model = init_model_channel_config(
                model,
                channel_config,
                heavy_channel_num=selector_args["r"],
                selected_channel=selector_args["selected_channel"],
            )

        print(f"Local state args: {local_state.__dict__}")

    # Some monkey patching
    # if compressor == CompressorType.SNAP_KV:
    #     model.prepare_inputs_for_generation = types.MethodType(
    #         prepare_inputs_for_generation_llama_snapkv, model
    #     )

    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # For longchat model
            # print(name)
            layer_id = int(name.split(".")[2])
            module.layer_id = layer_id

            # Hack the forward function
            module.forward = types.MethodType(attention_forward, module)

            module.state = local_state


def reset_sparse_config() -> None:
    global local_state
    local_state.arg_set = False
