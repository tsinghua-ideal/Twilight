# Python implementation of the paper:
#   "SnapKV : LLM Knows What You are Looking for Before Generation"
#   (https://arxiv.org/abs/2404.14469v1)

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.


import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from transformers.cache_utils import Cache, DynamicCache


def snap_kv_compression(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window_size: int = 64,
    max_capacity_prompt: int = 256 + 64,
    kernel_size: int = 5,
    pooling: str = "avgpool",
):
    # check if prefix phase
    assert key_states.shape[-2] == query_states.shape[-2]
    bsz, num_heads, q_len, head_dim = query_states.shape
    if q_len < max_capacity_prompt:
        return key_states, value_states
    else:
        attn_weights = torch.matmul(
            query_states[..., -window_size:, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (window_size, window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -window_size:, -window_size:] += attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim=-2)
        if pooling == "avgpool":
            attn_cache = F.avg_pool1d(
                attn_weights_sum,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        elif pooling == "maxpool":
            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError("Pooling method not supported")
        indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, :-window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, :-window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


def prepare_inputs_for_generation_llama_snapkv(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs
):
    if past_key_values is None:  # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
