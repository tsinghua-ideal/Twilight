# This is a Python implementation of Double Sparsity

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.


import torch

import math
from typing import Optional

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralAttention

from .quantize import min_max_per_token_quant_kv

from twilight.kernel import get_label_tensor

from .top_k import top_k


def init_model_channel_config(
    model, channel_config, heavy_channel_num, selected_channel="k"
):
    selected_channel = "." + selected_channel + "_proj"
    for name, module in model.named_modules():
        # if isinstance(module, Attention):
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # print(name)
            layer_idx = int(name.split(".")[2])
            key = "model.layers." + str(layer_idx) + ".self_attn" + selected_channel
            module.sorted_channel = (
                torch.tensor(channel_config[key])[:, :heavy_channel_num]
                .contiguous()
                .cuda()
            )

    return model


def permute_channel_config(sorted_channel):
    head_num = sorted_channel.shape[0]
    head_dim = sorted_channel.shape[1]
    return (sorted_channel * 2) % head_dim + (sorted_channel * 2) // head_dim


def double_sparse(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    sorted_channel: torch.Tensor,
    r: int,
    quant_bit: int,
) -> torch.Tensor:
    """SparQ weight estimator.

    r = head_dim * compression_rate

    e.g. compression_rate = 1/8, head_dim = 128, r = 16

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    bs, num_heads, kv_len, head_dim = key_states.shape
    query_label = torch.empty(
        (bs, num_heads, r),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    get_label_tensor(
        query_states.view(bs, num_heads, head_dim), sorted_channel, query_label, r
    )
    key_label = torch.empty(
        (bs * kv_len, num_heads, r),
        dtype=key_states.dtype,
        device=key_states.device,
    )
    get_label_tensor(
        key_states.transpose(1, 2).contiguous().view(bs * kv_len, num_heads, head_dim),
        sorted_channel,
        key_label,
        r,
    )
    query_label = query_label.view(bs, num_heads, 1, r)
    key_label = key_label.view(bs, kv_len, num_heads, r).transpose(1, 2)
    if quant_bit < 32:  # 32 means fp32 (no quantization)
        key_label = min_max_per_token_quant_kv(key_label, quant_bit)

    estimated_weights = torch.matmul(
        query_label, key_label.transpose(2, 3)
    ) / math.sqrt(
        query_states.shape[-1]
    )  # Divided by original d, not r

    return estimated_weights


def double_sparse_selector(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    sorted_channel: torch.Tensor,
    r: int,
    quant_bit: int,
    token_budget: Optional[int] = -1,
    sparsity_rate: Optional[float] = -1,
) -> torch.Tensor:
    """DS index selector.
    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    seq_length = key_states.shape[-2]

    # If sparsity rate is not -1, use it to calculate token_budget
    if sparsity_rate != -1:
        token_budget = int(seq_length * sparsity_rate)

    estimated_weights = double_sparse(
        query_states, key_states, sorted_channel, r, quant_bit
    )
    return top_k(estimated_weights, token_budget)


def double_sparse_selector1(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    sorted_channel: torch.Tensor,
    r: int,
    quant_bit: int,
    token_budget: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DS index selector.
    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    estimated_weights = double_sparse(
        query_states, key_states, sorted_channel, r, quant_bit
    )

    if attention_mask is not None:
        estimated_weights = estimated_weights + attention_mask
        estimated_weights = torch.max(
            estimated_weights, torch.tensor(torch.finfo(estimated_weights.dtype).min)
        )

    return top_k(estimated_weights, token_budget)
