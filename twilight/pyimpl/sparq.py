# This is a Python implementation of the paper:
#   "SparQ Attention: Bandwidth-Efficient LLM Inference"
#   (https://arxiv.org/pdf/2312.04985.pdf)

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.


import torch

import math

from .top_k import top_k


def sparq(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    r: int,
) -> torch.Tensor:
    """SparQ weight estimator.

    r = head_dim * compression_rate

    e.g. compression_rate = 1/8, head_dim = 128, r = 16

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    query_norm = torch.abs(query_states)

    _, channel_indices = torch.topk(query_norm, dim=-1, k=r)

    partial_query = torch.gather(query_states, dim=-1, index=channel_indices)
    channel_indices = channel_indices.repeat(1, 1, key_states.shape[-2], 1)
    partial_key = torch.gather(key_states, dim=-1, index=channel_indices)

    estimated_weights = torch.matmul(
        partial_query, partial_key.transpose(2, 3)
    ) / math.sqrt(
        query_states.shape[-1]
    )  # Divided by original d, not r
    return estimated_weights


def sparq_selector(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    r: int,
    token_budget: int,
) -> torch.Tensor:
    """SparQ index selector.
    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    estimated_weights = sparq(query_states, key_states, r)
    return top_k(estimated_weights, token_budget)
