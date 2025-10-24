# This is a Python implementation of Oracle TopK Sparse Attention.


import torch


def top_k(
    attn_weights: torch.Tensor,
    token_budget: int,
) -> torch.Tensor:
    k = min(token_budget, attn_weights.shape[-1])
    _, indices = attn_weights.topk(k=k, dim=-1)
    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)

    return mask


def oracle_topk_selector(
    attn_weights: torch.Tensor,
    token_budget: int,
) -> torch.Tensor:
    """Oracle TopK index selector.

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    return top_k(attn_weights, token_budget)
