# This is a Python implementation of StreamingLLM Attention.


import torch


def streaming_selector(
    attn_weights: torch.Tensor,
    token_budget: int,
    num_sinks: int,
) -> torch.Tensor:
    """StreamingLLM selector.

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    assert (
        len(mask.shape) == 4
    ), "StreamingLLM only supports 4D attention weights. [bs, num_heads, 1, seq_length]"

    mask[..., :num_sinks] = True
    mask[..., -token_budget:] = True
    return mask
