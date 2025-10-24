import torch

from typing import List


def tidal_decode_selector(
    state: "LocalState",
    attn_weights: torch.Tensor,
    token_budget: int,
    layer_id: int,
    reselection_layers: List,
) -> torch.Tensor:
    """Tidal Decode index selector.

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    if layer_id in reselection_layers:
        k = min(token_budget, attn_weights.shape[-1])
        _, indices = attn_weights.topk(k=k, dim=-1)
        mask = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        setattr(state, "td_mask_buffer", mask)
        return None, torch.ones_like(
            attn_weights, dtype=torch.bool
        )  # Use full attention in reselection layers

    assert hasattr(state, "td_mask_buffer"), "td_mask_buffer is not found in state"
    return None, getattr(state, "td_mask_buffer")
