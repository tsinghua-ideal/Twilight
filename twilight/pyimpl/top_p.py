# TopP


import torch
from torch import nn

from twilight.kernel import top_p_fp32_return_mask


def top_p_unnormalized(attn_weights: torch.Tensor, threshold: float) -> torch.Tensor:
    normalized_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    )
    mask = top_p_fp32_return_mask(normalized_weights, threshold)
    return mask
