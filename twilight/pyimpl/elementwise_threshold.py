# This is a Python implementation of Twi Sparse Attention.


import torch
from torch import nn
import numpy as np


def elementwise_threshold(attn_weights: torch.Tensor, threshold: float) -> torch.Tensor:
    """Perform element-wise threshold on weights.

    Return a mask.
    """

    first_weight = attn_weights[:, :, :, 0:1]
    # first_weight = torch.max(attn_weights, dim=-1, keepdims=True).values
    mask = attn_weights > first_weight - threshold
    return mask


def hierarchical_elementwise_threshold(attn_weights, rate: float) -> torch.Tensor:
    _, head_num, _, seq_len = attn_weights.shape

    threshold = [5, 7, 9, 10, 11, 12]
    masks = torch.zeros(
        [len(threshold), 1, head_num, 1, seq_len],
        dtype=torch.bool,
        device=attn_weights.device,
    )
    final_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

    tokens_num = []  # each item is a tensor with the shape (head_num,)
    tokens_num1 = []
    weights = torch.zeros(
        [len(threshold), head_num], dtype=attn_weights.dtype, device=attn_weights.device
    )

    for i, t in enumerate(threshold):
        mask_bottom = twi(attn_weights, t)
        masks[i, :] = mask_bottom
        tokens_num.append(torch.sum(mask_bottom, dim=-1).flatten())

    for i in range(len(tokens_num) - 1):
        tokens_num1.append(tokens_num[i + 1] - tokens_num[i])
    tokens_num1.append(seq_len - tokens_num[-1])

    for i in range(len(tokens_num1)):
        weights[len(tokens_num1) - i - 1, :] = tokens_num1[i] * np.exp(-threshold[i])

    weights = torch.cumsum(weights, dim=0)
    weights_pos = len(threshold) - (weights < 1 - rate).sum(dim=0)

    # print("Token num:")
    # for tn in tokens_num1:
    #     print(tn)
    # print(weights_pos)

    # This loop can be parallelized
    for i in range(head_num):
        final_mask[:, i, :, :] = masks[weights_pos[i], :, i, :, :]

    return final_mask
