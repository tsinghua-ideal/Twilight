# This is a Python implementation of the paper:
#   "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"
#   (https://arxiv.org/pdf/2406.10774.pdf)

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.

# Reference:
#   - https://github.com/mit-han-lab/Quest/blob/main/evaluation/quest_attention.py
#   - https://github.com/Infini-AI-Lab/MagicPIG/blob/main/RULER/RULER/scripts/pred/llama_quest.py


import torch
from typing import Optional


def quest_selector(
    model,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    chunk_size: int,
    token_budget: Optional[int] = -1,
    sparsity_rate: Optional[float] = -1,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quest index selector.

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    # Quest starts here
    # Must be decode stage and layer_id >= 2
    bs = query_states.shape[0]

    # Critical Estimation part for Quest
    sign = (query_states > 0) + (~(query_states > 0)) * -1
    postive_query = query_states * sign
    max_key = key_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]

    # If sparsity rate is not -1, use it to calculate token_budget
    if sparsity_rate != -1:
        token_budget = int(seq_length * sparsity_rate)

    padding_length = chunk_size - (seq_length % chunk_size)
    if padding_length < chunk_size:
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (
                        max_key.shape[0],
                        max_key.shape[1],
                        padding_length,
                        max_key.shape[3],
                    ),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // chunk_size,
        chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    if chunk_max_key.dtype == torch.float32:
        postive_query = postive_query.float()

    # Shape of chunk_max_key: [bs, head_num, chunk_num, head_dim]
    # Shape of estimae_weight: [bs, head_num, q_len, chunk_num] (q_len == 1)
    estimated_weight = torch.matmul(
        postive_query,
        chunk_max_key.transpose(2, 3),
    )

    # print(f"Layer {model.layer_id}, Esti Weights: ", estimated_weight)

    #  assert token_budget < 8192, "Budget too large, may cause OOM"

    # assert (
    #     token_budget % chunk_size == 0
    # ), "Budget must be divisible by chunk_size"
    chunk_budget = max(1, token_budget // chunk_size)
    if token_budget % chunk_size != 0:
        chunk_budget += 1

    # Select TopK chunks
    _chunk_budget = min(chunk_budget, estimated_weight.size(-1))
    _, topk = estimated_weight.topk(k=_chunk_budget, dim=-1)

    # print(f"Layer {model.layer_id}, Selected Weights: ", topk)

    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk1 = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)

    topk1 = topk1.reshape(topk1.shape[0], topk1.shape[1], topk1.shape[2], -1)

    # Note that we use the padded shape to avoid overflow
    padded_shape = [
        bs,
        model.num_heads,
        1,
        max_key.shape[-2],  # padded length
    ]
    mask_bottom = torch.zeros(padded_shape, dtype=torch.bool, device=topk1.device)
    mask_bottom.scatter_(dim=-1, index=topk1, value=True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]
    return mask_bottom


def quest_selector1(
    model,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    chunk_size: int,
    token_budget: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quest index selector.

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    # Quest starts here
    # Must be decode stage and layer_id >= 2
    bs = query_states.shape[0]

    # Critical Estimation part for Quest
    sign = (query_states > 0) + (~(query_states > 0)) * -1
    postive_query = query_states * sign
    max_key = key_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = chunk_size - (seq_length % chunk_size)
    if padding_length < chunk_size:
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (
                        max_key.shape[0],
                        max_key.shape[1],
                        padding_length,
                        max_key.shape[3],
                    ),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // chunk_size,
        chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    if chunk_max_key.dtype == torch.float32:
        postive_query = postive_query.float()

    # Shape of chunk_max_key: [bs, head_num, chunk_num, head_dim]
    # Shape of estimae_weight: [bs, head_num, q_len, chunk_num] (q_len == 1)
    estimated_weight = torch.matmul(
        postive_query,
        chunk_max_key.transpose(2, 3),
    )

    # print(f"Layer {model.layer_id}, Esti Weights: ", estimated_weight)

    #  assert token_budget < 8192, "Budget too large, may cause OOM"

    # assert (
    #     token_budget % chunk_size == 0
    # ), "Budget must be divisible by chunk_size"
    chunk_budget = max(1, token_budget // chunk_size)
    if token_budget % chunk_size != 0:
        chunk_budget += 1

    # Select TopK chunks
    _chunk_budget = min(chunk_budget, estimated_weight.size(-1))
    _, topk = estimated_weight.topk(k=_chunk_budget, dim=-1)

    # print(f"Layer {model.layer_id}, Selected Weights: ", topk)

    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk1 = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)

    topk1 = topk1.reshape(topk1.shape[0], topk1.shape[1], topk1.shape[2], -1)

    # Note that we use the padded shape to avoid overflow
    padded_shape = [
        bs,
        model.num_heads,
        1,
        max_key.shape[-2],  # padded length
    ]
    mask_bottom = torch.zeros(padded_shape, dtype=torch.bool, device=topk1.device)
    mask_bottom.scatter_(dim=-1, index=topk1, value=True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]
    return mask_bottom
