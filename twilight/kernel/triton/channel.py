# From: https://github.com/andy-yang-1/DoubleSparse/blob/main/models/triton_kernels/channel.py

import torch

import triton
import triton.language as tl
import math
import random


@triton.jit
def get_label_tensor_kernel(
    X,
    channel,
    Out,
    stride_x_ld,
    stride_x_h,
    stride_x_d,
    stride_channel_h,
    stride_channel_c,
    stride_out_ld,
    stride_out_h,
    stride_out_c,
    HEAVY_CHANNEL_NUM: tl.constexpr,
):

    # get the current head and current token
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)

    # load channel idx
    offs_channel = (
        channel + cur_head * stride_channel_h + tl.arange(0, HEAVY_CHANNEL_NUM)
    )
    heavy_channels = tl.load(offs_channel)

    # load X's heavy channel
    offs_X = (
        X
        + cur_token * stride_x_ld
        + cur_head * stride_x_h
        + heavy_channels * stride_x_d
    )
    label_tensor = tl.load(offs_X)

    # store to out
    offs_out = (
        Out
        + cur_token * stride_out_ld
        + cur_head * stride_out_h
        + tl.arange(0, HEAVY_CHANNEL_NUM) * stride_out_c
    )
    tl.store(offs_out, label_tensor)


def get_label_tensor(X, channel, Out, HEAVY_CHANNEL_NUM):

    L, H, _ = X.shape

    stride_x_ld, stride_x_h, stride_x_d = X.stride()
    stride_channel_h, stride_channel_c = channel.stride()
    stride_out_ld, stride_out_h, stride_out_c = Out.stride()

    grid = (L, H)

    get_label_tensor_kernel[grid](
        X,
        channel,
        Out,
        stride_x_ld,
        stride_x_h,
        stride_x_d,
        stride_channel_h,
        stride_channel_c,
        stride_out_ld,
        stride_out_h,
        stride_out_c,
        HEAVY_CHANNEL_NUM,
    )


def test_get_label_tensor():

    L, H, D = 10, 3, 8
    HEAVY_CHANNEL_NUM = 4

    X = torch.randn(L, H, D, dtype=torch.float32, device="cuda")
    # channel = torch.randint(0, D, (H, HEAVY_CHANNEL_NUM), dtype=torch.int64, device='cuda')
    channel = torch.zeros(H, HEAVY_CHANNEL_NUM, dtype=torch.int64, device="cuda")
    for h in range(H):
        channel[h] = torch.randperm(D, device="cuda")[:HEAVY_CHANNEL_NUM]
    print(channel)
    Out = torch.empty(L, H, HEAVY_CHANNEL_NUM, dtype=torch.float32, device="cuda")

    get_label_tensor(X, channel, Out, HEAVY_CHANNEL_NUM)

    Out_cpu = Out.cpu()
    for i in range(L):
        for j in range(H):
            expected = X[i, j, channel[j]].cpu()
            result = Out_cpu[i, j]
            if not torch.allclose(expected, result):
                print(f"Discrepancy found at token {i}, head {j}")
                return False
    print("Test passed!")
    return True


if __name__ == "__main__":
    test_get_label_tensor()
