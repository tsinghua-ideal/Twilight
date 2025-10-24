import torch


def min_max_per_token_quant_kv(kv: torch.Tensor, qbit: int) -> torch.Tensor:
    max_val = kv.amax(dim=-1, keepdim=True)
    min_val = kv.amin(dim=-1, keepdim=True)

    max_int = 2**qbit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-9) / max_int
    zeros = -min_val
    # print(zeros)
    return torch.clamp(torch.round((kv + zeros) / scales), min_int, max_int) * scales - zeros


def max_per_token_quant_kv(kv: torch.Tensor, qbit: int, smooth: bool) -> torch.Tensor:
    if smooth:
        # print(kv.shape)
        kv = kv - torch.min(kv, dim=-2, keepdim=True)[0]
    if qbit == 1:
        return torch.sign(kv)
    channel_max = torch.max(torch.abs(kv), dim=-1, keepdim=True)[0]
    scale = channel_max / (2**qbit - 1)
    scale = torch.max(scale, torch.tensor(1e-6))
    return torch.round(kv / scale) * scale  # quantize then dequantize back


if __name__ == "__main__":
    # kv = torch.Tensor([-4, 5, 6, -7, 8, 9])
    kv = torch.Tensor(
        [
            0.31,
            0.42,
            -0.57,
            0.23,
            -1.79,
            -2.18,
            1.90,
            4.32,
            -3.56,
            0.78,
            0.69,
            1.45,
            2.84,
            -2.09,
            -1.26,
        ]
    )
    print(min_max_per_token_quant_kv(kv, 4))
