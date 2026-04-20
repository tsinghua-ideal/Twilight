import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch import nn
import numpy as np
import time


from ftka.cuda_ops.gemv import batched_sparse_gemv, batched_sparse_gemv_int8_k, batched_sparse_gemv_int4_k, quest_sparse_gemv, single_sparse_gemv
from ftka.triton_ops.tokens_moving import discontinuous_move_tokens
from ftka.utils.benchmark import benchmark_forward_with_warmup


def test_single_sparse_gemv(check_correctness: bool = True):
    # Prepare data
    head_size = 128
    num_heads = 32

    print(f"head_size: {head_size}, num_heads: {num_heads}")

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        print(f"seq_len: {seq_len}")

        q = torch.ones(
            (num_heads, head_size), dtype=torch.float16, device="cuda:0"
        )
        k = torch.ones(
            (seq_len, num_heads, head_size), dtype=torch.float16, device="cuda:0"
        )
        o = torch.empty(
            (num_heads, seq_len), dtype=torch.float16, device="cuda:0"
        )

        single_sparse_gemv(q, o, k)

        ref_q = q.unsqueeze(1)
        ref_k = k.transpose(0, 1).transpose(1, 2)

        # Check output
        ref_o = torch.matmul(ref_q, ref_k).squeeze(1)

        if check_correctness:
            torch.testing.assert_close(o, ref_o, atol=1e-3, rtol=1e-2)

        _, avg_time = benchmark_forward_with_warmup(
            single_sparse_gemv,
            q,
            o,
            k,
            warmups=10,
            repeats=100,
        )

        print(f"[ftka single sparse gemv] seq_len: {seq_len}, time: {avg_time.mean*1e6} us")

        _, avg_time = benchmark_forward_with_warmup(
            torch.matmul,
            ref_q,
            ref_k,
            warmups=10,
            repeats=100,
        )

        print(f"[torch dot] seq_len: {seq_len}, time: {avg_time.mean*1e6} us")



def test_sparse_gemv_int8(batch_size: int, check_correctness: bool = True):
    # Prepare data
    head_size = 128
    num_heads = 32

    print(f"batch_size: {batch_size}, head_size: {head_size}, num_heads: {num_heads}")

    # token attention
    page_size = 1
    max_num_pages = 32768 // page_size

    # NOTICE: the quantized k cache should be INTERLEAVED and +128!!!!!

    k_cache_int8 = torch.randint(
        3,
        4,
        (
            max_num_pages,
            page_size,
            num_heads,
            head_size,
        ),
        dtype=torch.int8,
        device="cuda:0",
    )

    k_cache_uint8 = k_cache_int8.to(torch.uint8) + 128

    # print(k_cache_int8)
    # print(k_cache_uint8)

    quant_scales = torch.ones(
        max_num_pages,
        page_size,
        num_heads,
        dtype=torch.float16,
        device="cuda:0",
    )

    quant_zeros = torch.zeros(
        max_num_pages,
        page_size,
        num_heads,
        dtype=torch.float16,
        device="cuda:0",
    )

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:  # 32768 OOM
        # print(f"seq_len: {seq_len}")

        flops = 4 * num_heads * head_size * 1 * seq_len // 2

        # NHD
        q = torch.ones(
            (batch_size, num_heads, head_size), dtype=torch.float16, device="cuda:0"
        )
        o = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float16, device="cuda:0"
        )

        kv_page_indices = torch.arange(seq_len).int().to("cuda:0")
        kv_page_indices = kv_page_indices.repeat(batch_size)
        kv_page_indptr = (
            torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0") * seq_len
        )
        # 1 <= kv_last_page_len <= page_size
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

        batched_sparse_gemv_int8_k(
            q,
            o,
            k_cache_uint8,
            quant_scales,
            quant_zeros,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )

        torch.cuda.synchronize()

        _, avg_time = benchmark_forward_with_warmup(
            batched_sparse_gemv_int8_k,
            q,
            o,
            k_cache_uint8,
            quant_scales,
            quant_zeros,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            warmups=10,
            repeats=100,
        )

        print(f"{avg_time.mean*1e6},")


        q = q.unsqueeze(2)

        k_int8 = (
            k_cache_int8[:seq_len]
            .reshape(-1, num_heads, head_size)
            .transpose(0, 1)
            .transpose(1, 2)
            .unsqueeze(0)
        ).to(torch.float32)

        # print(q.shape, k_int8.shape)

        # scale: [num_heads, 1, seq_len]
        scales = quant_scales[:seq_len].reshape(-1, 1, num_heads).transpose(0, 2)
        zeros = quant_zeros[:seq_len].reshape(-1, 1, num_heads).transpose(0, 2)

        out1_int8 = (q.to(torch.float32) @ (k_int8 * scales + zeros)).to(q.dtype)

        # print(out_int8)
        # print(out1_int8)

        if check_correctness:
            torch.testing.assert_close(out_int8, out1_int8, rtol=1e-3, atol=1e-3)



def bench_sparse_gemv_fp16(batch_size: int, check_correctness: bool = True):
    # Prepare data
    head_size = 128
    num_heads = 32

    print(f"batch_size: {batch_size}, head_size: {head_size}, num_heads: {num_heads}")

    # token attention, page=1
    page_size = 1
    max_num_pages = 32768 // page_size

    k_cache = torch.randn(
        max_num_pages,
        page_size,
        num_heads,
        head_size,
        dtype=torch.float16,
        device="cuda:0",
    )

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:  # 32768 OOM
        flops = 4 * num_heads * head_size * 1 * seq_len // 2
        num_pages = min(max_num_pages, seq_len // page_size)

        # NHD
        q = torch.randn(
            (batch_size, num_heads, head_size), dtype=torch.float16, device="cuda"
        )
        o = torch.empty(
            (batch_size, num_heads, 1, seq_len),
            dtype=torch.float16,
            device="cuda:0",
        )

        kv_page_indices = torch.arange(num_pages).int().to("cuda:0")
        kv_page_indices = kv_page_indices.repeat(batch_size)
        kv_page_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0") * num_pages
        # print(kv_page_indices)
        # print(kv_page_indptr)
        
        # 1 <= kv_last_page_len <= page_size
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

        batched_sparse_gemv(
            q,
            o,
            k_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )

        _, avg_time = benchmark_forward_with_warmup(
            batched_sparse_gemv,
            q,
            o,
            k_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            warmups=10,
            repeats=100,
        )

        print(f"{avg_time.mean*1e6},")

        q = q.unsqueeze(2)

        # q: [batch_size, num_heads, 1, head_size]
        # k: [batch_size, num_heads, head_size, seq_len]
        k = (
            k_cache[:num_pages]
            .reshape(-1, num_heads, head_size)
            .transpose(0, 1)
            .transpose(1, 2)
            .repeat(batch_size, 1, 1, 1)
        )

        # print(q.shape, k.shape)

        k_cache_cpy = k_cache.clone().repeat(batch_size, 1, 1, 1).reshape(-1, num_heads, head_size)
        k_fake_indices = torch.arange(seq_len).int().repeat(batch_size).flatten().to("cuda:0")

        # print(k_cache_cpy.shape)
        # print(k_fake_indices.shape)

        # print(q.shape, k.shape, (q @ k).shape)

        out1 = q @ k

        def _torch_load_dot(q, k):
            # move seq_len tokens
            discontinuous_move_tokens(k_cache_cpy, k_cache_cpy, k_fake_indices, k_fake_indices)
            return torch.matmul(q, k), k_cache_cpy

        _, avg_time = benchmark_forward_with_warmup(
            _torch_load_dot,
            q,
            k,
            warmups=10,
            repeats=100,
        )

        # print(f"[torch load+dot] seq_len: {seq_len}, time: {avg_time.mean*1e6} us")

        if check_correctness:
            torch.testing.assert_close(out1, o, rtol=1e-3, atol=1e-3)


def test_quest_sparse_gemv(check_correctness: bool = True):
    # Prepare data
    batch_size = 2
    head_size = 128
    num_heads = 32

    print(f"batch_size: {batch_size}, head_size: {head_size}, num_heads: {num_heads}")

    # token attention, page=1
    page_size = 1
    max_num_pages = 32768 // page_size

    # NHD
    k_cache = torch.randn(
        max_num_pages,
        page_size,
        num_heads,
        head_size,
        dtype=torch.float16,
        device="cuda:0",
    )
    v_cache = torch.randn(
        max_num_pages,
        page_size,
        num_heads,
        head_size,
        dtype=torch.float16,
        device="cuda:0",
    )

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:  # 32768 OOM
        num_pages = min(max_num_pages, seq_len // page_size)

        # NHD
        q = torch.rand(
            (batch_size, num_heads, head_size), dtype=torch.float16, device="cuda"
        )

        o = torch.randn(
            (batch_size, num_heads, seq_len), dtype=torch.float16, device="cuda"
        )

        kv_page_indices = torch.arange(num_pages).int().to("cuda:0")
        kv_page_indices = kv_page_indices.repeat(batch_size)
        kv_page_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0") * num_pages
        # print(kv_page_indices)
        # print(kv_page_indptr)
        
        # 1 <= kv_last_page_len <= page_size
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

        # quest_sparse_gemv(
        #     q,
        #     o,
        #     k_cache,
        #     v_cache,
        #     kv_page_indices,
        #     kv_page_indptr,
        #     kv_last_page_len,
        # )

        _, avg_time = benchmark_forward_with_warmup(
            quest_sparse_gemv,
            q,
            o,
            k_cache,
            v_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            warmups=10,
            repeats=100,
        )

        print(f"{avg_time.mean*1e6},")


def test_sparse_gemv_int4(batch_size: int, check_correctness: bool = True):
    # Prepare data
    head_size = 128
    num_heads = 32

    print(f"batch_size: {batch_size}, head_size: {head_size}, num_heads: {num_heads}")

    # token attention
    page_size = 1
    max_num_pages = 32768 // page_size

    # NOTICE: the quantized k cache should be INTERLEAVED and +128!!!!!
    # 10000011
    k_cache_int4 = torch.randint(
        3,
        4,
        (
            max_num_pages,
            page_size,
            num_heads,
            head_size // 2,
        ),
        dtype=torch.int8,
        device="cuda:0",
    )

    k_cache_int4_uint8 = k_cache_int4.to(torch.uint8) # + 128

    # print(k_cache_int4)
    # print(k_cache_uint8)

    quant_scales = torch.ones(
        max_num_pages,
        page_size,
        num_heads,
        dtype=torch.float16,
        device="cuda:0",
    )

    quant_zeros = torch.zeros(
        max_num_pages,
        page_size,
        num_heads,
        dtype=torch.float16,
        device="cuda:0",
    )

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:  # 32768 OOM
        # print(f"seq_len: {seq_len}")

        flops = 4 * num_heads * head_size * 1 * seq_len // 2

        # NHD
        q = torch.ones(
            (batch_size, num_heads, head_size), dtype=torch.float16, device="cuda:0"
        )
        o = torch.empty(
            (batch_size, num_heads, seq_len), dtype=torch.float16, device="cuda:0"
        )

        kv_page_indices = torch.arange(seq_len).int().to("cuda:0")
        kv_page_indices = kv_page_indices.repeat(batch_size)
        kv_page_indptr = (
            torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0") * seq_len
        )
        # 1 <= kv_last_page_len <= page_size
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

        batched_sparse_gemv_int4_k(
            q,
            o,
            k_cache_int4_uint8,
            quant_scales,
            quant_zeros,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )

        torch.cuda.synchronize()

        _, avg_time = benchmark_forward_with_warmup(
            batched_sparse_gemv_int4_k,
            q,
            o,
            k_cache_int4_uint8,
            quant_scales,
            quant_zeros,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            warmups=10,
            repeats=100,
        )

        print(f"{avg_time.mean*1e6},")


        q = q.unsqueeze(2)

        k_int4 = (
            k_cache_int4[:seq_len]
            .reshape(-1, num_heads, head_size)
            .transpose(0, 1)
            .transpose(1, 2)
            .unsqueeze(0)
        ).to(torch.float32)

        # print(q.shape, k_int8.shape)

        # scale: [num_heads, 1, seq_len]
        scales = quant_scales[:seq_len].reshape(-1, 1, num_heads).transpose(0, 2)
        zeros = quant_zeros[:seq_len].reshape(-1, 1, num_heads).transpose(0, 2)

        # out1_int4 = (q.to(torch.float32) @ (k_int4 * scales + zeros)).to(q.dtype)

        # print(out_int8)
        # print(out1_int8)

        # if check_correctness:
        #     torch.testing.assert_close(out_int8, out1_int8, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    check_correctness = False

    test_single_sparse_gemv(check_correctness)
    torch.cuda.empty_cache()
    print()
    test_sparse_gemv_int8(64, check_correctness)
    torch.cuda.empty_cache()
    print()
    test_sparse_gemv_int4(64, check_correctness)
    torch.cuda.empty_cache()
    print()
    bench_sparse_gemv_fp16(64, check_correctness)
    
    # test_quest_sparse_gemv(check_correctness)
