/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_SAMPLING_CUH_
#define FLASHINFER_SAMPLING_CUH_

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/limits>
#include <numeric>

#include "flashinfer/math.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

using namespace cub;

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
#endif

template <typename T>
struct Pair {
  T value;
  int count;

  __device__ Pair operator+(const Pair& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ Pair& operator+=(const Pair& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

template <typename T, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    T deterministic_scan[BLOCK_THREADS / 32];
    typename BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<Pair<T>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_pair;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    union {
      T value;
      Pair<T> pair;
      T max_p;
    } block_aggregate;
  };
};

template <typename T, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
  union {
    typename BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<Pair<T>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_pair;
  } block_prim;
  struct {
    T max_val;
    T min_val;
    union {
      T value;
      int count;
      Pair<T> pair;
    } block_aggregate;
  };
};

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType>
__global__ void TopPReturnMaskKernel(DType* probs, uint8_t* mask, float* top_p_arr, float top_p_val,
                                     uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  DType p = DType(top_p_arr == nullptr ? top_p_val : top_p_arr[bx]);

  extern __shared__ __align__(alignof(RenormTempStorage<DType, BLOCK_THREADS, REDUCE_ALGO>))
      uint8_t smem_renorm[];
  auto& temp_storage =
      reinterpret_cast<RenormTempStorage<DType, BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
  temp_storage.max_val = DType(0);
  vec_t<DType, VEC_SIZE> probs_vec;
  DType probs_greater_than_pivot[VEC_SIZE];  // pivot initialized to 0

  DType threadlocal_max_val = DType(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_greater_than_pivot[j] = probs_vec[j];
    }
    threadlocal_max_val =
        max(threadlocal_max_val,
            BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                .Reduce<VEC_SIZE>(probs_greater_than_pivot, cub::Max()));
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.max_val = threadlocal_max_val;
  }
  __syncthreads();
  threadlocal_max_val = temp_storage.max_val;
  probs_vec.fill(DType(0));
  probs_vec.load(probs + row_idx * d);  // load attention sink

  DType low = 0.0, high = probs_vec[0];
  DType min_gt_low, max_le_high;
  DType sum_low(1);
  DType eps = DType(1e-5);

  // f(x) = sum(probs[probs > x]), f(x) is non-increasing
  // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
  // loop invariant:
  // - f(low) >= p, f(high) < p
  // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
  // stopping condition
  // - f(low) >= p, f(min_gt_low) == f(max_le_high) == f(high) < p
  do {
    DType threadlocal_sum(0);
    DType mid = (low + high) / DType(2);
    min_gt_low = high;
    max_le_high = low;
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_greater_than_pivot[j] = (probs_vec[j] > mid) ? probs_vec[j] : DType(0);
        if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
      }
      threadlocal_sum +=
          BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Sum<VEC_SIZE>(probs_greater_than_pivot);
      __syncthreads();
    }
    min_gt_low = BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, cub::Min());
    __syncthreads();
    max_le_high =
        BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Reduce(max_le_high, cub::Max());
    if (tx == 0) {
      temp_storage.block_aggregate.value = threadlocal_sum;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    threadlocal_sum = temp_storage.block_aggregate.value;
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;
    if (threadlocal_sum >= p) {
      low = mid;
      sum_low = threadlocal_sum;
    } else {
      high = min(mid, max_le_high);
    }
  } while (max_le_high - min_gt_low > eps);

  // set mask
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        mask[row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE + j] =
            (probs_vec[j] > low) ? 1 : 0;
      }
    }
  }
}

template <typename DType>
cudaError_t TopPReturnMask(DType* probs, uint8_t* mask, float* top_p_arr, uint32_t num_blocks,
                           float top_p_val, uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  const uint32_t smem_size = sizeof(RenormTempStorage<DType, BLOCK_THREADS, REDUCE_ALGO>);
  dim3 nblks(num_blocks);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &mask, &top_p_arr, &top_p_val, &d};
  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = TopPReturnMaskKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
