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

#include "pytorch_extension_utils.h"
#include "sampling.cuh"

using namespace flashinfer;

void top_p_fp16_return_mask(at::Tensor probs, at::Tensor mask,
                            std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                            int64_t cuda_stream) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(4, probs);  // probs: (batch_size, num_heads, qo_len, kv_seq_len)
  unsigned int batch_size = probs.size(0);
  unsigned int num_heads = probs.size(1);
  unsigned int kv_seq_len = probs.size(3);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(probs.scalar_type(), c_type, [&] {
    cudaError_t status = sampling::TopPReturnMask<c_type>(
        static_cast<c_type*>(probs.data_ptr()), static_cast<uint8_t*>(mask.data_ptr()),
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size * num_heads,
        top_p_val, kv_seq_len, stream);
    TORCH_CHECK(status == cudaSuccess,
                "TopPReturnMask failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}

void top_p_fp32_return_mask(at::Tensor probs, at::Tensor mask,
                            std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                            int64_t cuda_stream) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(4, probs);  // probs: (batch_size, num_heads, qo_len, kv_seq_len)
  unsigned int batch_size = probs.size(0);
  unsigned int num_heads = probs.size(1);
  unsigned int kv_seq_len = probs.size(3);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TopPReturnMask<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<uint8_t*>(mask.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size * num_heads,
      top_p_val, kv_seq_len, stream);
  TORCH_CHECK(status == cudaSuccess,
              "TopPReturnMask failed with error code " + std::string(cudaGetErrorString(status)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_p_fp16_return_mask", &top_p_fp16_return_mask,
        "Using top-p sampling to mask attention weights");
  m.def("top_p_fp32_return_mask", &top_p_fp32_return_mask,
        "Using top-p sampling to mask attention weights");
}
