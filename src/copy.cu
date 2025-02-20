#include <cstdint>
#include <cstdio>
#include <cuda/pipeline>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <memory>
#include <numeric>

#include "util.cu"

__global__ void copy(const int *__restrict__ src, int *__restrict__ dst,
                     int size) {
  constexpr int UNROLL = 32;
  constexpr int VEC_SIZE = sizeof(int4) / sizeof(int);

  int4 buffer[UNROLL];

  const int num_threads = blockDim.x * gridDim.x;
  const int stride = num_threads * UNROLL;

  const int4 *__restrict__ src_vec = reinterpret_cast<const int4 *>(src);
  int4 *__restrict__ dst_vec = reinterpret_cast<int4 *>(dst);
  size = size / VEC_SIZE;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += stride) {
#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      buffer[i] = src_vec[idx + i * num_threads];
    }

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      dst_vec[idx + i * num_threads] = buffer[i];
    }
  }
}

int main() {
  constexpr int THREADS = 256;
  constexpr int BLOCKS = 108;
  constexpr int N = 16384 * 1024 * 108;
  constexpr float BASELINE_ELEM_PER_CYCLE = 117.0f;
  constexpr float A100_SM_FREQ = 1.41e9f;

  auto host_src = std::make_unique<int[]>(N);
  auto host_dst = std::make_unique<int[]>(N);
  std::iota(host_src.get(), host_src.get() + N, 0);

  auto dev_src = make_cuda_unique<int>(N);
  auto dev_dst = make_cuda_unique<int>(N);

  CHECK_CUDA(cudaMemcpy(dev_src.get(), host_src.get(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  copy<<<BLOCKS, THREADS>>>(dev_src.get(), dev_dst.get(), N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaMemcpy(host_dst.get(), dev_dst.get(), N * sizeof(int),
                        cudaMemcpyDeviceToHost));

  const int cycles_taken = milliseconds * A100_SM_FREQ / 1e3;
  printf("%.0f elem/clk, Baseline cudaMemcpy %.0f elem/clk\n",
         static_cast<float>(N) / static_cast<float>(cycles_taken),
         BASELINE_ELEM_PER_CYCLE);

  if (memcmp(host_src.get(), host_dst.get(), N * sizeof(int)) != 0) {
    fprintf(stderr, "Arrays do not match\n");
    abort();
  }

  return 0;
  // No need for manual cudaFree or free - smart pointers handle cleanup
}