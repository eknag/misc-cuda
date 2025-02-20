#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cuda/pipeline>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <numeric>

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <int TILE_SIZE, int THREADS, typename VEC_T>
__global__ void copy(const int *__restrict__ A, int *__restrict__ B, int n) {

  constexpr int vector_elements = sizeof(VEC_T) / sizeof(int);
  constexpr int tile_vectors = TILE_SIZE / vector_elements;
  __shared__ __align__(16) VEC_T smem[tile_vectors];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int total_blocks = gridDim.x;
  const int items_per_block = n / total_blocks;
  const int block_start = bid * items_per_block;
  const int block_end =
      (bid == total_blocks - 1) ? n : block_start + items_per_block;
  const int iters = (block_end - block_start) / TILE_SIZE;

  for (int iter = 0; iter < iters; ++iter) {
    const int offset = (block_start + iter * TILE_SIZE) / vector_elements;

#pragma unroll
    for (int i = 0; i < tile_vectors; i += THREADS) {
      __pipeline_memcpy_async(
          &smem[i + tid], &reinterpret_cast<const VEC_T *>(A)[offset + i + tid],
          sizeof(VEC_T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

#pragma unroll
    for (int i = 0; i < tile_vectors; i += THREADS) {
      reinterpret_cast<VEC_T *>(B)[offset + i + tid] = smem[i + tid];
    }
  }
}

int main() {
  constexpr int TILE_SIZE = 8192;
  constexpr int THREADS = 256;
  constexpr int BLOCKS = 108; // Increased number of blocks
  constexpr int n = TILE_SIZE * 1024 * 108;

  int *A = static_cast<int *>(malloc(sizeof(int) * n));
  int *B = static_cast<int *>(malloc(sizeof(int) * n));

  // Initialize A with consecutive values starting from 0
  std::iota(A, A + n, 0);

  // Device pointers
  int *A_dev = nullptr;
  int *B_dev = nullptr;

  // Allocate device memory
  gpuErrchk(cudaMalloc(&A_dev, n * sizeof(int)));
  gpuErrchk(cudaMalloc(&B_dev, n * sizeof(int)));

  // Copy data from host to device
  gpuErrchk(cudaMemcpy(A_dev, A, n * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch kernel
  copy<TILE_SIZE, THREADS, int4><<<BLOCKS, THREADS>>>(A_dev, B_dev, n);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  gpuErrchk(cudaPeekAtLastError());

  // Copy back results
  gpuErrchk(cudaMemcpy(B, B_dev, n * sizeof(int), cudaMemcpyDeviceToHost));

  constexpr float A100_HBM_BW = 1.6e12;
  constexpr float A100_SM_FREQ = 1.41e9;
  constexpr float A100_SM_CYCLES_PER_MS = A100_SM_FREQ / 1e3;
  constexpr float bytes_per_cycle = A100_HBM_BW / A100_SM_FREQ;
  constexpr float cycles_per_element = 2 * sizeof(int) / bytes_per_cycle;

  const int cycles_taken = milliseconds * A100_SM_CYCLES_PER_MS;

  printf("%.3f cycles per element, best cast %.3f\n",
         static_cast<float>(cycles_taken) / n, cycles_per_element);

  printf("Time taken: %0.3f ms\n", milliseconds);

  for (int i = 0; i < n; i++) {
    if (A[i] != B[i]) {
      printf("Mismatch at index %d: %d != %d\n", i, A[i], B[i]);
    }
    assert(A[i] == B[i]);
  }

  // Cleanup

  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A);
  free(B);

  return 0;
}