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
__global__ void copy_and_measure(const int *__restrict__ A, int *__restrict__ B,
                                 int n, unsigned long long *start_cycle,
                                 unsigned long long *end_cycle) {

  constexpr int vector_elements = sizeof(VEC_T) / sizeof(int);
  constexpr int tile_vectors = TILE_SIZE / vector_elements;
  __shared__ __align__(16) VEC_T smem[tile_vectors];

  const int tid = threadIdx.x;
  const int iters = n / TILE_SIZE;

  start_cycle[tid] = clock64();

  for (int iter = 0; iter < iters; ++iter) {

#pragma unroll
    for (int i = 0; i < tile_vectors; i += THREADS) {
      __pipeline_memcpy_async(
          &smem[i + tid],
          &reinterpret_cast<const VEC_T *>(A)[i + tid + iter * tile_vectors],
          sizeof(VEC_T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
#pragma unroll
    for (int i = 0; i < tile_vectors; i += THREADS) {
      reinterpret_cast<VEC_T *>(B)[i + tid + iter * tile_vectors] =
          smem[i + tid];
    }
  }
  __threadfence(); // Ensure all threads have finished writing to global
                   // memory

  end_cycle[tid] = clock64();
}

int main() {
  constexpr int TILE_SIZE = 8192;
  constexpr int ITEMS_PER_WARP = TILE_SIZE * 1024;
  constexpr int BLOCKS = 1;
  constexpr int THREADS = 256;
  constexpr int n = BLOCKS * ITEMS_PER_WARP;

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

  // Create a device array to store the cycle count for each thread
  unsigned long long *start_cycle_dev = nullptr;
  unsigned long long *end_cycle_dev = nullptr;

  gpuErrchk(cudaMalloc(&start_cycle_dev,
                       BLOCKS * THREADS * sizeof(unsigned long long)));
  gpuErrchk(cudaMalloc(&end_cycle_dev,
                       BLOCKS * THREADS * sizeof(unsigned long long)));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch kernel
  copy_and_measure<TILE_SIZE, THREADS, int4>
      <<<BLOCKS, THREADS>>>(A_dev, B_dev, n, start_cycle_dev, end_cycle_dev);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  gpuErrchk(cudaPeekAtLastError());

  // Copy back results
  gpuErrchk(cudaMemcpy(B, B_dev, n * sizeof(int), cudaMemcpyDeviceToHost));

  // Now copy the timing data from the device
  unsigned long long *start_cycle_host =
      new unsigned long long[BLOCKS * THREADS];
  gpuErrchk(cudaMemcpy(start_cycle_host, start_cycle_dev,
                       BLOCKS * THREADS * sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost));

  unsigned long long *end_cycle_host = new unsigned long long[BLOCKS * THREADS];
  gpuErrchk(cudaMemcpy(end_cycle_host, end_cycle_dev,
                       BLOCKS * THREADS * sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost));

  // Print the per-thread cycle counts
  unsigned long long min_s = UINT64_MAX;
  unsigned long long max_e = 0;

  for (int i = 0; i < BLOCKS * THREADS; i++) {
    min_s = std::min(min_s, start_cycle_host[i]);
    max_e = std::max(max_e, end_cycle_host[i]);
  }

  unsigned long long max_cycles = max_e - min_s;

  printf("%.3f cycles per element\n", static_cast<float>(max_cycles) / n);

  printf("Time taken: %0.3f ms\n", milliseconds);

  for (int i = 0; i < n; i++) {
    if (A[i] != B[i]) {
      printf("Mismatch at index %d: %d != %d\n", i, A[i], B[i]);
    }
    assert(A[i] == B[i]);
  }

  // Cleanup
  delete[] start_cycle_host;
  delete[] end_cycle_host;
  cudaFree(start_cycle_dev);
  cudaFree(end_cycle_dev);
  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A);
  free(B);

  return 0;
}