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

template <int ITEMS_PER_WARP, int THREADS>
__global__ void copy_and_measure(const int *__restrict__ A, int *__restrict__ B,
                                 int n, unsigned long long *times) {

  constexpr int total_items = ITEMS_PER_WARP;
  constexpr int items_per_thread = total_items / THREADS;
  constexpr int vector_size =
      4; // Using int4, which processes 4 integers at a time
  constexpr int vectors_per_thread = items_per_thread / vector_size;

  static_assert(total_items % (THREADS * vector_size) == 0,
                "ITEMS_PER_WARP must be divisible by THREADS * vector_size");

  int tid = threadIdx.x;

  __shared__ __align__(16) int smem[total_items];

  unsigned long long start = clock64();

  // Vectorized load from global memory to shared memory
#pragma unroll
  for (int i = 0; i < vectors_per_thread; ++i) {
    int4 vec = reinterpret_cast<const int4 *>(A)[tid + i * THREADS];
    reinterpret_cast<int4 *>(smem)[tid + i * THREADS] = vec;
  }

  __syncthreads(); // Ensure all threads have finished writing to shared memory

  // Vectorized store from shared memory to global memory
#pragma unroll
  for (int i = 0; i < vectors_per_thread; ++i) {
    int4 vec = reinterpret_cast<int4 *>(smem)[tid + i * THREADS];
    reinterpret_cast<int4 *>(B)[tid + i * THREADS] = vec;
  }

  unsigned long long end = clock64();
  times[tid] = end - start;
}

int main() {
  constexpr int ITEMS_PER_WARP = 8192;
  constexpr int BLOCKS = 1;
  constexpr int THREADS = 32;
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
  unsigned long long *times_dev = nullptr;
  gpuErrchk(
      cudaMalloc(&times_dev, BLOCKS * THREADS * sizeof(unsigned long long)));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch kernel
  copy_and_measure<ITEMS_PER_WARP, THREADS>
      <<<BLOCKS, THREADS>>>(A_dev, B_dev, n, times_dev);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  gpuErrchk(cudaPeekAtLastError());

  // Copy back results
  gpuErrchk(cudaMemcpy(B, B_dev, n * sizeof(int), cudaMemcpyDeviceToHost));

  // Now copy the timing data from the device
  unsigned long long *times_host = new unsigned long long[BLOCKS * THREADS];
  gpuErrchk(cudaMemcpy(times_host, times_dev,
                       BLOCKS * THREADS * sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost));

  // Print the per-thread cycle counts
  for (int i = 0; i < BLOCKS * THREADS; i++) {
    printf("Thread %d took %.1f cycles per element\n", i,
           static_cast<float>(times_host[i]) / n);
  }

  printf("Time taken: %f ms\n", milliseconds);

  for (int i = 0; i < n; i++) {
    assert(A[i] == B[i]);
  }

  // Cleanup
  delete[] times_host;
  cudaFree(times_dev);
  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A);
  free(B);

  return 0;
}