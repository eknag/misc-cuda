#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <malloc.h>
#include <numeric>

template <int ITEMS_PER_THREAD, int THREADS>
__global__ void copy_and_measure(const int *__restrict__ A, int *__restrict__ B,
                                 int n, unsigned long long *times) {
  constexpr int total_items = THREADS * ITEMS_PER_THREAD;
  // Ensure n equals the expected total.
  assert(n == total_items);
  int tid = threadIdx.x;

  // Allocate shared memory for the copy.
  __shared__ int smem[total_items];

  // Record start clock
  unsigned long long start = clock64();

  int base = tid * ITEMS_PER_THREAD;

  // Using cp.async requires operating on 16 bytes (4 ints) at a time.
  constexpr int vec_items =
      ITEMS_PER_THREAD / 4; // assuming ITEMS_PER_THREAD is divisible by 4

  const int4 *src = reinterpret_cast<const int4 *>(A + base);
  int4 *smem_ptr = reinterpret_cast<int4 *>(&smem[base]);

#pragma unroll
  for (int i = 0; i < vec_items; ++i) {
    // Asynchronously copy 16 bytes from global memory to shared memory.
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 :
                 : "l"(smem_ptr + i), "l"(src + i), "n"(16));
  }
  // Wait until all asynchronous copies of the current group have completed.
  asm volatile("cp.async.wait_group 0;\n");

  __syncthreads();

  // Now move the data from shared memory (smem) to global B.
  int4 *dst = reinterpret_cast<int4 *>(B + base);
  int4 *s_ptr = reinterpret_cast<int4 *>(&smem[base]);
#pragma unroll
  for (int i = 0; i < vec_items; ++i) {
    dst[i] = s_ptr[i];
  }

  // Record end clock including the global write.
  unsigned long long end = clock64();
  times[tid] = end - start;
}

int main() {
  constexpr int ITEMS_PER_THREAD = 100;
  constexpr int BLOCKS = 1;
  constexpr int THREADS = 32;
  constexpr int n = BLOCKS * THREADS * ITEMS_PER_THREAD;

  int *A = static_cast<int *>(malloc(sizeof(int) * n));
  int *B = static_cast<int *>(malloc(sizeof(int) * n));

  // Initialize A with consecutive values starting from 0
  std::iota(A, A + n, 0);

  // Device pointers
  int *A_dev = nullptr;
  int *B_dev = nullptr;

  // Allocate device memory
  cudaMalloc(&A_dev, n * sizeof(int));
  cudaMalloc(&B_dev, n * sizeof(int));

  // Copy data from host to device
  cudaMemcpy(A_dev, A, n * sizeof(int), cudaMemcpyHostToDevice);

  // Create a device array to store the cycle count for each thread
  unsigned long long *times_dev = nullptr;
  cudaMalloc(&times_dev, BLOCKS * THREADS * sizeof(unsigned long long));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch kernel
  copy_and_measure<ITEMS_PER_THREAD, THREADS>
      <<<BLOCKS, THREADS>>>(A_dev, B_dev, n, times_dev);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy back results
  cudaMemcpy(B, B_dev, n * sizeof(int), cudaMemcpyDeviceToHost);

  // Now copy the timing data from the device
  unsigned long long *times_host = new unsigned long long[BLOCKS * THREADS];
  cudaMemcpy(times_host, times_dev,
             BLOCKS * THREADS * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  // Print the per-thread cycle counts
  for (int i = 0; i < BLOCKS * THREADS; i++) {
    printf("Thread %d took %llu cycles per element\n", i, times_host[i]);
  }

  printf("Time taken: %f ms\n", milliseconds);

  // Cleanup
  delete[] times_host;
  cudaFree(times_dev);
  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A);
  free(B);

  return 0;
}
