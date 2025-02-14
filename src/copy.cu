#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <malloc.h>
#include <numeric>

template <int ITEMS_PER_THREAD, int THREADS>
__global__ void copy_and_measure(const int *A, int *B, int n,
                                 unsigned long long *times) {
  constexpr int N = THREADS * ITEMS_PER_THREAD;
  assert(n == N);

  // Compute global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= THREADS) {
    return;
  }

  // Record start clock
  unsigned long long start = clock64();

  // Perform the copy in a grid-stride loop
  for (int i = tid; i < N; i += 32) {
    B[i] = A[i];
  }

  // Record end clock
  unsigned long long end = clock64();

  // Store the cycle count in the device array
  times[tid] = end - start;
}

int main() {
  constexpr int ITEMS_PER_THREAD = 10000;
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
    printf("Thread %d took %llu cycles per element\n", i,
           times_host[i] / ITEMS_PER_THREAD);
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
