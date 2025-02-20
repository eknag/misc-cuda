#include <assert.h>
#include <cstdio>
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
  // Ensure n equals the expected total.
  assert(n == total_items);

  int tid = threadIdx.x;
  int wid = threadIdx.x / 32;

  // Allocate shared memory for the copy, aligned to 16 bytes.
  __shared__ __align__(16) int smem[total_items];

  unsigned long long start = clock64();

  int base = wid * total_items;

  // Using cp.async requires operating on 16 bytes (4 ints) at a time.
  constexpr int vec_items = total_items / 4; // must be divisible by 4

  const int4 *src = reinterpret_cast<const int4 *>(A + base);
  int4 *smem_ptr = reinterpret_cast<int4 *>(&smem[base]);

#pragma unroll
  for (int i = 0; i < vec_items; ++i) {
    // Copy 16 bytes from global memory to shared memory asynchronously
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 :
                 : "r"(i * 16), "l"(src + i), "n"(16));
  }

  // Commit the group of asynchronous copies
  asm volatile("cp.async.commit_group;\n");
  // Wait until the committed group (1 group) finishes
  asm volatile("cp.async.wait_group 1;\n");

  __syncthreads(); // ensure data is ready

  // Now move the data from shared memory (smem) to global B.
  int4 *dst = reinterpret_cast<int4 *>(B + base);
  int4 *s_ptr = reinterpret_cast<int4 *>(&smem[base]);
#pragma unroll
  for (int i = 0; i < vec_items; ++i) {
    dst[i] = s_ptr[i];
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

  for (int i = 0; i < 10; i++) {
    assert(A[i] == B[i]);
  }

  // Print the per-thread cycle counts
  for (int i = 0; i < BLOCKS * THREADS; i++) {
    printf("Thread %d took %llu cycles per element\n", i, times_host[i] / n);
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