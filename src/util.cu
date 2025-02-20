#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(ans)                                                        \
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

struct CudaDeleter {
  void operator()(void *ptr) const { cudaFree(ptr); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
cuda_unique_ptr<T> make_cuda_unique(size_t n) {
  T *ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T)));
  return cuda_unique_ptr<T>(ptr);
}