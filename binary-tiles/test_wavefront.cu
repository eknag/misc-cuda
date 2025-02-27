#include <gtest/gtest.h>
#include <chrono>

#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "wavefront.h"  // your wavefront kernel declarations
#include "cpu_reference.h"           // the CPU reference from above

// Helper macro for CUDA error checks
#define CUDA_CHECK(expr) do {                                     \
    cudaError_t err = (expr);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",           \
                err, __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);                                                  \
    }                                                             \
} while(0)

// A function to find max(D) on the GPU-based DP array
// (or you can copy D back and do it on CPU)
int maxInDeviceArray(const int* d_D, int size) {
    // For simplicity, copy back to CPU and do a max reduction
    std::vector<int> h_D(size);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, size * sizeof(int), cudaMemcpyDeviceToHost));
    return *std::max_element(h_D.begin(), h_D.end());
}

//------------------------------------
// Now let's define the test fixture
//------------------------------------
class LargestSquareWavefrontTest : public ::testing::Test {
protected:
    // This fixture can hold shared data or do repeated setup
    void SetUp() override {
        // Nothing special here
    }

    void TearDown() override {
        // Cleanup if needed
    }
};

// Utility for running a single test (N x M, with given data).
int runTestCase(const std::vector<int>& h_M, int N, int M_cols)
{
    // 1. Allocate device arrays
    int size = N * M_cols * sizeof(int);
    int* d_M = nullptr;
    int* d_D = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));

    // 2. Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), size, cudaMemcpyHostToDevice));

    // 3. Optionally initialize d_D with the first row/col or just zero
    CUDA_CHECK(cudaMemset(d_D, 0, size));

    // 4. Call the wavefront kernel wrapper
    runLargestSquareNoGridSync(d_M, d_D, N, M_cols);

    // 5. Find the max value in d_D
    int gpuMaxSide = maxInDeviceArray(d_D, N*M_cols);

    // 6. Free device memory
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_D));

    return gpuMaxSide;
}

//------------------------------------
// TEST CASES
//------------------------------------

// Test 1: Small example
TEST_F(LargestSquareWavefrontTest, Small3x3Mixed) {
    int N = 3, M_cols = 3;
    // Example matrix:
    // 1 1 0
    // 1 1 1
    // 1 1 1
    std::vector<int> h_M = {
        1, 1, 0,
        1, 1, 1,
        1, 1, 1
    };
    // CPU reference
    int cpuResult = largestSquareCPU(h_M, N, M_cols);

    // GPU result
    int gpuResult = runTestCase(h_M, N, M_cols);

    EXPECT_EQ(cpuResult, gpuResult);
}

// Test 2: All zeros
TEST_F(LargestSquareWavefrontTest, AllZeros) {
    int N = 4, M_cols = 5;
    std::vector<int> h_M(N * M_cols, 0);

    int cpuResult = largestSquareCPU(h_M, N, M_cols);
    int gpuResult = runTestCase(h_M, N, M_cols);
    EXPECT_EQ(cpuResult, gpuResult);
}

// Test 3: All ones
TEST_F(LargestSquareWavefrontTest, AllOnes) {
    int N = 5, M_cols = 5;
    std::vector<int> h_M(N * M_cols, 1);

    int cpuResult = largestSquareCPU(h_M, N, M_cols);
    int gpuResult = runTestCase(h_M, N, M_cols);
    EXPECT_EQ(cpuResult, gpuResult);
}

// Test 4: Random small
TEST_F(LargestSquareWavefrontTest, RandomSmall) {
    int N = 8, M_cols = 10;
    std::vector<int> h_M(N * M_cols);
    srand(1234);
    for (auto & val : h_M) {
        val = (rand() % 2); // 0 or 1
    }

    int cpuResult = largestSquareCPU(h_M, N, M_cols);
    int gpuResult = runTestCase(h_M, N, M_cols);
    EXPECT_EQ(cpuResult, gpuResult);
}

// Test 5: Larger random (adjust sizes as you like)
TEST_F(LargestSquareWavefrontTest, LargerRandom) {
    int N = 256, M_cols = 300;
    std::vector<int> h_M(N * M_cols);
    srand(5678);
    for (auto & val : h_M) {
        val = (rand() % 2); // 0 or 1
    }

    int cpuResult = largestSquareCPU(h_M, N, M_cols);
    int gpuResult = runTestCase(h_M, N, M_cols);
    EXPECT_EQ(cpuResult, gpuResult);
}


TEST_F(LargestSquareWavefrontTest, ProfileKernelRuntime) {
    // Here we choose a bigger matrix to get a measurable runtime.
    const int N = 4096;
    const int M_cols = 4096;
    const int numIters = 5;  // how many times we run the kernel to measure

    // Create random input
    std::vector<int> h_M(N * M_cols);
    std::mt19937 rng(12345678);
    std::uniform_int_distribution<int> dist(0, 1000);
    for (int i = 0; i < N*M_cols; ++i) {
        h_M[i] = dist(rng) != 0;
    }

    // Profile CPU reference time
    std::chrono::high_resolution_clock::time_point cpuStart = std::chrono::high_resolution_clock::now();
    int cpuResult = largestSquareCPU(h_M, N, M_cols);
    std::chrono::high_resolution_clock::time_point cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;

    // Allocate device buffers
    size_t sizeBytes = N * M_cols * sizeof(int);
    int* d_M = nullptr;
    int* d_D = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_D, sizeBytes));

    // Copy input
    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), sizeBytes, cudaMemcpyHostToDevice));

    // Warm-up launch (optional)
    CUDA_CHECK(cudaMemset(d_D, 0, sizeBytes));
    runLargestSquareNoGridSync(d_M, d_D, N, M_cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // We'll measure multiple iterations
    std::vector<float> timings(numIters, 0.0f);

    for (int iter = 0; iter < numIters; iter++) {
        // Clear DP array
        CUDA_CHECK(cudaMemset(d_D, 0, sizeBytes));

        // Create events
        cudaEvent_t startEvt, stopEvt;
        CUDA_CHECK(cudaEventCreate(&startEvt));
        CUDA_CHECK(cudaEventCreate(&stopEvt));

        // Record start
        CUDA_CHECK(cudaEventRecord(startEvt));

        // Launch kernel
        runLargestSquareNoGridSync(d_M, d_D, N, M_cols);

        // Record stop
        CUDA_CHECK(cudaEventRecord(stopEvt));

        // Synchronize so we can measure time
        CUDA_CHECK(cudaEventSynchronize(stopEvt));

        // Compute elapsed
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvt, stopEvt));
        timings[iter] = ms;

        // Cleanup events
        CUDA_CHECK(cudaEventDestroy(startEvt));
        CUDA_CHECK(cudaEventDestroy(stopEvt));
    }

    // Copy back the final DP array to check correctness
    int gpuMaxSide = maxInDeviceArray(d_D, N*M_cols);

    // Compare correctness once
    EXPECT_EQ(cpuResult, gpuMaxSide);

    // Compute average kernel time
    float sum = 0.0f;
    for (auto t : timings) {
        sum += t;
    }
    float avgMs = sum / numIters;

    // Print or log the results
    std::cout << "[ProfileKernelRuntime] N=" << N 
              << " M_cols=" << M_cols << std::endl
              << " average CUDA time: " << avgMs << " ms\n" 
              << " speedup: " << cpuElapsed.count() / avgMs << "x\n" 
              << " Max side: " << gpuMaxSide << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_D));
}

//------------------------------------
// main() for Google Test
//------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // Optional: set a specific CUDA device
    CUDA_CHECK(cudaSetDevice(0));
    return RUN_ALL_TESTS();
}
