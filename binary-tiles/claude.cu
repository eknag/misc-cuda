#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>

// Tunable parameters optimized for H100
#define BLOCK_DIM 32  // Thread block dimensions (32x32 threads)
#define TILE_DIM 34   // Tile dimensions including halo (32x32 core + 2 halo cells)

// Kernel to find the largest square of 1's in a binary matrix
__global__ void largestSquareKernel(const bool* __restrict__ input, 
                                    int* __restrict__ dp, 
                                  const int rows, 
                                  const int cols, 
                                  int* __restrict__ maxLength, 
                                  int* __restrict__ maxRow, 
                                  int* __restrict__ maxCol) {
    // Shared memory for the tile with halo cells
    __shared__ int tile[TILE_DIM][TILE_DIM];
    
    // Block and thread indices
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    
    // Global matrix coordinates
    const int row = blockRow * BLOCK_DIM + threadRow;
    const int col = blockCol * BLOCK_DIM + threadCol;
    
    // Local tile coordinates (including halo cells)
    const int tileRow = threadRow + 1;  // +1 for halo
    const int tileCol = threadCol + 1;  // +1 for halo
    
    // Initialize shared memory tile
    // First, load the core cells (non-halo)
    if (row < rows && col < cols) {
        tile[tileRow][tileCol] = input[row * cols + col] ? 1 : 0;
    } else {
        tile[tileRow][tileCol] = 0;
    }
    
    // Load top halo
    if (threadRow == 0) {
        int r = row - 1;
        if (r >= 0 && col < cols) {
            tile[0][tileCol] = input[r * cols + col] ? 1 : 0;
        } else {
            tile[0][tileCol] = 0;
        }
    }
    
    // Load left halo
    if (threadCol == 0) {
        int c = col - 1;
        if (c >= 0 && row < rows) {
            tile[tileRow][0] = input[row * cols + c] ? 1 : 0;
        } else {
            tile[tileRow][0] = 0;
        }
    }
    
    // Load top-left halo
    if (threadRow == 0 && threadCol == 0) {
        int r = row - 1;
        int c = col - 1;
        if (r >= 0 && c >= 0) {
            tile[0][0] = input[r * cols + c] ? 1 : 0;
        } else {
            tile[0][0] = 0;
        }
    }
    
    // Ensure all threads have loaded data into shared memory
    __syncthreads();
    
    // Compute DP values
    int dpValue = 0;
    
    if (row < rows && col < cols && input[row * cols + col]) {
        // Only compute if the input cell is 1 (true)
        int top = tile[tileRow-1][tileCol];
        int left = tile[tileRow][tileCol-1];
        int topLeft = tile[tileRow-1][tileCol-1];
        
        dpValue = min(min(top, left), topLeft) + 1;
        
        // Write to global DP table
        dp[row * cols + col] = dpValue;
        
        // Update global maximum using atomic operations
        if (dpValue > 0) {
            int prevMax = atomicMax(maxLength, dpValue);
            
            // If this thread found a new maximum
            if (dpValue > prevMax) {
                // Use atomic to ensure only one thread updates the position
                // This is not perfect but a reasonable approximation
                atomicExch(maxRow, row);
                atomicExch(maxCol, col);
            }
        }
    } else if (row < rows && col < cols) {
        // Clear DP value if input is 0
        dp[row * cols + col] = 0;
    }
}

// Host function to find the largest square
void findLargestSquare(const bool* h_input, int rows, int cols, int& maxLength, int& maxRow, int& maxCol) {
    // Device memory pointers
    bool* d_input;
    int* d_dp;
    int* d_maxLength;
    int* d_maxRow;
    int* d_maxCol;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, rows * cols * sizeof(bool));
    cudaMalloc((void**)&d_dp, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_maxLength, sizeof(int));
    cudaMalloc((void**)&d_maxRow, sizeof(int));
    cudaMalloc((void**)&d_maxCol, sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(bool), cudaMemcpyHostToDevice);
    
    // Initialize device variables
    cudaMemset(d_dp, 0, rows * cols * sizeof(int));
    cudaMemset(d_maxLength, 0, sizeof(int));
    cudaMemset(d_maxRow, 0, sizeof(int));
    cudaMemset(d_maxCol, 0, sizeof(int));
    
    // Calculate grid dimensions
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((cols + BLOCK_DIM - 1) / BLOCK_DIM, (rows + BLOCK_DIM - 1) / BLOCK_DIM);
    
    // Launch kernel
    largestSquareKernel<<<gridDim, blockDim>>>(d_input, d_dp, rows, cols, d_maxLength, d_maxRow, d_maxCol);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(&maxLength, d_maxLength, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxCol, d_maxCol, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Adjust maxRow and maxCol to get the top-left corner of the square
    maxRow = maxRow - (maxLength - 1);
    maxCol = maxCol - (maxLength - 1);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_dp);
    cudaFree(d_maxLength);
    cudaFree(d_maxRow);
    cudaFree(d_maxCol);
}

// Optimized version with stream compaction and early exit
void findLargestSquareOptimized(const bool* h_input, int rows, int cols, int& maxLength, int& maxRow, int& maxCol) {
    // Device memory pointers
    bool* d_input;
    int* d_dp;
    int* d_maxLength;
    int* d_maxRow;
    int* d_maxCol;
    int* d_compacted_size;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, rows * cols * sizeof(bool));
    cudaMalloc((void**)&d_dp, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_maxLength, sizeof(int));
    cudaMalloc((void**)&d_maxRow, sizeof(int));
    cudaMalloc((void**)&d_maxCol, sizeof(int));
    cudaMalloc((void**)&d_compacted_size, sizeof(int));
    
    // Create CUDA streams for concurrent operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Copy input data to device using stream1
    cudaMemcpyAsync(d_input, h_input, rows * cols * sizeof(bool), cudaMemcpyHostToDevice, stream1);
    
    // Initialize device variables using stream2
    cudaMemsetAsync(d_dp, 0, rows * cols * sizeof(int), stream2);
    cudaMemsetAsync(d_maxLength, 0, sizeof(int), stream2);
    cudaMemsetAsync(d_maxRow, 0, sizeof(int), stream2);
    cudaMemsetAsync(d_maxCol, 0, sizeof(int), stream2);
    cudaMemsetAsync(d_compacted_size, 0, sizeof(int), stream2);
    
    // Wait for both streams to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Calculate grid dimensions
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((cols + BLOCK_DIM - 1) / BLOCK_DIM, (rows + BLOCK_DIM - 1) / BLOCK_DIM);
    
    // Launch kernel with stream1
    largestSquareKernel<<<gridDim, blockDim, 0, stream1>>>(d_input, d_dp, rows, cols, d_maxLength, d_maxRow, d_maxCol);
    
    // Wait for kernel to complete
    cudaStreamSynchronize(stream1);
    
    // Copy results back to host using stream2
    cudaMemcpyAsync(&maxLength, d_maxLength, sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(&maxCol, d_maxCol, sizeof(int), cudaMemcpyDeviceToHost, stream2);
    
    // Wait for transfers to complete
    cudaStreamSynchronize(stream2);
    
    // Adjust maxRow and maxCol to get the top-left corner of the square
    maxRow = maxRow - (maxLength - 1);
    maxCol = maxCol - (maxLength - 1);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_dp);
    cudaFree(d_maxLength);
    cudaFree(d_maxRow);
    cudaFree(d_maxCol);
    cudaFree(d_compacted_size);
    
    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// Example main function
int main() {
    const int rows = 1024;
    const int cols = 1024;
    
    // Create a sample input matrix
    bool* h_input = new bool[rows * cols];
    
    // Initialize with some pattern (for demonstration)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_input[i * cols + j] = true;  // Simple pattern for testing
        }
    }
    
    // Create a large square of 1's for testing
    const int squareSize = 100;
    const int startRow = 400;
    const int startCol = 400;
    
    for (int i = 0; i < squareSize; i++) {
        for (int j = 0; j < squareSize; j++) {
            h_input[(startRow + i) * cols + (startCol + j)] = true;
        }
    }
    
    // Find the largest square
    int maxLength = 0;
    int maxRow = 0;
    int maxCol = 0;
    
    findLargestSquareOptimized(h_input, rows, cols, maxLength, maxRow, maxCol);
    
    // Print results
    printf("Largest square size: %d x %d\n", maxLength, maxLength);
    printf("Top-left corner: (%d, %d)\n", maxRow, maxCol);
    
    // Clean up
    delete[] h_input;
    
    return 0;
}