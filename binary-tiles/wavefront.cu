#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Error‐checking wrapper (optional)
#define CUDA_CHECK(expr)  do {                         \
    cudaError_t ___err = (expr);                       \
    if (___err != cudaSuccess) {                       \
      fprintf(stderr,                                   \
              "CUDA error %d at %s:%d: %s\n",          \
              ___err, __FILE__, __LINE__,             \
              cudaGetErrorString(___err));            \
      exit(1);                                         \
    }                                                  \
} while(0)


// Each DP array entry is at D[i*M_cols + j], row-major.
__global__ void largestSquareWavefrontKernel(
    const int*  __restrict__ d_M, // 0/1 input, size N*M_cols
    int*        __restrict__ d_D, // DP array, size N*M_cols
    int         N,
    int         M_cols,
    int         tileH,  // tile size in row dimension
    int         tileW   // tile size in col dimension
) {
    // Cooperative groups handle for grid-wide sync
    cg::grid_group grid = cg::this_grid();
    
    // Identify which tile this block handles
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // The top-left corner of this tile in the full matrix
    int startRow = by * tileH;
    int startCol = bx * tileW;

    // Thread indices within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Allocate shared memory for a tile of size (tileH+1) x (tileW+1)
    // We add +1 if we want a small boundary region for convenience.
    extern __shared__ int sTile[]; 
    // Or you can do static if tile sizes are fixed: __shared__ int sTile[TILE_H+1][TILE_W+1];

    // Number of tile rows & cols we actually have, in case we are at boundary
    int localH = min(tileH, N - startRow);
    int localW = min(tileW, M_cols - startCol);

    // We will iterate over all wavefront steps. The total # of wavefront steps
    // for the grid = gridDim.x + gridDim.y - 1
    // But we do not know which diagonal we are on until we do the loop outside.
    // Instead we do a loop inside, and at each iteration we do:
    //    if (bx + by == diag) ... compute ...
    
    // We do not want to do the DP for every diagonal across the entire matrix in a single tile.
    // We only do it once we know the top & left neighbors are done for that diagonal of tiles.
    int totalDiag = gridDim.x + gridDim.y - 1;

    for (int diag = 0; diag < totalDiag; diag++) {
        // If it is *this* tile's turn (bx + by = diag), do the tile computation
        if ((bx + by) == diag) {
            //----------------------------------------
            // 1) Load data from global memory into shared mem
            //    We need the "border" from d_D if we want to do
            //    the DP update referencing top, left, diag for each cell.
            //    For safety, we can also load the M[] tile into shared memory
            //    (or just read from global directly).
            //----------------------------------------
            for (int localId = tx + ty * blockDim.x; 
                     localId < (localH * localW);
                     localId += (blockDim.x * blockDim.y)) {
                int rLocal = localId / localW;
                int cLocal = localId % localW;
                int rGlob  = startRow + rLocal;
                int cGlob  = startCol + cLocal;

                // We can directly read M and an old D. 
                // We'll store old D in shared memory for quick access.
                sTile[rLocal * localW + cLocal] = d_D[rGlob * M_cols + cGlob];
            }
            
            // We also need the boundary row and column from D if not out of range
            // (top boundary from row-1, left boundary from col-1).
            // This is somewhat tedious but basically you fetch:
            // d_D[(startRow - 1)*M_cols + (startCol + c)] or similar
            // We'll omit the full detail for brevity or do it inline in the DP step.

            __syncthreads();

            //----------------------------------------
            // 2) Compute the DP for the tile in local wavefront order
            //    Because each tile might still have an internal left‐to‐right dependency,
            //    we can do a row‐by‐row or mini‐diagonal approach inside the tile.
            //----------------------------------------
            for (int r = 0; r < localH; r++) {
                for (int c = 0; c < localW; c++) {
                    // Each thread can pick off some subset of (r,c).
                    // A simple approach: let a single warp or single thread do this, 
                    // or do a small wavefront inside the tile. 
                    // Here is a simplistic approach: use 1D thread assignment:
                    int lid = tx + ty * blockDim.x;
                    int wh  = blockDim.x * blockDim.y;
                    if (((r * localW) + c) % wh == lid) {
                        // global coords
                        int R = startRow + r;
                        int C = startCol + c;

                        // If M[R,C] == 0 => D[R,C] = 0
                        // else D[R,C] = 1 + min(D[R-1,C], D[R,C-1], D[R-1,C-1])
                        // We'll read from the global M array on the fly:
                        int valM = d_M[R * M_cols + C];
                        if (valM == 0) {
                            sTile[r * localW + c] = 0;
                        } else {
                            // Need neighbors from "top", "left", "diag"
                            // Possibly from sTile if r>0 && c>0, else from global d_D if on tile boundary
                            int top    = (r == 0) 
                                          ? ( (R>0) ? d_D[(R-1)*M_cols + C] : 0 )
                                          : sTile[(r-1)*localW + c];
                            int left   = (c == 0) 
                                          ? ( (C>0) ? d_D[R*M_cols + (C-1)] : 0 )
                                          : sTile[r*localW + (c-1)];
                            int diag   = (r == 0 || c == 0)
                                          ? ( (R>0 && C>0) ? d_D[(R-1)*M_cols + (C-1)] : 0 )
                                          : sTile[(r-1)*localW + (c-1)];
                            int newVal = 1 + min(top, min(left, diag));
                            sTile[r * localW + c] = newVal;
                        }
                    }
                }
                __syncthreads();
            }

            //----------------------------------------
            // 3) Write results back to global memory
            //----------------------------------------
            for (int localId = tx + ty * blockDim.x; 
                     localId < (localH * localW);
                     localId += (blockDim.x * blockDim.y)) {
                int rLocal = localId / localW;
                int cLocal = localId % localW;
                int rGlob  = startRow + rLocal;
                int cGlob  = startCol + cLocal;
                d_D[rGlob * M_cols + cGlob] = sTile[rLocal * localW + cLocal];
            }

        } // end if (bx+by == diag)

        // **Grid-wide** barrier to ensure that all tiles at diagonal `diag` 
        // have finished writing before the next diagonal reads them.
        grid.sync();
    }
}


void runLargestSquareWavefront(
    const int* d_M, // device pointer, M is NxM_cols (0/1)
    int*       d_D, // device pointer, same size, will hold the DP
    int        N,
    int        M_cols
) {
    //-------------------------------------------------------------------------
    // 1) Choose tile & block dimensions
    //    Use bigger tiles to reduce total blocks in the grid.
    //-------------------------------------------------------------------------
    int tileW = 64; 
    int tileH = 64;
    // Each block will conceptually process tileH x tileW cells.
    // We'll assign a moderate block size, e.g. (8,8).
    dim3 blockSize(8, 8);

    // Compute grid size in each dimension
    int gridDimX = (M_cols + tileW - 1) / tileW;
    int gridDimY = (N + tileH - 1) / tileH;
    dim3 gridSize(gridDimX, gridDimY);

    // Shared memory needed per block = tileH*tileW*sizeof(int) for sTile.
    // (You can add boundary overhead if needed.)
    size_t sharedMem = tileH * tileW * sizeof(int);

    //-------------------------------------------------------------------------
    // 2) Check if the device supports cooperative launch
    //-------------------------------------------------------------------------
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    if (!deviceProp.cooperativeLaunch) {
        fprintf(stderr, "Device does not support cooperativeLaunch!\n");
        exit(1);
    }

    //-------------------------------------------------------------------------
    // 3) Check occupancy to see if we can run all blocks concurrently
    //-------------------------------------------------------------------------
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        largestSquareWavefrontKernel,   // pointer to your __global__ kernel
        blockSize.x * blockSize.y,
        sharedMem
    ));
    int maxBlocks = blocksPerSM * deviceProp.multiProcessorCount;

    int requestedBlocks = gridDimX * gridDimY;
    if (requestedBlocks > maxBlocks) {
        fprintf(stderr,
                "ERROR: Cooperative launch requires %d blocks, but only %d can fit concurrently.\n"
                "Try bigger tiles or fewer blocks.\n",
                requestedBlocks, maxBlocks);
        exit(1);
    }

    //-------------------------------------------------------------------------
    // 4) Prepare kernel arguments
    //-------------------------------------------------------------------------
    void* kernelArgs[] = {
        (void*)&d_M,
        (void*)&d_D,
        (void*)&N,
        (void*)&M_cols,
        (void*)&tileH,
        (void*)&tileW
    };

    //-------------------------------------------------------------------------
    // 5) Launch in cooperative mode
    //-------------------------------------------------------------------------
    CUDA_CHECK( 
        cudaLaunchCooperativeKernel(
            (void*)largestSquareWavefrontKernel,
            gridSize,
            blockSize,
            kernelArgs,
            sharedMem,
            /*stream=*/0
        )
    );

    // If needed, sync after kernel
    CUDA_CHECK(cudaDeviceSynchronize());
}


#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C"
__global__ void largestSquareNoGridSyncKernel(
    const int* __restrict__ d_M,  // input matrix (N * M_cols)
    int*       __restrict__ d_D,  // DP array (N * M_cols)
    int        N, 
    int        M_cols,
    const int        tileH, 
    const int        tileW,
    int*       depCount,  // global array of size gridDimX * gridDimY
    const int        gridDimX,  // how many tiles horizontally
    const int        gridDimY   // how many tiles vertically
)
{
    extern __shared__ int sTile[]; 
    int blockThreads = blockDim.x * blockDim.y;
    int lid = threadIdx.x;



    for (int bx = blockIdx.x; bx < gridDimX; bx += gridDim.x){
        for (int by = blockIdx.y; by < gridDimY; by += gridDim.y){
            int startRow = by * tileH;
            int startCol = bx * tileW;
        
            int localH = min(tileH, N - startRow);
            int localW = min(tileW, M_cols - startCol);
            bool ready = false;
            while (!ready) {
                ready = true;
                // Check left tile (bx-1, by)
                if (bx > 0) {
                    if (atomicAdd(&depCount[by * gridDimX + (bx-1)], 0) == 0) 
                        ready = false;
                }
                // Check top tile (bx, by-1)
                if (by > 0) {
                    if (atomicAdd(&depCount[(by-1) * gridDimX + bx], 0) == 0) 
                        ready = false;
                }
                // Check diagonal tile (bx-1, by-1)
                if (bx > 0 && by > 0) {
                    if (atomicAdd(&depCount[(by-1) * gridDimX + (bx-1)], 0) == 0) 
                        ready = false;
                }
            }

            // -------------------------------------------------
            // 2) Load a wavefront from global D or M, compute in sTile
            // -------------------------------------------------

            // 2a) Initialize sTile to -1 (or 0) so we know "not computed yet"
            int tileSize = localH * localW;
            for (int idx = lid; idx < tileSize; idx += blockThreads) {
                sTile[idx] = -1;
            }
            __syncthreads();

            // 2b) Diagonal wavefront update in the tile
            //     For d=0..(localH+localW-2), the cells that satisfy r+c=d
            //     can be computed in parallel.

            for (int d = 0; d < (localH + localW - 1); d++) {
                // We'll assign columns in steps of 'blockThreads' to each thread
                for (int c = lid; c < localW; c += blockThreads) {
                    int r = d - c; // so r + c = d
                    if (r >= 0 && r < localH) {
                        // Global coords
                        int R = startRow + r;
                        int C = startCol + c;

                        int valM = d_M[R * M_cols + C];
                        if (valM == 0) {
                            sTile[r * localW + c] = 0;
                        } else {
                            // we need top, left, diag from sTile if in-bounds, else from d_D
                            int top  = 0;
                            int left = 0;
                            int diag = 0;

                            // top => (r-1, c) in tile if r>0
                            if (r > 0) {
                                top = sTile[(r - 1) * localW + c];
                            } else if (R > 0) {
                                // read from global if above tile
                                top = d_D[(R - 1) * M_cols + C];
                            }

                            // left => (r, c-1) in tile if c>0
                            if (c > 0) {
                                left = sTile[r * localW + (c - 1)];
                            } else if (C > 0) {
                                left = d_D[R * M_cols + (C - 1)];
                            }

                            // diag => (r-1, c-1) in tile if r>0 && c>0
                            if (r > 0 && c > 0) {
                                diag = sTile[(r - 1) * localW + (c - 1)];
                            } else if (R > 0 && C > 0) {
                                diag = d_D[(R - 1) * M_cols + (C - 1)];
                            }

                            int val = 1 + min(top, min(left, diag));
                            sTile[r * localW + c] = val;
                        }
                    }
                }
                __syncthreads(); // finish diagonal d before d+1
            }

            // -------------------------------------------------
            // 2c) Write final tile results back to d_D
            // -------------------------------------------------
            for (int idx = lid; idx < tileSize; idx += blockThreads) {
                int rLocal = idx / localW;
                int cLocal = idx % localW;
                int R = startRow + rLocal;
                int C = startCol + cLocal;

                d_D[R * M_cols + C] = sTile[idx];
            }
            __syncthreads();

            __threadfence();
            // Mark ourselves done
            depCount[by * gridDimX + bx] = 1;
        }
    }
}

void runLargestSquareNoGridSync(
    const int* d_M, // device pointer
    int*       d_D, // device pointer
    int        N,
    int        M_cols
){
    // 1) Tiling setup
    int tileW = 64;
    int tileH = 64;
    int gridDimX = (M_cols + tileW - 1) / tileW;
    int gridDimY = (N + tileH - 1) / tileH;

    // 2) Alloc depCount array [gridDimX * gridDimY]
    int totalTiles = gridDimX * gridDimY;
    int* d_depCount = nullptr;
    cudaMalloc(&d_depCount, totalTiles * sizeof(int));

    // 3) Initialize each tile's dependencies (the code that sets up
    //    how many dependencies each tile has, or sets them to 0 if we do the 'spin' approach).
    //    But here, we do "atomic sub approach" => each tile starts with '0'. 
    //    Actually we are checking neighbors, so let's just set them all to 0 for consistency.
    cudaMemset(d_depCount, 0, totalTiles*sizeof(int));

    // 4) 2D block & grid
    dim3 blockSize(32 * 32);
    dim3 gridSize(24, 12);

    // 5) Shared memory size = tileH*tileW * sizeof(int)
    size_t sharedMem = tileH * tileW * sizeof(int);

    // 6) Launch
    largestSquareNoGridSyncKernel<<<gridSize, blockSize, sharedMem>>>(
        d_M, d_D,
        N, M_cols,
        tileH, tileW,
        d_depCount,
        gridDimX, gridDimY
    );
    cudaDeviceSynchronize();

    cudaFree(d_depCount);
}