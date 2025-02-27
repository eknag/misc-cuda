// largestSquareWavefront.h (or .cuh if you prefer)
#pragma once

void runLargestSquareWavefront(
    const int* d_M,  // device pointer, matrix M
    int* d_D,        // device pointer, DP output
    int N, 
    int M_cols
);


extern __global__
void initDependencyCounts(int* depCount, int gridDimX, int gridDimY);

extern __global__
void largestSquareNoGridSyncKernel(
    const int* __restrict__ d_M,
    int*       __restrict__ d_D,
    int        N,
    int        M_cols,
    int        tileH,
    int        tileW,
    int*       depCount,
    int        gridDimX,
    int        gridDimY
);


void runLargestSquareNoGridSync(
    const int* d_M,
    int*       d_D,
    int        N,
    int        M_cols
);