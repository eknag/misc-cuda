#include <algorithm>
#include <vector>
#include <cassert>

// CPU reference for largest square of 1's
int largestSquareCPU(const std::vector<int>& M, int N, int M_cols)
{
    // DP array on CPU
    std::vector<int> D(N * M_cols, 0);
    int maxSide = 0;

    // Fill first row
    for (int j = 0; j < M_cols; ++j) {
        D[j] = M[j];
        maxSide = std::max(maxSide, D[j]);
    }

    // Fill first column
    for (int i = 1; i < N; ++i) {
        D[i*M_cols] = M[i*M_cols];
        maxSide = std::max(maxSide, D[i*M_cols]);
    }

    // Fill the rest
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < M_cols; ++j) {
            if (M[i*M_cols + j] == 1) {
                int top    = D[(i-1)*M_cols + j];
                int left   = D[i*M_cols + (j-1)];
                int diag   = D[(i-1)*M_cols + (j-1)];
                int val    = 1 + std::min({ top, left, diag });
                D[i*M_cols + j] = val;
                if (val > maxSide) {
                    maxSide = val;
                }
            } else {
                D[i*M_cols + j] = 0;
            }
        }
    }

    return maxSide;
}
