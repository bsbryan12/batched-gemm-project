#include "gemm.h"
#include <stdio.h>

// ---------------------------------------------------------------------------
// Naive batched GEMM
//
// Grid :  (ceil(N/BLK), ceil(M/BLK), batch_size)
// Block:  (BLK, BLK, 1)
//
// Each thread computes exactly one element of C[batch][row][col].
// No shared memory — every operand is fetched straight from global memory.
// This is the worst-case baseline: useful only for correctness reference
// and to show how much headroom the tiled/strided versions gain.
// ---------------------------------------------------------------------------

#define NAIVE_BLK 16

__global__ void naive_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int batch = blockIdx.z;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Offset to this batch's matrices
    const float* a = A + (long long)batch * M * K;
    const float* b = B + (long long)batch * K * N;
    float*       c = C + (long long)batch * M * N;

    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += a[row * K + k] * b[k * N + col];

    c[row * N + col] = sum;
}

void gemm_naive_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size)
{
    dim3 block(NAIVE_BLK, NAIVE_BLK, 1);
    dim3 grid(
        (N + NAIVE_BLK - 1) / NAIVE_BLK,
        (M + NAIVE_BLK - 1) / NAIVE_BLK,
        batch_size
    );
    naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
