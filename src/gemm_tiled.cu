#include "gemm.h"

// ---------------------------------------------------------------------------
// Tiled shared-memory batched GEMM  (Assignment 2 port)
//
// Same tiling strategy as A2, extended to a batch dimension via blockIdx.z.
// Each thread block loads TILE×TILE tiles of A and B into shared memory,
// accumulates the partial dot product, then advances to the next tile.
//
// This is the "safe fallback" if deeper optimizations don't work out.
// ---------------------------------------------------------------------------

#define TILE 32

__global__ void tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int batch = blockIdx.z;

    // Offset each pointer to this batch's data
    const float* a = A + (long long)batch * M * K;
    const float* b = B + (long long)batch * K * N;
    float*       c = C + (long long)batch * M * N;

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.f;

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {

        // Load tile of A — row-major, coalesced across threadIdx.x
        int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? a[row * K + a_col] : 0.f;

        // Load tile of B — row-major, coalesced across threadIdx.x
        int b_row = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? b[b_row * N + col] : 0.f;

        __syncthreads();

        // Accumulate partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        c[row * N + col] = sum;
}

void gemm_tiled_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size)
{
    dim3 block(TILE, TILE, 1);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE,
        batch_size                  // z dimension = batch index
    );
    tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
