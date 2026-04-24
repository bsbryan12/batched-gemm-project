#include "gemm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// CPU reference GEMM
// Plain triple loop — guaranteed correct, used as the gold standard.
// Only run on small matrices (M,N,K ≤ 256) to keep wall time reasonable.
// ---------------------------------------------------------------------------
void gemm_cpu_reference(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size)
{
    for (int b = 0; b < batch_size; ++b) {
        const float* a = A + (long long)b * M * K;
        const float* bm = B + (long long)b * K * N;
        float*       c  = C + (long long)b * M * N;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.f;
                for (int k = 0; k < K; ++k)
                    sum += a[i * K + k] * bm[k * N + j];
                c[i * N + j] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Max absolute error between two host buffers
// ---------------------------------------------------------------------------
float max_abs_error(
    const float* C_gpu, const float* C_ref,
    int M, int N, int batch_size)
{
    float max_err = 0.f;
    long long total = (long long)batch_size * M * N;
    for (long long i = 0; i < total; ++i) {
        float err = fabsf(C_gpu[i] - C_ref[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ---------------------------------------------------------------------------
// Allocate a strided batch buffer on device, filled with random floats in
// [-1, 1].  Caller is responsible for cudaFree.
// ---------------------------------------------------------------------------
float* alloc_random_device(int rows, int cols, int batch_size, unsigned seed)
{
    long long n = (long long)batch_size * rows * cols;

    // Fill on host first, then copy
    float* h = (float*)malloc(n * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }

    srand(seed);
    for (long long i = 0; i < n; ++i)
        h[i] = (float)rand() / RAND_MAX * 2.f - 1.f;

    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}
