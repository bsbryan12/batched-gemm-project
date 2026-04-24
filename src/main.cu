#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm.h"

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------
#define WARMUP_RUNS  3
#define TIMING_RUNS 20
#define ERR_THRESH   1e-3f   // max acceptable max-abs error vs CPU ref

// Error-checking macros
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",              \
                    cudaGetErrorString(_e), __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Compute theoretical TFLOPS for one batched GEMM config
//   ops = 2 * M * N * K * batch   (multiply-add = 2 ops)
// ---------------------------------------------------------------------------
static double tflops(int M, int N, int K, int batch, double ms)
{
    return 2.0 * M * N * K * batch / (ms * 1e-3) / 1e12;
}

// ---------------------------------------------------------------------------
// Run one kernel, time it, validate against CPU reference.
// Returns median time in ms.
// ---------------------------------------------------------------------------
typedef void (*KernelFn)(const float*, const float*, float*,
                         int, int, int, int);

static float time_kernel(
    KernelFn fn,
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, int batch_size)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < WARMUP_RUNS; ++i)
        fn(d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMING_RUNS; ++i)
        fn(d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / TIMING_RUNS;
}

// cuBLAS version needs special handling (takes a handle)
static float time_cublas(
    cublasHandle_t handle,
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, int batch_size)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; ++i)
        gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMING_RUNS; ++i)
        gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / TIMING_RUNS;
}

// ---------------------------------------------------------------------------
// Validate a kernel's output against the CPU reference.
// Copies d_C to host, runs CPU GEMM, computes max-abs error.
// Only called for the first (small) config to avoid huge CPU runtimes.
// ---------------------------------------------------------------------------
static void validate_kernel(
    const char* name,
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, int batch_size)
{
    long long size_a = (long long)batch_size * M * K;
    long long size_b = (long long)batch_size * K * N;
    long long size_c = (long long)batch_size * M * N;

    float* h_A   = (float*)malloc(size_a * sizeof(float));
    float* h_B   = (float*)malloc(size_b * sizeof(float));
    float* h_C   = (float*)malloc(size_c * sizeof(float));
    float* h_ref = (float*)malloc(size_c * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_A, d_A, size_a * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B, d_B, size_b * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c * sizeof(float), cudaMemcpyDeviceToHost));

    gemm_cpu_reference(h_A, h_B, h_ref, M, N, K, batch_size);

    float err = max_abs_error(h_C, h_ref, M, N, batch_size);
    printf("  [validate] %-20s max_abs_err = %.2e  %s\n",
           name, err, (err < ERR_THRESH) ? "PASS" : "FAIL ***");

    free(h_A); free(h_B); free(h_C); free(h_ref);
}

// ---------------------------------------------------------------------------
// Run a full sweep for one (M, N, K, batch_size) configuration
// ---------------------------------------------------------------------------
static void run_config(
    cublasHandle_t handle,
    int M, int N, int K, int batch_size,
    int do_validate)
{
    printf("\n=== M=%d N=%d K=%d  batch=%d ===\n", M, N, K, batch_size);

    float* d_A = alloc_random_device(M, K, batch_size, 42);
    float* d_B = alloc_random_device(K, N, batch_size, 99);
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_C, (long long)batch_size * M * N * sizeof(float)));

    // ---- Run and optionally validate each kernel ----
    struct { const char* name; KernelFn fn; } kernels[] = {
        { "naive",          gemm_naive_batched   },
        { "tiled",          gemm_tiled_batched   },
        { "strided_batched", gemm_strided_batched },
    };
    int nk = sizeof(kernels) / sizeof(kernels[0]);

    for (int i = 0; i < nk; ++i) {
        kernels[i].fn(d_A, d_B, d_C, M, N, K, batch_size);   // populate d_C
        if (do_validate)
            validate_kernel(kernels[i].name, d_A, d_B, d_C, M, N, K, batch_size);
        float ms = time_kernel(kernels[i].fn, d_A, d_B, d_C, M, N, K, batch_size);
        printf("  %-22s  %7.3f ms   %6.3f TFLOPS\n",
               kernels[i].name, ms, tflops(M, N, K, batch_size, ms));
    }

    // cuBLAS
    {
        gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
        if (do_validate)
            validate_kernel("cublas", d_A, d_B, d_C, M, N, K, batch_size);
        float ms = time_cublas(handle, d_A, d_B, d_C, M, N, K, batch_size);
        printf("  %-22s  %7.3f ms   %6.3f TFLOPS\n",
               "cublas", ms, tflops(M, N, K, batch_size, ms));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(void)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Print device name for the report
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Warmup runs: %d   Timing runs: %d\n\n", WARMUP_RUNS, TIMING_RUNS);
    printf("%-24s  %10s   %12s\n", "kernel", "time(ms)", "TFLOPS");
    printf("%-24s  %10s   %12s\n", "------", "--------", "------");

    // -------------------------------------------------------------------
    // Experiment 1: fixed matrix size (128×128), vary batch
    // Covers the "3 input sizes" requirement along the batch dimension.
    // -------------------------------------------------------------------
    printf("\n--- Vary batch size (M=N=K=128) ---\n");
    run_config(handle, 128, 128, 128,   8, /*validate=*/1);
    run_config(handle, 128, 128, 128,  64, /*validate=*/0);
    run_config(handle, 128, 128, 128, 512, /*validate=*/0);

    // -------------------------------------------------------------------
    // Experiment 2: fixed batch (64), vary matrix size
    // -------------------------------------------------------------------
    printf("\n--- Vary matrix size (batch=64) ---\n");
    run_config(handle, 32,  32,  32, 64, /*validate=*/1);
    run_config(handle, 128, 128, 128, 64, /*validate=*/0);
    run_config(handle, 256, 256, 256, 64, /*validate=*/0);

    cublasDestroy(handle);
    return 0;
}
