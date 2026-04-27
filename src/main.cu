#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm.h"

////////////////////////////////////////////////////////////////////////////////////////
// Timing, Validation, and Evaluation Harness
//
// This section handles all the evaluation 
// Lets GPU kernel be measured  against the cuBLAS baseline and verified against a trusted CPU reference
////////////////////////////////////////////////////////////////////////////////////////

#define WARMUP_RUNS  3
#define TIMING_RUNS 20
#define ERR_THRESH   1e-3f   
#define PEAK_TFLOPS  29.77   // peak FP32 for RTX 3080

// check for cuda errors to prevent silent failures
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",              \
                    cudaGetErrorString(_e), __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// tflops --
//
// compute theoretical TFLOPS for one batched GEMM config
// ops = 2 * M * N * K * batch (multiply-add counts as 2 operations)
static double tflops(int M, int N, int K, int batch, double ms) {
    return 2.0 * M * N * K * batch / (ms * 1e-3) / 1e12;
}

// percent_peak --
//
// compares our computed throughput to the theoretical hardware limit of the Gpu RTX 3080
static double percent_peak(double computed_tflops) {
    return (computed_tflops / PEAK_TFLOPS) * 100.0;
}

typedef void (*KernelFn)(const float*, const float*, float*, int, int, int, int);

// time_kernel --
//
// records the execution time of a  kernel.
// for accurate benchmarking, we do a few warm-up runs first 
// to ensure the GPU is fully awake and code is paged in.  
// we then use cuda events to measure the average over 20 runs for a stable reproducible results
static float time_kernel(KernelFn fn, const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int batch_size) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; ++i) fn(d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMING_RUNS; ++i) fn(d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / TIMING_RUNS;
}

// time_cublas --
//
// records the execution time of the NVIDIA cuBLAS library
// this is to compare performance against a strong baseline
// 
static float time_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int batch_size) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_RUNS; ++i) gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMING_RUNS; ++i) gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / TIMING_RUNS;
}

// validate_kernel --
//
// verifies the GPU output against a trusted sequential CPU implementation for correctnes validation
// 
static float validate_kernel(const char* name, const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int batch_size) {
    long long size_a = (long long)batch_size * M * K;
    long long size_b = (long long)batch_size * K * N;
    long long size_c = (long long)batch_size * M * N;

    float* h_A   = (float*)malloc(size_a * sizeof(float));
    float* h_B   = (float*)malloc(size_b * sizeof(float));
    float* h_C   = (float*)malloc(size_c * sizeof(float));
    float* h_ref = (float*)malloc(size_c * sizeof(float));

    // bring results back to host memory to check
    CUDA_CHECK(cudaMemcpy(h_A, d_A, size_a * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B, d_B, size_b * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c * sizeof(float), cudaMemcpyDeviceToHost));

    gemm_cpu_reference(h_A, h_B, h_ref, M, N, K, batch_size);
    float err = max_abs_error(h_C, h_ref, M, N, batch_size);

    free(h_A); free(h_B); free(h_C); free(h_ref);
    return err; 
}

// run_config --
//
// executes the  kernels for a specific matrix size and batch size and prints results
static void run_config(cublasHandle_t handle, int M, int N, int K, int batch_size, int do_validate) {
    printf("\n");
    printf("=========================================================================================\n");
    printf("  Configuration: M=%-4d N=%-4d K=%-4d | Batch Size: %d\n", M, N, K, batch_size);
    printf("-----------------------------------------------------------------------------------------\n");
    printf("  %-16s | %-19s | %-10s | %-13s | %-8s\n", "Kernel", "Validation(MaxErr)", "Time (ms)", "Performance", "% Peak");
    printf("-----------------------------------------------------------------------------------------\n");

    float* d_A = alloc_random_device(M, K, batch_size, 42);
    float* d_B = alloc_random_device(K, N, batch_size, 99);
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_C, (long long)batch_size * M * N * sizeof(float)));

    struct { const char* name; KernelFn fn; } kernels[] = {
        { "naive",          gemm_naive_batched   },
        { "tiled",          gemm_tiled_batched   },
        { "strided_batched", gemm_strided_batched },
    };
    int nk = sizeof(kernels) / sizeof(kernels[0]);

    for (int i = 0; i < nk; ++i) {
        kernels[i].fn(d_A, d_B, d_C, M, N, K, batch_size);   
        
        float err = -1.0f;
        const char* status = "N/A";
        if (do_validate) {
            err = validate_kernel(kernels[i].name, d_A, d_B, d_C, M, N, K, batch_size);
            status = (err < ERR_THRESH) ? "PASS" : "FAIL";
        }
        
        float ms = time_kernel(kernels[i].fn, d_A, d_B, d_C, M, N, K, batch_size);
        double tflops_val = tflops(M, N, K, batch_size, ms);
        
        if (do_validate) {
            printf("  %-16s | %8.2e (%-4s)     | %8.3f   | %6.3f TFLOPS | %5.1f%%\n",
                   kernels[i].name, err, status, ms, tflops_val, percent_peak(tflops_val));
        } else {
            printf("  %-16s |       N/A           | %8.3f   | %6.3f TFLOPS | %5.1f%%\n",
                   kernels[i].name, ms, tflops_val, percent_peak(tflops_val));
        }
    }

    // cuBLAS comparison
    {
        gemm_cublas_batched(handle, d_A, d_B, d_C, M, N, K, batch_size);
        float err = -1.0f;
        const char* status = "N/A";
        
        if (do_validate) {
            err = validate_kernel("cublas", d_A, d_B, d_C, M, N, K, batch_size);
            status = (err < ERR_THRESH) ? "PASS" : "FAIL";
        }
        
        float ms = time_cublas(handle, d_A, d_B, d_C, M, N, K, batch_size);
        double tflops_val = tflops(M, N, K, batch_size, ms);
        
        if (do_validate) {
            printf("  %-16s | %8.2e (%-4s)     | %8.3f   | %6.3f TFLOPS | %5.1f%%\n",
                   "cublas", err, status, ms, tflops_val, percent_peak(tflops_val));
        } else {
            printf("  %-16s |       N/A           | %8.3f   | %6.3f TFLOPS | %5.1f%%\n",
                   "cublas", ms, tflops_val, percent_peak(tflops_val));
        }
    }
    printf("=========================================================================================\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

////////////////////////////////////////////////////////////////////////////////////////
// main  block
////////////////////////////////////////////////////////////////////////////////////////

int main(void) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Peak FP32: ~%.1f TFLOPS\n", PEAK_TFLOPS);
    printf("Warmup runs: %d   Timing runs: %d\n\n", WARMUP_RUNS, TIMING_RUNS);

    // we test 3 input sizes  
    // These 3 sizes were chosen to demonstrate how the kernel 
    // scales from small memory bound workloads to massive compute bound workloads 

    // base case (small matrix, small batch)
    run_config(handle,  128,  128,  128,  8, /*validate=*/1); 
    
    // scaling case (medium matrix, large batch, shows high occupancy)
    run_config(handle,  512,  512,  512, 64, /*validate=*/1); 
    
    // peak performance case (massive matrix, pushes hardware limit)
    // no CPU validation, matrix size too large
    run_config(handle, 2048, 2048, 2048, 16, /*validate=*/0); 

    cublasDestroy(handle);
    return 0;
}
