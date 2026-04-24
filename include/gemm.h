#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Memory layout (ALL kernels use this convention):
//
//   A : [batch_size, M, K]   stride_a = M * K
//   B : [batch_size, K, N]   stride_b = K * N
//   C : [batch_size, M, N]   stride_c = M * N
//
//   Element access inside batch i:
//     A[i][row][col] = A_ptr[ i*M*K + row*K + col ]
//     B[i][row][col] = B_ptr[ i*K*N + row*N + col ]
//     C[i][row][col] = C_ptr[ i*M*N + row*N + col ]
// ---------------------------------------------------------------------------

// Naive: one thread computes one output element, blockIdx.z = batch index
void gemm_naive_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size);

// Tiled: shared-memory tiling ported from Assignment 2, looped over batch
void gemm_tiled_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size);


// Strided batched: single kernel launch, blockIdx.z = batch index,
// contiguous strided layout — the main optimization target
void gemm_strided_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size);


// cuBLAS reference wrapper (cublasGemmStridedBatched)
void gemm_cublas_batched(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size);

// CPU reference: plain triple loop, used for correctness checking
void gemm_cpu_reference(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size);

// Returns max absolute error between two C buffers (host pointers, size M*N*batch)
float max_abs_error(const float* C_gpu, const float* C_ref,
                    int M, int N, int batch_size);

// Convenience: allocate + fill a strided batch buffer with random floats
// Caller owns the returned pointer (cudaFree when done)
float* alloc_random_device(int rows, int cols, int batch_size, unsigned seed);
