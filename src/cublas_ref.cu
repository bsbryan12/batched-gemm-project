#include "gemm.h"
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// cuBLAS strided batched GEMM wrapper
//
// cublasGemmStridedBatched expects column-major matrices.
// Our matrices are row-major [M×K], [K×N], [M×N].
//
// The standard trick:
//   Row-major  C = A × B
//   ≡ Col-major C^T = B^T × A^T
//
// Since cuBLAS interprets our row-major A as col-major A^T (an N×K matrix),
// we swap A↔B and swap M↔N in the cuBLAS call.  The result written into C
// is then the correct row-major answer.
//
// In other words, we call:
//   cublas(B^T [N×K], A^T [K×M]) → C^T [N×M]   (col-major)
// which is the same bytes as our row-major C [M×N].
// ---------------------------------------------------------------------------

#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t _s = (call);                                     \
        if (_s != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",              \
                    _s, __FILE__, __LINE__);                            \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

void gemm_cublas_batched(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size)
{
    const float alpha = 1.f;
    const float beta  = 0.f;

    // Strides in elements
    long long stride_a = (long long)M * K;
    long long stride_b = (long long)K * N;
    long long stride_c = (long long)M * N;

    // Swap A↔B and M↔N to convert row-major → col-major (see header comment)
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,   // no transpose on the swapped operands
        N, M, K,                     // swapped: output is N×M in col-major = M×N row-major
        &alpha,
        B, N, stride_b,              // "A" for cuBLAS = our B
        A, K, stride_a,              // "B" for cuBLAS = our A
        &beta,
        C, N, stride_c,
        batch_size
    ));
    // Note: no cudaDeviceSynchronize here — caller owns sync if needed.
    // main.cu uses cudaEventRecord which inserts a sync point automatically.
}
