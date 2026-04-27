#include "gemm.h"

// -------------------------------------------------------------------------------------
// Key Ideas vs. the Naive and Tiled Versions 
// -------------------------------------------------------------------------------------
//
// 1. 3D Grid Mapping & Occupancy 
//    - Naive/Tiled: Processed one matrix at a time or used huge thread blocks.
//    - Batched: Maps the z dimension of the grid to the batch size (`blockIdx.z`). 
//      By using 256 threads per block instead of 1024, the GPU warp scheduler can fit 
//      up to 4 active blocks onto a single Streaming Multiprocessor (SM). The increase
//      in GPU occupancy allows the scheduler to hide memory latency.
//
// 2. Data Locality & Register Tiling 
//    - Tiled: Threads accumulated dot products using `float sum`, reading from shared 
//      memory for every single math operation (low arithmetic intensity).
//    - Batched: Implements 2D Register Tiling (`TM=4, TN=4`). Each thread loads a 
//      fragment of A and B into local registers, then computes 16 fused multiply-add 
//      (FMA) operations entirely in registers. This eliminates the shared memory 
//      bandwidth bottleneck.
//
// 3. Bank Conflicts 
//    - Tiled: Memory for A was stored sequentially. When threads read columns to 
//      compute the dot product, they hit the same memory banks, causing stalls.
//    - Batched: The shared memory tile for A is transposed (`sA[BK][BM]`). When warps 
//      read it during the math loop, they access consecutive memory banks, achieving 
//      no conflict free memory reads.
//
// 4. Memory Coalescing 
//    - Naive/Tiled: Fetched global memory one 32 bit float at a time.
//    - Batched: Casts pointers to `float4` to utilize 128 bit LDG/STG instructions. 
//      This quarters the total number of memory transactions on the global memory bus, 
//      saturating memory bandwidth.
//
// 5. Instruction Level Parallelism (ILP)
//    - Naive/Tiled: Relied on standard loops to compute the tiles.
//    - Batched: Uses `#pragma unroll` on the inner register-tiling loops. This 
//      forces the compiler to flatten the loops into a sequential block of FMA math 
//      instructions, removing loop overhead and maximizing ILP.
// -------------------------------------------------------------------------------------



// Tile sizes:
// BM, BN = 64  -- thread block output tile
// BK = 16      -- depth of the shared memory 
// TM = 4, TN = 4 -- each thread computes a 4x4 register sub tile of C
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// 256 threads per block 
// allows up to 4 blocks to reside on one SM
// maximizes GPU scheduling & occupancy
#define THREADS ((BM / TM) * (BN / TN))   


// strided_batched_kernel -- (CUDA device code)
//
// Computes a batched matrix multiplication (C = A * B) using a contiguous strided layout
// Uses register tiling , memory coalescing via 128 bit memory accesses, elimination of bank conflicts,
//  and instruction level parallelism ILP
//  
__launch_bounds__(THREADS)
__global__ void strided_batched_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    long long stride_a,
    long long stride_b,
    long long stride_c)
{
    // the z dimension of the grid represents our batch index.  
    // this guarantees massive grid sizes, allowing the hardware scheduler to maximize SM utilization
    int batch = blockIdx.z;

    // offset raw pointers to batch's matrices
    const float* a = A + batch * stride_a;
    const float* b = B + batch * stride_b;
    float*       c = C + batch * stride_c;

    // thread indexing: 2-D grid inside the block
    int tid   = threadIdx.x;
    int t_col = tid % (BN / TN);   
    int t_row = tid / (BN / TN);   

    // output tile origin in global C
    int c_row_start = blockIdx.y * BM;
    int c_col_start = blockIdx.x * BN;

    // shared memory tiles
    // sA is transposed ([BK][BM]) so that threads in a warp 
    // read consecutive banks during the math loop. This avoids shared memory bank conflicts 
    // sB is standard ([BK][BN]) because row accesses are already clean
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    // register accumulators (register tiling)
    // accumulators live in registers, not shared memory  
    // Each loaded sA/sB element is reused across 16 output elements, increasing arithmetic intensity and data locality
    
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {

        // load A tile into sA (transposed)
        // use a float4 load to pull 128 bits at once. 
        // this improves memory coalescing and halves the number of memory 
        // transactions on the memory bus.
        int a_load_row = c_row_start + tid / (BK / 4);
        int a_load_col = t * BK + (tid % (BK / 4)) * 4;

        // we only use the vectorized load if the matrix size is a safe multiple of 4
        // otherwise, we fall back to a safe scalar load to prevent out of bounds crash
        if ((K % 4 == 0) && a_load_row < M && a_load_col + 3 < K) {
            float4 vec = *reinterpret_cast<const float4*>(&a[a_load_row * K + a_load_col]);
            int sA_r = tid / (BK / 4);
            int sA_c = (tid % (BK / 4)) * 4;
            sA[sA_c + 0][sA_r] = vec.x;
            sA[sA_c + 1][sA_r] = vec.y;
            sA[sA_c + 2][sA_r] = vec.z;
            sA[sA_c + 3][sA_r] = vec.w;
        } else if (a_load_row < M) {
            int sA_r = tid / (BK / 4);
            int sA_c_base = (tid % (BK / 4)) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sA[sA_c_base + i][sA_r] =
                    (a_load_col + i < K) ? a[a_load_row * K + a_load_col + i] : 0.0f;
            }
        } else {
            int sA_r = tid / (BK / 4);
            int sA_c_base = (tid % (BK / 4)) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) sA[sA_c_base + i][sA_r] = 0.0f;
        }

        // load B tile into sB
        // same float4 optimization applied here to max memory coalescing
        int b_load_row = t * BK + tid / (BN / 4);
        int b_load_col = c_col_start + (tid % (BN / 4)) * 4;

        if ((N % 4 == 0) && b_load_row < K && b_load_col + 3 < N) {
            float4 vec = *reinterpret_cast<const float4*>(&b[b_load_row * N + b_load_col]);
            int sB_r = tid / (BN / 4);
            int sB_c = (tid % (BN / 4)) * 4;
            sB[sB_r][sB_c + 0] = vec.x;
            sB[sB_r][sB_c + 1] = vec.y;
            sB[sB_r][sB_c + 2] = vec.z;
            sB[sB_r][sB_c + 3] = vec.w;
        } else if (b_load_row < K) {
            int sB_r = tid / (BN / 4);
            int sB_c_base = (tid % (BN / 4)) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sB[sB_r][sB_c_base + i] =
                    (b_load_col + i < N) ? b[b_load_row * N + b_load_col + i] : 0.0f;
            }
        } else {
            int sB_r = tid / (BN / 4);
            int sB_c_base = (tid % (BN / 4)) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) sB[sB_r][sB_c_base + i] = 0.0f;
        }

        __syncthreads();

        // accumulate partial dot products from shared memory
        // the unroll pragmas force the compiler to flatten this loop 
        // eliminates branch overhead and allows the pipeline to issue independent FMAs back to back
        // maximizes Instruction Level Parallelism (ILP) 
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = sA[k][t_row * TM + i];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_reg[j] = sB[k][t_col * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // write output tile back to global memory
    // if N is a multiple of 4, use a vectorized 128 bit store to match the 128 bit loads,
    //  more efficient code than four fp32 stores
    if (N % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int row = c_row_start + t_row * TM + i;
            int col_base = c_col_start + t_col * TN;
            
            if (row < M && col_base + 3 < N) {
                float4 out_vec;
                out_vec.x = acc[i][0];
                out_vec.y = acc[i][1];
                out_vec.z = acc[i][2];
                out_vec.w = acc[i][3];
                *reinterpret_cast<float4*>(&c[row * N + col_base]) = out_vec;
            }
        }
    } else {
        
        // safe scalar fallback for odd matrix sizes
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int row = c_row_start + t_row * TM + i;
            if (row < M) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int col = c_col_start + t_col * TN + j;
                    if (col < N)
                        c[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// gemm_strided_batched --
//
// Set up grid and block dimensions and launch the strided batched GEMM kernel.
void gemm_strided_batched(
    const float* A, const float* B, float* C,
    int M, int N, int K, int batch_size)
{
    // 256 threads per block 
    dim3 block(THREADS, 1, 1);
    
    // map the z dimension of the grid to the batch size
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch_size);

    strided_batched_kernel<<<grid, block>>>(
        A, B, C, M, N, K,
        (long long)M * K,
        (long long)K * N,
        (long long)M * N
    );
    
    cudaDeviceSynchronize();
}
