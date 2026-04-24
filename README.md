# Batched GEMM — COSC 4397 Final Project

Strided batched GEMM in CUDA, benchmarked against cuBLAS on an RTX 3080.

## Build

```bash
make
```

Requires CUDA toolkit ≥ 11, `nvcc` on PATH, cuBLAS.  
Targets `sm_86` (RTX 3080 / Ampere).  Change `ARCH` in `Makefile` for other GPUs.

## Run

```bash
make run          # build + run, saves output to results/
# or
./batched_gemm
```

## Output format

```
=== M=128 N=128 K=128  batch=8 ===
  naive                  x.xxx ms   x.xxx TFLOPS
  tiled                  x.xxx ms   x.xxx TFLOPS
  strided_batched        x.xxx ms   x.xxx TFLOPS
  cublas                 x.xxx ms   x.xxx TFLOPS
```

## File structure

```
include/gemm.h          shared function signatures
src/main.cu             timing harness + experiment configs   
src/gemm_naive.cu       naive kernel, 1 thread/element        
src/gemm_tiled.cu       shared-memory tiled kernel            
src/gemm_batched.cu     strided batched kernel — main work    
src/cublas_ref.cu       cuBLAS wrapper                        
src/validate.cu         CPU reference + error checking        
results/                timing output files (gitignored)
```

## Reproducing results

1. SSH into the university GPU server (RTX 3080 node)
2. `git clone <repo> && cd batched-gemm`
3. `make run`
4. Results are saved to `results/timing_<timestamp>.txt`

No external dependencies beyond CUDA toolkit and cuBLAS (bundled with CUDA).