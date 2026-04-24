NVCC     = nvcc
ARCH     = -arch=sm_86        # RTX 3080 (Ampere)
CFLAGS   = -O3 -lineinfo $(ARCH)
INCLUDES = -Iinclude
LIBS     = -lcublas

SRC = src/main.cu         \
      src/gemm_naive.cu   \
      src/gemm_tiled.cu   \
      src/gemm_batched.cu \
      src/cublas_ref.cu   \
      src/validate.cu

TARGET = batched_gemm

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SRC) -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

# Run and save output to results/
run: $(TARGET)
	mkdir -p results
	./$(TARGET) | tee results/timing_$(shell date +%Y%m%d_%H%M%S).txt

# Quick sanity check (small matrices only)
test: $(TARGET)
	./$(TARGET)

.PHONY: all clean run test