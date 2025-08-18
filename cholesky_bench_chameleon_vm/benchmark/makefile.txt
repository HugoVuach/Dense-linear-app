# ===== Variables to adapt to your machine =====
CHAMELEON_DIR ?= /path/to/chameleon/install
STARPU_PREFIX ?= /path/to/starpu/install
CUDA_HOME     ?= /usr/local/cuda

# Your main source code
SRC_MAIN  = v6_test.c
BIN_MAIN  = v6_test

# Benchmark harness
SRC_BENCH = benchmark.c
BIN_BENCH = bench

# Common flags for the main binary
INCLUDES  = -I"$(CHAMELEON_DIR)/include" \
            -I"$(STARPU_PREFIX)/include/starpu/1.4" \
            -I"$(CUDA_HOME)/include"

LIBPATHS  = -L"$(CHAMELEON_DIR)/lib" \
            -L"$(STARPU_PREFIX)/lib" \
            -L"/usr/lib/x86_64-linux-gnu/openblas-pthread" \
            -L"$(CUDA_HOME)/lib64"

LIBS      = -lchameleon -lchameleon_starpu -lcoreblas -lstarpu-1.4 \
            -llapacke -lopenblas -lcublas -lcudart -lcuda \
            -lm -fopenmp -lpthread -ldl

RPATH     = -Wl,-rpath,"$(CHAMELEON_DIR)/lib:$(STARPU_PREFIX)/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:$(CUDA_HOME)/lib64"

# ===== Rules =====

.PHONY: all clean

all: $(BIN_MAIN) $(BIN_BENCH)

$(BIN_MAIN): $(SRC_MAIN)
	mpicc -O3 -o $@ $^ $(INCLUDES) $(LIBPATHS) $(RPATH) $(LIBS)

$(BIN_BENCH): $(SRC_BENCH)
	gcc -O2 -Wall -Wextra -o $@ $^

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) results/bench.csv
