# ===== Variables à adapter à ta machine =====
CHAMELEON_DIR ?= /chemin/vers/chameleon/install
STARPU_PREFIX ?= /chemin/vers/starpu/install
CUDA_HOME     ?= /usr/local/cuda

# Ton code source principal
SRC_MAIN  = v6_test.c
BIN_MAIN  = v6_test

# Le harness de bench
SRC_BENCH = benchmark.c
BIN_BENCH = bench

# Flags communs pour le binaire principal
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

# ===== Règles =====

.PHONY: all clean

all: $(BIN_MAIN) $(BIN_BENCH)

$(BIN_MAIN): $(SRC_MAIN)
	mpicc -O3 -o $@ $^ $(INCLUDES) $(LIBPATHS) $(RPATH) $(LIBS)

$(BIN_BENCH): $(SRC_BENCH)
	gcc -O2 -Wall -Wextra -o $@ $^

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) results/bench.csv
