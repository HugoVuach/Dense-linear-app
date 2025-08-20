# ===== Config overridable par l'environnement =====
CHAMELEON_DIR ?= /usr/local
STARPU_PREFIX ?= /usr/local
CUDA_HOME     ?= /usr/local/cuda
PKG_CONFIG    ?= pkg-config

# Paquets pkg-config
PC_PACKAGES   := chameleon starpu-1.4

# Sources / binaires
SRC_MAIN  = v6_test.c
BIN_MAIN  = v6_test
SRC_BENCH = benchmark.c
BIN_BENCH = bench

# Flags
CFLAGS    ?= -O3
WARNFLAGS ?=

# CFLAGS/INCLUDES depuis pkg-config + CUDA
CPPFLAGS  += $(shell $(PKG_CONFIG) --cflags $(PC_PACKAGES)) -I"$(CUDA_HOME)/include"

# Libs depuis pkg-config + BLAS/LAPACKE/CUDA
LDFLAGS   += $(shell $(PKG_CONFIG) --libs $(PC_PACKAGES)) \
             -lopenblas -llapacke \
             -lcublas -lcudart -lcuda \
             -lm -fopenmp -ldl -lpthread

# RPATH pour l'exécution dans l'image finale
RPATH     = -Wl,-rpath,"$(CHAMELEON_DIR)/lib:$(STARPU_PREFIX)/lib:/opt/hwloc_cuda/lib:$(CUDA_HOME)/lib64:/usr/local/lib"

# ===== Règles =====
.PHONY: all clean

all: $(BIN_MAIN) $(BIN_BENCH)

$(BIN_MAIN): $(SRC_MAIN)
	# Utiliser le wrapper MPI pour que les bons flags soient pris
	mpicc $(CFLAGS) $(WARNFLAGS) -o $@ $^ $(CPPFLAGS) $(RPATH) $(LDFLAGS)

$(BIN_BENCH): $(SRC_BENCH)
	gcc -O2 -Wall -Wextra -o $@ $^

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) results/bench.csv
