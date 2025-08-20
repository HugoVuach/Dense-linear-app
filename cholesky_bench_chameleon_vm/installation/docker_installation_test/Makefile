# ===== Emplacements (fournis par le Dockerfile via ENV) =====
CHAMELEON_DIR ?= /usr/local
STARPU_PREFIX ?= /usr/local
CUDA_HOME     ?= /usr/local/cuda

# Sources / binaires
SRC_MAIN  = v6_test.c
BIN_MAIN  = v6_test
SRC_BENCH = benchmark.c
BIN_BENCH = bench

# Flags récupérés via pkg-config
PKG_CFLAGS := $(shell pkg-config --cflags chameleon starpu-1.4)
PKG_LIBS   := $(shell pkg-config --libs   chameleon starpu-1.4)

# On ajoute les includes CUDA si besoin
CFLAGS   += -O3 $(PKG_CFLAGS) -I$(CUDA_HOME)/include
LDFLAGS  += -Wl,-rpath,"$(CHAMELEON_DIR)/lib:$(STARPU_PREFIX)/lib:$(CUDA_HOME)/lib64"
LIBPATHS += -L$(CHAMELEON_DIR)/lib -L$(STARPU_PREFIX)/lib -L$(CUDA_HOME)/lib64

# Bibliothèques additionnelles :
# - lapacke est fourni par libopenblas.so, donc pas de -llapacke
# - on ajoute openblas et CUDA runtime
LDLIBS   += -lopenblas -lcublas -lcudart -lm -fopenmp -ldl -lpthread

.PHONY: all clean

all: $(BIN_MAIN) $(BIN_BENCH)

# Utiliser mpicc pour le binaire MPI
$(BIN_MAIN): $(SRC_MAIN)
	mpicc $(CFLAGS) $(LIBPATHS) $(LDFLAGS) -o $@ $^ $(PKG_LIBS) $(LDLIBS)

$(BIN_BENCH): $(SRC_BENCH)
	gcc -O2 -Wall -Wextra -o $@ $^

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) results/bench.csv
