# =============================================================================
# Stage 1 — builder 
# Builds the full stack: OpenBLAS, OpenMPI, CUDA with hwloc, StarPU & Chameleon
# =============================================================================

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC 

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libhwloc-dev libnuma-dev libomp-dev \
    libtool libtool-bin autoconf automake m4 \
    gfortran curl wget unzip \
    zlib1g-dev libffi-dev libssl-dev \
    libevent-dev libudev-dev libcap-dev \
    libx11-dev libxext-dev libxrender-dev libxtst-dev \
    python3 python3-pip python3-dev \
    libxml2-dev libxslt1-dev \
    libnuma1 \
    libpciaccess0 libpciaccess-dev \
    ninja-build ocl-icd-opencl-dev \
    ca-certificates gnupg2 lsb-release software-properties-common \
    openmpi-bin libopenmpi-dev \
    libnvidia-ml-dev \
 && rm -rf /var/lib/apt/lists/*

# OpenBLAS 
RUN set -eux; cd /tmp; \
    git clone --depth 1 --branch v0.3.26 https://github.com/xianyi/OpenBLAS.git; \
    cd OpenBLAS; make USE_LAPACK=1 LAPACKE=1 -j"$(nproc)"; \
    make install PREFIX=/usr/local; ldconfig; rm -rf /tmp/OpenBLAS

# CUDA toolkit (nvcc et headers pour la compilation)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-9 && \
    rm -rf /var/lib/apt/lists/*

# Rendre visibles les libs CUDA
RUN printf "/usr/local/cuda/lib64\n/usr/local/lib\n" > /etc/ld.so.conf.d/cuda.conf && ldconfig

# ENV propres (avec valeurs par défaut pour éviter UndefinedVar)
ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:/opt/hwloc_cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/hwloc_cuda/lib:${LD_LIBRARY_PATH:-}" \
    PKG_CONFIG_PATH="/usr/local/cuda/lib64/pkgconfig:/opt/hwloc_cuda/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:${PKG_CONFIG_PATH:-}"


# hwloc (version 2.x stable) avec CUDA, NVML désactivé
RUN set -eux; cd /tmp; \
    git clone --branch hwloc-2.10.0 https://github.com/open-mpi/hwloc.git; \
    cd hwloc; ./autogen.sh; \
    export ac_cv_lib_nvidia_ml_nvmlInit=no; \
    ./configure --enable-cuda --with-cuda=/usr/local/cuda --disable-nvml --prefix=/opt/hwloc_cuda; \
    make -j"$(nproc)"; make install; ldconfig; rm -rf /tmp/hwloc


# StarPU (bindings Python désactivés)
RUN set -eux; cd /tmp; \
  git clone --depth 1 --branch starpu-1.4 https://gitlab.inria.fr/starpu/starpu.git; \
  cd starpu; ./autogen.sh; \
  ./configure --prefix=/usr/local \
      --with-hwloc=/opt/hwloc_cuda \
      --enable-cuda --enable-opencl --enable-mpi \
      --with-cuda=/usr/local/cuda \
      --disable-build-doc \
      --disable-python --without-python --disable-starpupy \
      PYTHON=no \
      --with-blas-lib="-lopenblas" \
      --with-blas-inc="/usr/local/include"; \
  make -j"$(nproc)"; \
  mkdir -p /usr/local/lib/python3.12/site-packages || true; \
  make install; ldconfig; rm -rf /tmp/starpu

# Chameleon
RUN set -eux; cd /tmp; \
    git clone --depth 1 --recurse-submodules https://gitlab.inria.fr/solverstack/chameleon.git; \
    cd chameleon; mkdir build && cd build; \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON \
      -DCHAMELEON_ENABLE_CUDA=ON -DCHAMELEON_USE_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCHAMELEON_ENABLE_MPI=ON -DCHAMELEON_ENABLE_OPENCL=ON \
      -DCHAMELEON_SCHED_STARPU=ON -DCHAMELEON_SCHED_PARSEC=OFF -DCHAMELEON_SCHED_QUARK=OFF \
      -DCHAMELEON_ENABLE_TESTING=ON -DCHAMELEON_PREC_D=ON \
      -DBLAS_LIBRARIES="/usr/local/lib/libopenblas.so" \
      -DBLAS_INCLUDE_DIRS="/usr/local/include" \
      -DLAPACKE_INCLUDE_DIR="/usr/local/include" \
      -DLAPACKE_LIBRARY="/usr/local/lib/libopenblas.so" \
      -DCBLAS_LIBRARY="/usr/local/lib/libopenblas.so" \
      -DSTARPU_DIR=/usr/local -DHWLOC_ROOT=/opt/hwloc_cuda; \
    make -j"$(nproc)"; make install; ldconfig; rm -rf /tmp/chameleon




# =============================================================================
# Stage 2 — runtime base 
# =============================================================================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnuma1 libgomp1 libevent-2.1-7t64 libcap2 libudev1 \
    libx11-6 libxext6 libxrender1 libxtst6 \
    libxml2 ocl-icd-libopencl1 libgfortran5 libquadmath0 \
    libpciaccess0 openmpi-bin pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Copy runtime artifacts from builder (sans CUDA toolkit)
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/hwloc_cuda/ /opt/hwloc_cuda/

ENV CUDA_HOME="/usr/local/cuda" \
    HWLOC_PREFIX="/opt/hwloc_cuda" \
    STARPU_PREFIX="/usr/local" \
    CHAMELEON_DIR="/usr/local" \
    PATH="/usr/local/bin:/opt/hwloc_cuda/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/hwloc_cuda/lib:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    PKG_CONFIG_PATH="/opt/hwloc_cuda/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/local/cuda/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"


WORKDIR /work
CMD ["/bin/bash", "-lc", "echo 'CUDA:' && nvcc --version || true; echo 'MPI:' && mpicc --version | head -n1; echo 'StarPU:' && pkg-config --modversion starpu-1.4 || true; echo 'Chameleon:' && pkg-config --modversion chameleon || true"]
 

# =============================================================================
# Stage 3 — app-build - v6_test.c benchmark.c
# =============================================================================
FROM builder AS app-build

ENV CHAMELEON_DIR="/usr/local" \
    STARPU_PREFIX="/usr/local" \
    CUDA_HOME="/usr/local/cuda" \
    PKG_CONFIG_PATH="/opt/hwloc_cuda/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/local/cuda/lib64/pkgconfig:${PKG_CONFIG_PATH:-}" \
    LD_LIBRARY_PATH="/opt/hwloc_cuda/lib:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    PATH="/usr/local/bin:/opt/hwloc_cuda/bin:/usr/local/cuda/bin:${PATH}"

WORKDIR /app
COPY v6_test.c benchmark.c Makefile ./
RUN make -j"$(nproc)"


# =============================================================================
# Stage 4 — app 
# =============================================================================
FROM runtime AS app

ENV CHAMELEON_DIR="/usr/local" \
    STARPU_PREFIX="/usr/local" \
    CUDA_HOME="/usr/local/cuda" \
    PKG_CONFIG_PATH="/opt/hwloc_cuda/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/local/cuda/lib64/pkgconfig:${PKG_CONFIG_PATH:-}" \
    LD_LIBRARY_PATH="/opt/hwloc_cuda/lib:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    PATH="/usr/local/bin:/opt/hwloc_cuda/bin:/usr/local/cuda/bin:${PATH}"

WORKDIR /app
COPY --from=app-build /app/v6_test /app/bench ./
RUN mkdir -p /app/results

CMD ["/bin/bash"]
