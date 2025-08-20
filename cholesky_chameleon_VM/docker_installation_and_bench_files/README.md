# How to perform complete Cholesky benchmark?

To run the benchmark files, take a look at the section **"How to run the Benchmark files?"** of the documentation.

This repo contains:

* **`Dockerfile`** (multi-stage) that builds the entire HPC stack (OpenBLAS → CUDA → hwloc-CUDA → StarPU → Chameleon) and then compiles your executables.
* **`Makefile`** (minimal) that builds two binaries from the sources:
  * `v6_test` (main code, linked with MPI + Chameleon + StarPU + CUDA)
  * `bench`

You can **freely modify `v6_test.c`** to test your ideas; rebuilding will only affect the `app-build` stage (the heavy layers remain cached).

---

## Dockerfile structure

The Dockerfile has 4 stages:

1. **`builder`** (base: `ubuntu:24.04`)  
   Installs toolchain + dependencies, compiles:
   * OpenBLAS  
   * CUDA Toolkit (build environment)  
   * hwloc **with CUDA support** (NVML disabled)  
   * StarPU (Python bindings disabled)  
   * Chameleon (StarPU scheduler, CUDA, MPI)

   > Artifacts are installed under `/usr/local` and `/opt/hwloc_cuda`.

2. **`runtime`** (base: `nvidia/cuda:12.9.0-runtime-ubuntu24.04`)  
   Minimal runtime image (runtime libs + MPI).  
   Only the required artifacts from `builder` are copied here.

3. **`app-build`**  
   Copies `v6_test.c`, `benchmark.c`, `Makefile` and runs `make`.  
   Produces the binaries `/app/v6_test` and `/app/bench`.

4. **`app`**  
   Lightweight final image containing the binaries, ready to run.

---

## Host prerequisites

* **NVIDIA driver**  
* **nvidia-container-toolkit** installed/configured for `docker run --gpus all`

---

## Build the image

```bash
docker build --target app-build -t cholesky_app_build .
```

## to run the benchmark
```bash
docker run --rm -it cholesky_app_build ./bench
```
