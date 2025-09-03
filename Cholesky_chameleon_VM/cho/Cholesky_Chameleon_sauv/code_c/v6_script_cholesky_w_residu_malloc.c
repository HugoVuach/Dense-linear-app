#include <chameleon.h>
#include <starpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 17) {
        fprintf(stderr, "Usage: %s <num_cpu> <num_gpu> <matrix_size_N> <tile_size_NB> <mb> <nb> <bsiz> <lm> <ln> <ioff> <joff> <m> <n> <p> <q> <seed>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int ncpu = atoi(argv[1]);
    int ngpu = atoi(argv[2]);
    int N = atoi(argv[3]);
    int NB = atoi(argv[4]);
    int mb = atoi(argv[5]);
    int nb = atoi(argv[6]);
    int bsiz = atoi(argv[7]);
    int lm = atoi(argv[8]);
    int ln = atoi(argv[9]);
    int ioff = atoi(argv[10]);
    int joff = atoi(argv[11]);
    int m = atoi(argv[12]);
    int n = atoi(argv[13]);
    int p = atoi(argv[14]);
    int q = atoi(argv[15]);
    int seed = atoi(argv[16]);

    int info;

    CHAM_desc_t *descA, *descAorig, *descR;

    // Initialisation de CHAMELEON et StarPU
    CHAMELEON_Init(ncpu, ngpu);

    // Allocation de mémoire pinée via StarPU
    size_t size_in_elems = (size_t)lm * (size_t)ln;
    double *dataA, *dataAorig, *dataR;
    starpu_malloc((void **)&dataA,     size_in_elems * sizeof(double));
    starpu_malloc((void **)&dataAorig, size_in_elems * sizeof(double));
    starpu_malloc((void **)&dataR,     size_in_elems * sizeof(double));

    // Création des descripteurs avec buffers utilisateurs

    void *A_data;
    starpu_malloc((void**)&A_data, lm * ln * sizeof(double));


    CHAMELEON_Desc_Create(&descA, dataA, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);

    CHAMELEON_Desc_Create(&descAorig, dataAorig, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);

    CHAMELEON_Desc_Create(&descR, dataR, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);

    // Génération SPD et copie
    CHAMELEON_dplgsy_Tile((double)N, ChamLower, descA, seed);
    CHAMELEON_dlacpy_Tile(ChamUpperLower, descA, descAorig);

    // Factorisation de Cholesky
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    info = CHAMELEON_dpotrf_Tile(ChamLower, descA);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (1.0 / 3.0) * pow(N, 3) / (time_sec * 1e9);

    printf("N = %d, NB = %d\n", N, NB);
    printf("Time: %.3f s\n", time_sec);
    printf("Performance: %.2f Gflop/s\n", gflops);

    if (info != 0) {
        fprintf(stderr, "Erreur dans CHAMELEON_dpotrf_Tile: %d\n", info);
    }

    // === VALIDATION NUMÉRIQUE ===

    CHAMELEON_dlacpy_Tile(ChamLower, descA, descR);
    CHAMELEON_dlauum_Tile(ChamLower, descR); // descR ← LL^T

    // descAorig ← descAorig - LL^T
    CHAMELEON_dgeadd_Tile(ChamNoTrans, -1.0, descR, 1.0, descAorig);

    // Normes
    double normA = CHAMELEON_dlange_Tile(ChamInfNorm, descR);
    double residual = CHAMELEON_dlange_Tile(ChamInfNorm, descAorig);
    double relative_error = residual / normA;

    printf("||A - LL^T||_inf / ||A||_inf = %.2e\n", relative_error);
    if (relative_error < 1e-10) {
        printf("Validation numérique : ✅ PASS\n");
    } else {
        printf("Validation numérique : ❌ FAIL\n");
    }

    // Libération des ressources
    CHAMELEON_Desc_Destroy(&descA);
    CHAMELEON_Desc_Destroy(&descAorig);
    CHAMELEON_Desc_Destroy(&descR);
    starpu_free(dataA);
    starpu_free(dataAorig);
    starpu_free(dataR);
    CHAMELEON_Finalize();

    return (info != 0);
}
