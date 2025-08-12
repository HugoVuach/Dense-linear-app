#include <chameleon.h>
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

    if (bsiz != mb*nb) {
        fprintf(stderr, "Warning: bsiz (%d) != mb*nb (%d)\n", bsiz, mb*nb);
    }

    const char *sched = getenv("STARPU_SCHED");
    printf("[setup] ncpu=%d ngpu=%d N=%d NB=%d scheduler=%s\n",
           ncpu, ngpu, N, NB, sched ? sched : "(default)");

    int info;
    CHAM_desc_t *descA = NULL, *descAorig = NULL, *descR = NULL;

    CHAMELEON_Init(ncpu, ngpu);

    // A SPD
    CHAMELEON_Desc_Create(&descA, NULL, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
    CHAMELEON_dplgsy_Tile((double)N, ChamLower, descA, seed);

    // Copie pour validation
    CHAMELEON_Desc_Create(&descAorig, NULL, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
    CHAMELEON_dlacpy_Tile(ChamUpperLower, descA, descAorig);

    // Factorisation LL^T
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    info = CHAMELEON_dpotrf_Tile(ChamLower, descA);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (1.0/3.0) * (double)N * (double)N * (double)N / (time_sec * 1e9);

    printf("N = %d, NB = %d\n", N, NB);
    printf("Time: %.3f s\n", time_sec);
    printf("Performance: %.2f Gflop/s\n", gflops);

    if (info != 0) {
        fprintf(stderr, "Erreur dans CHAMELEON_dpotrf_Tile: %d\n", info);
    }

    // === VALIDATION NUMÉRIQUE ===
    // On veut ||A - LL^T|| / ||A|| : calculer ||A|| maintenant (descAorig contient A)
    double normA = CHAMELEON_dlange_Tile(ChamInfNorm, descAorig);

    // Reconstruire LL^T -> descR
    CHAMELEON_Desc_Create(&descR, NULL, ChamRealDouble,
                          mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
    CHAMELEON_dlacpy_Tile(ChamLower, descA, descR);
    CHAMELEON_dlauum_Tile(ChamLower, descR); // descR ← LL^T

    // descAorig ← Aorig - LL^T
    CHAMELEON_dgeadd_Tile(ChamNoTrans, -1.0, descR, 1.0, descAorig);

    double residual = CHAMELEON_dlange_Tile(ChamInfNorm, descAorig);
    double relative_error = residual / (normA > 0 ? normA : 1.0);

    printf("||A - LL^T||_inf / ||A||_inf = %.2e\n", relative_error);
    printf("Validation numérique : %s\n", (relative_error < 1e-10) ? "✅ PASS" : "❌ FAIL");

    // Nettoyage
    CHAMELEON_Desc_Destroy(&descA);
    CHAMELEON_Desc_Destroy(&descAorig);
    CHAMELEON_Desc_Destroy(&descR);
    CHAMELEON_Finalize();

    return (info != 0);
}
