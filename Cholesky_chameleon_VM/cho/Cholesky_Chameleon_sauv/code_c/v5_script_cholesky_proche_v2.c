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




    //CHAMELEON_Init(ncpu, ngpu);
    // CHAMELEON_Init(1,1);

    //CHAMELEON_Desc_Create(&descA, mat_ptr,      dtyp,       mb, nb,     bsiz,   lm,   ln,    ioff, joff,   m, n,    p, q);
    //CHAMELEON_Desc_Create(&descA, NULL,    ChamRealDouble,  NB, NB,    NB * NB,  N,    N,       0,    0,    N, N,   1, 1);

    int ncpu = atoi(argv[1]); // Number of CPU threads
    int ngpu = atoi(argv[2]); // Number of GPU threads
    int N = atoi(argv[3]);   // Taille de la matrice (ex: 3000)
    int NB = atoi(argv[4]); // Taille des blocs (ex: 256)
    int mb = atoi(argv[5]); // Taille des lignes par bloc
    int nb = atoi(argv[6]); // Taille des colonnes par bloc
    int bsiz = atoi(argv[7]); // Taille d'allocation par bloc
    int lm = atoi(argv[8]); // Nombre total de lignes de la matrice globale
    int ln = atoi(argv[9]); // Nombre total de colonnes globale
    int ioff = atoi(argv[10]); // Offset ligne de la sous-matrice
    int joff = atoi(argv[11]); // Offset colonne de la sous-matrice
    int m = atoi(argv[12]); // Nombre de lignes de la sous-matrice
    int n = atoi(argv[13]); // Nombre de colonnes de la sous-matrice
    int p = atoi(argv[14]); // Nombre de lignes de la sous-matrice
    int q = atoi(argv[15]); // Nombre de colonnes de la sous-matrice
    int seed = atoi(argv[16]); // Graine pour le générateur de nombres aléatoires

    int info;

    CHAM_desc_t *descA;
    CHAMELEON_Init(ncpu,ngpu);
    CHAMELEON_Desc_Create(&descA, NULL, ChamRealDouble,
                          mb, nb, bsiz,
                          lm, ln, ioff, joff, m, n,
                          p, q);
    CHAMELEON_dplgsy_Tile((double)N, ChamLower, descA, seed);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    info = CHAMELEON_dpotrf_Tile(ChamLower, descA);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (1.0 / 3.0) * pow(N, 3) / (time_sec * 1e9);
    printf("N=%d, NB=%d\n", N, NB);
    printf("Time: %.3f s\n", time_sec);
    printf("Performance: %.2f Gflop/s\n", gflops);
    if (info != 0) {
        fprintf(stderr, "Erreur dans CHAMELEON_dpotrf_Tile: %d\n", info);
    }

    CHAMELEON_Desc_Destroy(&descA);
    CHAMELEON_Finalize();

    return (info != 0);
}

// exemple de commande 
// ./v5_script_cholesky_proche_v2 1 1 3000 256 256 256 65536 3000 3000 0 0 3000 3000 1 1 51
