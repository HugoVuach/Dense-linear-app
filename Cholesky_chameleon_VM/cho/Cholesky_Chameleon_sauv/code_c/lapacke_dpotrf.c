#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>

// Calcule ||A||_inf
double norm_inf(double* A, int N) {
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            row_sum += fabs(A[i*N + j]);
        }
        if (row_sum > norm) norm = row_sum;
    }
    return norm;
}

int main() {
    const int N = 12000;            // Taille de la matrice
    const char uplo = 'L';         // Cholesky sur triangle inférieur
    int info;

    // Allocation de la matrice A et copie
    double* A = (double*) malloc(N*N * sizeof(double));
    double* A_orig = (double*) malloc(N*N * sizeof(double));
    if (!A || !A_orig) {
        printf("Erreur d'allocation\n");
        return 1;
    }

    // Génération d'une matrice SPD : A = L * L^T
    srand(0);
    for (int i = 0; i < N*N; i++) {
        A[i] = ((double)rand()) / RAND_MAX;
    }
    // Rendre la matrice symétrique et diagonale dominante
    for (int i = 0; i < N; i++) {
        A[i*N + i] += N;
        for (int j = 0; j < i; j++) {
            A[i*N + j] = A[j*N + i];
        }
    }
    // Sauvegarde de A
    for (int i = 0; i < N*N; i++) A_orig[i] = A[i];

    // Mesure de temps
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Factorisation Cholesky
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, N, A, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    if (info != 0) {
        printf("Erreur dans dpotrf: %d\n", info);
        return 1;
    }

    // Temps en secondes
    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;

    // Performance théorique
    double gflops = (1.0/3.0)*N*N*N / (time * 1e9);

    // Reconstruction de LL^T
    double* R = (double*) calloc(N*N, sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1.0, A, N, A, N, 0.0, R, N);

    // Calcul du résidu
    for (int i = 0; i < N*N; i++) {
        R[i] -= A_orig[i];
    }

    double normA = norm_inf(A_orig, N);
    double normR = norm_inf(R, N);
    double rel_residual = normR / normA;

    // Résultats
    printf("Taille N = %d\n", N);
    printf("Temps d'exécution = %.4f s\n", time);
    printf("Performance = %.2f Gflop/s\n", gflops);
    printf("Résidu relatif ||A - LL^T||_inf / ||A||_inf = %.2e\n", rel_residual);
    printf("Validation : %s\n", rel_residual < 1e-10 ? "✅ PASS" : "❌ FAIL");

    // Libération
    free(A);
    free(A_orig);
    free(R);

    return 0;
}
