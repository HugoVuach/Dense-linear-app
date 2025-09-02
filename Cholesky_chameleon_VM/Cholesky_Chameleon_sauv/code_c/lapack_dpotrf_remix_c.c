#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <string.h>
#include <time.h>

#define min(a,b) ((a)<(b)?(a):(b))

// Version simplifiée de dpotrf bloqué (uplo='L')
int dpotrf_c(const char uplo, int N, double *A, int lda, int block_size) {
    if (uplo != 'L') {
        fprintf(stderr, "UPLO = 'U' not supported in this version\n");
        return -1;
    }

    for (int j = 0; j < N; j += block_size) {
        int jb = min(block_size, N - j);

        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, jb, j,
                    -1.0, &A[j*lda], lda, 1.0, &A[j*lda + j], lda);

        // Factorisation bloc diagonal
        for (int k = 0; k < jb; ++k) {
            double *Ajj = &A[(j + k)*lda + j];
            if (Ajj[k] <= 0.0) return j + k + 1;
            Ajj[k] = sqrt(Ajj[k]);
            for (int i = k + 1; i < jb; ++i) {
                Ajj[i] /= Ajj[k];
            }
            for (int i = k + 1; i < jb; ++i) {
                for (int l = k + 1; l <= i; ++l) {
                    A[(j + i)*lda + j + l] -= Ajj[i] * Ajj[l];
                }
            }
        }

        if (j + jb < N) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N - j - jb, jb, j,
                        -1.0, &A[(j + jb)*lda], lda, &A[j*lda], lda,
                         1.0, &A[(j + jb)*lda + j], lda);

            cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower,
                        CblasTrans, CblasNonUnit,
                        N - j - jb, jb, 1.0,
                        &A[j*lda + j], lda, &A[(j + jb)*lda + j], lda);
        }
    }

    return 0;
}

int main() {
    int N = 12000;
    int lda = N;
    int nb = 256;
    double *A = malloc(N * N * sizeof(double));

    // Initialiser une matrice SPD : A = A^T * A + N * I
    for (int i = 0; i < N * N; i++) A[i] = drand48();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, 1.0, A, N, 0.0, A, N);
    for (int i = 0; i < N; i++) A[i*N + i] += N;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int info = dpotrf_c('L', N, A, lda, nb);

    clock_gettime(CLOCK_MONOTONIC, &t2);

    if (info != 0) {
        printf("Cholesky failed at leading minor %d\n", info);
    } else {
        double time_sec = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        double gflops = (1.0/3.0) * N * N * N / (time_sec * 1e9);

        printf("Cholesky successful\n");
        printf("Matrix size N = %d\n", N);
        printf("Time = %.4f seconds\n", time_sec);
        printf("Performance = %.2f Gflop/s\n", gflops);
    }

    free(A);
    return 0;
}
