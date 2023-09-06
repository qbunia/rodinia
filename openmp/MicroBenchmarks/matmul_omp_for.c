/*
 * Square matrix multiplication
 * A[N][N] * B[N][N] = C[N][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include "omp.h"

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int N, REAL *A) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}


void matmul_cilkplus(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
#pragma omp parallel for shared(N,A,B,C) private(i,j,k,temp)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0;
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    int N;

    int num_threads = 4; /* 4 is default number of threads */
    if (argc < 2) {
        fprintf(stderr, "Usage: mm <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc >=3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double elapsed_cilkplus;

    REAL *A = malloc(sizeof(REAL)*N*N);
    REAL *B = malloc(sizeof(REAL)*N*N);
    REAL *C_cilkplus = malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i;
    int num_runs = 10;

    elapsed_cilkplus = read_timer();
    for (i=0; i<num_runs; i++)
        matmul_cilkplus(N, A, B, C_cilkplus);
    elapsed_cilkplus = (read_timer() - elapsed_cilkplus)/num_runs;
    /* you should add the call to each function and time the execution */

    printf("======================================================================================================\n");
    printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_omp:\t\t%4f\t%4f\n", elapsed_cilkplus * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cilkplus)));
    return 0;
}


