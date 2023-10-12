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

double check(REAL *A, REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

void matmul_serial(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
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

void matmul_P3(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    #pragma omp parallel shared(N,A,B,C) private(i,j,k,temp)
    #pragma omp for schedule(dynamic,64)
    {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                temp = 0;
                #pragma omp simd reduction(+:temp)
                for (k = 0; k < N; k++) {
                    temp += (A[i * N + k] * B[k * N + j]);
                }
                C[i * N + j] = temp;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N;

    int num_threads = 8; /* 8 is default number of threads */
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
    REAL *C_serial = malloc(sizeof(REAL)*N*N);
    REAL *C_omp = malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i;
    int num_runs = 10;
    
    matmul_serial(N, A, B, C_serial);
    
    //warm-up
    matmul_P3(N, A, B, C_omp);
    
    double error = check(C_serial,C_omp, N);
    printf("error:%g\n", error);

    elapsed_cilkplus = read_timer();
    for (i=0; i<num_runs; i++)
        matmul_P3(N, A, B, C_omp);
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

