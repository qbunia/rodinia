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
#include "mm_cuda.h"

#define ALLOWED_DIFF 0.0001

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

#define REAL double

void init(int N, REAL *A) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
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

double check(REAL *A, REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int N;

    N = atoi(argv[1]);


    double elapsed_serial;
    double elapsed_cuda;

    REAL *A = malloc(sizeof(REAL)*N*N);
    REAL *B = malloc(sizeof(REAL)*N*N);
    REAL *C_serial = malloc(sizeof(REAL)*N*N);
    REAL *C = malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i, j;
    int num_runs = 10;

    elapsed_serial = read_timer();
    for (i=0; i<num_runs; i++)
        matmul_serial(N, A, B, C_serial);
    elapsed_serial = (read_timer() - elapsed_serial)/num_runs;
    /* you should add the call to each function and time the execution */

    elapsed_cuda = read_timer();
    for (i=0; i<num_runs; i++)
        mm_kernel(A, B, C, N, 0);
    elapsed_cuda = (read_timer() - elapsed_cuda)/num_runs;
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fabs(C[i * N + j] - C_serial[i * N + j]) > ALLOWED_DIFF) {
                printf("C[%d][%d]: %g, C_serial[%d][%d]: %g\n", i, j, C[i * N + j], i, j, C_serial[i * N + j]);
                break;
            }
        }
    };

    printf("======================================================================================================\n");
    printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_serial:\t\t%4f\t%4f\n", elapsed_serial * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_serial)));
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_cuda:\t\t%4f\t%4f\n", elapsed_cuda * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda)));
    
    double error = check(C_serial,C, N);
    printf("error:%g\n", error);
    
    return 0;
}


